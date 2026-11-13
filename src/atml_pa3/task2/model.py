"""Frozen Qwen with continuous visual insertion and example-weighted SFT loss."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

QWEN_ID = "Qwen/Qwen3-0.6B"
QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
IGNORE = -100


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    visual_position: int
    question_tokens: int
    target_tokens: int
    question_truncated: bool
    target_truncated: bool


def encode_example(tokenizer, question, target, *, caption=False):
    """Token caps apply to content only; the assistant EOT is always supervised.

    Tokenize content and template segments independently to guarantee exact
    boundaries. The Qwen non-thinking template supplies all role/scaffold text.
    """
    qm, am = "TASK2_QUESTION_SENTINEL_6304", "TASK2_ANSWER_SENTINEL_6304"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": qm}, {"role": "assistant", "content": am}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False)
    if rendered.count(qm) != 1 or rendered.count(am) != 1:
        raise ValueError("Unexpected Qwen chat template")
    prefix, rest = rendered.split(qm)
    middle, suffix = rest.split(am)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    qraw, araw = encode(question), encode(target)
    cap = 64 if caption else 16
    qids, aids = qraw[:80], araw[:cap]
    if not aids:
        raise ValueError("Empty supervised answer")
    # Content cannot inject role delimiters or premature end-of-turn markers.
    specials = set(tokenizer.all_special_ids)
    if specials.intersection(qraw + araw):
        raise ValueError("Source content contains a reserved chat token")
    pids, mids, suffix_ids = encode(prefix), encode(middle), encode(suffix)
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not suffix_ids or suffix_ids[0] != eot:
        raise ValueError("Expected assistant end-of-turn immediately after target")
    ids = pids + qids + mids + aids + suffix_ids
    labels = [IGNORE] * (len(pids)+len(qids)+len(mids)) + aids + [eot] + [IGNORE]*(len(suffix_ids)-1)
    return EncodedExample(ids, labels, len(pids), len(qids), len(aids), len(qraw)>80, len(araw)>cap)


def example_weighted_loss(logits, labels):
    """Mean valid causal token loss within each example, then across examples."""
    shifted = labels[:, 1:]
    mask = shifted != IGNORE
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Every example must contain supervised causal targets")
    valid_logits = logits[:, :-1][mask].float()
    losses = F.cross_entropy(valid_logits, shifted[mask], reduction="none")
    batch_indices = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(mask)[mask]
    sums = torch.zeros(labels.shape[0], device=logits.device, dtype=losses.dtype)
    sums.scatter_add_(0, batch_indices, losses)
    per_example = sums / counts
    return per_example.mean(), per_example


class VisualLanguageModel(nn.Module):
    """Part 2 can inject Qwen LoRA after construction; the connector API stays fixed."""
    def __init__(self, language_model, *, vision_size=768, hidden_size=1024, visual_tokens=196):
        super().__init__()
        self.language_model = language_model
        language_model.requires_grad_(False)
        self.visual_tokens = visual_tokens
        if language_model.get_input_embeddings().weight.shape[1] != hidden_size:
            raise ValueError("Connector/Qwen embedding dimensions disagree")
        self.connector = nn.Sequential(nn.Linear(vision_size, hidden_size), nn.GELU(),
                                       nn.Linear(hidden_size, hidden_size)).float()
        self.register_buffer("visual_scale", torch.tensor(1.0, dtype=torch.float32))
        self.vision_size = vision_size

    def build_inputs(self, examples, features):
        if not examples or len(examples) != len(features):
            raise ValueError("One optional feature tensor is required per example")
        embedding = self.language_model.get_input_embeddings()
        device = embedding.weight.device
        sequences, all_labels, spans = [], [], []
        for example, feature in zip(examples, features):
            ids = torch.tensor(example.input_ids, dtype=torch.long, device=device)
            text = embedding(ids)
            labels = torch.tensor(example.labels, dtype=torch.long, device=device)
            if feature is not None:
                if tuple(feature.shape) != (self.visual_tokens, self.vision_size):
                    raise ValueError("Unexpected visual feature dimensions")
                visual = self.connector(feature.to(device=device, dtype=torch.float32)) * self.visual_scale
                visual = visual.to(dtype=text.dtype)
                position = example.visual_position
                text = torch.cat([text[:position], visual, text[position:]], dim=0)
                labels = torch.cat([labels[:position], torch.full((self.visual_tokens,), IGNORE, device=device), labels[position:]])
                spans.append((position, position+self.visual_tokens))
            else:
                spans.append(None)
            sequences.append(text)
            all_labels.append(labels)
        length = max(x.shape[0] for x in sequences)
        embeds = torch.stack([F.pad(x, (0, 0, 0, length-x.shape[0])) for x in sequences])
        labels = torch.stack([F.pad(x, (0, length-x.shape[0]), value=IGNORE) for x in all_labels])
        mask = torch.stack([torch.arange(length, device=device)<x.shape[0] for x in sequences]).long()
        positions = (mask.cumsum(dim=1)-1).clamp_min(0)
        return {"inputs_embeds": embeds, "labels": labels, "attention_mask": mask,
                "position_ids": positions, "visual_spans": spans}

    def forward(self, examples, features):
        batch = self.build_inputs(examples, features)
        # Frozen weights still participate in autograd for visual-input gradients.
        output = self.language_model(inputs_embeds=batch["inputs_embeds"],
                                     attention_mask=batch["attention_mask"], position_ids=batch["position_ids"],
                                     use_cache=False)
        loss, per_example = example_weighted_loss(output.logits, batch["labels"])
        return {"loss": loss, "per_example_loss": per_example, "logits": output.logits,
                **{k: v for k, v in batch.items() if k != "inputs_embeds"}}

    @torch.no_grad()
    def calibrate(self, features, text_token_ids, *, split, image_ids):
        if split != "train" or len(features) != len(image_ids) or not features or not text_token_ids:
            raise ValueError("Calibration requires identified training images/text only")
        embedding = self.language_model.get_input_embeddings()
        device = embedding.weight.device
        text = embedding(torch.tensor(text_token_ids, device=device)).float()
        text_rms = text.square().mean().sqrt()
        total, count = torch.zeros((), device=device), 0
        with torch.autocast(device.type, enabled=False):
            for feature in features:
                visual = self.connector(feature.to(device=device, dtype=torch.float32))
                total += visual.square().sum()
                count += visual.numel()
        visual_rms = (total/count).sqrt()
        scale = text_rms / visual_rms.clamp_min(1e-12)
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid connector calibration scale")
        self.visual_scale.copy_(scale)
        return {"split": split, "image_ids": list(image_ids), "text_token_count": len(text_token_ids),
                "text_token_sha256": hashlib.sha256(str(text_token_ids).encode()).hexdigest(),
                "rule": "match global RMS of unscaled connector outputs to training content-token embeddings",
                "text_rms": text_rms.item(), "unscaled_visual_rms": visual_rms.item(), "scale": scale.item()}


def load_vlm(device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION,
                                               local_files_only=True, use_fast=True)
    language_model = AutoModelForCausalLM.from_pretrained(QWEN_ID, revision=QWEN_REVISION,
                                                        local_files_only=True, dtype=torch.float16,
                                                        attn_implementation="eager").to(device)
    language_model.config.use_cache = False
    language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.manual_seed(6304)
    model = VisualLanguageModel(language_model)
    model.connector.to(device=device, dtype=torch.float32)
    model.visual_scale = model.visual_scale.to(device)
    return model, tokenizer
