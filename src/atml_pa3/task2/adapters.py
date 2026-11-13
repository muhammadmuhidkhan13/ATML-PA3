"""Task 2 attention-only LoRA; no dependency on Task 1 code."""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from .curriculum import Recipe, lr_factor

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class AttentionLoRA(nn.Module):
    def __init__(self, base, seed, rank=16, alpha=32):
        super().__init__()
        if base.weight.dtype != torch.float16:
            raise ValueError("LoRA requires a frozen FP16 base")
        self.base = base.requires_grad_(False)
        generator = torch.Generator().manual_seed(seed)
        a = torch.empty(rank, base.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        self.lora_A = nn.Parameter(a.to(base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        self.dropout = nn.Dropout(0.05)
        self.scale = alpha/rank

    def forward(self, x):
        original = self.base(x)
        delta = F.linear(F.linear(self.dropout(x).float(), self.lora_A), self.lora_B) * self.scale
        return original + delta.to(original.dtype)


def attach_attention_lora(model, *, expected_layers=28):
    layers = model.language_model.model.layers
    if len(layers) != expected_layers:
        raise ValueError("Unexpected Qwen layer count")
    existing = [m for m in model.language_model.modules() if isinstance(m, AttentionLoRA)]
    if existing:
        if len(existing) != expected_layers*4:
            raise ValueError("Partial LoRA installation")
        return
    for i, layer in enumerate(layers):
        for j, name in enumerate(TARGETS):
            base = getattr(layer.self_attn, name)
            if not isinstance(base, nn.Linear):
                raise TypeError("Expected original attention linear")
            setattr(layer.self_attn, name, AttentionLoRA(base, 6304+i*4+j))


def assert_trainables(model, stage):
    connector = set(map(id, model.connector.parameters()))
    adapters = {id(p) for name, p in model.language_model.named_parameters() if name.endswith((".lora_A", ".lora_B"))}
    expected = connector | (adapters if stage >= 2 else set())
    if stage >= 2 and not adapters:
        raise AssertionError("Stage 2/3 requires LoRA")
    for name, p in model.named_parameters():
        if p.requires_grad != (id(p) in expected):
            raise AssertionError(f"Incorrect trainability: {name}")
        if p.requires_grad and p.dtype != torch.float32:
            raise AssertionError(f"Trainable parameter not FP32: {name}")
        if id(p) not in expected and (p.dtype != torch.float16 or p.grad is not None):
            raise AssertionError(f"Backbone precision/gradient violation: {name}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_stage_optimizer(model, stage, recipe=Recipe()):
    assert_trainables(model, stage)
    groups = [{"params": list(model.connector.parameters()), "lr": recipe.connector_lrs[stage-1], "name": "connector"}]
    if stage >= 2:
        groups.append({"params": [p for name, p in model.language_model.named_parameters()
                                  if name.endswith((".lora_A", ".lora_B"))],
                       "lr": recipe.adapter_lrs[stage-1], "name": "lora"})
    optimizer = torch.optim.AdamW(groups, weight_decay=recipe.weight_decay, betas=(.9,.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda completed: lr_factor(completed+1, recipe.budgets[stage-1]))
    return optimizer, scheduler


def backbone_hash(model):
    """Canonical names survive adapter wrapping; never store the frozen backbone."""
    digest = hashlib.sha256()
    parameters = [(name.replace(".base.", "."), p) for name, p in model.language_model.named_parameters()
                  if not name.endswith((".lora_A", ".lora_B"))]
    for name, p in sorted(parameters):
        digest.update(name.encode())
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
