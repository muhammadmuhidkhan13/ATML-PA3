from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from atml_pa3.task2.model import VisualLanguageModel, encode_example, example_weighted_loss


class Tokenizer:
    all_special_ids = [1]
    def encode(self, text, **kwargs):
        ids = []
        for i, piece in enumerate(text.split("<|im_end|>")):
            if i:
                ids.append(1)
            ids.extend(ord(c)+2 for c in piece)
        return ids
    def convert_tokens_to_ids(self, text):
        return 1
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "USER\n"+messages[0]["content"]+"<|im_end|>\nASSISTANT\n<think>\n\n</think>\n\n"+messages[1]["content"]+"<|im_end|>\n"


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 8)
        self.head = nn.Linear(8, 256)
    def get_input_embeddings(self):
        return self.embedding
    def forward(self, inputs_embeds, **kwargs):
        # Causal cumulative mixing lets visual-prefix inputs affect answer logits.
        return SimpleNamespace(logits=self.head(inputs_embeds.cumsum(dim=1)))


def fixture():
    torch.manual_seed(6304)
    model = VisualLanguageModel(TinyLM(), vision_size=3, hidden_size=8, visual_tokens=4)
    return model, Tokenizer()


@pytest.mark.parametrize("caption,cap", [(False, 16), (True, 64)])
def test_caps_eot_and_exact_masking(caption, cap):
    example = encode_example(Tokenizer(), "q"*90, "a"*100, caption=caption)
    assert example.question_tokens == 80 and example.target_tokens == cap
    assert example.question_truncated and example.target_truncated
    supervised = [v for v in example.labels if v != -100]
    assert supervised == [ord("a")+2]*cap+[1]


def test_image_positions_padding_and_no_image():
    model, tokenizer = fixture()
    a = encode_example(tokenizer, "Question", "answer")
    b = encode_example(tokenizer, "Q", "a")
    batch = model.build_inputs([a,b], [torch.ones(4,3), None])
    start, end = batch["visual_spans"][0]
    assert end-start == 4 and start == a.visual_position
    assert batch["labels"][0,start:end].tolist() == [-100]*4
    assert batch["visual_spans"][1] is None
    assert batch["attention_mask"].sum(1).tolist() == [len(a.input_ids)+4, len(b.input_ids)]
    assert batch["position_ids"][0,:len(a.input_ids)+4].tolist() == list(range(len(a.input_ids)+4))
    assert (batch["labels"][1,len(b.input_ids):] == -100).all()
    no_image = model.build_inputs([a], [None])
    assert no_image["labels"][0].tolist() == a.labels
    assert no_image["inputs_embeds"].shape[1] == batch["inputs_embeds"].shape[1]-4


def test_frozen_lm_still_propagates_connector_gradients():
    model, tokenizer = fixture()
    example = encode_example(tokenizer, "Question", "answer")
    result = model([example], [torch.randn(4,3)])
    result["loss"].backward()
    assert all(p.grad is None and not p.requires_grad for p in model.language_model.parameters())
    assert all(p.dtype == torch.float32 and p.grad is not None and torch.isfinite(p.grad).all()
               and torch.count_nonzero(p.grad) for p in model.connector.parameters())


def test_loss_is_example_weighted_not_token_weighted():
    logits = torch.tensor([[[4.,0.],[0.,0.],[0.,0.],[0.,0.]],
                           [[4.,0.],[4.,0.],[4.,0.],[0.,0.]]], requires_grad=True)
    labels = torch.tensor([[-100,0,-100,-100],[-100,1,1,1]])
    loss, per = example_weighted_loss(logits, labels)
    a = F.cross_entropy(logits[0,:1], labels[0,1:2])
    b = F.cross_entropy(logits[1,:3], labels[1,1:])
    torch.testing.assert_close(loss, (a+b)/2)
    assert not torch.isclose(loss, (a+3*b)/4)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_calibration_uses_train_only_and_matches_rms():
    model, _ = fixture()
    features = [torch.randn(4,3), torch.randn(4,3)]
    ids = [5,7,9]
    info = model.calibrate(features, ids, split="train", image_ids=[1,2])
    visual = torch.stack([model.connector(x)*model.visual_scale for x in features])
    expected = model.language_model.get_input_embeddings()(torch.tensor(ids)).square().mean().sqrt()
    torch.testing.assert_close(visual.square().mean().sqrt(), expected)
    assert info["image_ids"] == [1,2]
    with pytest.raises(ValueError, match="training"):
        model.calibrate(features, ids, split="dev", image_ids=[1,2])


def test_invalid_features_empty_targets_and_empty_loss():
    model, tokenizer = fixture()
    with pytest.raises(ValueError, match="Empty"):
        encode_example(tokenizer, "q", "")
    example = encode_example(tokenizer, "q", "a")
    with pytest.raises(ValueError, match="dimensions"):
        model([example], [torch.ones(3,3)])
    with pytest.raises(ValueError, match="supervised"):
        example_weighted_loss(torch.randn(1,3,2), torch.full((1,3), -100))
