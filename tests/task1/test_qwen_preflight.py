"""CPU-only checks for Qwen smoke-test helpers."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from atml_pa3.task1.model_setup import AdapterLinear
from atml_pa3.task1.qwen_preflight import (
    check_label_roundtrip,
    select_short_examples,
    single_backward_probe,
)


def example(i):
    return {
        "source_index": i,
        "input_ids": [1, 2, 3, 4],
        "labels": [-100, -100, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "source": {"response": "3,4"},
    }


def test_sample_source_order_no_truncation(tmp_path):
    path = tmp_path / "examples.jsonl"
    long_row = example(0)

    long_row["input_ids"] += [5] * 40
    long_row["labels"] += [-100] * 40
    long_row["attention_mask"] += [1] * 40

    rows = [
        long_row, example(1), example(2), example(3)
    ]

    path.write_text(
        "".join(
            json.dumps(x) + "\n" for x in rows
        ),
        encoding="utf-8",
    )

    selected = select_short_examples(
        path, max_tokens=32, count=2
    )

    assert [
        r["source_index"] for r in selected
    ] == [1, 2]

    assert (
        selected[0]["input_ids"]
        == rows[1]["input_ids"]
    )


def test_sample_explicit_length_band(tmp_path):
    path = tmp_path / "examples.jsonl"
    rows = [example(i) for i in range(5)]
    for row, length in zip(rows, (40, 300, 520, 700, 900)):
        extra = length - len(row["input_ids"])
        row["input_ids"] += [5] * extra
        row["labels"] += [-100] * extra
        row["attention_mask"] += [1] * extra
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    selected = select_short_examples(
        path, max_tokens=800, count=2, min_tokens=500
    )
    assert [row["source_index"] for row in selected] == [2, 3]
    assert [len(row["input_ids"]) for row in selected] == [520, 700]


def test_sample_rejects_invalid_counts(tmp_path):
    path = tmp_path / "samples.jsonl"

    path.write_text(
        json.dumps(example(0)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="Not enough"
    ):
        select_short_examples(
            path, max_tokens=32, count=2
        )

    with pytest.raises(
        ValueError, match="min_tokens"
    ):
        select_short_examples(
            path, max_tokens=16, count=1
        )

    with pytest.raises(
        ValueError, match="min_tokens"
    ):
        select_short_examples(
            path, max_tokens=128, count=1, min_tokens=129
        )


def test_label_roundtrip_detection():
    class FakeTokenizer:
        def decode(self, ids, **kwargs):
            return "".join(
                {
                    3: "3,4",
                    4: "<|im_end|>",
                }[x]
                for x in ids
            )

    row = example(0)

    check_label_roundtrip(
        [row], FakeTokenizer()
    )

    row["source"]["response"] = "wrong"

    with pytest.raises(
        ValueError, match="exact response"
    ):
        check_label_roundtrip(
            [row], FakeTokenizer()
        )


class TinyAdapterLM(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = torch.nn.Embedding(
            12, 8
        ).half()

        self.adapter = AdapterLinear(
            torch.nn.Linear(8, 8).half(),
            4, 8, 6304,
        )

        self.head = torch.nn.Linear(
            8, 12
        ).half()

        for module in (
            self.embed, self.head
        ):
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(
        self, input_ids, attention_mask,
        labels, use_cache=False
    ):
        hidden = self.adapter(
            self.embed(input_ids)
        )

        logits = self.head(hidden).float()

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 12),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        return SimpleNamespace(loss=loss)


def test_backward_without_optimizer_update():
    torch.manual_seed(6304)

    model = TinyAdapterLM()
    before = copy.deepcopy(model.state_dict())

    result = single_backward_probe(
        model,
        [example(0), example(1)],
        0,
        "cpu",
    )

    assert result["optimizer_steps"] == 0
    assert result["adapter_modules"] == 1
    assert result["valid_response_targets"] == 4
    assert result["nonzero_B_gradients"] == 1
    assert result["gradient_tensors"] == 2

    for name, weight in model.state_dict().items():
        torch.testing.assert_close(
            weight,
            before[name],
            atol=0,
            rtol=0,
        )
