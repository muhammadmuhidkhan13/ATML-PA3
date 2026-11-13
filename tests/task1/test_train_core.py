"""CPU-only regression tests for the Task 1 training core."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from atml_pa3.task1.train_core import (
    ACCUMULATION_STEPS,
    collate_microbatch,
    iter_effective_batches,
    lr_multiplier,
    make_adamw,
    make_scheduler,
    supervised_token_count,
    train_effective_batch,
    verify_dataset,
)


def record(index, labels):
    ids = [1, 2, 3, 4, 5, 6][:len(labels)]
    return {
        "source_index": index,
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
    }


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 5)
        self.output = torch.nn.Linear(5, 10)

    def forward(
        self, input_ids, attention_mask,
        labels, use_cache=False
    ):
        logits = self.output(self.embedding(input_ids))
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 10),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss)


def test_padding_and_shifted_label_count():
    a = record(0, [-100, -100, 3, 4])
    b = record(1, [-100, 2, 3, 4, 5, -100])

    batch = collate_microbatch(
        [a, b], 0, torch.device("cpu")
    )

    assert batch["labels"].shape == (2, 6)
    assert batch["labels"][0].tolist() == [
        -100, -100, 3, 4, -100, -100
    ]
    assert batch["attention_mask"][0].tolist() == [
        1, 1, 1, 1, 0, 0
    ]
    assert supervised_token_count(batch) == 6


def test_streaming_source_order_and_extra_rows(tmp_path):
    path = tmp_path / "examples.jsonl"
    rows = [
        record(i, [-100, -100, 3, 4])
        for i in range(4)
    ]

    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8",
    )

    groups = list(
        iter_effective_batches(
            path,
            count=4,
            microbatch_size=1,
            accumulation_steps=2,
        )
    )

    actual = [
        [
            [r["source_index"] for r in part]
            for part in group
        ]
        for group in groups
    ]

    assert actual == [
        [[0], [1]],
        [[2], [3]],
    ]

    with pytest.raises(ValueError, match="more rows"):
        list(
            iter_effective_batches(
                path, count=2,
                microbatch_size=1,
                accumulation_steps=2,
            )
        )

    rows[2]["source_index"] = 0
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="strictly increasing"
    ):
        list(
            iter_effective_batches(
                path, count=4,
                microbatch_size=1,
                accumulation_steps=2,
            )
        )


def test_weight_decay_and_schedule():
    model = torch.nn.Linear(4, 2)
    optimizer = make_adamw(model, 2e-5)

    assert {
        group["weight_decay"]
        for group in optimizer.param_groups
    } == {0.0, 0.01}

    assert lr_multiplier(1) == pytest.approx(1 / 38)
    assert lr_multiplier(38) == pytest.approx(1)
    assert lr_multiplier(1250) == pytest.approx(0)

    make_scheduler(optimizer)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        2e-5 / 38
    )


def test_accumulation_matches_full_batch():
    torch.manual_seed(6304)

    base = TinyCausalLM()
    measured = copy.deepcopy(base)
    reference = copy.deepcopy(base)

    part = [
        record(0, [-100, -100, 3, 4]),
        record(1, [-100, 2, 3, 4, 5, -100]),
    ]

    parts = [
        [
            dict(
                r,
                source_index=r["source_index"] + 2 * i,
            )
            for r in part
        ]
        for i in range(16)
    ]

    optimizer = torch.optim.SGD(
        measured.parameters(), lr=0.025
    )
    reference_optimizer = torch.optim.SGD(
        reference.parameters(), lr=0.025
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda _: 1.0
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=False
    )

    result = train_effective_batch(
        measured,
        optimizer,
        scheduler,
        scaler,
        parts,
        0,
        torch.device("cpu"),
    )

    combined = collate_microbatch(
        [r for group in parts for r in group],
        0,
        torch.device("cpu"),
    )

    reference_optimizer.zero_grad(set_to_none=True)
    reference(**combined).loss.backward()

    torch.nn.utils.clip_grad_norm_(
        reference.parameters(), 1.0
    )
    reference_optimizer.step()

    assert result["optimizer_step_completed"] is True
    assert result["supervised_tokens"] == 16 * 6

    for actual, expected in zip(
        measured.parameters(),
        reference.parameters(),
    ):
        torch.testing.assert_close(
            actual, expected,
            rtol=1e-5, atol=1e-6,
        )


def test_dataset_hash_lock(tmp_path):
    (tmp_path / "examples.jsonl").write_bytes(b"test\n")
    sha = hashlib.sha256(b"test\n").hexdigest()

    metadata = {
        "mode": "full_build",
        "seed": 6304,
        "selection": {"accepted": 40000},
        "rules": {
            "shuffle": False,
            "truncate": False,
        },
        "output_sha256": {
            "examples.jsonl": sha,
        },
    }

    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    (tmp_path / "SHA256SUMS").write_text(
        f"{sha}  examples.jsonl\n",
        encoding="utf-8",
    )

    assert verify_dataset(tmp_path, sha).name == (
        "examples.jsonl"
    )

    with pytest.raises(ValueError, match="hash"):
        verify_dataset(tmp_path, "0" * 64)


def test_skipped_update_does_not_advance_scheduler():
    class SkippingScaler:
        def __init__(self):
            self.value = 32.0

        def scale(self, x):
            return x

        def unscale_(self, _):
            pass

        def get_scale(self):
            return self.value

        def step(self, _):
            pass

        def update(self):
            self.value /= 2

        def is_enabled(self):
            return True

    model = TinyCausalLM()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01
    )
    scheduler = make_scheduler(optimizer)
    before = scheduler.last_epoch

    rec = record(0, [-100, -100, 3, 4])
    groups = [
        [rec, rec]
        for _ in range(ACCUMULATION_STEPS)
    ]

    report = train_effective_batch(
        model, optimizer, scheduler,
        SkippingScaler(), groups, 0,
        torch.device("cpu"),
    )

    assert report["optimizer_step_completed"] is False
    assert scheduler.last_epoch == before
