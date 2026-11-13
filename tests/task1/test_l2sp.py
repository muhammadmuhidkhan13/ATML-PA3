"""CPU regression tests for exact L2-SP training."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from atml_pa3.task1.l2sp import L2SPRegularizer
from atml_pa3.task1.train_core import (
    collate_microbatch,
    train_effective_batch,
    make_scheduler,
    make_adamw,
)
from atml_pa3.task1.run_controller import (
    run_training,
    FROZEN_DATA_SHA256,
    MODEL_SHA,
)


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 6)
        self.head = torch.nn.Linear(6, 10)

    def forward(
        self, input_ids, attention_mask,
        labels, use_cache=False
    ):
        logits = self.head(self.embed(input_ids))
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 10),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss)


def row(i):
    return {
        "source_index": i,
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "labels": [-100, -100, 3, 4],
    }


def parts(start=0):
    return [
        [row(start + 2*i), row(start + 2*i + 1)]
        for i in range(16)
    ]


def no_scaler():
    return torch.amp.GradScaler("cuda", enabled=False)


def test_tied_parameters_and_all_parameter_types():
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(3, 3)
            self.b = torch.nn.Linear(3, 3)
            self.b.weight = self.a.weight
            self.norm = torch.nn.LayerNorm(3)

    model = Tied()
    anchor = L2SPRegularizer(model)

    unique = list(
        model.named_parameters(remove_duplicate=True)
    )

    assert len(unique) == 5
    assert anchor.reference_parameters == sum(
        p.numel() for _, p in unique
    )
    assert anchor.reference_bytes == (
        4 * anchor.reference_parameters
    )

    original = [
        p.detach().clone()
        for _, p in unique
    ]

    with torch.no_grad():
        for _, p in unique:
            p.add_(0.25)

    omega = anchor.add_to_unscaled_grad()

    expected = 0.5 * sum(
        float((p.detach() - ref).square().sum())
        for (_, p), ref in zip(unique, original)
    )

    assert omega == pytest.approx(expected)

    for _, p in unique:
        torch.testing.assert_close(
            p.grad,
            torch.full_like(p, 0.25e-3),
        )


def test_regularizer_gradients_match_autograd():
    torch.manual_seed(6304)

    model = TinyLM()
    anchor = L2SPRegularizer(model)
    original = [
        p.detach().clone()
        for p in model.parameters()
    ]

    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02)

    x = torch.tensor([[1, 2, 3, 4]])

    batch = {
        "input_ids": x,
        "attention_mask": torch.ones_like(x),
        "labels": torch.tensor([[-100, -100, 3, 4]]),
    }

    model(**batch).loss.backward()
    omega = anchor.add_to_unscaled_grad()

    actual = [
        p.grad.detach().clone()
        for p in model.parameters()
    ]

    reference = copy.deepcopy(model)
    reference.zero_grad(set_to_none=True)

    penalty = 0.5 * sum(
        (p - ref).square().sum()
        for p, ref in zip(reference.parameters(), original)
    )

    (reference(**batch).loss + 1e-3 * penalty).backward()

    assert omega == pytest.approx(
        penalty.item(), rel=1e-6
    )

    for result, p in zip(
        actual, reference.parameters()
    ):
        torch.testing.assert_close(
            result, p.grad,
            atol=1e-6, rtol=1e-5,
        )


def test_effective_batch_matches_full_objective():
    torch.manual_seed(6304)

    base = TinyLM()
    measured = copy.deepcopy(base)
    reference = copy.deepcopy(base)

    anchor = L2SPRegularizer(measured)

    original = [
        p.detach().clone()
        for p in reference.parameters()
    ]

    with torch.no_grad():
        measured.head.weight.add_(0.3)
        reference.head.weight.add_(0.3)

    optimizer = torch.optim.SGD(
        measured.parameters(), lr=0.01
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda _: 1
    )

    result = train_effective_batch(
        measured,
        optimizer,
        scheduler,
        no_scaler(),
        parts(),
        0,
        torch.device("cpu"),
        regularizer=anchor,
    )

    ref_optimizer = torch.optim.SGD(
        reference.parameters(), lr=0.01
    )

    full = collate_microbatch(
        [r for group in parts() for r in group],
        0,
        torch.device("cpu"),
    )

    omega = 0.5 * sum(
        (p - ref).square().sum()
        for p, ref in zip(reference.parameters(), original)
    )

    loss = reference(**full).loss + 1e-3 * omega
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        reference.parameters(), 1.0
    )

    ref_optimizer.step()

    assert result["omega"] == pytest.approx(
        omega.item()
    )
    assert result["lambda_omega"] == pytest.approx(
        1e-3 * omega.item()
    )
    assert result["total_loss"] == pytest.approx(
        result["sft_loss"] + result["lambda_omega"]
    )

    for actual, expected in zip(
        measured.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            actual, expected,
            rtol=1e-5, atol=1e-6,
        )


def test_controller_logging_and_resume(tmp_path):
    data = tmp_path / "data.jsonl"

    data.write_text(
        "".join(json.dumps(row(i)) + "\n" for i in range(64)),
        encoding="utf-8",
    )

    def components():
        torch.manual_seed(6304)
        model = TinyLM()
        anchor = L2SPRegularizer(model)
        opt = make_adamw(model, 2e-5)
        sched = make_scheduler(
            opt, total_updates=2, warmup=1
        )
        return model, anchor, opt, sched, no_scaler()

    def run(folder, resume=None, stop=None):
        model, anchor, opt, sched, scaler = components()

        report = run_training(
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            data_path=data,
            run_dir=folder,
            run_spec={"test": "l2sp"},
            pad_token_id=0,
            device="cpu",
            total_examples=64,
            checkpoint_steps=(1, 2),
            stop_after_update=stop,
            resume_from=resume,
            regularizer=anchor,
        )

        return model, report

    expected, _ = run(tmp_path / "continuous")

    _, partial = run(
        tmp_path / "resume", stop=1
    )

    restored, finished = run(
        tmp_path / "resume",
        resume=(
            tmp_path / "resume"
            / "checkpoints" / "update_00001"
        ),
    )

    assert partial["completed_updates"] == 1
    assert finished["completed_updates"] == 2

    for a, b in zip(
        expected.parameters(), restored.parameters()
    ):
        torch.testing.assert_close(
            a, b, atol=0, rtol=0
        )

    events = [
        json.loads(s)
        for s in (
            tmp_path / "resume" / "attempts.jsonl"
        ).read_text().splitlines()
    ]

    attempts = [
        e for e in events
        if e["event"] == "optimizer_attempt"
    ]

    assert len(attempts) == 2
    assert all(e["omega"] >= 0 for e in attempts)
    assert all(e["lambda_omega"] >= 0 for e in attempts)
    assert all(
        e["total_loss"] == pytest.approx(
            e["sft_loss"] + e["lambda_omega"]
        )
        for e in attempts
    )


def test_official_run_requires_anchor(tmp_path):
    model = TinyLM()
    optimizer = make_adamw(model, 2e-5)
    scheduler = make_scheduler(optimizer)

    spec = {
        "method": "l2sp",
        "seed": 6304,
        "dataset_sha256": FROZEN_DATA_SHA256,
        "model_revision": MODEL_SHA,
        "total_examples": 40000,
        "l2sp_lambda": 1e-3,
    }

    with pytest.raises(
        ValueError, match="Official L2-SP"
    ):
        run_training(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=no_scaler(),
            data_path=tmp_path / "missing.jsonl",
            run_dir=tmp_path / "unused",
            run_spec=spec,
            pad_token_id=0,
            device="cpu",
        )


def test_invalid_reference_rejected():
    model = torch.nn.Linear(2, 2)

    with pytest.raises(
        ValueError, match="coefficient"
    ):
        L2SPRegularizer(model, 0)

    anchor = L2SPRegularizer(model)

    with pytest.raises(
        ValueError, match="different model"
    ):
        anchor.assert_matches(
            torch.nn.Linear(2, 2)
        )

    model.weight.requires_grad_(False)

    with pytest.raises(
        ValueError, match="all FP32 trainable"
    ):
        L2SPRegularizer(model)
