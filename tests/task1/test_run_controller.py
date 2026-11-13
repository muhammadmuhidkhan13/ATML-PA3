"""CPU-only controller tests. Never load or train Qwen."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from atml_pa3.task1.run_controller import run_training
from atml_pa3.task1.train_core import make_adamw, make_scheduler


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 6)
        self.head = torch.nn.Linear(6, 12)

    def forward(
        self, input_ids, attention_mask,
        labels, use_cache=False
    ):
        logits = self.head(self.embedding(input_ids))

        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, 12),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        return SimpleNamespace(loss=loss)


def create_data(path: Path, n=96):
    with path.open("w", encoding="utf-8") as stream:
        for i in range(n):
            row = {
                "source_index": i,
                "input_ids": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1],
                "labels": [-100, -100, 3, 4],
            }
            stream.write(json.dumps(row) + "\n")


def components(seed):
    torch.manual_seed(seed)
    model = TinyModel()
    optimizer = make_adamw(model, 0.002)
    scheduler = make_scheduler(
        optimizer, total_updates=3, warmup=1
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=False
    )
    return model, optimizer, scheduler, scaler


def run(path, folder, initial_seed, **kwargs):
    model, optimizer, scheduler, scaler = components(
        initial_seed
    )

    report = run_training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        data_path=path,
        run_dir=folder,
        run_spec={"test": "cpu_control"},
        pad_token_id=0,
        device=torch.device("cpu"),
        total_examples=96,
        checkpoint_steps=(1, 2, 3),
        **kwargs,
    )

    return report, model, optimizer, scheduler


def test_uninterrupted_equals_interrupted_then_resumed(tmp_path):
    data = tmp_path / "data.jsonl"
    create_data(data)

    uninterrupted, model1, opt1, sched1 = run(
        data, tmp_path / "uninterrupted", 6304
    )

    partial, _, _, _ = run(
        data, tmp_path / "partial", 6304,
        stop_after_update=1,
    )

    assert partial["completed_updates"] == 1
    assert partial["examples_consumed"] == 32

    resume_path = (
        tmp_path / "partial"
        / "checkpoints"
        / "update_00001"
    )

    final, model3, opt3, sched3 = run(
        data,
        tmp_path / "partial",
        88,
        resume_from=resume_path,
    )

    assert final["completed_updates"] == 3
    assert final["examples_consumed"] == 96
    assert final["skipped_attempts"] == 0
    assert uninterrupted["completed_updates"] == 3

    for key in model1.state_dict():
        torch.testing.assert_close(
            model1.state_dict()[key],
            model3.state_dict()[key],
        )

    assert sched1.state_dict() == sched3.state_dict()

    for i in opt1.state_dict()["state"]:
        for key in opt1.state_dict()["state"][i]:
            a = opt1.state_dict()["state"][i][key]
            b = opt3.state_dict()["state"][i][key]

            if torch.is_tensor(a):
                torch.testing.assert_close(a, b)

    assert (
        tmp_path / "partial"
        / "checkpoints"
        / "update_00003"
        / "state.pt"
    ).is_file()


def test_invalid_budget_or_official_config_rejected(tmp_path):
    data = tmp_path / "fake.jsonl"
    create_data(data)

    model, optimizer, scheduler, scaler = components(6304)

    common = dict(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        data_path=data,
        run_dir=tmp_path / "run",
        run_spec={"method": "full"},
        pad_token_id=0,
        device="cpu",
    )

    with pytest.raises(ValueError, match="32-example"):
        run_training(
            **common,
            total_examples=95,
            checkpoint_steps=(1, 2),
        )

    with pytest.raises(ValueError, match="official|Official"):
        run_training(**common)

    with pytest.raises(ValueError, match="increasing"):
        run_training(
            **common,
            total_examples=96,
            checkpoint_steps=(2, 1, 3),
        )

    with pytest.raises(ValueError, match="early stop"):
        run_training(
            **common,
            total_examples=96,
            checkpoint_steps=(1, 2, 3),
            stop_after_update=4,
        )


def test_new_run_cannot_overwrite_attempt_history(tmp_path):
    data = tmp_path / "data.jsonl"
    create_data(data)

    report, *_ = run(
        data, tmp_path / "run", 6304,
        stop_after_update=1,
    )

    assert report["completed_updates"] == 1

    with pytest.raises(FileExistsError, match="Attempt log"):
        run(data, tmp_path / "run", 6304)


def test_skip_retries_same_data_and_logs(
    monkeypatch, tmp_path
):
    data = tmp_path / "data.jsonl"
    create_data(data, 32)

    from atml_pa3.task1 import run_controller as mod

    received = []

    def fake_attempt(
        model, optimizer, scheduler, scaler,
        microbatches, pad_token_id, device
    ):
        received.append([
            r["source_index"]
            for part in microbatches
            for r in part
        ])

        ok = len(received) > 1

        return {
            "optimizer_step_completed": ok,
            "sft_loss": 1.0,
            "supervised_tokens": 64,
            "grad_norm_before_clip": 1.0,
            "scale_before": 32.0,
            "scale_after": 32.0 if ok else 16.0,
        }

    monkeypatch.setattr(
        mod, "train_effective_batch", fake_attempt
    )

    model, optimizer, scheduler, scaler = components(6304)

    report = run_training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        data_path=data,
        run_dir=tmp_path / "run",
        run_spec={"test": "skip"},
        pad_token_id=0,
        device="cpu",
        total_examples=32,
        checkpoint_steps=(1,),
    )

    assert len(received) == 2
    assert received[0] == received[1] == list(range(32))
    assert report["completed_updates"] == 1
    assert report["skipped_attempts"] == 1

    attempts = [
        json.loads(x)
        for x in (
            tmp_path / "run" / "attempts.jsonl"
        ).read_text().splitlines()
    ]

    assert [
        x["completed"]
        for x in attempts
        if x["event"] == "optimizer_attempt"
    ] == [False, True]
