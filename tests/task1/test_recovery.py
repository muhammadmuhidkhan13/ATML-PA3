"""CPU tests for exact checkpoint/resume behavior."""

import copy
import random

import numpy as np
import pytest
import torch

from atml_pa3.task1.recovery import (
    load_checkpoint,
    save_checkpoint,
)
from atml_pa3.task1.train_core import make_scheduler


def make_parts():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(5, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = make_scheduler(
        optimizer, total_updates=8, warmup=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scheduler, scaler


def step(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.randn(3, 3)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


def test_recovery_matches_uninterrupted_training_and_rng(tmp_path):
    torch.manual_seed(6304)
    np.random.seed(6304)
    random.seed(6304)

    model, opt, sched, scaler = make_parts()

    for _ in range(2):
        step(model, opt, sched)

    spec = {
        "method": "full",
        "seed": 6304,
        "data_hash": "abc",
        "lr": 0.01,
    }

    checkpoint = tmp_path / "update_00002"

    saved = save_checkpoint(
        checkpoint,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=scaler,
        run_spec=spec,
        completed_updates=2,
        skipped_updates=1,
    )

    assert saved["next_example_offset"] == 64
    assert (checkpoint / "state.pt").is_file()

    reference_draws = (
        random.random(),
        np.random.rand(),
        torch.rand(1).item(),
    )

    expected_loss = step(model, opt, sched)
    expected_weights = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(opt.state_dict())

    random.seed(77)
    np.random.seed(77)
    torch.manual_seed(77)

    new_model, new_opt, new_sched, new_scaler = make_parts()

    metadata = load_checkpoint(
        checkpoint,
        model=new_model,
        optimizer=new_opt,
        scheduler=new_sched,
        scaler=new_scaler,
        expected_run_spec=spec,
    )

    assert metadata["skipped_updates"] == 1

    actual_draws = (
        random.random(),
        np.random.rand(),
        torch.rand(1).item(),
    )

    assert actual_draws == pytest.approx(reference_draws)

    actual_loss = step(new_model, new_opt, new_sched)

    assert actual_loss == pytest.approx(expected_loss, rel=1e-6)

    for key, weight in expected_weights.items():
        torch.testing.assert_close(
            new_model.state_dict()[key], weight
        )

    assert new_sched.last_epoch == sched.last_epoch

    for key, state in expected_optimizer["state"].items():
        for name, value in state.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(
                    new_opt.state_dict()["state"][key][name],
                    value,
                )


def test_no_overwrite_and_spec_mismatch(tmp_path):
    model, opt, sched, scaler = make_parts()
    spec = {"method": "full", "seed": 6304}
    checkpoint = tmp_path / "cp"

    save_checkpoint(
        checkpoint,
        model=model, optimizer=opt,
        scheduler=sched, scaler=scaler,
        run_spec=spec, completed_updates=0,
    )

    with pytest.raises(FileExistsError):
        save_checkpoint(
            checkpoint,
            model=model, optimizer=opt,
            scheduler=sched, scaler=scaler,
            run_spec=spec, completed_updates=0,
        )

    with pytest.raises(ValueError, match="specification"):
        load_checkpoint(
            checkpoint,
            model=model, optimizer=opt,
            scheduler=sched, scaler=scaler,
            expected_run_spec={"method": "lora"},
        )


def test_corruption_detected(tmp_path):
    model, opt, sched, scaler = make_parts()
    spec = {"method": "full"}
    checkpoint = tmp_path / "cp"

    save_checkpoint(
        checkpoint,
        model=model, optimizer=opt,
        scheduler=sched, scaler=scaler,
        run_spec=spec, completed_updates=1,
    )

    with (checkpoint / "state.pt").open("ab") as output:
        output.write(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(
            checkpoint,
            model=model, optimizer=opt,
            scheduler=sched, scaler=scaler,
            expected_run_spec=spec,
        )


def test_trainable_only_keeps_frozen_backbone(tmp_path):
    class MiniAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.adapter = torch.nn.Linear(
                2, 2, bias=False
            )

            for p in self.backbone.parameters():
                p.requires_grad_(False)

        def forward(self, x):
            return self.backbone(x) + self.adapter(x)

    torch.manual_seed(6304)

    original = MiniAdapter()
    initial = copy.deepcopy(original)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, original.parameters()),
        lr=0.01,
    )
    scheduler = make_scheduler(
        optimizer, total_updates=8, warmup=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    original.adapter.weight.data.add_(1.0)
    spec = {"method": "test_adapter", "seed": 6304}
    checkpoint = tmp_path / "adapter"

    meta = save_checkpoint(
        checkpoint,
        model=original,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        run_spec=spec,
        completed_updates=3,
        state_kind="trainable",
    )

    assert meta["saved_model_keys"] == ["adapter.weight"]

    opt2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, initial.parameters()),
        lr=0.01,
    )
    sch2 = make_scheduler(
        opt2, total_updates=8, warmup=2
    )
    scl2 = torch.amp.GradScaler("cuda", enabled=False)

    load_checkpoint(
        checkpoint,
        model=initial,
        optimizer=opt2,
        scheduler=sch2,
        scaler=scl2,
        expected_run_spec=spec,
    )

    torch.testing.assert_close(
        initial.adapter.weight,
        original.adapter.weight,
    )
    torch.testing.assert_close(
        initial.backbone.weight,
        original.backbone.weight,
    )


def test_invalid_cursor_rejected(tmp_path):
    model, opt, sched, scaler = make_parts()

    with pytest.raises(ValueError, match="update"):
        save_checkpoint(
            tmp_path / "cp",
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            run_spec={"a": 1},
            completed_updates=1251,
        )
