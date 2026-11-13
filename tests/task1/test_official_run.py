"""Entrypoint checks with local fixtures; never download Qwen or train on GPU."""
import json
from types import SimpleNamespace

import pytest
import torch

from atml_pa3.task1 import official_run as entry
from atml_pa3.task1.model_setup import ModelInfo
from atml_pa3.task1.recovery import load_checkpoint, save_checkpoint
from atml_pa3.task1.train_core import make_adamw, make_scheduler


@pytest.fixture
def data(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    metadata = {
        "model": {"id": entry.MODEL_ID, "snapshot_sha": entry.MODEL_REVISION},
        "rules": {"thinking": False, "shuffle": False, "truncate": False},
        "datasets": {"metamath": {"revision": "pinned-test-revision"}},
        "normalization": {"version": "fixture"},
        "chat_template_sha256": "fixture",
    }
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    # Data verification is covered by train_core tests; no large test dataset.
    monkeypatch.setattr(entry, "verify_dataset", lambda *args: folder / "examples.jsonl")
    return folder


@pytest.mark.parametrize("method,rank,lr", [
    ("full", None, 2e-5), ("l2sp", None, 2e-5),
    ("lora4", 4, 2e-4), ("lora16", 16, 2e-4), ("oplora16", 16, 2e-4),
])
def test_locked_recipe(data, method, rank, lr):
    spec = entry.build_run_spec(method, data)
    assert spec["optimizer"]["peak_lr"] == lr
    assert spec["total_examples"] == 40000
    assert spec["checkpoint_updates"] == [312, 625, 937, 1250]
    assert spec["l2sp_lambda"] == (0.001 if method == "l2sp" else 0.0)
    if rank is None:
        assert spec["adapter"] is None
    else:
        assert spec["adapter"]["rank"] == rank
        assert spec["adapter"]["alpha"] == 2 * rank
        assert spec["adapter"]["protection_rank"] == (128 if method == "oplora16" else None)
        assert len(spec["adapter"]["targets"]) == 7
    assert len(spec["implementation_sha256"]) == 6
    assert all(len(h) == 64 for h in spec["implementation_sha256"].values())


def test_default_cli_does_not_load_or_write(data, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Validation must not load a model or start training")
    monkeypatch.setattr(entry, "load_qwen_model", forbidden)
    monkeypatch.setattr(entry, "execute_run", forbidden)
    dest = tmp_path / "unused"
    entry.main(["--method", "lora4", "--data-dir", str(data), "--run-dir", str(dest)])
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "validation_only"
    assert output["training_started"] is False
    assert not dest.exists()


def test_bad_provenance_and_changed_recipe_rejected(data, tmp_path):
    spec = entry.build_run_spec("full", data)
    spec["total_examples"] = 32
    with pytest.raises(ValueError, match="fixed recipe"):
        entry.execute_run(spec=spec, data_dir=data, run_dir=tmp_path / "run")
    metadata = json.loads((data / "metadata.json").read_text())
    metadata["model"]["snapshot_sha"] = "main"
    (data / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="provenance"):
        entry.build_run_spec("full", data)


def test_existing_run_and_foreign_or_stale_resume_rejected(tmp_path):
    with pytest.raises(FileExistsError):
        entry.validate_destination(tmp_path, None)
    with pytest.raises(ValueError, match="belong"):
        entry.validate_destination(tmp_path / "run", tmp_path)
    (tmp_path / "run_spec.json").write_text("{}")
    for name in ("update_00312", "update_00625"):
        cp = tmp_path / "checkpoints" / name
        cp.mkdir(parents=True)
        (cp / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="latest"):
        entry.validate_destination(tmp_path, tmp_path / "checkpoints/update_00312")
    entry.validate_destination(tmp_path, tmp_path / "checkpoints/update_00625")


def mock_runtime(monkeypatch, method, loader_calls):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "CPU test fixture")
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *args: None)
    # Use disabled CPU scaler even though the runner is exercising CUDA wiring.
    scaler_type = torch.amp.GradScaler
    monkeypatch.setattr(entry.torch.amp, "GradScaler",
                        lambda *args, **kwargs: scaler_type("cuda", enabled=False))
    def loader(name, **kwargs):
        loader_calls.append((name, kwargs))
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)
        info = ModelInfo(name, entry.METHOD_SETTINGS[name][2],
                         6, 6, 0, "full", (), 0.0)
        return model, SimpleNamespace(pad_token_id=0), info
    monkeypatch.setattr(entry, "load_qwen_model", loader)


def test_l2sp_reference_precedes_real_checkpoint_restore(data, tmp_path, monkeypatch):
    spec = entry.build_run_spec("l2sp", data)
    dest = tmp_path / "run"
    checkpoint = dest / "checkpoints/update_00625"
    dest.mkdir()
    (dest / "run_spec.json").write_text(json.dumps(spec))
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2.0)
    opt = make_adamw(model, 2e-5)
    sched = make_scheduler(opt)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    save_checkpoint(checkpoint, model=model, optimizer=opt, scheduler=sched,
                    scaler=scaler, run_spec=spec, completed_updates=625)
    calls = []
    mock_runtime(monkeypatch, "l2sp", calls)

    def controller(**kwargs):
        regularizer = kwargs["regularizer"]
        # A genuine restore changes all six parameters from 1 to 2.
        meta = load_checkpoint(kwargs["resume_from"], model=kwargs["model"],
                               optimizer=kwargs["optimizer"], scheduler=kwargs["scheduler"],
                               scaler=kwargs["scaler"], expected_run_spec=spec)
        assert meta["next_example_offset"] == 20000
        assert regularizer.add_to_unscaled_grad() == 3.0
        assert kwargs["stop_after_update"] == 937
        return {"completed_updates": 937}
    monkeypatch.setattr(entry, "run_training", controller)
    report = entry.execute_run(spec=spec, data_dir=data, run_dir=dest,
                               resume_from=checkpoint, stop_after_update=937)
    assert report["completed_updates"] == 937
    assert calls[0][1]["resume_oplora"] is False
    assert json.loads((dest / "run_spec.json").read_text()) == spec
    assert len(list(dest.glob("session_*.json"))) == 1


@pytest.mark.parametrize("method", tuple(entry.METHOD_SETTINGS))
def test_fresh_execution_wires_fixed_controller_settings(data, tmp_path, monkeypatch, method):
    spec = entry.build_run_spec(method, data)
    dest = tmp_path / "new_run"
    calls = []
    mock_runtime(monkeypatch, method, calls)

    def controller(**kwargs):
        # Exercise actual AdamW/scheduler construction without a model update.
        lr = entry.METHOD_SETTINGS[method][2]
        # mock_runtime uses the same method learning rates as the factory.
        assert kwargs["optimizer"].param_groups[0]["lr"] == pytest.approx(lr / 38)
        assert kwargs["scheduler"].last_epoch == 0
        assert kwargs["run_spec"] == spec
        assert kwargs["data_path"] == data / "examples.jsonl"
        assert kwargs["resume_from"] is None
        assert (kwargs["regularizer"] is not None) == (method == "l2sp")
        assert json.loads((dest / "run_spec.json").read_text()) == spec
        return {"completed_updates": 312}
    monkeypatch.setattr(entry, "run_training", controller)
    entry.execute_run(spec=spec, data_dir=data, run_dir=dest, stop_after_update=312)
    assert calls[0][1]["resume_oplora"] is False
    assert len(list(dest.glob("result_*.json"))) == 1


def test_oplora_resume_uses_saved_bases_and_rejects_spec_changes(data, tmp_path, monkeypatch):
    spec = entry.build_run_spec("oplora16", data)
    dest = tmp_path / "run"
    checkpoint = dest / "checkpoints/update_00312"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text("{}")
    (dest / "run_spec.json").write_text(json.dumps(spec))
    calls = []
    mock_runtime(monkeypatch, "oplora16", calls)
    def controller(**kwargs):
        assert kwargs["regularizer"] is None
        assert kwargs["resume_from"] == checkpoint
        return {"completed_updates": 625}
    monkeypatch.setattr(entry, "run_training", controller)
    entry.execute_run(spec=spec, data_dir=data, run_dir=dest, resume_from=checkpoint)
    assert calls[0][1]["resume_oplora"] is True
    saved = {**spec, "seed": 1}
    (dest / "run_spec.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="differs"):
        entry.execute_run(spec=spec, data_dir=data, run_dir=dest, resume_from=checkpoint)
    assert len(calls) == 1


def test_execution_requires_cuda(data, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        entry.execute_run(spec=entry.build_run_spec("full", data),
                          data_dir=data, run_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()
