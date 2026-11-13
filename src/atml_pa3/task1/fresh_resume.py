"""Fresh-process Qwen adapter checkpoint continuation smoke check."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .model_setup import MODEL_REVISION, SEED, load_qwen_model
from .qwen_preflight import check_label_roundtrip
from .run_controller import FROZEN_DATA_SHA256, run_training
from .smoke_step import assert_state_equal, select_short_rows
from .train_core import make_adamw, make_scheduler, verify_dataset

METHODS = ("lora4", "lora16", "oplora16")
STAGES = ("reference", "interrupted", "resume")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_workspace(
    data_dir: Path,
    workspace: Path,
    *,
    method: str = "lora4",
    max_tokens: int = 128,
) -> dict:
    """Create shared inputs for three independent worker processes."""
    data_dir, workspace = Path(data_dir), Path(workspace)
    if method not in METHODS:
        raise ValueError(f"Unsupported fresh-process method: {method}")
    if workspace.exists():
        raise FileExistsError(f"Refusing existing output directory: {workspace}")
    if not torch.cuda.is_available():
        raise RuntimeError("Fresh-process Qwen smoke requires CUDA")

    source = verify_dataset(data_dir, FROZEN_DATA_SHA256)
    rows = select_short_rows(source, max_tokens=max_tokens, count=64)
    workspace.mkdir(parents=True, exist_ok=False)
    sample_path = workspace / "smoke_examples.jsonl"
    with sample_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    source_dir = Path(__file__).parent
    spec = {
        "smoke_only": True,
        "check": "fresh_process_exact_continuation",
        "method": method,
        "seed": SEED,
        "model_revision": MODEL_REVISION,
        "frozen_data_sha256": FROZEN_DATA_SHA256,
        "max_tokens": max_tokens,
        "selected_source_indices": [row["source_index"] for row in rows],
        "train_examples": 64,
        "microbatch_size": 2,
        "accumulation_steps": 16,
        "scheduler": {"total_updates": 2, "warmup": 1},
        "initial_gradient_scale": 256.0,
        "implementation_sha256": {
            name: _sha256(source_dir / name)
            for name in (
                "fresh_resume.py", "model_setup.py", "train_core.py",
                "run_controller.py", "recovery.py", "smoke_step.py",
            )
        },
    }
    config = {
        "schema_version": 1,
        "data_path": str(sample_path.resolve()),
        "workspace": str(workspace.resolve()),
        "run_spec": spec,
    }
    (workspace / "fresh_resume_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return config


def _seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_worker(workspace: Path, stage: str) -> dict:
    """Run exactly one stage; the CLI launches each stage in a new process."""
    workspace = Path(workspace).resolve()
    if stage not in STAGES:
        raise ValueError(f"Unknown worker stage: {stage}")
    config = json.loads(
        (workspace / "fresh_resume_config.json").read_text(encoding="utf-8")
    )
    if Path(config["workspace"]).resolve() != workspace:
        raise ValueError("Workspace/config path mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("Fresh-process Qwen smoke requires CUDA")

    spec = config["run_spec"]
    method = spec["method"]
    device = torch.device("cuda:0")
    _seed_everything()
    model, tokenizer, info = load_qwen_model(
        method,
        device=device,
        resume_oplora=(stage == "resume" and method == "oplora16"),
    )
    if info.state_kind != "trainable" or info.adapter_modules != 196:
        raise AssertionError("Unexpected adapter model layout")

    rows = [
        json.loads(line)
        for line in Path(config["data_path"]).read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != 64:
        raise ValueError("Fresh-process smoke requires exactly 64 examples")
    check_label_roundtrip(rows, tokenizer)

    optimizer = make_adamw(model, learning_rate=info.peak_lr)
    scheduler = make_scheduler(optimizer, total_updates=2, warmup=1)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=256.0)
    run_dir = workspace / ("reference" if stage == "reference" else "resumed")
    resume_from = None
    stop_after = None
    if stage in ("reference", "interrupted"):
        run_dir.mkdir(exist_ok=False)
        stop_after = 1 if stage == "interrupted" else None
    else:
        resume_from = run_dir / "checkpoints" / "update_00001"
        if not resume_from.is_dir():
            raise FileNotFoundError("Interrupted checkpoint is missing")

    report = run_training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        data_path=Path(config["data_path"]),
        run_dir=run_dir,
        run_spec=spec,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        resume_from=resume_from,
        state_kind="trainable",
        extra_state_keys=info.extra_state_keys,
        total_examples=64,
        checkpoint_steps=(1, 2),
        stop_after_update=stop_after,
        max_skips_per_batch=8,
    )
    expected = 1 if stage == "interrupted" else 2
    if report["completed_updates"] != expected:
        raise AssertionError(f"{stage} ended at the wrong update")
    result = {
        "stage": stage,
        "process_id": __import__("os").getpid(),
        "completed_updates": report["completed_updates"],
        "examples_consumed": report["examples_consumed"],
        "skipped_attempts": report["skipped_attempts"],
        "peak_gpu_memory_allocated_bytes": report["peak_gpu_memory_allocated_bytes"],
        "basis_preprocessing_seconds": info.basis_preprocessing_seconds,
    }
    (workspace / f"worker_{stage}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _verified_payload(checkpoint: Path) -> tuple[dict, dict]:
    checkpoint = Path(checkpoint)
    metadata = json.loads(
        (checkpoint / "metadata.json").read_text(encoding="utf-8")
    )
    state_path = checkpoint / "state.pt"
    if _sha256(state_path) != metadata["state_sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint}")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    return metadata, payload


def compare_final_checkpoints(workspace: Path) -> dict:
    """Require exact final state after uninterrupted and fresh-process paths."""
    workspace = Path(workspace).resolve()
    config = json.loads(
        (workspace / "fresh_resume_config.json").read_text(encoding="utf-8")
    )
    worker_results = {
        stage: json.loads((workspace / f"worker_{stage}.json").read_text(encoding="utf-8"))
        for stage in STAGES
    }
    process_ids = [worker_results[stage]["process_id"] for stage in STAGES]
    if len(set(process_ids)) != len(process_ids):
        raise AssertionError("Worker stages did not use distinct processes")

    reference_dir = workspace / "reference" / "checkpoints" / "update_00002"
    resumed_dir = workspace / "resumed" / "checkpoints" / "update_00002"
    reference_meta, reference = _verified_payload(reference_dir)
    resumed_meta, resumed = _verified_payload(resumed_dir)
    for metadata in (reference_meta, resumed_meta):
        if metadata["completed_updates"] != 2 or metadata["next_example_offset"] != 64:
            raise AssertionError("Final checkpoint cursor is incorrect")

    # Serialization hashes need not match because torch.save storage identifiers
    # may differ; every semantically relevant nested value must match exactly.
    assert_state_equal(resumed, reference)
    result = {
        "status": "FRESH-PROCESS EXACT CONTINUATION PASSED",
        "smoke_only": True,
        "method": config["run_spec"]["method"],
        "worker_process_ids": process_ids,
        "distinct_worker_processes": True,
        "completed_optimizer_updates": 2,
        "next_example_offset": 64,
        "exact_states_compared": [
            "all_adapters", "optimizer", "scheduler", "scaler", "rng", "cursor"
        ],
        "projection_bases_included": config["run_spec"]["method"] == "oplora16",
        "reference_checkpoint_sha256": reference_meta["state_sha256"],
        "resumed_checkpoint_sha256": resumed_meta["state_sha256"],
    }
    (workspace / "fresh_resume_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
