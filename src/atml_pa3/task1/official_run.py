"""Fixed Task 1 training recipe and explicit execution boundary.

Validation reads local artifacts only. Importing or planning never loads Qwen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import time
import uuid

import numpy as np
import torch

from .l2sp import L2SPRegularizer
from .model_setup import (
    DROPOUT, METHOD_SETTINGS, MODEL_ID, MODEL_REVISION,
    PROTECTION_RANK, SEED, TARGETS, load_qwen_model,
)
from .run_controller import FROZEN_DATA_SHA256, OFFICIAL_CHECKPOINTS, run_training
from .train_core import make_adamw, make_scheduler, verify_dataset

ROOT = Path(__file__).resolve().parents[3]


def _digest(path: Path, *, source=False) -> str:
    digest = hashlib.sha256()
    if source:
        # Source hashes survive a Windows/Linux newline conversion.
        digest.update(path.read_text(encoding="utf-8-sig").encode("utf-8"))
    else:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, value: dict):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_run_spec(method: str, data_dir: Path) -> dict:
    """Verify the frozen artifact and describe a path-independent recipe."""
    if method not in METHOD_SETTINGS:
        raise ValueError(f"Unknown method: {method}")
    data_dir = Path(data_dir)
    verify_dataset(data_dir, FROZEN_DATA_SHA256)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["model"] != {"id": MODEL_ID, "snapshot_sha": MODEL_REVISION}:
        raise ValueError("Frozen dataset model provenance differs")
    if metadata["rules"].get("thinking") is not False:
        raise ValueError("Dataset must use non-thinking formatting")
    rank, alpha, lr = METHOD_SETTINGS[method]
    code_files = (
        "official_run.py", "model_setup.py", "train_core.py",
        "run_controller.py", "recovery.py", "l2sp.py",
    )
    return {
        "schema_version": 1,
        "method": method,
        "seed": SEED,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_sha256": FROZEN_DATA_SHA256,
        "dataset_metadata_sha256": _digest(data_dir / "metadata.json"),
        "dataset_revisions": metadata["datasets"],
        "filtering_rules": metadata["rules"],
        "normalization": metadata["normalization"],
        "chat_template_sha256": metadata["chat_template_sha256"],
        "total_examples": 40000,
        "total_updates": 1250,
        "epochs": 1,
        "microbatch_size": 2,
        "accumulation_steps": 16,
        "max_tokens": 2048,
        "shuffle": False,
        "truncate": False,
        "enable_thinking": False,
        "optimizer": {"name": "AdamW", "peak_lr": lr, "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": 0.01,
                      "decay_exclusions": "biases and parameters with ndim < 2"},
        "scheduler": {"name": "cosine", "warmup_updates": 38},
        "grad_clip_norm": 1.0,
        "precision": "FP16 autocast; FP32 trainables and AdamW states; no quantization",
        "gradient_checkpointing": True,
        "attention_implementation": "eager",
        "grad_scaler": {"init_scale": 256.0, "growth_factor": 2.0,
                        "backoff_factor": 0.5, "growth_interval": 2000},
        "max_skips_per_batch": 8,
        "skipped_update_policy": "retry same effective batch; advance LR only on success",
        "l2sp_lambda": 0.001 if method == "l2sp" else 0.0,
        "adapter": None if rank is None else {
            "rank": rank, "alpha": alpha, "dropout": DROPOUT,
            "targets": [name for _, name in TARGETS],
            "trainable_biases": False,
            "protection_rank": PROTECTION_RANK if method == "oplora16" else None,
        },
        "checkpoint_updates": list(OFFICIAL_CHECKPOINTS),
        "implementation_sha256": {
            name: _digest(Path(__file__).parent / name, source=True)
            for name in code_files
        },
        "environment": {"python": platform.python_version(), **{
            package: version(package)
            for package in ("torch", "transformers", "numpy", "tokenizers")
        }},
    }


def validate_destination(run_dir: Path, resume_from: Path | None):
    """Reject accidental overwrites and branching over later checkpoints."""
    run_dir = Path(run_dir)
    if resume_from is None:
        if run_dir.exists():
            raise FileExistsError(f"Use an unused run directory: {run_dir}")
        return
    resume_from = Path(resume_from).resolve()
    checkpoint_root = run_dir.resolve() / "checkpoints"
    if resume_from.parent != checkpoint_root or not resume_from.is_dir():
        raise ValueError("Resume checkpoint must belong to this run directory")
    if not (run_dir / "run_spec.json").is_file():
        raise ValueError("Resume requires the saved run_spec.json")
    complete = sorted(
        path for path in checkpoint_root.glob("update_*")
        if path.is_dir() and (path / "metadata.json").is_file()
    )
    if not complete or resume_from != complete[-1]:
        raise ValueError("Resume must use the latest complete checkpoint")


def execute_run(
    *, spec: dict, data_dir: Path, run_dir: Path, device: str = "cuda:0",
    resume_from: Path | None = None, stop_after_update: int | None = None,
) -> dict:
    """Explicit execution only; callers must obtain training authorization."""
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    validate_destination(run_dir, resume_from)
    # Do not allow programmatic callers to smuggle altered settings into a run.
    if spec != build_run_spec(spec["method"], data_dir):
        raise ValueError("Run specification differs from the fixed recipe")
    if resume_from is not None:
        saved_spec = json.loads((run_dir / "run_spec.json").read_text(encoding="utf-8"))
        if spec != saved_spec:
            raise ValueError("Resume configuration, code, or environment differs")
    if stop_after_update is not None and stop_after_update not in OFFICIAL_CHECKPOINTS:
        raise ValueError("Stop must coincide with a required checkpoint")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Official FP16 execution requires CUDA")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Avoid TF32 changing the FP32 projection/reference computations.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if resume_from is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_new(run_dir / "run_spec.json", spec)
    session_id = uuid.uuid4().hex
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model, tokenizer, info = load_qwen_model(
        spec["method"], device=device,
        resume_oplora=(resume_from is not None and spec["method"] == "oplora16"),
    )
    torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - started
    # Capture theta0 from the original pinned model BEFORE controller restore.
    reference_started = time.perf_counter()
    regularizer = L2SPRegularizer(model) if spec["method"] == "l2sp" else None
    torch.cuda.synchronize(device)
    reference_seconds = time.perf_counter() - reference_started
    optimizer = make_adamw(model, learning_rate=info.peak_lr)
    scheduler = make_scheduler(optimizer)
    scaler = torch.amp.GradScaler("cuda", **spec["grad_scaler"])
    session = {
        "session_id": session_id,
        "resume_from": str(resume_from) if resume_from else None,
        "stop_after_update": stop_after_update,
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(), "model_info": asdict(info),
        "model_setup_seconds": setup_seconds,
        "setup_peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "reference_creation_seconds": reference_seconds,
        "reference_bytes": regularizer.reference_bytes if regularizer else 0,
        "basis_storage_bytes": sum(
            b.numel() * b.element_size() for name, b in model.named_buffers()
            if name.endswith((".basis_u", ".basis_v", ".basis_ready"))
        ),
    }
    _write_new(run_dir / f"session_{session_id}.json", session)
    report = run_training(
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        data_path=data_dir / "examples.jsonl", run_dir=run_dir, run_spec=spec,
        pad_token_id=tokenizer.pad_token_id, device=device,
        resume_from=resume_from, state_kind=info.state_kind,
        extra_state_keys=info.extra_state_keys, regularizer=regularizer,
        stop_after_update=stop_after_update,
    )
    report["checkpoint_sizes_bytes"] = {
        path.name: sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        for path in (run_dir / "checkpoints").glob("update_*") if path.is_dir()
    }
    _write_new(run_dir / f"result_{session_id}.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHOD_SETTINGS), required=True)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "artifacts/task1/data/metamath-40000-v2")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stop-after-update", type=int, choices=OFFICIAL_CHECKPOINTS)
    parser.add_argument("--execute-training", action="store_true",
                        help="Explicitly launch official training; default only validates")
    args = parser.parse_args(argv)
    validate_destination(args.run_dir, args.resume_from)
    spec = build_run_spec(args.method, args.data_dir)
    if args.resume_from is not None:
        saved = json.loads((args.run_dir / "run_spec.json").read_text(encoding="utf-8"))
        if saved != spec:
            raise ValueError("Resume configuration, code, or environment differs")
    if not args.execute_training:
        print(json.dumps({"mode": "validation_only", "model_loaded": False,
                          "training_started": False, "run_spec": spec}, indent=2))
        return
    print(json.dumps(execute_run(
        spec=spec, data_dir=args.data_dir, run_dir=args.run_dir, device=args.device,
        resume_from=args.resume_from, stop_after_update=args.stop_after_update,
    ), indent=2))
