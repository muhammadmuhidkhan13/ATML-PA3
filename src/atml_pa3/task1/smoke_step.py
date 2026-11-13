"""A single non-official Qwen LoRA optimizer update with checkpoint restore."""
from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from .model_setup import AdapterLinear, MODEL_REVISION, SEED, load_qwen_model
from .oplora_preflight import projection_metrics
from .qwen_preflight import check_label_roundtrip
from .recovery import load_checkpoint, _rng_state
from .run_controller import FROZEN_DATA_SHA256, run_training
from .train_core import make_adamw, make_scheduler, verify_dataset


def trained_projection_metrics(model) -> list[dict]:
    """Validate nonzero learned OPLoRA corrections without changing factors."""
    results = []
    for name, module in model.named_modules():
        if isinstance(module, AdapterLinear) and module.protection_rank is not None:
            if not torch.count_nonzero(module.lora_B):
                raise AssertionError(f"OPLoRA B did not change: {name}")
            metrics = projection_metrics(module)
            if any(not np.isfinite(v) or v > 1e-4 for v in metrics.values()):
                raise AssertionError(f"Learned projection constraints failed: {name}: {metrics}")
            results.append({"name": name, **metrics})
    return results


def assert_state_equal(actual, expected):
    """Require exact nested checkpoint state equality, including tensor dtype."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(
            actual.cpu(), expected.cpu(), rtol=0, atol=0,
            check_device=False,
        )
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_state_equal(a, b)
    else:
        assert actual == expected


def select_short_rows(
    data_path: Path,
    max_tokens: int = 128,
    count: int = 32,
) -> list[dict]:
    """Take complete source-ordered examples, never truncate."""
    if not 32 <= max_tokens <= 128:
        raise ValueError(
            "Smoke limit must be between 32 and 128 tokens"
        )
    if not isinstance(count, int) or count < 1:
        raise ValueError("Smoke example count must be positive")

    selected = []

    with Path(data_path).open(
        "r", encoding="utf-8"
    ) as handle:
        for raw_line in handle:
            row = json.loads(raw_line)
            ids = row["input_ids"]

            if len(ids) > max_tokens:
                continue

            if not (
                0 < len(ids) == len(row["labels"])
                == len(row["attention_mask"])
            ):
                raise ValueError("Malformed selected row")

            if not any(
                label != -100 for label in row["labels"][1:]
            ):
                raise ValueError(
                    "Selected row has no supervised targets"
                )

            selected.append(row)

            if len(selected) == count:
                break

    if len(selected) != count:
        raise ValueError(
            f"Fewer than {count} complete short training examples"
        )

    indices = [
        row["source_index"] for row in selected
    ]

    if indices != sorted(set(indices)):
        raise ValueError("Selected source indices out of order")

    return selected


def run_smoke_update(
    data_dir: Path,
    run_dir: Path,
    max_tokens: int = 128,
    *,
    method: str = "lora4",
) -> dict:
    """One smoke optimizer update, never an official training run."""
    data_dir = Path(data_dir)
    run_dir = Path(run_dir)
    if method not in ("lora4", "lora16", "oplora16"):
        raise ValueError("Optimizer smoke supports lora4, lora16, or oplora16 only")

    if run_dir.exists():
        raise FileExistsError(
            f"Refusing existing smoke directory: {run_dir}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The real Qwen optimizer smoke requires CUDA"
        )

    source = verify_dataset(
        data_dir, FROZEN_DATA_SHA256
    )

    rows = select_short_rows(source, max_tokens)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda:0")
    if method == "oplora16":
        torch.backends.cuda.matmul.allow_tf32 = False

    print(
        f"Loading pinned Qwen3-0.6B {method}...",
        flush=True,
    )

    model, tokenizer, info = load_qwen_model(
        method, device=device
    )

    if (
        info.adapter_modules != 196
        or info.state_kind != "trainable"
    ):
        raise AssertionError(
            "Unexpected LoRA model configuration"
        )

    check_label_roundtrip(rows, tokenizer)
    if method == "oplora16" and len(info.extra_state_keys) != 588:
        raise AssertionError("OPLoRA must checkpoint all 588 basis buffers")

    optimizer = make_adamw(
        model,
        learning_rate=info.peak_lr,
    )

    # Smoke-only schedule; NOT the official 38/1250 schedule.
    scheduler = make_scheduler(
        optimizer,
        total_updates=2,
        warmup=1,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=256.0,
    )

    run_dir.mkdir(parents=True, exist_ok=False)

    sample_path = run_dir / "smoke_examples.jsonl"

    with sample_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    spec = {
        "smoke_only": True,
        "method": method,
        "seed": SEED,
        "model_revision": MODEL_REVISION,
        "frozen_data_sha256": FROZEN_DATA_SHA256,
        "max_tokens": max_tokens,
        "selected_source_indices": [
            row["source_index"] for row in rows
        ],
        "train_examples": 32,
        "microbatch_size": 2,
        "accumulation_steps": 16,
        "peak_lr": info.peak_lr,
        "scheduler": {"total_updates": 2, "warmup": 1},
        "initial_gradient_scale": 256.0,
        "protection_rank": 128 if method == "oplora16" else None,
        "svd_driver": "gesvd" if method == "oplora16" else None,
        "implementation_sha256": {
            name: hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()
            ).hexdigest()
            for name in (
                "smoke_step.py", "model_setup.py", "train_core.py",
                "run_controller.py", "recovery.py", "oplora_preflight.py",
            )
        },
    }

    print(
        "Running ONE non-official optimizer update...",
        flush=True,
    )

    report = run_training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        data_path=sample_path,
        run_dir=run_dir,
        run_spec=spec,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        total_examples=32,
        checkpoint_steps=(1,),
        state_kind="trainable",
        extra_state_keys=info.extra_state_keys,
        max_skips_per_batch=8,
    )

    if report["completed_updates"] != 1:
        raise AssertionError(
            "The smoke run did not complete exactly one update"
        )

    checkpoint = (
        run_dir / "checkpoints" / "update_00001"
    )

    first_b = next(
        param
        for name, param in model.named_parameters()
        if name.endswith(".lora_B")
    )

    original = first_b.detach().clone()

    if torch.count_nonzero(original).item() == 0:
        raise AssertionError(
            "The optimizer did not change a LoRA B factor"
        )

    # Read only our own newly saved, trusted checkpoint. The loader below
    # independently verifies its hash before restoring it.
    payload = torch.load(
        checkpoint / "state.pt", map_location="cpu", weights_only=False
    )
    projections_before = trained_projection_metrics(model) if method == "oplora16" else []
    if method == "oplora16" and len(projections_before) != 196:
        raise AssertionError("Expected 196 learned projected corrections")

    # Perturb every saved adapter and clear/alter runtime states so equality
    # cannot pass merely because load_checkpoint left them untouched.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.5)
        for name, buffer in model.named_buffers():
            if name in info.extra_state_keys:
                buffer.zero_()
    optimizer.state.clear()
    scheduler.last_epoch = -1
    scaler.update(new_scale=scaler.get_scale() / 2)
    random.random()
    np.random.rand()
    torch.rand(1)
    torch.rand(1, device=device)

    restored = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_run_spec=spec,
    )

    torch.testing.assert_close(
        first_b.detach(),
        original,
        rtol=0,
        atol=0,
    )

    for name, expected in payload["model"].items():
        assert_state_equal(model.state_dict()[name], expected)
    assert_state_equal(optimizer.state_dict(), payload["optimizer"])
    assert_state_equal(scheduler.state_dict(), payload["scheduler"])
    assert_state_equal(scaler.state_dict(), payload["scaler"])
    assert_state_equal(_rng_state(), payload["rng"])
    projections_after = trained_projection_metrics(model) if method == "oplora16" else []
    assert_state_equal(projections_after, projections_before)

    if (
        restored["completed_updates"] != 1
        or restored["next_example_offset"] != 32
    ):
        raise AssertionError(
            "Checkpoint update index changed on restore"
        )

    result = {
        "status": "REAL QWEN OPTIMIZER/RECOVERY SMOKE PASSED",
        "method": method,
        "trainable_parameters": info.trainable_parameters,
        "model_revision": MODEL_REVISION,
        "frozen_data_sha256": FROZEN_DATA_SHA256,
        "smoke_only": True,
        "selected_source_indices": spec[
            "selected_source_indices"
        ],
        "completed_optimizer_updates": report[
            "completed_updates"
        ],
        "skipped_fp16_attempts": report[
            "skipped_attempts"
        ],
        "elapsed_seconds": round(
            report["elapsed_seconds"], 2
        ),
        "peak_cuda_memory_gib": round(
            report["peak_gpu_memory_allocated_bytes"]
            / (1024 ** 3),
            3,
        ),
        "checkpoint_restored": True,
        "restored_states": [
            "all_adapters", "optimizer", "scheduler", "scaler", "rng", "cursor"
        ],
        "checkpoint_sha256": restored["state_sha256"],
        "basis_preprocessing_seconds": info.basis_preprocessing_seconds,
        "restored_projection_buffers": len(info.extra_state_keys),
        "learned_projection_targets_checked": len(projections_after),
        "run_directory": str(run_dir),
    }
    if method == "oplora16":
        result["restored_states"].append("all_projection_bases")
        (run_dir / "trained_projection_metrics.json").write_text(
            json.dumps(projections_after, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    (run_dir / "smoke_result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
