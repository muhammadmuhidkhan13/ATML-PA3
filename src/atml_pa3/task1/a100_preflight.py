"""Bounded real-Qwen A100 readiness checks; never official training."""
from __future__ import annotations

import gc
import hashlib
import heapq
import json
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .l2sp import L2SPRegularizer
from .model_setup import METHOD_SETTINGS, MODEL_ID, MODEL_REVISION, SEED, load_qwen_model
from .qwen_preflight import check_label_roundtrip
from .run_controller import FROZEN_DATA_SHA256
from .train_core import (
    ACCUMULATION_STEPS,
    MICROBATCH_SIZE,
    collate_microbatch,
    make_adamw,
    make_scheduler,
    train_effective_batch,
    verify_dataset,
)

METHODS = tuple(METHOD_SETTINGS)
GIB = 1024 ** 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_longest_rows(data_path: Path, count: int = 32) -> list[dict]:
    """Select the longest complete saved rows, then restore source order."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    longest: list[tuple[int, int, dict]] = []
    with Path(data_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            ids = row["input_ids"]
            labels = row["labels"]
            masks = row["attention_mask"]
            source = row["source_index"]
            if not (0 < len(ids) <= 2048 and len(ids) == len(labels) == len(masks)):
                raise ValueError(f"Malformed row at source index {source}")
            if not any(label != -100 for label in labels[1:]):
                raise ValueError(f"No supervised targets at source index {source}")
            item = (len(ids), source, row)
            if len(longest) < count:
                heapq.heappush(longest, item)
            elif item[:2] > longest[0][:2]:
                heapq.heapreplace(longest, item)
    if len(longest) != count:
        raise ValueError(f"Dataset has fewer than {count} examples")
    rows = [item[2] for item in longest]
    rows.sort(key=lambda row: row["source_index"])
    if len({row["source_index"] for row in rows}) != count:
        raise ValueError("Selected source indices are not unique")
    return rows


def _microbatches(rows: list[dict]) -> list[list[dict]]:
    expected = MICROBATCH_SIZE * ACCUMULATION_STEPS
    if len(rows) != expected:
        raise ValueError(f"Expected exactly {expected} stress examples")
    return [
        rows[index:index + MICROBATCH_SIZE]
        for index in range(0, expected, MICROBATCH_SIZE)
    ]


def inspect_a100(min_vram_gib: float = 75.0) -> dict:
    """Require the intended single A100 80GB-class CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Preflight requires exactly one visible GPU")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    total_gib = properties.total_memory / GIB
    if "A100" not in properties.name.upper():
        raise RuntimeError(f"Expected an A100, found: {properties.name}")
    if total_gib < min_vram_gib:
        raise RuntimeError(
            f"Expected an 80GB-class A100 (>= {min_vram_gib:.1f} GiB); "
            f"found {total_gib:.2f} GiB"
        )
    return {
        "device": str(device),
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_total_memory_gib": round(total_gib, 3),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def run_method_probe(
    *, method: str, rows: list[dict], output_dir: Path,
    hardware: dict,
) -> dict:
    """Run one official-shaped effective-batch update without checkpointing."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    setup_started = time.perf_counter()
    model, tokenizer, info = load_qwen_model(method, device=device)
    regularizer = L2SPRegularizer(model) if method == "l2sp" else None
    optimizer = make_adamw(model, learning_rate=info.peak_lr)
    scheduler = make_scheduler(optimizer, total_updates=2, warmup=1)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=256.0)
    check_label_roundtrip(rows, tokenizer)
    # Materialize the largest padded microbatch before timing for an explicit
    # shape check. It is discarded immediately; the real update collates again.
    largest_pair = max(_microbatches(rows), key=lambda part: max(len(r["input_ids"]) for r in part))
    shape = tuple(collate_microbatch(
        largest_pair, tokenizer.pad_token_id, device
    )["input_ids"].shape)
    torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - setup_started
    update_started = time.perf_counter()
    metrics = train_effective_batch(
        model,
        optimizer,
        scheduler,
        scaler,
        _microbatches(rows),
        tokenizer.pad_token_id,
        device,
        **({"regularizer": regularizer} if regularizer is not None else {}),
    )
    torch.cuda.synchronize(device)
    update_seconds = time.perf_counter() - update_started
    if not metrics["optimizer_step_completed"]:
        raise AssertionError(f"GradScaler skipped the {method} smoke update")
    result = {
        "status": "A100 METHOD PREFLIGHT PASSED",
        "smoke_only": True,
        "official_training": False,
        "method": method,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "microbatch_size": MICROBATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "effective_batch_size": len(rows),
        "selected_source_indices": [row["source_index"] for row in rows],
        "selected_sequence_lengths": [len(row["input_ids"]) for row in rows],
        "largest_padded_microbatch_shape": list(shape),
        "trainable_parameters": info.trainable_parameters,
        "total_parameters": info.total_parameters,
        "adapter_modules": info.adapter_modules,
        "basis_preprocessing_seconds": info.basis_preprocessing_seconds,
        "l2sp_reference_bytes": regularizer.reference_bytes if regularizer else 0,
        "setup_seconds": setup_seconds,
        "optimizer_update_seconds": update_seconds,
        "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "peak_gpu_memory_allocated_gib": round(torch.cuda.max_memory_allocated(device) / GIB, 3),
        "peak_gpu_memory_reserved_gib": round(torch.cuda.max_memory_reserved(device) / GIB, 3),
        "hardware": hardware,
        "metrics": metrics,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    del metrics, scaler, scheduler, optimizer, regularizer, tokenizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_a100_preflight(
    *, data_dir: Path, output_dir: Path,
    methods: tuple[str, ...] = METHODS, min_vram_gib: float = 75.0,
) -> dict:
    """Execute bounded probes and write one portable summary."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Methods must be nonempty and unique")
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    data_path = verify_dataset(data_dir, FROZEN_DATA_SHA256)
    rows = select_longest_rows(data_path, count=32)
    hardware = inspect_a100(min_vram_gib=min_vram_gib)
    output_dir.mkdir(parents=True, exist_ok=False)
    selection = {
        "frozen_data_sha256": FROZEN_DATA_SHA256,
        "selected_source_indices": [row["source_index"] for row in rows],
        "selected_sequence_lengths": [len(row["input_ids"]) for row in rows],
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    results = []
    started = time.perf_counter()
    for method in methods:
        print(f"\n=== A100 bounded probe: {method} ===", flush=True)
        result = run_method_probe(
            method=method, rows=rows,
            output_dir=output_dir / method, hardware=hardware,
        )
        results.append(result)
        print(json.dumps({
            "method": method,
            "status": result["status"],
            "peak_allocated_gib": result["peak_gpu_memory_allocated_gib"],
            "peak_reserved_gib": result["peak_gpu_memory_reserved_gib"],
            "setup_seconds": round(result["setup_seconds"], 3),
            "optimizer_update_seconds": round(result["optimizer_update_seconds"], 3),
        }, indent=2), flush=True)
    summary = {
        "status": "A100 80GB READINESS PREFLIGHT PASSED",
        "smoke_only": True,
        "official_training": False,
        "frozen_data_sha256": FROZEN_DATA_SHA256,
        "hardware": hardware,
        "methods": list(methods),
        "total_elapsed_seconds": time.perf_counter() - started,
        "results": [{
            "method": result["method"],
            "peak_gpu_memory_allocated_gib": result["peak_gpu_memory_allocated_gib"],
            "peak_gpu_memory_reserved_gib": result["peak_gpu_memory_reserved_gib"],
            "setup_seconds": result["setup_seconds"],
            "optimizer_update_seconds": result["optimizer_update_seconds"],
        } for result in results],
        "environment": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("torch", "transformers", "numpy", "tokenizers")},
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
