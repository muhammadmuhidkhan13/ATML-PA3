"""Source-order training controller with resumable milestone checkpoints.

Importing this module does not start training.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import torch

from .recovery import load_checkpoint, save_checkpoint
from .train_core import (
    EFFECTIVE_BATCH_SIZE,
    TOTAL_EXAMPLES,
    iter_effective_batches,
    train_effective_batch,
    verify_dataset,
)

OFFICIAL_CHECKPOINTS = (312, 625, 937, 1250)
FROZEN_DATA_SHA256 = "f8ed6892b608c40579c849c49437ebd6f5097e06377ba3a1605fd9a60f5ce0e6"
MODEL_SHA = "c1899de289a04d12100db370d81485cdf75e47ca"
METHODS = {"full", "l2sp", "lora4", "lora16", "oplora16"}


def _event(path: Path, event: dict):
    """Append a JSON event to the attempt log."""
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(event, sort_keys=True, allow_nan=False)
            + "\n"
        )
        stream.flush()


def run_training(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    data_path: Path,
    run_dir: Path,
    run_spec: dict,
    pad_token_id: int,
    device: torch.device,
    resume_from: Path | None = None,
    state_kind: str = "full",
    extra_state_keys=(),
    total_examples: int = TOTAL_EXAMPLES,
    checkpoint_steps: tuple[int, ...] = OFFICIAL_CHECKPOINTS,
    max_skips_per_batch: int = 8,
    stop_after_update: int | None = None,
    regularizer=None,
) -> dict:
    """Train in source order with exact resume and checkpoint boundaries.

    Skipped optimizer attempts are retried on the same effective batch.
    They do not advance the training cursor or optimizer-update count.
    """
    data_path = Path(data_path)
    run_dir = Path(run_dir)
    device = torch.device(device)

    if not isinstance(run_spec, dict) or not run_spec:
        raise ValueError("Nonempty run_spec required")

    if (
        not isinstance(total_examples, int)
        or total_examples <= 0
        or total_examples % EFFECTIVE_BATCH_SIZE
    ):
        raise ValueError(
            "Example count must divide into complete 32-example updates"
        )

    total_updates = total_examples // EFFECTIVE_BATCH_SIZE
    steps = tuple(checkpoint_steps)

    if (
        not steps
        or steps != tuple(sorted(set(steps)))
        or steps[-1] != total_updates
    ):
        raise ValueError(
            "Checkpoints must be unique, increasing, and include final update"
        )

    if any(not (1 <= s <= total_updates) for s in steps):
        raise ValueError("Checkpoint index out of range")

    if (
        not isinstance(max_skips_per_batch, int)
        or max_skips_per_batch < 1
    ):
        raise ValueError("max_skips_per_batch must be positive")

    if state_kind not in ("full", "trainable"):
        raise ValueError("Invalid checkpoint state_kind")

    if (
        stop_after_update is not None
        and stop_after_update not in steps
    ):
        raise ValueError(
            "An early stop must coincide with a saved checkpoint"
        )

    # Check that the regularizer belongs to this model.
    if regularizer is not None:
        regularizer.assert_matches(model)

    # Lock the official assignment configuration.
    if total_examples == TOTAL_EXAMPLES:
        if run_spec.get("method") == "l2sp":
            if (
                regularizer is None
                or regularizer.coefficient != 1e-3
                or run_spec.get("l2sp_lambda") != 1e-3
            ):
                raise ValueError(
                    "Official L2-SP requires lambda=0.001 and a starting-point reference"
                )
        elif regularizer is not None:
            raise ValueError(
                "Only L2-SP may use a starting-point regularizer"
            )
        if steps != OFFICIAL_CHECKPOINTS:
            raise ValueError(
                "Official checkpoint schedule cannot be changed"
            )

        if run_spec.get("dataset_sha256") != FROZEN_DATA_SHA256:
            raise ValueError(
                "Official run must use independently validated dataset hash"
            )

        if (
            run_spec.get("model_revision") != MODEL_SHA
            or run_spec.get("seed") != 6304
        ):
            raise ValueError(
                "Official model revision or seed does not match assignment"
            )

        if run_spec.get("method") not in METHODS:
            raise ValueError(
                "Unrecognized official fine-tuning method"
            )

        if run_spec.get("total_examples") != TOTAL_EXAMPLES:
            raise ValueError(
                "Run specification lacks official example count"
            )

        verify_dataset(data_path.parent, FROZEN_DATA_SHA256)

    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    checkpoint_root = run_dir / "checkpoints"
    log_path = run_dir / "attempts.jsonl"

    if resume_from is None and log_path.exists():
        raise FileExistsError(
            "Attempt log already exists: use an unused run directory "
            "or explicitly resume"
        )

    checkpoint_root.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped_total = 0

    if resume_from is not None:
        resume_from = Path(resume_from)

        if (
            resume_from.resolve().parent
            != checkpoint_root.resolve()
        ):
            raise ValueError(
                "Resume checkpoint must belong to this run directory"
            )

        restored = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_run_spec=run_spec,
        )

        if restored["state_kind"] != state_kind:
            raise ValueError("Resume state kind mismatch")

        completed = restored["completed_updates"]
        skipped_total = restored["skipped_updates"]

        if completed not in steps:
            raise ValueError(
                "Resume checkpoint is not a required checkpoint stage"
            )

    if (
        stop_after_update is not None
        and stop_after_update < completed
    ):
        raise ValueError(
            "Cannot stop before a previously completed update"
        )

    session = uuid.uuid4().hex

    _event(log_path, {
        "event": "resume" if resume_from else "start",
        "session": session,
        "completed_updates": completed,
        "skipped_attempts": skipped_total,
        "source": str(resume_from) if resume_from else None,
    })

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    saved = []

    batches = iter_effective_batches(
        data_path,
        count=total_examples,
    )

    # Replay the dataset cursor without doing any model updates.
    for _ in range(completed):
        next(batches)

    ending = (
        total_updates
        if stop_after_update is None
        else stop_after_update
    )

    while completed < ending:
        microbatches = next(batches)
        skipped_this_batch = 0

        while True:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            step_started = time.perf_counter()
            lr_used = float(optimizer.param_groups[0]["lr"])

            kwargs = {"regularizer": regularizer} if regularizer is not None else {}
            metrics = train_effective_batch(
                model,
                optimizer,
                scheduler,
                scaler,
                microbatches,
                pad_token_id,
                device,
                **kwargs,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            stepped = metrics["optimizer_step_completed"]

            _event(log_path, {
                "event": "optimizer_attempt",
                "session": session,
                "target_update": completed + 1,
                "completed": stepped,
                "lr_used": lr_used,
                "elapsed_seconds": (
                    time.perf_counter() - step_started
                ),
                **metrics,
            })

            if stepped:
                break

            skipped_total += 1
            skipped_this_batch += 1

            if skipped_this_batch >= max_skips_per_batch:
                raise RuntimeError(
                    f"Optimizer update {completed + 1} skipped "
                    f"{skipped_this_batch} times; aborting without "
                    "consuming the next effective batch"
                )

        completed += 1

        if completed in steps:
            destination = (
                checkpoint_root / f"update_{completed:05d}"
            )

            metadata = save_checkpoint(
                destination,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                run_spec=run_spec,
                completed_updates=completed,
                skipped_updates=skipped_total,
                state_kind=state_kind,
                extra_state_keys=extra_state_keys,
            )

            saved.append(str(destination))

            _event(log_path, {
                "event": "checkpoint",
                "session": session,
                "completed_updates": completed,
                "next_example_offset": (
                    metadata["next_example_offset"]
                ),
                "path": str(destination),
            })

    if ending == total_updates:
        try:
            next(batches)
        except StopIteration:
            pass
        else:
            raise AssertionError(
                "Unexpected extra effective batch"
            )

    elapsed = time.perf_counter() - started
    peak_bytes = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    )

    _event(log_path, {
        "event": "end",
        "session": session,
        "completed_updates": completed,
        "skipped_attempts": skipped_total,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_allocated_bytes": peak_bytes,
    })

    return {
        "completed_updates": completed,
        "examples_consumed": (
            completed * EFFECTIVE_BATCH_SIZE
        ),
        "skipped_attempts": skipped_total,
        "checkpoints_saved_this_session": saved,
        "attempt_log": str(log_path),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_allocated_bytes": peak_bytes,
    }
