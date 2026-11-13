"""Shared Task 1 training primitives. Importing never starts training."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterator

import torch

IGNORE_INDEX = -100
MICROBATCH_SIZE = 2
ACCUMULATION_STEPS = 16
EFFECTIVE_BATCH_SIZE = MICROBATCH_SIZE * ACCUMULATION_STEPS
TOTAL_EXAMPLES = 40000
TOTAL_UPDATES = TOTAL_EXAMPLES // EFFECTIVE_BATCH_SIZE
WARMUP_UPDATES = 38
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0


def verify_dataset(folder: Path, expected_sha256: str) -> Path:
    """Verify the frozen training artifact before it is consumed."""
    folder = Path(folder)
    data_path = folder / "examples.jsonl"
    metadata_path = folder / "metadata.json"
    sums_path = folder / "SHA256SUMS"

    if not all(
        p.is_file()
        for p in (data_path, metadata_path, sums_path)
    ):
        raise FileNotFoundError(
            "Missing examples.jsonl, metadata.json, or SHA256SUMS"
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if (
        metadata.get("mode") != "full_build"
        or metadata.get("seed") != 6304
    ):
        raise ValueError("Dataset must be the seed-6304 full build")

    if (
        metadata.get("selection", {}).get("accepted")
        != TOTAL_EXAMPLES
    ):
        raise ValueError("Expected 40000 accepted examples")

    rules = metadata.get("rules", {})
    if (
        rules.get("shuffle") is not False
        or rules.get("truncate") is not False
    ):
        raise ValueError("Dataset shuffling/truncation is forbidden")

    hasher = hashlib.sha256()

    with data_path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            hasher.update(chunk)

    actual = hasher.hexdigest()

    if actual != expected_sha256:
        raise ValueError("Dataset hash differs from validated artifact")

    if (
        metadata.get("output_sha256", {}).get("examples.jsonl")
        != actual
    ):
        raise ValueError("Dataset hash differs from metadata")

    sums = dict(
        (name, digest)
        for digest, name in (
            line.strip().split(maxsplit=1)
            for line in sums_path.read_text(
                encoding="utf-8"
            ).splitlines()
        )
    )

    if sums.get("examples.jsonl") != actual:
        raise ValueError("Dataset hash differs from SHA256SUMS")

    return data_path


def iter_effective_batches(
    data_path: Path,
    *,
    count: int = TOTAL_EXAMPLES,
    microbatch_size: int = MICROBATCH_SIZE,
    accumulation_steps: int = ACCUMULATION_STEPS,
) -> Iterator[list[list[dict]]]:
    """Stream full effective batches in original source order."""
    if min(count, microbatch_size, accumulation_steps) < 1:
        raise ValueError("Batch settings must be positive")

    effective = microbatch_size * accumulation_steps

    if count % effective:
        raise ValueError(
            "Example count must divide evenly into effective batches"
        )

    previous_source = -1
    read_count = 0
    effective_rows = []

    with Path(data_path).open(
        "r", encoding="utf-8"
    ) as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("Empty dataset line")

            row = json.loads(line)
            read_count += 1

            if read_count > count:
                raise ValueError("Dataset has more rows than expected")

            source = row["source_index"]

            if (
                not isinstance(source, int)
                or source <= previous_source
            ):
                raise ValueError(
                    "Source indices are not strictly increasing"
                )

            previous_source = source

            ids = row["input_ids"]
            masks = row["attention_mask"]
            labels = row["labels"]

            if not (
                0 < len(ids) <= 2048
                and len(ids) == len(masks) == len(labels)
            ):
                raise ValueError(
                    f"Invalid sequence lengths at source {source}"
                )

            if any(m != 1 for m in masks):
                raise ValueError(
                    f"Unpadded attention mask invalid: {source}"
                )

            if not any(
                label != IGNORE_INDEX
                for label in labels[1:]
            ):
                raise ValueError(
                    f"No supervised tokens at source {source}"
                )

            if any(
                label != IGNORE_INDEX and label != token
                for label, token in zip(labels, ids)
            ):
                raise ValueError(
                    f"Label/token mismatch at source {source}"
                )

            effective_rows.append(row)

            if len(effective_rows) == effective:
                yield [
                    effective_rows[i:i + microbatch_size]
                    for i in range(
                        0, effective, microbatch_size
                    )
                ]
                effective_rows = []

    if read_count != count or effective_rows:
        raise ValueError(
            f"Expected {count} complete records; found {read_count}"
        )


def collate_microbatch(
    records: list[dict],
    pad_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Right-pad a microbatch, masking all padding labels."""
    if not records:
        raise ValueError("Empty microbatch")

    length = max(
        len(record["input_ids"])
        for record in records
    )

    result = {}

    for name, fill in (
        ("input_ids", pad_token_id),
        ("attention_mask", 0),
        ("labels", IGNORE_INDEX),
    ):
        result[name] = torch.tensor(
            [
                record[name] + [fill] * (
                    length - len(record[name])
                )
                for record in records
            ],
            dtype=torch.long,
            device=device,
        )

    return result


def supervised_token_count(
    batch: dict[str, torch.Tensor],
) -> int:
    """Count valid causal next-token prediction targets."""
    if batch["labels"].ndim != 2:
        raise ValueError("Expected two-dimensional labels")

    return int(
        (
            batch["labels"][:, 1:] != IGNORE_INDEX
        ).sum().item()
    )


def make_adamw(
    model: torch.nn.Module,
    learning_rate: float,
) -> torch.optim.AdamW:
    """AdamW with the assignment's weight decay exclusions."""
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dtype != torch.float32:
            raise ValueError(
                f"Trainable parameter must be FP32: "
                f"{name} ({param.dtype})"
            )

        if (
            name.split(".")[-1] == "bias"
            or param.ndim < 2
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    if not decay and not no_decay:
        raise ValueError("No trainable parameters")

    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def lr_multiplier(
    update_number: int,
    total_updates: int = TOTAL_UPDATES,
    warmup: int = WARMUP_UPDATES,
) -> float:
    """Linear warmup followed by cosine decay to zero."""
    if not 0 < warmup < total_updates:
        raise ValueError("Invalid warmup configuration")

    if update_number < 1:
        raise ValueError("Update numbers begin at 1")

    if update_number <= warmup:
        return update_number / warmup

    progress = min(
        1.0,
        (update_number - warmup) / (
            total_updates - warmup
        ),
    )

    return 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_updates: int = TOTAL_UPDATES,
    warmup: int = WARMUP_UPDATES,
):
    """Create the shared learning-rate scheduler."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda completed: lr_multiplier(
            completed + 1,
            total_updates,
            warmup,
        ),
    )


def train_effective_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    microbatches: list[list[dict]],
    pad_token_id: int,
    device: torch.device,
    *,
    regularizer=None,
) -> dict:
    """Attempt one optimizer update with token-weighted accumulation.

    The caller manages the data cursor and skipped-update retries.
    Checkpointing is implemented separately.
    """
    if (
        len(microbatches) != ACCUMULATION_STEPS
        or any(
            len(part) != MICROBATCH_SIZE
            for part in microbatches
        )
    ):
        raise ValueError(
            "Expected 16 microbatches of 2 examples"
        )

    targets = sum(
        sum(
            label != IGNORE_INDEX
            for label in record["labels"][1:]
        )
        for part in microbatches
        for record in part
    )

    if targets <= 0:
        raise ValueError("No supervised response tokens")

    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    model.train()

    for part in microbatches:
        batch = collate_microbatch(
            part, pad_token_id, device
        )

        n_tokens = supervised_token_count(batch)

        if n_tokens == 0:
            raise ValueError(
                "Microbatch has no supervised tokens"
            )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            output = model(
                **batch,
                use_cache=False,
            )

            # The model returns mean cross-entropy
            # over non-masked causal targets.
            unweighted = output.loss

            scaled_loss = unweighted * (
                n_tokens / targets
            )

        if not torch.isfinite(unweighted).item():
            raise FloatingPointError(
                "Non-finite microbatch SFT loss"
            )

        weighted_loss += (
            float(unweighted.detach().float().item())
            * n_tokens / targets
        )

        scaler.scale(scaled_loss).backward()

    # Unscale before computing and clipping gradient norm.
    scaler.unscale_(optimizer)
    omega = 0.0
    lambda_omega = 0.0
    if regularizer is not None:
        regularizer.assert_matches(model)
        omega = regularizer.add_to_unscaled_grad()
        lambda_omega = regularizer.coefficient * omega

    params = [
        p
        for group in optimizer.param_groups
        for p in group["params"]
        if p.grad is not None
    ]

    grad_norm = torch.nn.utils.clip_grad_norm_(
        params,
        GRAD_CLIP_NORM,
        error_if_nonfinite=False,
    )

    raw_grad_norm = float(grad_norm.detach().float().item())
    grad_norm_finite = math.isfinite(raw_grad_norm)
    if not grad_norm_finite and not scaler.is_enabled():
        raise FloatingPointError(
            "Non-finite gradient norm without enabled GradScaler"
        )

    previous_scale = scaler.get_scale()

    scaler.step(optimizer)
    scaler.update()

    stepped = (
        scaler.get_scale() >= previous_scale
        if scaler.is_enabled()
        else True
    )

    # Never advance LR when GradScaler skips an update.
    if stepped:
        scheduler.step()

    return {
        "sft_loss": weighted_loss,
        "omega": omega,
        "lambda_omega": lambda_omega,
        "total_loss": weighted_loss + lambda_omega,
        "supervised_tokens": targets,
        "grad_norm_before_clip": (
            raw_grad_norm if grad_norm_finite else None
        ),
        "grad_norm_finite": grad_norm_finite,
        "optimizer_step_completed": bool(stepped),
        "scale_before": float(previous_scale),
        "scale_after": float(scaler.get_scale()),
    }
