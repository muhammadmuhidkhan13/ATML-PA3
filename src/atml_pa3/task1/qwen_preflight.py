"""Read-only, single-microbatch Qwen adapter compatibility check."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .model_setup import AdapterLinear
from .train_core import (
    collate_microbatch,
    make_adamw,
    supervised_token_count,
)


def select_short_examples(
    data_path: Path,
    max_tokens: int = 128,
    count: int = 2,
    min_tokens: int = 1,
) -> list[dict]:
    """Choose complete examples in a length band; never truncate."""
    if (
        not 1 <= min_tokens <= max_tokens <= 2048
        or max_tokens < 32
        or count not in (1, 2)
    ):
        raise ValueError(
            "Use 1 <= min_tokens <= max_tokens <= 2048, "
            "max_tokens >= 32, and a microbatch of 1 or 2"
        )

    chosen = []

    with Path(data_path).open(
        "r", encoding="utf-8"
    ) as source:
        for line in source:
            row = json.loads(line)
            ids = row["input_ids"]
            labels = row["labels"]

            if not min_tokens <= len(ids) <= max_tokens:
                continue

            if (
                len(labels) != len(ids)
                or len(row["attention_mask"]) != len(ids)
            ):
                raise ValueError("Malformed saved training example")

            if not any(v != -100 for v in labels[1:]):
                raise ValueError(
                    "Example lacks response supervision"
                )

            chosen.append(row)

            if len(chosen) == count:
                return chosen

    raise ValueError(
        f"Not enough complete examples in the "
        f"{min_tokens}..{max_tokens} token band"
    )


def check_label_roundtrip(
    rows: list[dict],
    tokenizer,
) -> None:
    """Verify exact response-plus-EOT supervision."""
    for row in rows:
        labels = row["labels"]
        tokens = row["input_ids"]

        if any(
            label != -100 and label != token
            for token, label in zip(tokens, labels)
        ):
            raise ValueError(
                "Supervision labels contain wrong token IDs"
            )

        supervised = [
            v for v in labels if v != -100
        ]

        actual = tokenizer.decode(
            supervised,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        expected = (
            row["source"]["response"] + "<|im_end|>"
        )

        if actual != expected:
            raise ValueError(
                "Supervised text does not match exact response + EOT"
            )


def single_backward_probe(
    model,
    rows: list[dict],
    pad_token_id: int,
    device,
) -> dict:
    """Forward/backward and gradient inspection; no optimizer step."""
    device = torch.device(device)

    adapters = [
        m for m in model.modules()
        if isinstance(m, AdapterLinear)
    ]

    if not adapters:
        raise ValueError(
            "Expected at least one custom LoRA adapter"
        )

    model.train()

    optimizer = make_adamw(model, 2e-4)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
        init_scale=256.0,
    )

    optimizer.zero_grad(set_to_none=True)

    batch = collate_microbatch(
        rows, pad_token_id, device
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        output = model(
            **batch,
            use_cache=False,
        )

        loss = output.loss

    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(
            "Non-finite Qwen SFT loss"
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    norm_sq = 0.0
    b_grads_nonzero = 0
    grad_tensors = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            if param.grad is not None:
                raise AssertionError(
                    f"Frozen parameter acquired gradient: {name}"
                )

            continue

        if param.grad is None:
            raise AssertionError(
                f"Trainable parameter missing gradient: {name}"
            )

        gradient = param.grad.detach().float()

        if not bool(
            torch.isfinite(gradient).all().item()
        ):
            raise FloatingPointError(
                f"Non-finite gradient: {name}"
            )

        norm_sq += float(
            gradient.square().sum().item()
        )

        grad_tensors += 1

        if (
            name.endswith(".lora_B")
            and bool(
                torch.count_nonzero(gradient).item()
            )
        ):
            b_grads_nonzero += 1

    if not b_grads_nonzero:
        raise AssertionError(
            "No LoRA B matrix received a nonzero gradient"
        )

    report = {
        "loss": float(
            loss.detach().float().item()
        ),
        "valid_response_targets": supervised_token_count(
            batch
        ),
        "max_microbatch_tokens": int(
            batch["input_ids"].shape[1]
        ),
        "adapter_modules": len(adapters),
        "gradient_tensors": grad_tensors,
        "nonzero_B_gradients": b_grads_nonzero,
        "unscaled_gradient_norm": math.sqrt(norm_sq),
        "peak_cuda_memory_gib": (
            round(
                torch.cuda.max_memory_allocated(device)
                / (1024 ** 3),
                3,
            )
            if device.type == "cuda"
            else None
        ),
        "optimizer_steps": 0,
    }

    optimizer.zero_grad(set_to_none=True)

    return report
