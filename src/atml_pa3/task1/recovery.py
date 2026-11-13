"""Atomic, resumable Task 1 checkpoints. Does not start training."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch

from .train_core import EFFECTIVE_BATCH_SIZE, TOTAL_UPDATES

VERSION = 1


def _json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
    }


def _set_rng_state(state):
    if state["torch_cuda"]:
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count()
            != len(state["torch_cuda"])
        ):
            raise RuntimeError(
                "Checkpoint CUDA RNG state does not match GPU count"
            )

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    destination,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    run_spec,
    completed_updates,
    skipped_updates=0,
    state_kind="full",
    extra_state_keys=(),
):
    """Publish checkpoint atomically after a complete optimizer update."""

    if (
        not isinstance(completed_updates, int)
        or not (0 <= completed_updates <= TOTAL_UPDATES)
    ):
        raise ValueError("Invalid completed optimizer update number")

    if (
        not isinstance(skipped_updates, int)
        or skipped_updates < 0
    ):
        raise ValueError("Invalid skipped-update count")

    if state_kind not in ("full", "trainable"):
        raise ValueError("Invalid checkpoint state_kind")

    if not isinstance(run_spec, dict) or not run_spec:
        raise ValueError("run_spec must be a nonempty dictionary")

    spec_hash = hashlib.sha256(_json_bytes(run_spec)).hexdigest()
    destination = Path(destination)

    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace checkpoint: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    full_state = model.state_dict()

    trainable_keys = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    selected = trainable_keys | set(extra_state_keys)

    if state_kind == "trainable":
        if not trainable_keys:
            raise ValueError("No trainable parameters to checkpoint")

        if not selected.issubset(full_state):
            raise ValueError("Required state keys are missing")

        model_state = {
            name: tensor
            for name, tensor in full_state.items()
            if name in selected
        }
    else:
        model_state = full_state

    metadata = {
        "format_version": VERSION,
        "run_spec_sha256": spec_hash,
        "completed_updates": completed_updates,
        "next_example_offset": (
            completed_updates * EFFECTIVE_BATCH_SIZE
        ),
        "skipped_updates": skipped_updates,
        "state_kind": state_kind,
        "saved_model_keys": sorted(model_state),
        "extra_state_keys": sorted(extra_state_keys),
    }

    payload = {
        "metadata": metadata,
        "run_spec": run_spec,
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": _rng_state(),
    }

    temporary = destination.parent / (
        ".incomplete-"
        + destination.name
        + "-"
        + uuid.uuid4().hex
    )

    temporary.mkdir()

    try:
        state_path = temporary / "state.pt"
        torch.save(payload, state_path)

        metadata["state_sha256"] = _sha256_file(state_path)

        (temporary / "metadata.json").write_bytes(
            _json_bytes(metadata) + b"\n"
        )

        if destination.exists():
            raise FileExistsError(
                f"Checkpoint appeared during save: {destination}"
            )

        temporary.rename(destination)

    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return metadata


def load_checkpoint(
    source,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    expected_run_spec,
):
    """Restore a trusted locally generated checkpoint."""

    source = Path(source)

    metadata = json.loads(
        (source / "metadata.json").read_text(encoding="utf-8")
    )

    if metadata.get("format_version") != VERSION:
        raise ValueError("Unrecognized checkpoint format")

    expected_hash = hashlib.sha256(
        _json_bytes(expected_run_spec)
    ).hexdigest()

    if metadata.get("run_spec_sha256") != expected_hash:
        raise ValueError(
            "Run specification differs; cannot safely resume"
        )

    if metadata.get("next_example_offset") != (
        metadata.get("completed_updates") * EFFECTIVE_BATCH_SIZE
    ):
        raise ValueError("Invalid checkpoint data cursor")

    if not (
        0 <= metadata["completed_updates"] <= TOTAL_UPDATES
    ):
        raise ValueError("Invalid completed-update count")

    if metadata.get("skipped_updates", -1) < 0:
        raise ValueError("Invalid skipped-update count")

    state_path = source / "state.pt"

    if _sha256_file(state_path) != metadata["state_sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch")

    # Only load checkpoints created by our own trusted code.
    payload = torch.load(
        state_path,
        map_location="cpu",
        weights_only=False,
    )

    embedded_metadata = {
        key: value
        for key, value in metadata.items()
        if key != "state_sha256"
    }

    if payload["metadata"] != embedded_metadata:
        raise ValueError("Checkpoint payload/metadata mismatch")

    if _json_bytes(payload["run_spec"]) != (
        _json_bytes(expected_run_spec)
    ):
        raise ValueError(
            "Checkpoint payload run specification mismatch"
        )

    saved_keys = set(payload["model"])

    if saved_keys != set(metadata["saved_model_keys"]):
        raise ValueError("Saved model keys differ from metadata")

    state_kind = metadata["state_kind"]

    if state_kind == "full":
        model.load_state_dict(payload["model"], strict=True)

    elif state_kind == "trainable":
        required_keys = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        required_keys |= set(metadata["extra_state_keys"])

        if (
            not required_keys
            or not required_keys.issubset(saved_keys)
        ):
            raise ValueError(
                "Checkpoint does not cover trainable/extra tensors"
            )

        if not saved_keys.issubset(model.state_dict()):
            raise ValueError("Unknown model state keys")

        result = model.load_state_dict(
            payload["model"],
            strict=False,
        )

        if result.unexpected_keys:
            raise ValueError("Unexpected adapter checkpoint keys")

    else:
        raise ValueError("Unknown checkpoint state_kind")

    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])

    optimizer.zero_grad(set_to_none=True)
    _set_rng_state(payload["rng"])

    return metadata
