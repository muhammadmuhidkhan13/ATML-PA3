"""Immutable official Task 1 checkpoint and benchmark selection."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Condition:
    name: str
    method: str | None
    update: int

CONDITIONS = {
    item.name: item for item in (
        Condition("baseline", None, 0),
        Condition("full_mid", "full", 625),
        Condition("full_final", "full", 1250),
        Condition("l2sp_final", "l2sp", 1250),
        Condition("lora4_final", "lora4", 1250),
        Condition("lora16_final", "lora16", 1250),
        Condition("oplora16_final", "oplora16", 1250),
    )
}

RUN_DIRECTORY_DEFAULTS = {
    "full": "official-full-v1", "l2sp": "official-l2sp-v1",
    "lora4": "official-lora4-v1", "lora16": "official-lora16-v1",
    "oplora16": "official-oplora16-v1",
}

BENCHMARKS = {
    "gsm8k_cot": {"count": 1319, "split": "test", "metric": "exact_match,flexible-extract", "chat": True},
    "hellaswag": {"count": 10042, "split": "validation", "metric": "acc_norm,none", "chat": False},
    "boolq": {"count": 3270, "split": "validation", "metric": "acc,none", "chat": False},
    "arc_easy": {"count": 2376, "split": "test", "metric": "acc_norm,none", "chat": False},
}


def checkpoint_directory(artifacts: Path, condition: str, run_dirs: dict | None = None) -> Path | None:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown evaluation condition: {condition}")
    c = CONDITIONS[condition]
    if c.method is None:
        return None
    folder = (run_dirs or {}).get(c.method, RUN_DIRECTORY_DEFAULTS[c.method])
    return Path(artifacts) / folder / "checkpoints" / f"update_{c.update:05d}"
