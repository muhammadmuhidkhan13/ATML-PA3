"""Smoke-test pinned Qwen3 without updating model weights."""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0, str(ROOT / "src")
)

import numpy as np
import torch

from atml_pa3.task1.model_setup import (
    MODEL_REVISION,
    SEED,
    load_qwen_model,
)
from atml_pa3.task1.qwen_preflight import (
    check_label_roundtrip,
    select_short_examples,
    single_backward_probe,
)
from atml_pa3.task1.run_controller import (
    FROZEN_DATA_SHA256,
)
from atml_pa3.task1.train_core import verify_dataset


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--method",
        choices=("lora4", "lora16"),
        default="lora4",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--min-tokens",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(1, 2),
        default=2,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            ROOT / "artifacts" / "task1"
            / "data" / "metamath-40000-v2"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    args = parser.parse_args()

    if args.output_dir is not None and args.output_dir.exists():
        raise FileExistsError(
            f"Refusing existing output directory: {args.output_dir}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Real Qwen preflight requires CUDA"
        )

    device = torch.device("cuda:0")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(
        "Verifying frozen dataset SHA-256...",
        flush=True,
    )

    data_path = verify_dataset(
        args.data_dir,
        FROZEN_DATA_SHA256,
    )

    rows = select_short_examples(
        data_path,
        args.max_tokens,
        args.batch_size,
        args.min_tokens,
    )

    print(
        "Selected saved source indices:",
        [r["source_index"] for r in rows],
        flush=True,
    )

    print(
        "Selected complete sequence lengths:",
        [len(r["input_ids"]) for r in rows],
        flush=True,
    )

    print(
        "Loading pinned Qwen3-0.6B with",
        args.method,
        flush=True,
    )

    model, tokenizer, info = load_qwen_model(
        args.method,
        device=device,
    )

    check_label_roundtrip(
        rows, tokenizer
    )

    if (
        info.adapter_modules != 196
        or info.state_kind != "trainable"
    ):
        raise AssertionError(
            "Unexpected real Qwen adapter layout"
        )

    report = single_backward_probe(
        model,
        rows,
        tokenizer.pad_token_id,
        device,
    )

    print(
        "REAL QWEN PREFLIGHT PASSED",
        flush=True,
    )

    result = {
        "status": "REAL QWEN LENGTH-BAND PREFLIGHT PASSED",
        "smoke_only": True,
        "method": args.method,
        "model_revision": MODEL_REVISION,
        "frozen_data_sha256": FROZEN_DATA_SHA256,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "selected_source_indices": [r["source_index"] for r in rows],
        "selected_sequence_lengths": [len(r["input_ids"]) for r in rows],
        "trainable_parameters": info.trainable_parameters,
        **report,
    }

    print(json.dumps(result, indent=2), flush=True)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print(
        "No optimizer step, no checkpoint, "
        "no official training.",
        flush=True,
    )


if __name__ == "__main__":
    main()
