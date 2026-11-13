"""Pinned streaming Task 1 data preparation; no model weights or GPU required."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile

from .normalization import NORMALIZATION_VERSION, build_leakage_set
from .selection import canonical_bytes, select_examples


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--full-build", action="store_true")

    parser.add_argument("--count", type=int)
    parser.add_argument("--max-scanned", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--metamath-revision", default="main")
    parser.add_argument("--gsm8k-revision", default="main")

    args = parser.parse_args(argv)

    if args.dry_run:
        args.count = 16 if args.count is None else args.count
        args.max_scanned = (
            1000 if args.max_scanned is None else args.max_scanned
        )

        if not 1 <= args.count <= 100:
            parser.error("Dry runs accept only 1..100 examples")
    else:
        args.count = 40000 if args.count is None else args.count

        if args.count != 40000:
            parser.error(
                "A full build must contain exactly 40000 eligible examples"
            )

    if (
        args.max_scanned is not None
        and args.max_scanned < args.count
    ):
        parser.error("max-scanned must be at least count")

    if args.output_dir.exists():
        parser.error(
            "Output directory already exists; "
            "choose a new path to preserve prior work"
        )

    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    api = HfApi()

    model_id = "Qwen/Qwen3-0.6B"
    meta_id = "meta-math/MetaMathQA"
    gsm_id = "openai/gsm8k"

    model_sha = api.model_info(
        model_id,
        revision=args.model_revision,
    ).sha

    meta_sha = api.dataset_info(
        meta_id,
        revision=args.metamath_revision,
    ).sha

    gsm_sha = api.dataset_info(
        gsm_id,
        revision=args.gsm8k_revision,
    ).sha

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_sha,
        use_fast=True,
    )

    config_path = Path(
        hf_hub_download(
            model_id,
            "tokenizer_config.json",
            revision=model_sha,
        )
    )

    if config_path.parent.name != model_sha:
        raise AssertionError(
            "Tokenizer cache path does not match resolved snapshot SHA"
        )

    gsm_rows = list(
        load_dataset(
            gsm_id,
            "main",
            split="test",
            streaming=True,
            revision=gsm_sha,
        )
    )

    if len(gsm_rows) != 1319:
        raise AssertionError(
            f"Expected complete GSM8K test set of 1319, "
            f"found {len(gsm_rows)}"
        )

    leakage = build_leakage_set(gsm_rows)

    rows = load_dataset(
        meta_id,
        split="train",
        streaming=True,
        revision=meta_sha,
    )

    args.output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.partial-",
            dir=args.output_dir.parent,
        )
    )

    data_path = staging / "examples.jsonl"
    data_hash = hashlib.sha256()
    written = 0

    try:
        with data_path.open("wb") as data_file:

            def write_example(example):
                nonlocal written

                line = canonical_bytes(example) + b"\n"
                data_file.write(line)
                data_hash.update(line)
                written += 1

                if written % 1000 == 0:
                    print(
                        f"Prepared {written}/{args.count} "
                        "eligible examples",
                        flush=True,
                    )

            _, stats = select_examples(
                rows,
                tokenizer,
                leakage,
                count=args.count,
                max_scanned=args.max_scanned,
                on_accept=write_example,
                retain_examples=False,
            )

        if written != args.count:
            raise AssertionError(
                f"Writer produced {written}/{args.count} examples"
            )

        metadata = {
            "schema_version": 1,
            "mode": (
                "dry_run"
                if args.dry_run
                else "full_build"
            ),
            "seed": 6304,
            "model": {
                "id": model_id,
                "snapshot_sha": model_sha,
            },
            "datasets": {
                "metamath": {
                    "id": meta_id,
                    "revision": meta_sha,
                    "split": "train",
                },
                "gsm8k": {
                    "id": gsm_id,
                    "revision": gsm_sha,
                    "config": "main",
                    "split": "test",
                    "rows": len(gsm_rows),
                    "normalized_unique_questions": len(leakage),
                    "raw_rows_sha256": hashlib.sha256(
                        canonical_bytes(gsm_rows)
                    ).hexdigest(),
                    "leakage_set_sha256": hashlib.sha256(
                        canonical_bytes(sorted(leakage))
                    ).hexdigest(),
                },
            },
            "normalization": {
                "version": NORMALIZATION_VERSION,
                "punctuation": "preserved",
                "approved": True,
            },
            "rules": {
                "max_formatted_tokens": 2048,
                "shuffle": False,
                "truncate": False,
                "thinking": False,
                "source_index": "zero-based",
                "raw_source_preserved": True,
                "rejection_precedence": [
                    "empty_query",
                    "empty_response",
                    "gsm8k_overlap",
                    "over_length",
                ],
                "supervision": (
                    "supplied response and assistant EOT only; "
                    "padding label -100"
                ),
                "special_token_source_text": (
                    "fail visibly; do not silently reject"
                ),
            },
            "selection": stats,
            "tokenizer_files_sha256": {
                path.name: sha256(path)
                for path in sorted(config_path.parent.iterdir())
                if path.is_file()
                and path.suffix in {
                    ".json",
                    ".jinja",
                    ".txt",
                }
            },
            "chat_template_sha256": hashlib.sha256(
                tokenizer.chat_template.encode("utf-8")
            ).hexdigest(),
            "environment": {
                "python": platform.python_version(),
                **{
                    package: version(package)
                    for package in [
                        "transformers",
                        "tokenizers",
                        "datasets",
                        "huggingface_hub",
                    ]
                },
            },
            "command": (
                sys.argv
                if argv is None
                else argv
            ),
        }

        root = Path(__file__).resolve().parents[3]

        metadata["implementation_sha256"] = {
            str(path.relative_to(root)).replace("\\", "/"): sha256(path)
            for path in sorted(
                Path(__file__).parent.glob("*.py")
            )
        }

        metadata["output_sha256"] = {
            data_path.name: data_hash.hexdigest()
        }

        meta_path = staging / "metadata.json"

        meta_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        sums_path = staging / "SHA256SUMS"

        sums_path.write_text(
            f"{data_hash.hexdigest()}  examples.jsonl\n"
            f"{sha256(meta_path)}  metadata.json\n",
            encoding="utf-8",
        )

        staging.replace(args.output_dir)

    except BaseException:
        shutil.rmtree(
            staging,
            ignore_errors=True,
        )
        raise

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                **stats,
            },
            indent=2,
        )
    )

    print(
        "Pinned revisions:",
        model_sha,
        meta_sha,
        gsm_sha,
    )


if __name__ == "__main__":
    main()
