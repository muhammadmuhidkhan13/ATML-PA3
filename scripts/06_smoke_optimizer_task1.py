"""One explicit LoRA smoke optimizer update, never official training."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atml_pa3.task1.smoke_step import run_smoke_update


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--execute-smoke", action="store_true"
    )
    parser.add_argument(
        "--method", choices=("lora4", "lora16", "oplora16"), default="lora4"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            ROOT / "artifacts/task1/data/metamath-40000-v2"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args(argv)

    if not args.execute_smoke:
        parser.error(
            "Pass --execute-smoke to authorize one optimizer update"
        )

    result = run_smoke_update(
        args.data_dir,
        args.run_dir or ROOT / f"artifacts/task1/smoke-{args.method}-update1-v1",
        args.max_tokens,
        method=args.method,
    )

    print(
        json.dumps(result, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
