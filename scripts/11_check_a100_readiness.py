"""Run bounded one-update readiness probes on one A100 80GB GPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atml_pa3.task1.a100_preflight import METHODS, run_a100_preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-smoke", action="store_true")
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "artifacts/task1/data/metamath-40000-v2",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--min-vram-gib", type=float, default=75.0)
    args = parser.parse_args(argv)
    if not args.execute_smoke:
        parser.error("Pass --execute-smoke to authorize bounded optimizer updates")
    result = run_a100_preflight(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        methods=tuple(args.methods),
        min_vram_gib=args.min_vram_gib,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
