"""Compare uninterrupted Qwen smoke training with a fresh-process resume."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atml_pa3.task1.fresh_resume import (
    METHODS,
    STAGES,
    compare_final_checkpoints,
    prepare_workspace,
    run_worker,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-smoke", action="store_true")
    parser.add_argument("--method", choices=METHODS, default="lora4")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "artifacts/task1/data/metamath-40000-v2",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-stage", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--workspace", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker_stage is not None:
        if args.workspace is None:
            parser.error("Internal worker requires --workspace")
        print(json.dumps(run_worker(args.workspace, args.worker_stage), indent=2))
        return
    if not args.execute_smoke:
        parser.error("Pass --execute-smoke to authorize bounded optimizer updates")

    output_dir = args.output_dir or (
        ROOT / f"artifacts/task1/fresh-resume-{args.method}-v1"
    )
    prepare_workspace(
        args.data_dir, output_dir, method=args.method, max_tokens=args.max_tokens
    )
    script = Path(__file__).resolve()
    for stage in STAGES:
        subprocess.run(
            [
                sys.executable, "-X", "utf8", "-u", str(script),
                "--worker-stage", stage, "--workspace", str(output_dir.resolve()),
            ],
            check=True,
        )
    print(json.dumps(compare_final_checkpoints(output_dir), indent=2), flush=True)


if __name__ == "__main__":
    main()
