"""Read-only, one-benchmark Task 1 evaluation CLI. Never trains."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atml_pa3.task1.evaluation.conditions import CONDITIONS, BENCHMARKS, checkpoint_directory
from atml_pa3.task1.evaluation.checkpoints import validate_training_checkpoint, read_inference_state


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--condition', choices=tuple(CONDITIONS), required=True)
    ap.add_argument('--benchmark', choices=tuple(BENCHMARKS), required=True)
    ap.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts/task1')
    ap.add_argument('--checkpoint', type=Path, help='Override default official checkpoint directory')
    ap.add_argument('--export', type=Path, help='Verified inference export directory')
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--limit', type=int, help='SMOKE ONLY: evaluate N examples (not for official results)')
    ap.add_argument('--execute-evaluation', action='store_true', help='Actually load model and evaluate; default validates only')
    args = ap.parse_args(argv)
    checkpoint = args.checkpoint
    if args.condition != 'baseline' and checkpoint is None and args.export is None:
        checkpoint = checkpoint_directory(args.artifacts, args.condition)
    if args.condition == 'baseline' and (checkpoint or args.export):
        ap.error('Baseline cannot accept a checkpoint or export')
    if checkpoint is not None and args.export is not None:
        ap.error('Choose --checkpoint or --export')
    if checkpoint is not None:
        validate_training_checkpoint(checkpoint, args.condition)
    if args.export is not None:
        read_inference_state(args.export, args.condition)
    destination = args.output_dir or args.artifacts / 'evaluation' / args.condition / args.benchmark
    details = {
        'condition': args.condition, 'benchmark': args.benchmark,
        'expected_samples': BENCHMARKS[args.benchmark]['count'],
        'checkpoint': str(checkpoint) if checkpoint else None,
        'export': str(args.export) if args.export else None,
        'output_dir': str(destination),
        'mode': 'execution' if args.execute_evaluation else 'validation_only',
        'smoke_only': args.limit is not None,
    }
    print(json.dumps(details, indent=2))
    if not args.execute_evaluation:
        return
    from atml_pa3.task1.evaluation.harness import run_benchmark
    result = run_benchmark(condition=args.condition, benchmark=args.benchmark,
                           output_dir=destination, device=args.device,
                           checkpoint=checkpoint, export=args.export,
                           batch_size=args.batch_size, limit=args.limit)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
