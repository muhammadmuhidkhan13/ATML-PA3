"""Export compact *trusted* Task 1 recovery checkpoints, no model loading.

Usage: python scripts/13_export_task1_inference.py --condition full_mid \\
  --checkpoint <run>/checkpoints/update_00625 --output <new>/full_mid
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atml_pa3.task1.evaluation.conditions import CONDITIONS
from atml_pa3.task1.evaluation.checkpoints import validate_training_checkpoint, export_inference_state


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--condition', required=True, choices=[n for n in CONDITIONS if n != 'baseline'])
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--execute-export', action='store_true')
    args = ap.parse_args(argv)
    meta = validate_training_checkpoint(args.checkpoint, args.condition)
    print(json.dumps({'condition': args.condition, 'verified_update': meta['completed_updates'],
                      'source': str(args.checkpoint), 'output': str(args.output),
                      'mode': 'export' if args.execute_export else 'validation_only'}, indent=2))
    if args.execute_export:
        manifest = export_inference_state(args.checkpoint, args.condition, args.output)
        print(json.dumps({'status': 'EXPORTED', 'manifest': manifest}, indent=2, default=str))


if __name__ == '__main__':
    main()
