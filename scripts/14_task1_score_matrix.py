"""Collect complete Task 1 benchmark results as CSV; no model required."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atml_pa3.task1.evaluation.report import score_matrix, write_csv

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluation-dir', type=Path, default=ROOT/'artifacts/task1/evaluation')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows = score_matrix(args.evaluation_dir)
    write_csv(rows, args.output)
    print(f'Wrote {len(rows)} verified score rows to {args.output}')
