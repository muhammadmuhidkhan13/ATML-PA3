"""Reproducible scalar score matrix from verified full-split evaluation outputs."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from .conditions import CONDITIONS, BENCHMARKS


def score_matrix(root: Path) -> list[dict]:
    root = Path(root)
    scores = {}
    for condition in CONDITIONS:
        for benchmark, spec in BENCHMARKS.items():
            path = root / condition / benchmark / 'summary.json'
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding='utf-8'))
            if (record.get('status') != 'COMPLETE_FULL_SPLIT' or
                record.get('condition') != condition or record.get('benchmark') != benchmark or
                record.get('examples') != spec['count'] or
                record.get('official_metric_key') != spec['metric']):
                raise ValueError(f'Incomplete or incorrect result: {path}')
            scores[(condition, benchmark)] = float(record['primary_metric'])
    rows = []
    for condition in CONDITIONS:
        for benchmark in BENCHMARKS:
            if (condition, benchmark) not in scores:
                continue
            value = scores[(condition, benchmark)]
            baseline = scores.get(('baseline', benchmark))
            rows.append({'condition': condition, 'benchmark': benchmark,
                         'primary_metric_key': BENCHMARKS[benchmark]['metric'],
                         'score': value,
                         'delta_percentage_points': (100 * (value - baseline)
                                                     if baseline is not None else None)})
    return rows


def write_csv(rows: list[dict], destination: Path):
    if not rows:
        raise ValueError('No complete results')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('condition', 'benchmark', 'primary_metric_key',
                                               'score', 'delta_percentage_points'))
        writer.writeheader()
        writer.writerows(rows)
