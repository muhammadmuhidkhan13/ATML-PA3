"""Task 1 report EVIDENCE collection. No plots or report writing.

Examples (from /workspace/ATML-PA3-linux):
  python scripts/15_collect_task1_evidence.py --stage preview
  python scripts/15_collect_task1_evidence.py --stage diagnostics --smoke --execute
  python scripts/15_collect_task1_evidence.py --stage qualitative --smoke --execute --condition baseline
  python scripts/15_collect_task1_evidence.py --stage diagnostics --execute
  python scripts/15_collect_task1_evidence.py --stage qualitative --execute
  python scripts/15_collect_task1_evidence.py --stage validate
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atml_pa3.task1.evidence import (
    ALL, TRAINED, collect_diagnostics, collect_qualitative,
    read_qualitative_yaml, select_probes, validate_collection,
    export_collection, QUALITATIVE_YAML_SHA256,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['preview', 'diagnostics', 'qualitative', 'validate', 'export'], required=True)
    ap.add_argument('--execute', action='store_true', help='Required for model loading and data collection')
    ap.add_argument('--smoke', action='store_true', help='Use isolated smoke output and only 4 probes/3 questions')
    ap.add_argument('--condition', choices=ALL, help='Optional single condition')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts' / 'task1')
    ap.add_argument('--yaml', type=Path, default=ROOT / 'artifacts' / 'task1' /
                    'qualitative-inputs' / 'first_plot_questions.yaml')
    args = ap.parse_args()
    artifacts = args.artifacts.resolve()
    out = artifacts / ('report-data-smoke-v1' if args.smoke else 'report-data-v1')
    if args.stage == 'preview':
        from atml_pa3.task1.evaluation.checkpoints import sha256_file
        if sha256_file(args.yaml) != QUALITATIVE_YAML_SHA256:
            raise ValueError('Qualitative YAML differs from pinned SHA256')
        prompts = read_qualitative_yaml(args.yaml)
        print(json.dumps({'mode': 'preview', 'yaml': str(args.yaml),
                          'qualitative_prompts': len(prompts),
                          'diagnostic_conditions': TRAINED,
                          'qualitative_conditions': ALL,
                          'output': str(out), 'gpu_used': False}, indent=2))
        # Probe preview does create the fixed selection file, but does not load a model.
        ids = select_probes(artifacts, out, smoke=args.smoke)
        for item in ids['entries'][:3]:
            print(f"{item['benchmark']} doc_id={item['doc_id']} prompt={item['prompt'][:120]!r}")
        return
    if args.stage == 'validate':
        print(json.dumps(validate_collection(out, smoke=args.smoke), indent=2))
        return
    if args.stage == 'export':
        if args.smoke or args.condition:
            ap.error('Export only applies to full official collection')
        print(json.dumps(export_collection(artifacts, out, args.yaml), indent=2))
        return
    if not args.execute:
        print(json.dumps({'status': 'DRY_RUN_ONLY', 'stage': args.stage,
                          'execute_required': True, 'output': str(out)}, indent=2))
        return
    if not args.device.startswith('cuda'):
        raise ValueError('Official data collection requires CUDA; no CPU model fallback')
    if args.stage == 'diagnostics':
        if args.condition == 'baseline':
            ap.error('No baseline weight diagnostic; choose a trained condition')
        conditions = (args.condition,) if args.condition else TRAINED
        collect_diagnostics(artifacts, out, args.device, smoke=args.smoke, conditions=conditions)
    elif args.stage == 'qualitative':
        conditions = (args.condition,) if args.condition else ALL
        collect_qualitative(artifacts, out, args.device, args.yaml,
                            smoke=args.smoke, conditions=conditions)
    print(json.dumps({'status': 'STAGE_FINISHED', 'stage': args.stage,
                      'smoke': args.smoke, 'conditions': conditions,
                      'output': str(out)}, indent=2))


if __name__ == '__main__':
    main()
