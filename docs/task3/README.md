# Task 3 · Test-time compute

## Offline fixed-budget and adaptive evidence

`artifacts/task3/prepared-v1/` contains the IDs and ten rollout orders for each MATH and MathVista question. Each dataset has 200 development and 800 held-out questions, with 32 supplied rollouts per question. `offline-v1/` contains the frozen configuration, per-question score arrays and outcomes, aggregates, signal diagnostics, fusion ablations, illustrations and figures. It covers FirstAns, oracle coverage, self-consistency, confidence measures, VisConf and RWC. `semantics-v2/` contains the frozen scorer settings and cached correctness/equivalence records. `adaptive-v1/` contains the development threshold grid, frozen choices, per-question decisions and costs, held-out comparisons and failure examples. The analysis seed is 6304.

Run these read-only checks on the published fixed-budget evidence.

```sh
python scripts/task3_export.py --split development --expected-questions 200 --validate-only
python scripts/task3_export.py --split evaluation --expected-questions 800 --validate-only
```

To rebuild figures, copy `artifacts/task3/offline-v1` to a scratch output tree and call the same entrypoint with `--output` pointing to the copy and without `--validate-only`. Run `scripts/task3_compute_analysis.py --output` against a copy of `adaptive-v1` to regenerate the derived compute-analysis tables.

## Rebuild offline results with instructor inputs

Obtain the instructor's original folder and retain its `README.MD` and five Parquet files under `ATML-PA3-Data/`. The format, field definitions and original export notes are in [the rollout data README](../task3-rollout-data.md). The public provenance projection records the input hashes and sizes. [Instructor rollout folder](https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23?usp=sharing)

Use a scratch checkout for a new run. Restore the original private `prepared-v1/source_manifest.json` if independently validating the original adaptive freeze with `task3_part2_validate.py`. The published copy of its provenance has private paths removed and an explicitly different checksum. A fresh preparation writes its own source manifest and therefore creates a separate freeze. Move or copy the published prepared, semantic, fixed-budget and adaptive directories outside the scratch experiment's default output locations before building new outputs. Keep `reference-v1/manifest.json` so the fetch script uses the pinned upstream metadata.

```sh
python scripts/task3_fetch_resources.py
python scripts/task3_prepare.py
python scripts/task3_cache.py
python scripts/task3_offline.py --split development
python scripts/task3_offline.py --split evaluation
python scripts/task3_illustrations.py --split evaluation
python scripts/task3_adaptive.py --stage development
python scripts/task3_adaptive.py --stage evaluation
python scripts/task3_part2_export.py
python scripts/task3_part2_validate.py
```

This pipeline is CPU work. It checks full file hashes, rebuilds answer-equivalence caches and evaluates every prefix and policy. The saved outputs and earlier adaptive audit are available for checks without rebuilding the cache.

## Completed live run and CPU verification

The RTX 3090 run compares three parallel 1,024-token draws, a single 3,072-token draw, and a 2,048-token refinement seeded by the first parallel answer. There are 250 unique measured calls and 150 strategy-question outcomes for 50 held-out MATH questions. `artifacts/task3/live-assets-v1/` holds the selected questions and references, scorer and model locks. `artifacts/task3/runs/live-v1/` holds the original run, call and grade indexes, device/preflight proof, raw responses and token/time/memory measurements. `artifacts/task3/exports/live-v1/` holds the validated tables, qualitative examples and seven figures.

```sh
python scripts/task3_live_export.py --assets artifacts/task3/live-assets-v1 --run artifacts/task3/runs/live-v1 --output results/task3/live-replay
```

This reconstructs strategies from hashed call and grading records, checks completion and regenerates tables and figures without model weights or CUDA.

## Run a new live experiment

Use one dedicated RTX 3090/3090 Ti or supported 80 GB A100 for all strategies, with the v6 runtime stack (Python 3.12.15, PyTorch 2.14.1+cu130 and `environment/task3-live-requirements.txt`). The model-file lock checks the pinned snapshot. The two shards total about 5.75 GiB and the preflight exercises the full length envelope. The two synthetic stress prompts may exceed their nominal input targets by up to 256 tokens. Official rendered prompt limits are 4,096 initial tokens and 8,192 revision tokens.

```sh
hf download Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 --local-dir model
python scripts/task3_gpu_preflight.py --assets artifacts/task3/live-assets-v1 --model-path model --output results/task3/preflight --execute-preflight
python scripts/task3_live.py --assets artifacts/task3/live-assets-v1 --model-path model --preflight results/task3/preflight --output results/task3/new-live --execute-generation
python scripts/task3_live_export.py --assets artifacts/task3/live-assets-v1 --run results/task3/new-live --output results/task3/new-export
```

Repeat the generation command to resume on the same compatible runtime. A completed output directory is sealed by the live validator. See `scripts/task3_live.py --help` for validation-only mode.

## Sources

- Released question metadata and scoring conventions. [MathVista official implementation](https://github.com/lupantech/MathVista/tree/c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb)
- Semantic answer verification. [Math-Verify](https://github.com/huggingface/Math-Verify)
- Rollout and live model provenance. [MATH dataset](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval) · [Pinned Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/aa8e72537993ba99e69dfaafa59ed015b17504d1)
- Selection, confidence and refinement references. [Self-Certainty](https://arxiv.org/abs/2502.18581) · [VisConf](https://arxiv.org/abs/2610.24526) · [Self-Consistency](https://arxiv.org/abs/2203.11171) · [Self-Refine](https://arxiv.org/abs/2303.17651) · [Adaptive-Consistency](https://arxiv.org/abs/2305.11860)
