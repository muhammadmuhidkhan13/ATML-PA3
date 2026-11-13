# Task 1 · LLM fine-tuning

The experiment uses Qwen/Qwen3-0.6B at `c1899de289a04d12100db370d81485cdf75e47ca`, seed 6304, the first 40,000 eligible MetaMathQA rows in source order, and the frozen GSM8K test leakage filter. Method settings and checkpoint schedules are recorded in `artifacts/task1/official-*/run_spec.json`. The official run has 1,250 updates and checkpoints at 312, 625, 937 and 1,250. Training code includes full fine-tuning, an L2-SP penalty and three low-rank adapter variants.

## Rebuild saved tables on CPU

```sh
python scripts/14_task1_score_matrix.py --output results/task1/benchmark_scores.csv
python task1-report-review/RQ4_statistical_analysis/run_task1_rq4_stats.py --project . --output results/task1/rq4
```

`artifacts/task1/evaluation/` contains 28 verified condition/benchmark cells for GSM8K, HellaSwag, BoolQ and ARC-Easy. Each cell includes its evaluation summary and a compact per-example projection with ordered example IDs, hashes and primary metric values. The GSM8K cells also retain generation-token audits. The projections support the score and paired-statistics scripts. Text-dependent RQ1 scripts require the full local harness sample records at their documented locations. Saved compact RQ1 tables and figures are included.

`artifacts/task1/report-data-v1/` contains the weight-space, activation, training and resource evidence and exported CSVs. Probe IDs and starting-input hashes are under `activation/`. `task1-report-review/RQ2_lora_spectrum/` contains singular-value outputs from the pinned LoRA-16 final checkpoint. `task1-report-review/RQ5_qualitative_analysis/` contains the 168 responses and the 42-row manual annotation workbook.

## Rebuild inputs, train and evaluate

The following commands load external data or model assets. Restore an exact checkpoint or run training on suitable CUDA hardware for non-baseline evaluation. Official training records specify FP16 and the original A100 environment. The dataset preparation command preserves the source order and rejects overlap and overlength examples.

```sh
python scripts/03_prepare_task1_data.py --full-build --output-dir results/task1/data --model-revision c1899de289a04d12100db370d81485cdf75e47ca --metamath-revision aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18 --gsm8k-revision 740312add88f781978c0658806c59bc2815b9866
python scripts/07_train_task1.py --method full --data-dir results/task1/data --run-dir results/task1/full --execute-training
python scripts/12_evaluate_task1.py --condition full_final --benchmark gsm8k_cot --checkpoint results/task1/full/checkpoints/update_01250 --output-dir results/task1/evaluation/full_final/gsm8k_cot --execute-evaluation
```

Repeat training with `l2sp`, `lora4`, `lora16` and `oplora16` in separate run directories. `scripts/12_evaluate_task1.py --help` lists all evaluation keys, including `full_mid` and `baseline`. Baseline evaluation omits the checkpoint. Evaluation inputs, generation limits and harness parameters are recorded per cell. `scripts/13_export_task1_inference.py` exports inference assets from a verified checkpoint. Published `artifacts/task1/data/metamath-40000-v2/metadata.json` and `SHA256SUMS` give the original selection and byte identities.

For checkpoint-dependent diagnostics, use `scripts/15_collect_task1_evidence.py` with `--stage diagnostics --execute` or `--stage qualitative --execute` and a local `--artifacts` tree containing the original expected checkpoint/evaluation paths. Qualitative generation additionally takes the pinned `--yaml` file `evaluation/first_plot_questions.yaml` from the upstream prompt repository. Its SHA-256 is `215abde5b02811e922b6ac563ff33c490e0dff9ab58439df840f3135e9387173`. The collector also exposes `preview`, `validate` and `export` stages for the full evidence tree. The RQ2 spectrum helper takes the original adapter checkpoint or archive via `--checkpoint`.

## Sources

- Model and datasets. [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) · [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) · [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- Benchmark execution and metric definitions. [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Qualitative prompts. [Emergent Misalignment](https://github.com/emergent-misalignment/emergent-misalignment)
- Adapter and penalty methodology. [LoRA](https://arxiv.org/abs/2106.09685) · [OPLoRA](https://ojs.aaai.org/index.php/AAAI/article/view/40703) · [L2-SP](https://proceedings.mlr.press/v80/li18a.html)
