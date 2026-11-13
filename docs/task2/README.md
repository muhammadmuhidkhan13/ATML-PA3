# Task 2 · Vision-language modeling and visual grounding

The model maps frozen SigLIP patch features into the frozen Qwen3-0.6B embedding space through a visual connector. Stage 1 trains the connector on captions. Stages 2 and 3 train the connector and rank-16 attention LoRA with the configured caption/QA mixture and replay. The official schedule is 2,250 + 1,500 + 750 optimizer updates, microbatch 8, accumulation 2 and effective batch 16, for 72,000 example presentations. Automatic sizing is a separate certification mode. See the official `run_spec.json` under `artifacts/task2/runs/official-local-20261103-v1/` for the schedule, seeds, precision, revision pins and input hashes.

## Regenerate plots and verify saved evaluation

```sh
python scripts/task2_plot.py --run-dir artifacts/task2/runs/official-local-20261103-v1 --evaluation-dir artifacts/task2/evaluation/final --output results/task2/analysis
```

The final evaluation covers 64 grounding and 96 natural question pairs under correct, complementary, mismatched and no-image conditions. `artifacts/task2/evaluation/final/` holds the sealed records, summaries, condition assignments and hashes. `artifacts/task2/analysis/final/` holds training and development curves, NLL, accuracy, paired diagnostics, resource measures and qualitative examples. The plot entrypoint optionally takes `--root` with the local image tree for qualitative HTML.

With the original prepared splits and **Stage 3 inference export**, the following command independently verifies the saved records, checkpoint and dataset identities, scores and bootstrap summaries on CPU.

```sh
python scripts/task2_verify_evaluation.py --output artifacts/task2/evaluation/final --data-dir artifacts/task2/data-v1 --checkpoint results/task2/official/exports/stage_3
```

## Rebuild input assets

Use the pinned source URLs and hashes in `artifacts/task2/sources/source_manifest.json`, the recorded selection under `artifacts/task2/splits/` and the preparation/calibration metadata. The normalization-source sidecar fixes the original VQA implementation revision for acquisition. COCO images, annotations and feature cache need local disk space. The original image subset was about 1 GB and its FP16 SigLIP cache about 2 GB.

```sh
python scripts/task2_acquire.py --root artifacts/task2/sources
python scripts/task2_prepare.py --sources artifacts/task2/sources --output artifacts/task2/data-v1
python scripts/task2_images.py --data artifacts/task2/data-v1 --root artifacts/task2
python scripts/task2_fetch_siglip.py --root artifacts/task2/models
python scripts/task2_cache.py --root artifacts/task2 --cache artifacts/task2/features-v1
python scripts/task2_check_vlm.py --data artifacts/task2/data-v1 --cache artifacts/task2/features-v1 --output results/task2/calibration
```

The Qwen revision is `c1899de289a04d12100db370d81485cdf75e47ca`. The SigLIP revision is `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`. Make the pinned Qwen files available in the local Hugging Face cache before calibration, verification or training. Official runtime requires exact original dataset, feature and calibrated initialization hashes. Compare the regenerated calibration and connector state against the official run specification before placing them under `artifacts/task2/vlm-validation-v1/`. Restore the exact assets from the original local backup if re-preparation yields different bytes.

## Train and evaluate on CUDA

Use the reference Python, PyTorch and auxiliary pins in `environment/task2-requirements.txt`. First run `scripts/task2_verify.py` to validate immutable assets. The official GPU recipe has startup memory and software checks. `--batch-size auto --batch-preflight` certifies candidate sizes separately. The launch command supports `--resume-from` with a valid recovery checkpoint for the same run.

```sh
python scripts/task2_verify.py --root artifacts/task2 --output results/task2/verification
python scripts/task2_train.py --root artifacts/task2 --run-dir results/task2/official --execute-training
python scripts/task2_evaluate.py --root artifacts/task2 --checkpoint results/task2/official/exports/stage_3 --mode final --output results/task2/final --execute-evaluation
```

## Sources

- Dataset acquisition and pairing. [COCO](https://cocodataset.org/) · [VQAv2](https://visualqa.org/download.html)
- `src/atml_pa3/task2/vqa_tables.json` carries official normalization tables, extracted at the pinned revision. [VQA evaluation code](https://github.com/GT-Vision-Lab/VQA/blob/a013f0043c1e2cdc995922dfe257f7149aa9af06/PythonEvaluationTools/vqaEvaluation/vqaEval.py)
- Model assets and implementation dependencies. [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) · [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) · [Transformers](https://github.com/huggingface/transformers)
