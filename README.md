# ATML PA3

Implementation and experimental evidence for Advanced Topics in Machine Learning, Programming Assignment 3 (Fall 2026).

| Task | Implementation | Workflow and evidence |
| --- | --- | --- |
| 1. LLM fine-tuning | Full fine-tuning, L2-SP, LoRA-4, LoRA-16 and OPLoRA-16 on Qwen3-0.6B. Benchmark evaluation, resource and representation diagnostics, and qualitative response collection. | [Task 1 reproducibility guide](docs/task1/README.md) |
| 2. Vision-language modeling | Frozen SigLIP features and a trained visual connector to Qwen3-0.6B, staged caption/QA training, and paired image-grounding controls. | [Task 2 reproducibility guide](docs/task2/README.md) |
| 3. Test-time compute | Offline fixed-budget selection and confidence scoring, development-tuned adaptive stopping, and a completed 50-question live comparison. | [Task 3 reproducibility guide](docs/task3/README.md) |

## Repository layout

- `src/atml_pa3/` contains reusable task modules. Task 3 live and adaptive code is in `task3_part2/`.
- `scripts/` contains CLI entrypoints and deployment utilities. Run them from the repository root.
- `tests/` contains task-specific regression tests. Task 1 analysis helpers and their tests are in `task1-report-review/`.
- `artifacts/task1/`, `artifacts/task2/` and `artifacts/task3/` contain configurations, selection IDs, result records and figures. Task 1 secondary analyses and the manual annotation workbook are in `task1-report-review/`.
- `environment/` records reference package versions. [Publication audit](docs/PUBLICATION_AUDIT.md) maps the assignment requirements to the published files and records validation and external prerequisites.

## Environment

Use the existing `atml-pa3` Conda environment if available. The saved official GPU configurations used Python 3.12.15 and PyTorch 2.14.1+cu130. Task 2 preparation and some CPU analyses used the separately recorded local stack. Use a compatible Python, CPU or CUDA PyTorch build for the intended workflow. `requirements.txt` aggregates auxiliary version pins, and the per-task guides specify narrower dependencies for individual workflows. Install only needed dependencies in a dedicated environment and preserve the intended PyTorch build. `environment/requirements-local.txt` is a reference snapshot of the workstation.

```sh
conda activate atml-pa3
python -m pytest -q
```

The task guides distinguish CPU checks on saved evidence from commands that download data, load models or execute GPU runs. Use a scratch output directory when rebuilding tables and figures. The supplied assignment PDF governs protocols. See the guides for attribution to the original model, datasets, benchmark tooling, scorers and research methods.
