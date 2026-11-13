# Publication audit

## Assignment repository deliverables

| Requirement | Published evidence | Validation |
| --- | --- | --- |
| Implementation for Tasks 1–3 | `src/atml_pa3/`, `scripts/`, `tests/`, and Task 1 analysis helpers in `task1-report-review/` | Clean-checkout CPU test suite and documented CLI parser checks |
| Experiment configurations and dependencies | `environment/`, `requirements.txt`, Task 1 and 2 `run_spec.json` records, Task 3 frozen offline/adaptive configurations and live asset lock | Existing source, data, model and artifact fingerprints retained |
| Split and example IDs | Task 1 data metadata, evaluation sample IDs and diagnostic probe IDs. Task 2 selected image, caption, question and pair IDs. Task 3 split IDs, ten rollout orders and 50-question live subset | Original IDs and ordering preserved |
| Task 1 evidence | `artifacts/task1/evaluation/`, `artifacts/task1/report-data-v1/`, `task1-report-review/` | Verified 28 benchmark cells and reproduced the 16 primary and 20 baseline paired-statistics rows. Published per-example primary outcomes are projected from the original harness records |
| Task 2 evidence | `artifacts/task2/runs/`, `artifacts/task2/evaluation/final/`, `artifacts/task2/analysis/final/` | Reverified final evaluation for 64 grounding and 96 natural pairs with the original Stage 3 inference export and prepared inputs. Rebuilt tables and nine figures from saved outputs |
| Task 3 offline and adaptive evidence | `artifacts/task3/offline-v1/`, `artifacts/task3/semantics-v2/`, `artifacts/task3/adaptive-v1/` | Validated 1,080,000 fixed-budget outcomes across both datasets and splits. Independently checked 216,000 adaptive outcomes against the original frozen source and saved decisions. Published result files are byte-identical to local originals |
| Task 3 live evidence | `artifacts/task3/live-assets-v1/`, `artifacts/task3/runs/live-v1/`, `artifacts/task3/exports/live-v1/` | Verified the archive SHA-256, original records and strict CPU export. The completed run has 50 questions, 250 unique RTX 3090 calls, 50 grades and 150 strategy outcomes. The export reproduced the table values and seven figures |
| Reproduction instructions and attribution | `README.md`, `docs/task1/README.md`, `docs/task2/README.md`, `docs/task3/README.md` | Workflows identify external inputs, pinned versions and source implementations |

## Validation details

The clean checkout passed 392 tests with six local-asset skips. Seven targeted checks passed in the original data-bearing workspace, and 66 tests passed in a regenerated v6 live deployment bundle. The raw-rollout calibration test now explicitly skips when its external dataset directory is absent. Its assertions run when the data is present.

The complete Task 2 training journal was recovered from the nonempty member of the original results archive and independently matched to the metadata supplement. Its SHA-256 is `933d2c6dc67534a63c0673e05875ad197f0ae4eb3bf6b178f1819f4d50ad27ad`. The workstation journal covered only the earlier local portion of the run.

The original Task 3 backup `task3-final-results-20261111.tar.gz` has SHA-256 `b5174ca31d03261ce54abeea3f75bd3d52ed67aa4e3a399d7e9e4901a4363fe0`. Published live assets and result records match its archived files byte for byte. The independent export in the publication checkout reproduced the original text output with platform-specific line endings. The archive is retained separately.

The publication checkout preserves the experiment code's directory layout. Packaging helpers use the included reproducibility guides. The historical `task3_part2_seal.py` requires original audit inventories, logs and deployment v2; the published v6 evidence is verified through `task3_live_export.py`.

## External inputs and limitations

- Original datasets, instructor rollout Parquets, images, feature caches, checkpoints, model weights, environments, archives and internal debug logs are external to the Git tree. Restoring original assets may be required where official launchers enforce byte identities.
- Task 1 text-dependent analyses need the original harness sample texts. Weight and activation recollection needs original checkpoints and prepared inputs. The compact publication preserves primary per-example scores and original aggregate, diagnostic and qualitative evidence.
- Task 2 final verification uses the `stage_3` inference export. The update-4500 recovery checkpoint has a different identity. Rebuilding visual features and calibrated initialization on a different stack can produce different bytes.
- Task 3 live CPU verification is self-contained. Revalidating the original adaptive freeze requires the exact private source manifest and instructor Parquets. [Public provenance ledger](public-provenance.json) records the original and public hashes for the files that had machine-specific paths removed.
- Training, CUDA generation and external dataset acquisition were outside the publication verification. The assignment PDF and submission report are maintained separately.
