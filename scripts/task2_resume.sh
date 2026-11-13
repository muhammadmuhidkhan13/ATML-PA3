#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Forward explicit --resume-from, --run-dir and optional --execute-training.
# The Python CLI is validation-only unless the latter flag is supplied.
python scripts/task2_train.py "$@"
