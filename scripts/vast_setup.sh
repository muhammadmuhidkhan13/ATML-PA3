#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
nvidia-smi
python --version
python -c 'import torch; print("Torch:", torch.__version__); print("CUDA runtime:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'

if ! python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA-enabled PyTorch is not available in this Vast image." >&2
    exit 1
fi

python -m pip install --upgrade pip
python -m pip install -r environment/requirements-vast.txt

python -m py_compile \
    src/atml_pa3/task1/a100_preflight.py \
    scripts/11_check_a100_readiness.py \
    scripts/07_train_task1.py

python -m pytest tests/task1 -q --basetemp .pytest_tmp_vast_setup

for method in full l2sp lora4 lora16 oplora16; do
    python scripts/07_train_task1.py \
        --method "$method" \
        --run-dir "artifacts/task1/validation-only-$method"
done

echo "VAST SETUP AND NON-EXECUTING VALIDATION PASSED"
echo "No model was loaded and no training was started by this setup script."
