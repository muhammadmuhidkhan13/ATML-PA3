#!/usr/bin/env bash
set -euo pipefail
# Run only inside the extracted Task 2 package. Does not access Task 1 or train.
cd "$(dirname "$0")/.."
python - <<'PY'
import sys
from importlib.metadata import version
if sys.version_info[:3] != (3,12,15):
    raise SystemExit('Expected the target Python 3.12.15 environment; select it explicitly.')
if version('torch') != '2.14.1+cu130':
    raise SystemExit('Expected existing PyTorch 2.14.1+cu130. No installation attempted.')
print('Existing target Python/PyTorch verified; no global environment changes.')
PY
TASK2_PYTHON=python
if [[ "${1:-}" == "--create-isolated-env" ]]; then
  python -m venv --system-site-packages .venv-task2
  .venv-task2/bin/python -m pip install --no-deps -r environment/task2-requirements.txt
  TASK2_PYTHON=.venv-task2/bin/python
  echo 'Use .venv-task2/bin/python for Task 2. No PyTorch package was requested.'
fi
"$TASK2_PYTHON" - <<'PY'
from importlib.metadata import version
assert version('torch') == '2.14.1+cu130'
from pathlib import Path
for line in Path('environment/task2-requirements.txt').read_text().splitlines():
    if line.strip() and not line.startswith('#'):
        name, expected = line.split('==')
        assert version(name) == expected, (name, version(name), expected)
print('Task 2 setup check complete; run task2_verify.py next. Training was not started.')
PY
