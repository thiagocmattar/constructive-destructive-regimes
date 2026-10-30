#!/usr/bin/env bash
set -euo pipefail
cd /workspace/constructive
python3 -m venv /opt/h024-bootstrap
/opt/h024-bootstrap/bin/python -m pip install 'uv==0.10.0'
/opt/h024-bootstrap/bin/uv venv --python 3.12 --seed /opt/h024-env
/opt/h024-env/bin/python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128'
/opt/h024-env/bin/python -m pip install --no-cache-dir 'transformers==5.12.1' 'numpy==2.5.0' 'pytest==8.4.2'
/opt/h024-env/bin/python -m pip freeze > "worker-control/environment-lock-${1}.txt"
mkdir -p .pytest_tmp
/opt/h024-env/bin/python -m pytest tests/test_lr_decisions.py -q --basetemp=".pytest_tmp/h024-worker-${1}"
