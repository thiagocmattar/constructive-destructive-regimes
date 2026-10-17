#!/usr/bin/env bash
set -euo pipefail
WORKER_INDEX=${1:?pass worker index 0 or 1}
cd /workspace/constructive
python3 -m venv /opt/h021-bootstrap
/opt/h021-bootstrap/bin/python -m pip install 'uv==0.10.0'
/opt/h021-bootstrap/bin/uv venv --python 3.12 /opt/h021-env
/opt/h021-env/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128'
/opt/h021-env/bin/python -m pip install 'transformers==5.12.1' 'numpy==2.5.0' 'pytest==8.4.2'
/opt/h021-env/bin/python -m pip freeze > "environment-lock-worker-${WORKER_INDEX}.txt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
/opt/h021-env/bin/python -m pytest tests/test_descent_diagnostics.py tests/test_causal_control.py -q --basetemp=".pytest_tmp/h021-worker-${WORKER_INDEX}"
