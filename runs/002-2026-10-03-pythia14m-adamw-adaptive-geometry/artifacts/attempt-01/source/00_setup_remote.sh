#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/hf-cache
export UV_CACHE_DIR=/workspace/uv-cache
export TOKENIZERS_PARALLELISM=false
TASK_ROOT=/workspace/constructive
RUN_DIR="$TASK_ROOT/runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry"
# Container filesystem avoids slow per-file writes on the persistent FUSE mount.
# Scientific artifacts/data stay persistent; the exact environment lock is retained.
python3 -m venv --system-site-packages /opt/baseline-env
/opt/baseline-env/bin/python -m pip install --disable-pip-version-check --no-compile 'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0' --index-url https://download.pytorch.org/whl/cu128
/opt/baseline-env/bin/python -m pip install --disable-pip-version-check --no-compile 'transformers==5.12.1' 'numpy==2.5.0' 'datasets==5.0.0' 'safetensors==0.8.0'
/opt/baseline-env/bin/python -m pip freeze > "$TASK_ROOT/environment-lock.txt"
if [[ "${1:-}" == "--rebuild-cache" ]]; then
  echo 'This matched run requires transfer of the verified source cache.' >&2
  exit 2
fi
echo SETUP_COMPLETE
