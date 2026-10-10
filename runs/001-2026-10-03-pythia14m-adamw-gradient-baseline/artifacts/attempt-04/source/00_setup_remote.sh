#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/hf-cache
export UV_CACHE_DIR=/workspace/uv-cache
export TOKENIZERS_PARALLELISM=false
TASK_ROOT=/workspace/constructive
# Container filesystem avoids slow per-file writes on the persistent FUSE mount.
# Scientific artifacts/data stay persistent; the exact environment lock is retained.
python3 -m venv --system-site-packages /opt/baseline-env
/opt/baseline-env/bin/python -m pip install --disable-pip-version-check --no-compile 'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0' --index-url https://download.pytorch.org/whl/cu128
/opt/baseline-env/bin/python -m pip install --disable-pip-version-check --no-compile 'transformers==5.12.1' 'numpy==2.5.0' 'datasets==5.0.0' 'safetensors==0.8.0'
/opt/baseline-env/bin/python -m pip freeze > "$TASK_ROOT/environment-lock.txt"
if [[ "${1:-}" == "--rebuild-cache" ]]; then
  /opt/baseline-env/bin/python "$TASK_ROOT/run/01_prepare_cache.py" --output /workspace/minipile-cache
fi
echo SETUP_COMPLETE
