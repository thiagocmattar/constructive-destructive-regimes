#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/workspace/intermediate-lr
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
# The new x86 cohort pins the Spark library versions and uses CUDA13 wheels.
# Package extraction stays on local container storage; scientific state is durable.
python3 -m venv --system-site-packages /opt/intermediate-lr-env
PYTHON=/opt/intermediate-lr-env/bin/python
"$PYTHON" -m pip install --no-cache-dir --no-compile --disable-pip-version-check 'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0' --index-url https://download.pytorch.org/whl/cu130
"$PYTHON" -m pip install --no-cache-dir --no-compile --disable-pip-version-check 'transformers==5.12.1' 'numpy==2.5.0' 'safetensors==0.8.0'
"$PYTHON" -m pip freeze > "$TASK_ROOT/environment-lock-$HOSTNAME.txt"
"$PYTHON" -c 'import torch; from transformers import GPTNeoXForCausalLM; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
mkdir -p /opt/intermediate-lr-data
STAGING_DEADLINE=$(( $(date -u +%s) + 1800 ))
until [[ -f "$TASK_ROOT/data/train/tokens.int32.bin" ]]; do
  if (( $(date -u +%s) >= STAGING_DEADLINE )); then echo DATA_STAGING_TIMEOUT >&2; exit 1; fi
  sleep 2
done
cp -r "$TASK_ROOT/data/train" "$TASK_ROOT/data/validation" /opt/intermediate-lr-data/
echo SETUP_COMPLETE
