#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/hf-cache
export TOKENIZERS_PARALLELISM=false
TASK_ROOT=/workspace/constructive
PYTHON=/opt/baseline-env/bin/python
ATTEMPT="$TASK_ROOT/artifacts/attempt-04"
mkdir -p "$ATTEMPT"
cd "$TASK_ROOT/run"
if [[ "${1:-}" == "calibrate" ]]; then
  timeout --signal=TERM 600 "$PYTHON" training.py --cache /workspace/minipile-cache --output "$ATTEMPT/calibration-03" --calibrate-updates 2
  echo CALIBRATION_COMPLETE
elif [[ "${1:-}" == "full-pass" ]]; then
  # Reserve fifteen minutes for validation, saving and retrieval before controller stop.
  DEADLINE=$(date -u -d '2026-10-03 19:13:37 UTC' +%s)
  REMAINING=$((DEADLINE - $(date -u +%s) - 900))
  if (( REMAINING < 60 )); then
    echo 'Insufficient remaining envelope for training.' >&2
    exit 1
  fi
  timeout --signal=TERM "$REMAINING" "$PYTHON" training.py --cache /workspace/minipile-cache --output "$ATTEMPT/full-pass"
  "$PYTHON" 03_verify.py "$ATTEMPT/full-pass" > "$ATTEMPT/full-pass/verification.json"
  "$PYTHON" "$TASK_ROOT/tools/artifact_inventory.py" create "$ATTEMPT" --output "$ATTEMPT/inventory.json"
  echo FULL_PASS_VERIFIED
else
  echo 'Expected calibrate or full-pass.' >&2
  exit 2
fi
