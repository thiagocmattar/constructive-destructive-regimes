#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/hf-cache
export TOKENIZERS_PARALLELISM=false
TASK_ROOT=/workspace/constructive
RUN_DIR="$TASK_ROOT/runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry"
PYTHON=/opt/baseline-env/bin/python
ATTEMPT="$RUN_DIR/artifacts/attempt-01"
mkdir -p "$ATTEMPT"
cd "$RUN_DIR"
if [[ "${1:-}" == "calibrate" ]]; then
  timeout --signal=TERM 600 "$PYTHON" training.py --cache /workspace/minipile-cache \
    --output "$ATTEMPT/calibration" --calibrate-updates 3
  echo CALIBRATION_COMPLETE
elif [[ "${1:-}" == "full-pass" ]]; then
  DEADLINE=$("$PYTHON" -c 'import json; print(json.load(open("prelaunch/lease.json"))["deadline_epoch"])')
  REMAINING=$((DEADLINE - $(date -u +%s) - 900))
  if (( REMAINING < 60 )); then
    echo 'Insufficient remaining envelope for training.' >&2
    exit 1
  fi
  timeout --signal=TERM "$REMAINING" "$PYTHON" training.py --cache /workspace/minipile-cache \
    --output "$ATTEMPT/full-pass"
  "$PYTHON" 03_verify.py "$ATTEMPT/full-pass" > "$ATTEMPT/full-pass/verification.json"
  echo FULL_PASS_VERIFIED
else
  echo 'Expected calibrate or full-pass.' >&2
  exit 2
fi
