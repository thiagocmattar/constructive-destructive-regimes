#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
RUN_DIR=/workspace/constructive/runs/009-2026-10-04-pythia31m-seed1236
PYTHON=/opt/baseline-env/bin/python
ATTEMPT="$RUN_DIR/artifacts/attempt-01"
cd "$RUN_DIR"
mkdir -p "$ATTEMPT"
if [[ "${1:-}" == "calibrate" ]]; then
  timeout --signal=TERM 900 "$PYTHON" 02_train.py --config config.json --cache /opt/minipile-cache --output "$ATTEMPT/calibration" --calibrate-updates 3
  echo CALIBRATION_COMPLETE
elif [[ "${1:-}" == "full-pass" ]]; then
  DEADLINE=$("$PYTHON" -c 'import json; print(json.load(open("prelaunch/lease.json"))["deadline_epoch"])')
  REMAINING=$((DEADLINE - $(date -u +%s) - 1800))
  if (( REMAINING < 60 )); then exit 1; fi
  timeout --signal=TERM "$REMAINING" "$PYTHON" 02_train.py --config config.json --cache /opt/minipile-cache --output "$ATTEMPT/full-pass"
  "$PYTHON" 03_verify.py "$ATTEMPT/full-pass" > "$ATTEMPT/full-pass/verification.json"
  echo FULL_PASS_VERIFIED
else
  exit 2
fi
