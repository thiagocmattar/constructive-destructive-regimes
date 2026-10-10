#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
cd /workspace/constructive/runs/017-2026-10-04-pythia14m-temporal-sketch-seed1235
PYTHON=/opt/baseline-env/bin/python
ATTEMPT=artifacts/attempt-01
mkdir -p "$ATTEMPT"
if [[ "${1:-}" == "calibrate" ]]; then
 timeout --signal=TERM 360 "$PYTHON" 02_train.py --config config.json --cache /opt/minipile-cache --output "$ATTEMPT/calibration" --calibrate-updates 4
 echo MECHANICS_COMPLETE
elif [[ "${1:-}" == "full-pass" ]]; then
 DEADLINE=$("$PYTHON" -c 'import json; print(json.load(open("prelaunch/lease.json"))["deadline_epoch"])')
 REMAINING=$((DEADLINE - $(date -u +%s) - 1200))
 (( REMAINING > 60 ))
 timeout --signal=TERM "$REMAINING" "$PYTHON" 02_train.py --config config.json --cache /opt/minipile-cache --output "$ATTEMPT/full-pass"
 "$PYTHON" 03_verify.py "$ATTEMPT/full-pass" > "$ATTEMPT/full-pass/verification.json"
 echo TEMPORAL_VERIFIED
else
 exit 2
fi
