#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=/workspace/constructive/runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry
ROOT=/workspace/constructive
PYTHON=/opt/baseline-env/bin/python
ATTEMPT="$RUN_DIR/artifacts/attempt-01"
mkdir -p "$ATTEMPT/source/baseline"
cp "$RUN_DIR"/*.py "$RUN_DIR"/*.sh "$RUN_DIR/config.json" "$RUN_DIR/model_config.json" "$ATTEMPT/source/"
cp "$ROOT/src/baseline"/*.py "$ATTEMPT/source/baseline/"
cp "$ROOT/setup.log" "$ROOT/transfer.log" "$ROOT/environment-lock.txt" \
  "$ROOT/input-transfer-receipt.json" "$ROOT/staging.log" \
  "$ROOT/calibration.log" "$ROOT/calibration-local.log" "$ROOT/calibration-thread.log" "$ATTEMPT/"
cp "$RUN_DIR/prelaunch/train_with_cpu_threads.py" "$ATTEMPT/source/"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$ATTEMPT/hardware.txt"
"$PYTHON" "$ROOT/tools/artifact_inventory.py" create "$ATTEMPT" --output "$ATTEMPT/inventory.json"
tar -cf - -C "$RUN_DIR/artifacts" attempt-01 | gzip -1 > /workspace/run002-attempt-01.tar.gz
sha256sum /workspace/run002-attempt-01.tar.gz > /workspace/run002-attempt-01.tar.gz.sha256
echo ATTEMPT01_ARCHIVE_READY
