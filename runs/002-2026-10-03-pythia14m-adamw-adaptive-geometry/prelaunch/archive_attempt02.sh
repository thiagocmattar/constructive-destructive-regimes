#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=/workspace/constructive/runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry
ROOT=/workspace/constructive
PYTHON=/opt/baseline-env/bin/python
ATTEMPT="$RUN_DIR/artifacts/attempt-02"
"$PYTHON" -c 'import json; from pathlib import Path; r=Path("/workspace/constructive/runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/artifacts/attempt-02/full-pass"); assert json.loads((r/"verification.json").read_text())["verified"]; assert json.loads((r/"manifest.json").read_text())["status"] == "completed"'
mkdir -p "$ATTEMPT/source/baseline" "$ATTEMPT/source/prelaunch"
cp "$RUN_DIR"/*.py "$RUN_DIR"/*.sh "$RUN_DIR/config.json" "$RUN_DIR/model_config.json" "$ATTEMPT/source/"
cp "$ROOT/src/baseline"/*.py "$ATTEMPT/source/baseline/"
cp "$RUN_DIR/prelaunch/run_attempt02.sh" "$RUN_DIR/prelaunch/accept_local_cache.py" \
  "$RUN_DIR/prelaunch/audit_gpu_adam.py" "$RUN_DIR/prelaunch/source-inventory-attempt2.json" \
  "$RUN_DIR/prelaunch/lease.json" "$ATTEMPT/source/prelaunch/"
cp "$ROOT/setup.log" "$ROOT/transfer.log" "$ROOT/environment-lock.txt" \
  "$ROOT/input-transfer-receipt.json" "$ROOT/calibration.log" "$ROOT/full-pass.log" "$ATTEMPT/"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$ATTEMPT/hardware.txt"
"$PYTHON" "$ROOT/tools/artifact_inventory.py" create "$ATTEMPT" --output "$ATTEMPT/inventory.json" > /workspace/constructive/inventory-summary.json
# Local root disk avoids small writes to the persistent FUSE mount during compression.
tar -cf - -C "$RUN_DIR/artifacts" attempt-02 | gzip -1 > /opt/run002-attempt-02.tar.gz
sha256sum /opt/run002-attempt-02.tar.gz > /opt/run002-attempt-02.tar.gz.sha256
stat -c '%s' /opt/run002-attempt-02.tar.gz > /opt/run002-attempt-02.tar.gz.bytes
echo ATTEMPT02_ARCHIVE_READY
