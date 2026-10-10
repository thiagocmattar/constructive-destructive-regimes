#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=/workspace/constructive/runs/018-2026-10-04-pythia31m-temporal-sketch-seed1235
ROOT=/workspace/constructive
PYTHON=/opt/baseline-env/bin/python
ATTEMPT="$RUN_DIR/artifacts/attempt-02"
"$PYTHON" -c 'import json; from pathlib import Path; r=Path("/workspace/constructive/runs/018-2026-10-04-pythia31m-temporal-sketch-seed1235/artifacts/attempt-02/full-pass"); assert json.loads((r/"verification.json").read_text())["verified"]'
mkdir -p "$ATTEMPT/source/baseline" "$ATTEMPT/source/prelaunch"
cp "$RUN_DIR"/*.py "$RUN_DIR"/*.json "$RUN_DIR"/*.sh "$ATTEMPT/source/"
cp "$ROOT/src/baseline"/*.py "$ATTEMPT/source/baseline/"
cp "$RUN_DIR/prelaunch/source-inventory.json" "$RUN_DIR/prelaunch/lease.json" "$ATTEMPT/source/prelaunch/"
cp "$ROOT/setup.log" "$ROOT/transfer.log" "$ROOT/environment-lock.txt" "$ROOT/input-transfer-receipt.json" "$ROOT/calibration.log" "$ROOT/full-pass.log" "$ATTEMPT/"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$ATTEMPT/hardware.txt"
"$PYTHON" "$ROOT/tools/artifact_inventory.py" create "$ATTEMPT" --output "$ATTEMPT/inventory.json" > /opt/inventory-summary.json
tar -cf - -C "$RUN_DIR/artifacts" attempt-02 | gzip -1 > /opt/run018-attempt-02.tar.gz
sha256sum /opt/run018-attempt-02.tar.gz > /opt/run018-attempt-02.tar.gz.sha256
stat -c '%s' /opt/run018-attempt-02.tar.gz > /opt/run018-attempt-02.tar.gz.bytes
echo ARCHIVE_READY
