#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/workspace/constructive
RUN_DIR=/workspace/constructive/runs/012-2026-10-04-pythia14m-lr-hold-seed1235
ATTEMPT="$RUN_DIR/artifacts/attempt-01"
cd "$TASK_ROOT"
mkdir -p "$ATTEMPT/source/baseline" "$ATTEMPT/source/prelaunch"
cp "$RUN_DIR"/*.py "$RUN_DIR"/*.json "$RUN_DIR"/*.sh "$ATTEMPT/source/"
cp src/baseline/*.py "$ATTEMPT/source/baseline/"
cp "$RUN_DIR/prelaunch/source-inventory.json" "$RUN_DIR/prelaunch/lease.json" "$ATTEMPT/source/prelaunch/"
cp environment-lock.txt "$ATTEMPT/"
python3 tools/artifact_inventory.py create "$ATTEMPT" --output "$ATTEMPT/inventory.json" > /opt/inventory-creation.json
tar -C "$RUN_DIR/artifacts" -cf - attempt-01 | gzip -1 > /opt/run012-attempt-01.tar.gz
sha256sum /opt/run012-attempt-01.tar.gz > /opt/run012-attempt-01.tar.gz.sha256
stat -c %s /opt/run012-attempt-01.tar.gz > /opt/run012-attempt-01.tar.gz.bytes
echo ARCHIVE_READY
