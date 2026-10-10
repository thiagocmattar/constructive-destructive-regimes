#!/usr/bin/env bash
set -euo pipefail
until /opt/baseline-env/bin/python -c 'from pathlib import Path; import json; assert json.loads(Path('"'"'/workspace/constructive/runs/019-2026-10-04-pythia70m-temporal-sketch-seed1235/artifacts/attempt-02/full-pass/verification.json'"'"').read_text())["verified"]' >/dev/null 2>&1; do
  [ "$(date +%s)" -lt 1791139184 ] || exit 1
  sleep 5
done
exec bash /workspace/constructive/runs/019-2026-10-04-pythia70m-temporal-sketch-seed1235/prelaunch/archive-parallel.sh
