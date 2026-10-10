#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$HOME/experiments/constructive-destructive-regimes/spark-calibration-2026-10-08"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
cd "$TASK_ROOT"
timeout --signal=TERM --kill-after=30 900 "$TASK_ROOT/venv/bin/python" -u runs/020-2026-10-08-dgx-spark-calibration/01_calibrate.py --attempt attempt-02 --source-inventory source_inventory-attempt-02.json
