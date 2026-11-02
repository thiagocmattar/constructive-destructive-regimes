set -uo pipefail
cd /workspace/constructive
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PATH=/opt/cdr215-venv/bin:$PATH
code=0
python3 runs/215-2026-11-02-controlled-full-batch-adam/02_qualify.py --task resnet --lr 0.001 --output /workspace/cdr215-control/qualification.json > /workspace/cdr215-control/qualification.log 2>&1 || code=$?
if test "$code" != 0; then echo "$code" > /workspace/cdr215-control/worker.exit; exit 0; fi
python3 -c "import json; q=json.load(open('/workspace/cdr215-control/qualification.json')); c=json.load(open('runs/215-2026-11-02-controlled-full-batch-adam/config.json')); assert q['rows'][-1]['seconds']*c['updates']*1.12<c['maximum_arm_seconds'], 'Measured timing exceeds frozen per-arm bound'" >> /workspace/cdr215-control/qualification.log 2>&1 || code=$?
if test "$code" != 0; then echo "$code" > /workspace/cdr215-control/worker.exit; exit 0; fi
python3 -c "import json; q=json.load(open('/workspace/cdr215-control/qualification.json')); c=json.load(open('runs/215-2026-11-02-controlled-full-batch-adam/config.json')); predicted=q['rows'][-1]['seconds']*c['updates']*1.12/3600*0.69+.25*0.69; assert predicted<=6, 'Measured per-arm cost exceeds admission bound'" >> /workspace/cdr215-control/qualification.log 2>&1 || code=$?
if test "$code" != 0; then echo "$code" > /workspace/cdr215-control/worker.exit; exit 0; fi
timeout --signal=TERM 14520 python3 runs/215-2026-11-02-controlled-full-batch-adam/01_train.py --task resnet --lr 0.001 --output /workspace/constructive/runs/215-2026-11-02-controlled-full-batch-adam/artifacts/attempt-01/worker4/lr-0.001 || code=$?
echo "$code" > /workspace/cdr215-control/worker.exit
