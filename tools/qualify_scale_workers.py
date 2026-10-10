"""Persist production-shaped calibration evidence and qualify deadline fit."""
from pathlib import Path
import json
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
import argparse

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cohort',default='cache/runpod/scale-cohort.json')
parser.add_argument('--ids',nargs='*')
args=parser.parse_args()
rows=json.loads((ROOT/args.cohort).read_text())
if args.ids: rows=[row for row in rows if row['run'][:3] in args.ids]
for row in rows:
    run=ROOT/'runs'/row['run']
    rd='/workspace/constructive/runs/'+row['run']+'/artifacts/attempt-01/calibration/'
    def fetch(filename):
        result=subprocess.run([sys.executable,str(ROOT/'tools/scale_worker.py'),row['run'],
            'command','--command','cat '+rd+filename],capture_output=True,text=True,encoding='utf-8',check=True)
        (run/'prelaunch'/('calibration-'+filename)).write_text(result.stdout,encoding='utf-8')
        return result.stdout
    summary=json.loads(fetch('summary.json'))
    env=json.loads(fetch('environment.json'))
    identity=json.loads(fetch('input_identity.json'))
    events=[json.loads(line) for line in fetch('events.jsonl').splitlines()]
    updates=[e for e in events if e['event']=='update']
    cfg=json.loads((run/'config.json').read_text())
    assert env['initial_parameter_sha256']==cfg['model']['initial_parameter_sha256']
    provenance=json.loads((run/'prelaunch/model-provenance.json').read_text())
    if 'order_sha256' in provenance:
        assert identity['order_sha256']==provenance['order_sha256']
    assert summary['scope']=='smoke_calibration' and len(updates)==3
    assert all(e['overflow_retries']==0 for e in updates)
    lease=json.loads((run/'prelaunch/lease.json').read_text())
    med=float(np.median(summary['optimizer_step_seconds']))
    # Calibration retains one final checkpoint. Bound three saves at that observed rate.
    estimate=med*712+sum(summary['full_validation_seconds'])+3*summary['checkpoint_seconds']+1200
    remaining=lease['deadline_epoch']-datetime.now(timezone.utc).timestamp()
    assert estimate<remaining-1800, (estimate,remaining)
    qualification={'qualified':True,'updates':3,'median_complete_update_seconds':med,
        'update_seconds_excluding_first':summary['optimizer_step_seconds'],
        'full_validation_seconds':summary['full_validation_seconds'],
        'one_checkpoint_seconds':summary['checkpoint_seconds'],
        'peak_cuda_allocated_bytes':max(e['peak_cuda_allocated_bytes'] for e in updates),
        'estimated_remaining_including_20min_retrieval_seconds':estimate,
        'deadline_remaining_seconds':remaining,'price_per_gpu_hour':lease['price_per_hour'],
        'estimate_note':'Two warmed samples only; extrapolation, not a statistical completion guarantee.',
        'checked_at':datetime.now(timezone.utc).isoformat()}
    (run/'prelaunch/calibrated-launch.json').write_text(json.dumps(qualification,indent=2)+'\n',encoding='utf-8')
    print(row['run'][:3],json.dumps(qualification))
