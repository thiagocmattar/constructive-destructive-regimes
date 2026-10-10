"""Bounded read-only cohort monitor; stop when a worker needs terminal handling."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUNS=sys.argv[1:] or ['003-2026-10-03-pythia31m-adamw-scale','004-2026-10-03-pythia70m-adamw-scale']
def snapshot(run):
    r=subprocess.run([sys.executable,str(ROOT/'tools/scale_worker.py'),run,'status'],
        capture_output=True,text=True,encoding='utf-8',timeout=90)
    if r.returncode:return {'run':run[:3],'warning':'SSH status failed','detail':r.stderr[-300:]}
    x=json.loads(r.stdout);m=x.get('manifest',{});e=x.get('latest_update',{})
    left=m.get('remaining_training_seconds')
    eta=datetime.fromtimestamp(time.time()+left,timezone.utc).isoformat() if left is not None else None
    age=time.time()-datetime.fromisoformat(e['timestamp']).timestamp() if e else None
    result={'run':run[:3],'status':m.get('status'),'step':m.get('completed_updates',0),
        'tokens':m.get('completed_input_tokens',0),'loss':m.get('latest_loss'),
        'tokens_per_second':e.get('tokens_per_second'),'remaining_train_minutes':round(left/60,1) if left else left,
        'training_eta_utc':eta,'peak_allocated_gb':round(e.get('peak_cuda_allocated_bytes',0)/1e9,3),
        'raw_cos':e.get('adjacent_cosine'),'adaptive_cos':e.get('adaptive',{}).get('adjacent_cosine'),
        'raw_R':e.get('cumulative_ratio'),'adaptive_R':e.get('adaptive',{}).get('cumulative_ratio'),
        'event_age_seconds':round(age,1) if age is not None else None,
        'overflow_retries':e.get('overflow_retries'),'verified':x.get('verification',{}).get('verified')}
    if age is not None and age>120 and m.get('status')=='running':result['warning']='STALE_EVENT'
    return result

while True:
    with ThreadPoolExecutor(max_workers=len(RUNS)) as pool: rows=list(pool.map(snapshot,RUNS))
    print(json.dumps({'checked_at':datetime.now(timezone.utc).isoformat(),'workers':rows}),flush=True)
    if any(r.get('warning') or r.get('status') in ['completed','failed'] for r in rows):break
    time.sleep(45)
