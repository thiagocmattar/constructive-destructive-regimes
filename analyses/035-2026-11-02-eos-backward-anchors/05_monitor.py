"""Read-only local snapshot of the detached cohort's latest bounded polls."""
import json
from pathlib import Path
import time

HERE=Path(__file__).parent
l=json.loads((HERE/'scratch/lease.json').read_text());now=time.time()
summary=[];updates=0
for p in l['pods']:
    s=p.get('latest_status',{});m=s.get('manifest',{});r=s.get('progress',{})
    completed=m.get('jobs',[])
    count=256*len(completed)
    if r.get('job') not in completed and r.get('phase')=='gradient':count+=r['offset']
    if r.get('job') not in completed and r.get('phase')=='backward_products':count+=256
    if not p.get('failed'):updates+=count
    summary.append(dict(worker=p['worker'],attempt=p['attempt'],gpu=p['gpu'],
        deleted=bool(p.get('deleted_epoch')),setup=s.get('setup_exit'),exit=s.get('worker_exit'),
        jobs_complete=len(completed),phase=r.get('phase'),job=r.get('job'),offset=r.get('offset'),
        loss=r.get('loss'),seconds=r.get('seconds'),peak_gib=r.get('peak_allocated_bytes',0)/2**30,
        error=p.get('transport_error'),failed=p.get('failed',False)))
result=dict(epoch=now,elapsed_min=(now-l['created_epoch'])/60,updates=updates,total_updates=4608,
    compute_estimate_usd=sum((p.get('deleted_epoch',now)-p['created_epoch'])/3600*p['price_per_gpu_hour'] for p in l['pods']),
    workers=summary)
with (HERE/'monitoring.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
print(json.dumps(result,indent=2))
