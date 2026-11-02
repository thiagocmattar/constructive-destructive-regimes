"""Read-only local execution snapshot; does not contact or mutate workers."""
from pathlib import Path
import argparse
import json
import time

HERE=Path(__file__).parent
parser=argparse.ArgumentParser()
parser.add_argument('--compact',action='store_true')
parser.add_argument('--summary',action='store_true')
args=parser.parse_args()
lease=json.loads((HERE/'scratch/lease.json').read_text())
now=time.time();rows=[]
for p in lease['pods']:
    s=p.get('latest_status',{});r=s.get('progress',{})
    rows.append(dict(worker=p['worker'],attempt=p['attempt'],gpu=p['gpu'],phase=p['phase'],
        deleted=bool(p.get('deleted_epoch')),calibration_complete=p.get('calibration_complete',False),
        setup=s.get('setup_exit'),worker_exit=s.get('worker_exit'),
        task=r.get('task'),parent=r.get('parent'),anchor=r.get('anchor'),lr=r.get('lr'),lag=r.get('lag'),
        horizon=r.get('horizon'),loss=r.get('loss'),adjacent_cosine=r.get('adjacent_cosine'),
        seconds=r.get('seconds'),peak_allocated_gb=r.get('peak_allocated_bytes',0)/1e9,
        error=p.get('transport_error'),scientific_status=s.get('manifest',{}).get('status')))
cost=sum((p.get('deleted_epoch',now)-p['created_epoch'])/3600*p['price_per_gpu_hour'] for p in lease['pods'])
if args.compact or args.summary:
    receipts=list(HERE.glob('retrieval-210-*.json'))
    profiles=0
    for path in receipts:
        receipt=json.loads(path.read_text())
        if receipt['scientific_status']=='complete':
            root=HERE.parents[1]/receipt['artifact_root']
            profiles+=len(list(root.glob('*-*/result.json')))
    print(f'Elapsed {(now-lease["created_epoch"])/60:.1f} min; compute USD {cost:.2f}; '
          f'deleted {sum(r["deleted"] for r in rows)}/{len(rows)}; retrieved profiles {profiles}/27')
    if args.summary:
        live=[r for r in rows if not r['deleted'] and r['lag'] is not None]
        for task in sorted({r['task'] for r in live}):
            group=[r for r in live if r['task']==task]
            print(f'{task}: {len(group)} live workers; current lags '
                  f'{min(r["lag"] for r in group)}..{max(r["lag"] for r in group)}; '
                  f'errors={sum(bool(r["error"]) for r in group)}')
        raise SystemExit
    for r in rows:
        if not r['deleted']:
            print(f'w{r["worker"]} {r["task"]}/{r["parent"]}@{r["anchor"]} '
                  f'{r["lag"]}/{r["horizon"]} '
                  f'{r["seconds"] or 0:.2f}s/gradient '
                  f'loss={r["loss"]} peakGB={r["peak_allocated_gb"]:.2f} '
                  f'{r["scientific_status"]} error={r["error"]}')
    raise SystemExit
print(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(now)),
    elapsed_minutes=(now-lease['created_epoch'])/60,compute_estimate_usd=cost,
    source='bounded-interval controller snapshot; stale during transfer/setup',workers=rows),indent=2))
