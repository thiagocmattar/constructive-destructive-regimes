"""Read-only concurrent task-worker status with compact progress/cost output."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud037',HERE/'05_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def one(pod):
    row={k:pod[k] for k in ('task','seed','id','gpu')}
    try:
        s=c.status(pod)
        row['scientific_status']=s.get('worker_manifest',{}).get('status','running')
        row.update({k:s[k] for k in ('setup_exit','calibration_exit','calibration216_exit','production_exit','microbatch_exit','worker','event','gpu','disk') if k in s})
        profile=s.get('profile216',s.get('profile'))
        if profile:
            row['timings']=[dict(batch=x['batch'],seconds=x['seconds']) for x in profile['samples'] if x['repetition']==2]
            row['profile_status']=profile.get('status','running')
        if s.get('microbatch') and not pod.get('production_pid'):row['microbatch']=s['microbatch']
    except Exception as error:row['error']=repr(error)
    return row


def main():
    state=c.lease();pods=[p for p in state['pods'] if not p.get('deleted_epoch')]
    with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(one,pods))
    snapshot=dict(utc=c.stamp(),estimated_task_usd=c.spend(state),deadline_epoch=state['deadline_epoch'],workers=rows)
    c.save(c.SCRATCH/'latest-status.json',snapshot)
    if '--compact' in sys.argv:
        print(snapshot['utc'],'estimated task USD',round(snapshot['estimated_task_usd'],2))
        for r in rows:
            event=r.get('event',{});worker=r.get('worker',{});job=worker.get('job',{})
            if not event:
                print(f'{r["task"]} seed{r["seed"]}: {job.get("name","?")} initializing'
                    +(f', warning={r["error"]}' if 'error' in r else ''))
                continue
            print(f'{r["task"]} seed{r["seed"]}: {r.get("scientific_status","?")}, '
                f'{job.get("name","?")} {event.get("update","?")}/{event.get("requested_updates","?")}, '
                f'loss={event.get("loss",0):.4f}, {event.get("seconds_per_update",0):.3f}s/update, '
                f'peak={event.get("peak_allocated_bytes",0)/1e9:.2f}GB'
                +(f', warning={r["error"]}' if 'error' in r else ''))
        return
    for r in rows:
        if 'microbatch' in r:
            d=r['microbatch'];r['microbatch']=[dict(physical=x['physical'],seconds=x['seconds'],relative_gradient_l2_error=x['relative_gradient_l2_error'],peak_gb=round(x['peak_allocated_bytes']/1e9,2)) for x in d['samples']]
    print(json.dumps(snapshot))


if __name__=='__main__':main()
