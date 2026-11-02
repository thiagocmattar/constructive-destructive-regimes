"""Read-only concurrent task-worker status with compact progress/cost output."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud037',HERE/'05_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def one(pod):
    row={k:pod[k] for k in ('task','seed','id','gpu')}
    try:
        s=c.status(pod)
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
    for r in rows:
        if 'microbatch' in r:
            d=r['microbatch'];r['microbatch']=[dict(physical=x['physical'],seconds=x['seconds'],relative_gradient_l2_error=x['relative_gradient_l2_error'],peak_gb=round(x['peak_allocated_bytes']/1e9,2)) for x in d['samples']]
    print(json.dumps(snapshot))


if __name__=='__main__':main()
