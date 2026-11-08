"""Small read-only progress snapshot and early cross-worker replay gate."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud',HERE/'48_cloud_conventional.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def status(name):
    d=c.status(name);q=d.pop('qualification',None);p=c.get(name)
    if q:
        c.dump(c.ROOT/p['run']/(name+'_qualification.json'),q)
        d['replay']=q['eight_step_fingerprint'][:12]
        d['qualification_seconds_per_update']=round(q['measured_seconds_per_update'],2)
    if d.get('progress'):
        a=d['progress'];d['progress']={k:a[k] for k in ['arm','state','loss','remaining_seconds','peak_cuda_bytes','processed_units']}
        d['progress']['remaining_minutes']=round(d['progress'].pop('remaining_seconds')/60,1)
    return d


def main():
    names=[p.stem[6:] for p in c.S.glob('lease-*.json')]
    with ThreadPoolExecutor(max_workers=7) as pool:
        values=list(pool.map(status,names))
    for d in values:print(json.dumps(d))
    qualifications=[]
    for name in names:
        p=c.get(name);f=c.ROOT/p['run']/(name+'_qualification.json')
        if name!='resnet' and f.exists():qualifications.append((name,json.loads(f.read_text())))
    if len(qualifications)==6:
        reference=qualifications[0][1]
        for n,q in qualifications:
            for key in ['environment','fork_fingerprint','eight_step_fingerprint','eight_step_trace']:
                assert q[key]==reference[key],(n,key,'Matched control required')
        c.dump(HERE/'conventional_control_gate.json',dict(status='passed',workers=[n for n,_ in qualifications],
            exact_fields=['environment','fork_fingerprint','eight_step_fingerprint','eight_step_trace'],
            eight_step_fingerprint=reference['eight_step_fingerprint']))
        print('All six Transformer workers pass exact common-control replay.')
    spend=0.;bound=0.
    for f in list(c.S.glob('lease-*.json'))+list(c.S.glob('aborted-*.json')):
        p=json.loads(f.read_text());spend+=(p.get('deleted_epoch',p.get('stopped_epoch',time.time()))-p['created_epoch'])/3600*p['price_per_hour']
        bound+=(p.get('deleted_epoch',p['deadline_epoch'])-p['created_epoch'])/3600*p['bounded_hourly_usd']
    print(json.dumps(dict(provisional_compute_usd=round(spend,3),maximum_committed_including_storage_allowance=round(bound,3),task_ceiling_usd=20)))
    c.dump(c.S/'latest-status.json',dict(utc=c.x.stamp(),workers=values,provisional_compute_usd=spend,maximum_committed_usd=bound))


if __name__=='__main__':main()
