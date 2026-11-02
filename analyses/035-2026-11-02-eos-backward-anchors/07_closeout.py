"""Fresh cleanup/billing reads and minimal public provider receipts."""
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('cloud035',HERE/'01_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def main():
    close=json.loads((HERE/'closeout.json').read_text())
    assert close['status']=='complete'
    before=json.loads((HERE/'preflight.json').read_text())
    pods=c.c.cli('pod','list');volumes=c.c.cli('network-volume','list')
    assert not any(p['id'] in {x['id'] for x in close['pods']} or p['name'].startswith(c.PREFIX) for p in pods)
    assert {x['id'] for x in before['baseline_volumes']}.issubset({x['id'] for x in volumes})
    c.save(HERE/'cleanup_verification.json',dict(status='passed',epoch=time.time(),task_pods_remaining=0,
        task_storage_remaining_gb=0,unrelated_volumes_preserved=True,
        remaining_pods=[dict(id=p['id'],name=p['name']) for p in pods],
        remaining_volumes=[{k:v[k] for k in ['id','name','size','dataCenterId']} for v in volumes]))
    start=datetime.fromtimestamp(min(p['created_epoch'] for p in close['pods']),timezone.utc).isoformat()
    end=datetime.now(timezone.utc).isoformat()
    try:
        raw=c.c.cli('billing','pods','--start-time',start,'--end-time',end,'--bucket-size','hour')
        ids={p['id'] for p in close['pods']}
        assert isinstance(raw,list)
        raw=[r for r in raw if r['podId'] in ids]
        c.save(HERE/'billing.json',dict(status='provisional',queried_utc=end,start_utc=start,
            task_pod_ids=[p['id'] for p in close['pods']],provider_response=raw,
            posted_gpu_usd=sum(r['gpuAmount'] for r in raw),posted_disk_usd=sum(r['diskAmount'] for r in raw),
            posted_total_usd=sum(r['totalAmount'] for r in raw),unrelated_resources_excluded=True,
            compute_allocation_estimate_usd=close['compute_estimate_usd'],
            caveat='Provider bucket posting may lag final deletion; allocation estimate is not a complete posted invoice.'))
    except Exception as error:
        c.save(HERE/'billing.json',dict(status='provisional; provider read unavailable',error=str(error),
            compute_allocation_estimate_usd=close['compute_estimate_usd'],queried_utc=end))
    fields=['id','name','costPerHr','createdAt','gpuCount','gpuId','imageName','containerDiskInGb',
        'volumeInGb','volumeMountPath','desiredStatus','runtimeStatus','dataCenterId','computeType',
        'minCudaVersion','machineId']
    private=HERE/'scratch/control-reads';private.mkdir(parents=True,exist_ok=True)
    for pattern in ['create-worker*.json','final-pod-worker*.json']:
        for path in HERE.glob(pattern):
            copy=private/path.name
            if not copy.exists():copy.write_bytes(path.read_bytes())
            raw=json.loads(path.read_text())
            c.save(path,{k:raw[k] for k in fields if k in raw})
    print(json.dumps(dict(task_pods_remaining=0,task_storage_gb=0,
        unrelated_volumes_preserved=True,compute_estimate_usd=close['compute_estimate_usd'],billing='provisional')))


if __name__=='__main__':main()
