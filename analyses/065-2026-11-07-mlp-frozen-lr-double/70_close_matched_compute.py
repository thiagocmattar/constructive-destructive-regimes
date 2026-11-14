"""Verify owned-resource teardown, durable evidence and provisional task cost."""
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('fleet',HERE/'58_matched_fleet.py');f=importlib.util.module_from_spec(s);s.loader.exec_module(f)


def main():
    results=json.loads((HERE/'matched/results.json').read_text())
    assert results['complete_matrix'] and results['conditions']==36
    leases=[json.loads(p.read_text()) for p in f.S.glob('lease-*.json')]
    assert leases and all(p.get('deleted_epoch') for p in leases),'Workers remain; verify/download before deleting'
    for p in leases:
        assert p.get('retrieval_verified') or p.get('no_scientific_artifacts'),p['name']
    ids={p['id'] for p in leases};live=f.x.cli('pod','list')
    remaining=[p['id'] for p in live if p['id'] in ids or p.get('name','').startswith('cdr-h051-')]
    assert not remaining,remaining
    records=[]
    for p in leases:
        hours=(p['deleted_epoch']-p['created_epoch'])/3600
        records.append(dict(name=p['name'],id=p['id'],gpu=p['gpu'],hours_including_startup=hours,
            compute_usd=hours*p['price_per_hour'],reserved_disk_usd=hours*p['storage_hourly_usd'],
            verified_retrieval=p.get('retrieval_verified',False),no_scientific_artifacts=p.get('no_scientific_artifacts',False)))
    total=sum(v['compute_usd']+v['reserved_disk_usd'] for v in records)
    assert total<50,total
    execution_hours=(max(p['deleted_epoch'] for p in leases)-min(p['created_epoch'] for p in leases))/3600
    assert execution_hours<=8,execution_hours
    retention=[]
    for case,run in f.RUNS.items():
        receipt=json.loads((f.ROOT/run/'spark_durable_retention.json').read_text());assert receipt['status']=='passed'
        inventory=json.loads((f.ROOT/run/'artifacts/attempt-01/durable_inventory.json').read_text())
        assert len(inventory['files'])==receipt['files']
        assert sum(v['bytes'] for v in inventory['files'])==receipt['raw_bytes']
        for path in (f.ROOT/run).glob('*_immutable_prestage.json'):
            pre=json.loads(path.read_text());assert pre['status']=='passed'
            worker=path.name.removesuffix('_immutable_prestage.json')
            final={v['path']:v for v in inventory['files']}
            for item in pre['inventory']:
                exact=final[worker+'/'+item['path']]
                assert exact['bytes']==item['bytes'] and exact['sha256']==item['sha256']
        retention.append(dict(case=case,**receipt))
    # Stop only task-owned monitors. Their loops exit at their next bounded poll.
    (f.S/'STOP_WATCHER').write_text('All task workers deleted after verified retrieval.\n')
    (f.S/'STOP_PRESTAGE').write_text('All immutable evidence retained and verified.\n')
    cleanup=f.spark(f'''set -e
{f.SPARKPY} - <<'PY'
from pathlib import Path
import json,subprocess
root=Path({f.SPARKROOT!r}).resolve()
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
active=[line for line in processes if '69_copy_matched_inventory.py' in line and '/scratch/' in line and 'python' in line]
assert not active,active
removed=[]
for suffix in ['', '.pub', '.hosts']:
 p=Path({f.KEY!r}+suffix).resolve();assert p.is_relative_to(root)
 if p.exists():p.unlink();removed.append(str(p.relative_to(root)))
print(json.dumps(dict(status='passed',task_transfer_key_removed=True,removed=removed)))
PY
''',timeout=30)
    report=dict(status='passed',utc=f.x.stamp(),conditions=36,updates=11904,gradient_update_audits=108,
        task_pods_remaining=remaining,retained_cloud_volumes=0,provisional_total_usd=total,
        provisional_compute_usd=sum(v['compute_usd'] for v in records),provisional_disk_usd=sum(v['reserved_disk_usd'] for v in records),
        billing_basis='Live allocated hourly rates times recorded creation-to-deletion duration, including cold starts; 60GB reserved temporary storage at the recorded running-storage rate. Provider reconciliation may differ.',
        maximum_authorized_total_usd=50,workers=records,retention=retention,
        cloud_execution_hours_including_startup_and_retrieval=execution_hours,
        maximum_authorized_execution_hours=8,
        total_raw_bytes_on_owned_spark=sum(v['raw_bytes'] for v in retention),
        cleanup=json.loads(cleanup.strip().splitlines()[-1]),
        source_result_sha256=f.x.file_hash(HERE/'matched/results.json'))
    f.dump(HERE/'matched/compute_closeout.json',report)
    print(json.dumps({k:report[k] for k in ['status','conditions','updates','task_pods_remaining','retained_cloud_volumes','provisional_total_usd','total_raw_bytes_on_owned_spark']}))


if __name__=='__main__':main()
