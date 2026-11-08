"""Read back all task resources and task-filtered billing after verified deletion."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def cli(*args):
    result=subprocess.run(['runpodctl',*args],capture_output=True,text=True,check=True,
        timeout=75,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return json.loads(result.stdout)


def main():
    ledger=read(HERE/'cross_architecture_leases.json')
    assert len(ledger['pods'])==5
    assert all(p.get('deleted_epoch') and all(j.get('verified_retrieval') for j in p['jobs']) for p in ledger['pods'])
    task_ids={p['id'] for p in ledger['pods']}
    resources=cli('pod','list','--all')  # Default list omits stopped pods.
    assert not [p for p in resources if p['id'] in task_ids or p.get('name','').startswith('cdr-h048-')]
    now=datetime.now(timezone.utc)
    start=datetime.fromtimestamp(min(p['created_epoch'] for p in ledger['pods']),timezone.utc).replace(minute=0,second=0,microsecond=0)
    begin=start.isoformat(timespec='seconds').replace('+00:00','Z')
    end=now.isoformat(timespec='seconds').replace('+00:00','Z')
    def billing(pod):
        rows=cli('billing','pods','--bucket-size','hour','--start-time',begin,'--end-time',end,'--pod-id',pod['id'])
        assert all(row['podId']==pod['id'] for row in rows)
        return dict(name=pod['name'],id=pod['id'],rows=rows,
            provider_reported_total_usd=sum(row['totalAmount'] for row in rows))
    with ThreadPoolExecutor(max_workers=5) as pool:pods=list(pool.map(billing,ledger['pods']))
    bound=sum((p['deleted_epoch']-p['created_epoch'])/3600*p['bounded_hourly_usd'] for p in ledger['pods'])
    assert bound<=ledger['budget_usd']
    reported=sum(p['provider_reported_total_usd'] for p in pods)
    record=dict(status='verified_resource_closeout',utc=now.isoformat(),generator_sha256=sha(Path(__file__)),
        ledger_sha256=sha(HERE/'cross_architecture_leases.json'),
        task_pod_ids_absent_including_stopped=sorted(task_ids),remaining_task_pods=0,
        unrelated_account_resources_preserved=len(resources),retained_task_cloud_storage_gb=0,
        storage_note='Only Pod-local container/volume disks were created; no network volume was provisioned.',
        billing_start=begin,billing_end=end,pods=pods,provider_reported_total_usd=reported,
        conservative_elapsed_compute_storage_bound_usd=bound,user_ceiling_usd=ledger['budget_usd'],
        provider_report_within_ceiling=reported<=ledger['budget_usd'],
        billing_status='Provisional: provider history can lag deletion and omit recent buckets; elapsed-lease accounting bounds this task independently.')
    (HERE/'transfer_cloud_closeout.json').write_text(json.dumps(record,indent=2)+'\n')
    (HERE/'scratch/cross-architecture/WATCHER_DONE').touch()
    print(json.dumps({k:record[k] for k in ['remaining_task_pods','retained_task_cloud_storage_gb',
        'provider_reported_total_usd','conservative_elapsed_compute_storage_bound_usd','provider_report_within_ceiling']}))


if __name__=='__main__':main()
