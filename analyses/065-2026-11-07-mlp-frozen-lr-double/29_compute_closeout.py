"""Summarize verified H048 execution, controller work, storage and bounded spend."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    summary=read(HERE/'transfer_summary.json');assert summary['status']=='verified'
    assert summary['generator_sha256']==sha(HERE/'24_summarize_transfer.py')
    workers=[]
    for filename,digest in summary['sources'].items():
        path=HERE/filename;assert sha(path)==digest
        data=read(path);assert data['generator_sha256']==sha(HERE/'17_reduce_transfer.py')
        workers+=data['workers']
    arms=[]
    for worker in workers:
        folder=ROOT/worker['worker'];manifest=read(folder/'manifest.json')
        baseline=next(r for r in manifest['results'] if r['arm']['id']=='control')
        for result in manifest['results']:
            name=result['arm']['id'];reused=bool(result.get('reused_from'))
            rows=[json.loads(s) for s in (folder/name/'updates.jsonl').read_text().splitlines()]
            trials=[];zero=0
            for row in rows:
                for gate in row['gates'].values():
                    detail=gate.get('functional_budget') or gate.get('loss_guard')
                    if detail:
                        trials.append(len(detail['trials']));zero+=int(detail['trials'][-1]['factor']==0)
                        break
            arms.append(dict(run=worker['run_id'],case=worker['case'],seed=worker['seed'],worker=worker['worker'],arm=name,
                reused_control=reused,updates=result['updates'],executed_updates=0 if reused else result['updates'],
                continuation_seconds=0. if reused else result['elapsed_seconds'],source_continuation_seconds=result['elapsed_seconds'],
                seconds_per_update=None if reused else result['elapsed_seconds']/result['updates'],
                observed_seconds_per_update_change_pct=None if reused else 100*((result['elapsed_seconds']/result['updates'])/(baseline['elapsed_seconds']/baseline['updates'])-1),
                peak_cuda_bytes=result['peak_cuda_bytes'],candidate_trials_mean=float(np.mean(trials)) if trials else None,
                candidate_trials_max=max(trials) if trials else None,zero_accepted_updates=zero,
                manifest_sha256=sha(folder/name/'manifest.json')))
    storage=[];identities={}
    for number in range(269,286):
        run=next((ROOT/'runs').glob(str(number)+'-*'));files=[p for p in (run/'artifacts').rglob('*') if p.is_file()]
        configured={a['id'] for a in read(run/'config.json')['arms']}
        executed={name for w in workers if w['run_id']==number for name in w['curves']}
        for path in files:
            stat=path.stat();identity=(stat.st_dev,stat.st_ino)
            assert stat.st_ino;identities.setdefault(identity,stat.st_size)
        receipts=[]
        for path in run.glob('*retrieval.json'):
            receipt=read(path)
            if receipt.get('verified'):receipts.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),files=receipt['files'],raw_bytes=receipt['raw_bytes']))
        storage.append(dict(run=number,local_artifact_files=len(files),logical_bytes=sum(p.stat().st_size for p in files),
            input_files=len(read(run/'input_manifest.json')['files']),input_manifest_sha256=sha(run/'input_manifest.json'),
            configured_but_not_executed_arms=sorted(configured-executed),verified_receipts=receipts))
    ledger=read(HERE/'cross_architecture_leases.json');pods=[]
    for pod in ledger['pods']:
        assert pod.get('deleted_epoch') and all(j.get('verified_retrieval') for j in pod['jobs'])
        hours=(pod['deleted_epoch']-pod['created_epoch'])/3600
        pods.append(dict(name=pod['name'],id=pod['id'],hours=hours,compute_estimate_usd=hours*pod['price_per_hour'],
            conservative_compute_storage_bound_usd=hours*pod['bounded_hourly_usd'],retained_cloud_storage_gb=0))
    bound=sum(p['conservative_compute_storage_bound_usd'] for p in pods);assert bound<=ledger['budget_usd']
    output=dict(status='verified',generator_sha256=sha(Path(__file__)),summary_sha256=sha(HERE/'transfer_summary.json'),
        scientific_workers=len(workers),scientific_arm_records=len(arms),reused_control_records=sum(a['reused_control'] for a in arms),
        executed_continuation_updates=sum(a['executed_updates'] for a in arms),
        continuation_seconds_excluding_reused_controls=sum(a['continuation_seconds'] for a in arms),
        timing_scope='Arm elapsed time includes tracing, audit checkpoint writes, and terminal held-out evaluation; excludes prefix, mechanics, independent replay, transfers and setup. Relative timing is against the instrumented control, not plain Adam. Concurrent Spark timings are not isolated benchmarks.',
        arms=arms,storage=storage,local_logical_artifact_bytes=sum(s['logical_bytes'] for s in storage),
        local_unique_file_bytes=sum(identities.values()),
        storage_note='Unique bytes identify Windows volume/file IDs, counting hardlinks once. Original paths may be junctions; see artifact_storage.json. Spark retains its original raw evidence, including metadata-only mechanics retrievals.',
        pods=pods,conservative_task_spend_bound_usd=bound,user_ceiling_usd=ledger['budget_usd'],
        billing_status='Provisional elapsed-lease estimate; reconcile separately with task-filtered provider billing.',retained_cloud_storage_gb=0)
    (HERE/'transfer_compute_closeout.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['scientific_workers','scientific_arm_records','reused_control_records','executed_continuation_updates','local_unique_file_bytes','conservative_task_spend_bound_usd']}))


if __name__=='__main__':main()
