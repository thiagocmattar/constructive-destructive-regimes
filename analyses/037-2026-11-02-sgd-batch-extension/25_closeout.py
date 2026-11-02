"""Read back task cleanup, frozen inputs, retained storage and provisional cost."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def main():
    c=module('cloud037','05_cloud.py');t=module('spark037','16_spark_helper.py')
    assert c.read(HERE/'cohort_acceptance.json')['status']=='passed'
    barrier=c.read(HERE/'tail-barrier.json')
    assert barrier.get('finished_epoch') and barrier['status'] not in ('waiting','queue_paused')
    state=c.lease();pods=state['pods']
    assert all(p.get('deleted_epoch') for p in pods)
    actual=c.cli('pod','list');own_ids={p['id'] for p in pods}
    assert not any(p['id'] in own_ids or p['name'].startswith(c.PREFIX) for p in actual)
    volumes=c.cli('network-volume','list')
    assert not any(v['name'].startswith(c.PREFIX) for v in volumes)
    spark=json.loads(t.ssh(f'''python3 - <<'PY'
import json,subprocess
from pathlib import Path
root={t.REMOTE!r};tasks=[]
for path in Path('/proc').glob('[0-9]*'):
 try:command=(path/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 except OSError:continue
 if root in command and ('/01_measure.py' in command or '/17_spark_forks.py' in command):tasks.append(int(path.name))
old=Path('/proc/458832/status');old_state=None
if old.exists():old_state=next(line.split(':',1)[1].strip() for line in old.read_text().splitlines() if line.startswith('State:'))
print(json.dumps(dict(active_task_measurement_pids=tasks,unrelated_old_queue_state=old_state,
 retained_project_bytes=int(subprocess.check_output(['du','-sb',root]).split()[0]))))
PY
'''))
    assert not spark['active_task_measurement_pids']
    inputs=[]
    for name in ['214-2026-11-02-sgd-batch-extension','216-2026-11-02-sgd-transformer-microbatch512']:
        run=c.ROOT/'runs'/name;manifest=c.read(run/'input_manifest.json')
        for entry in manifest['files']:
            path=c.ROOT/entry['path']
            assert path.stat().st_size==entry['bytes'] and c.file_hash(path)==entry['sha256']
        inputs.append(dict(run=name,files=len(manifest['files']),config_sha256=c.file_hash(run/'config.json'),
            input_manifest_sha256=c.file_hash(run/'input_manifest.json')))
    old=c.read(HERE/'executed-inputs-verification.json');assert old['runs']==inputs
    old['closeout_rechecked_utc']=c.stamp();c.save(HERE/'executed-inputs-verification.json',old)
    prices=[];ledger=[];events=[]
    for pod in pods:
        receipt=c.read(HERE/f'create-{pod["task"]}-{pod["seed"]}-a{pod["attempt"]}.json')
        assert receipt['response']['costPerHr']==pod['price_per_gpu_hour']
        if 'env' in receipt['response']:
            del receipt['response']['env'];receipt['environment_values_omitted']=True
            c.save(HERE/f'create-{pod["task"]}-{pod["seed"]}-a{pod["attempt"]}.json',receipt)
        prices.append(dict(pod_id=pod['id'],name=pod['name'],catalog_and_allocated_usd_per_hour=pod['price_per_gpu_hour']))
        hours=(pod['deleted_epoch']-pod['created_epoch'])/3600
        ledger.append(dict(pod_id=pod['id'],name=pod['name'],gpu=pod['gpu'],
            created_epoch=pod['created_epoch'],deleted_epoch=pod['deleted_epoch'],hours=hours,
            gpu_usd=hours*pod['price_per_gpu_hour'],conservative_disk_usd=hours*65*.10/720))
        events.extend([(pod['created_epoch'],pod.get('gpu_count',1)),(pod['deleted_epoch'],-pod.get('gpu_count',1))])
    active=peak=0
    for _,delta in sorted(events):active+=delta;peak=max(peak,active)
    total=c.spend(state);ended=max(p['deleted_epoch'] for p in pods)
    assert total<40 and ended<=state['deadline_epoch'] and peak<=6
    c.save(HERE/'allocated-price-verification.json',dict(status='passed',utc=c.stamp(),resources=prices,
        storage_estimate='Conservative65GB per Pod at USD0.10/GB-month, using720hours/month; retained until deletion.',
        billing_status='provisional; elapsed-time estimate, not a posted invoice.'))
    result=dict(status='passed',utc=c.stamp(),all_task_pods_deleted=True,task_pods=len(pods),
        task_cloud_storage_gb=0,unrelated_pods_preserved=len(actual),unrelated_volumes_preserved=len(volumes),
        max_concurrent_cloud_gpus=peak,start_epoch=state['start_epoch'],last_deletion_epoch=ended,
        elapsed_hours=(ended-state['start_epoch'])/3600,approved_usd=40,approved_wall_hours=6,
        estimated_total_usd=total,billing_status='provisional; elapsed-time GPU plus conservative disk estimate, not posted billing',
        ledger=ledger,spark=spark,
        queue_barrier_receipt_sha256=c.file_hash(HERE/'tail-barrier.json'),
        production_retrieval='All six primary blocks and both helpers verified before cloud cleanup.',
        calibration_limitation='One calibration-only archive was lost; no production ran there. See retrieval-incident.json.')
    c.save(HERE/'closeout.json',result)
    c.save(c.SCRATCH/'closed.json',dict(utc=c.stamp(),closeout_sha256=c.file_hash(HERE/'closeout.json')))
    print(json.dumps({k:result[k] for k in ['status','task_pods','estimated_total_usd','elapsed_hours','task_cloud_storage_gb']}))


if __name__=='__main__':main()
