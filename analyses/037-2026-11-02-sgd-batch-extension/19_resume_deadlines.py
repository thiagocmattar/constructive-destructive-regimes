"""Resume naturally bounded queues under the user's extended, unchanged US$40 lease."""
import importlib.util
from pathlib import Path
import shutil
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud037',HERE/'05_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def resume(pod,receipt):
    lease=c.lease();assert c.spend(lease)<32
    target=c.ROOT/receipt['path'];manifest=c.read(target/'worker_manifest.json')
    assert manifest['status']=='bounded_stop' and receipt['verified']
    assert lease['scientific_deadline_epoch']>manifest['deadline_epoch']+120
    assert time.time()<lease['scientific_deadline_epoch']-600
    assert c.file_hash(target/'worker_inventory.json')==receipt['inventory_sha256']
    inventory=c.read(target/'worker_inventory.json')
    for name,entry in inventory.items():
        path=target/name;assert path.resolve().is_relative_to(target.resolve())
        assert path.stat().st_size==entry['bytes'] and c.file_hash(path)==entry['sha256']
    segment=1+len(pod.get('production_segments',[]))
    snapshot=c.ROOT/'tmp/cdr037-resume'/pod['name']/f'segment-{segment:02d}'
    assert target.resolve().is_relative_to(c.ROOT.resolve()) and snapshot.resolve().is_relative_to(c.ROOT.resolve())
    assert not snapshot.exists()
    shutil.copytree(target,snapshot)
    for name,entry in inventory.items():assert c.file_hash(snapshot/name)==entry['sha256']
    preserved=dict(receipt,path=snapshot.relative_to(c.ROOT).as_posix(),preserved_before_resume=True)
    c.save(HERE/f'stream-retrieval-{pod["name"]}-segment-{segment:02d}.json',preserved)
    progress_path=c.SCRATCH/(pod['name']+'-stream.json');stream=c.read(progress_path)
    assert stream['done']
    complete=[]
    for job in manifest['identity']['jobs']:
        folder=target/job['name']
        if (folder/'summary.json').exists():
            assert c.read(folder/'summary.json')['status']=='complete';complete.append(job['name'])
        elif folder.exists():
            moved=c.ROOT/'tmp/cdr037-resume-mutables'/pod['name']/f'segment-{segment:02d}'/job['name']
            assert folder.resolve().is_relative_to(target.resolve())
            assert moved.resolve().is_relative_to(c.ROOT.resolve()) and not moved.exists()
            moved.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(folder),str(moved))
    # Keep done=True until the stale terminal inventory is moved and a new
    # detached process has started, preventing premature terminal retrieval.
    remote_output=c.outpath(pod)
    script=f'''set -euo pipefail
test "$(cat {c.CONTROL}/production.exit)" = 0
test ! -e {c.REMOTE}/{c.runrel(pod)}/STOP
mkdir -p {c.CONTROL}/resume-segment-{segment:02d}
mv {remote_output}/worker_inventory.json {c.CONTROL}/resume-segment-{segment:02d}/worker_inventory.json
for name in production.pid production.exit production.log production.sh; do
 mv {c.CONTROL}/$name {c.CONTROL}/resume-segment-{segment:02d}/$name
done
'''
    c.ssh(pod,script)
    old=dict(pid=pod['production_pid'],started_epoch=pod['production_started_epoch'],
        previous_deadline_epoch=manifest['deadline_epoch'],snapshot=snapshot.relative_to(c.ROOT).as_posix())
    pod.setdefault('production_segments',[]).append(old)
    c.launch(pod,'production')
    latest=c.lease();next(p for p in latest['pods'] if p['id']==pod['id']).update(pod)
    c.save(c.SCRATCH/'lease.json',latest)
    stream.update(done=False,jobs=complete);c.save(progress_path,stream)
    c.save(HERE/f'resume-{pod["name"]}-segment-{segment:02d}.json',dict(utc=c.stamp(),status='resumed',
        pod_id=pod['id'],previous=old,new_pid=pod['production_pid'],new_deadline_epoch=lease['scientific_deadline_epoch'],
        preserved_inventory_sha256=receipt['inventory_sha256'],scientific_inputs_unchanged=True))
    print(c.stamp(),pod['name'],'resumed under extended lease',flush=True)


def main():
    while time.time()<c.lease()['scientific_deadline_epoch']-600 and not (c.SCRATCH/'closed.json').exists():
        for pod in c.lease()['pods']:
            if pod.get('deleted_epoch') or not pod.get('production_pid'):continue
            receipt=HERE/('stream-retrieval-'+pod['name']+'.json')
            if not receipt.exists():continue
            item=c.read(receipt)
            if item['scientific_status']!='bounded_stop':continue
            progress=c.read(c.SCRATCH/(pod['name']+'-stream.json'))
            if not progress['done']:continue
            try:resume(dict(pod),item)
            except Exception as error:print(c.stamp(),pod['name'],repr(error),flush=True)
        time.sleep(30)


if __name__=='__main__':main()
