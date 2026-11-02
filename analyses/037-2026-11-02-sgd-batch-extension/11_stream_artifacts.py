"""Retrieve immutable completed jobs during training, then verify terminal inventory."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import tarfile
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud037',HERE/'05_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def tick(pod):
    c.connect(pod)
    progress=c.SCRATCH/(pod['name']+'-stream.json')
    state=c.read(progress) if progress.exists() else dict(jobs=[],chunks=[],done=False)
    if state['done']:return
    output=c.outpath(pod)
    script='''python3 - <<'PY'
import json,subprocess,hashlib
from pathlib import Path
root=Path(OUTPUT);cached=set(CACHED)
ready=[]
if root.exists():
 for p in sorted(root.iterdir()):
  if p.is_dir() and p.name not in cached and (p/'inventory.json').exists() and (p/'summary.json').exists():ready.append(p.name)
terminal=(root/'worker_inventory.json').exists()
if terminal:
 ready=sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in cached)
selected=ready[:8 if terminal else 2];metadata=terminal and not ready
if metadata:selected=sorted(p.name for p in root.iterdir() if p.is_file())
if not selected:print(json.dumps({'ready':False,'terminal':terminal}));raise SystemExit
archive=Path('/workspace/cdr214-stream.tar.gz')
subprocess.run(['tar','-I','gzip -1','-cf',str(archive),'-C',str(root),*selected],check=True)
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps({'ready':True,'terminal':terminal,'selected':selected,'metadata':metadata,'sha256':digest,'bytes':archive.stat().st_size}))
PY
'''.replace('OUTPUT',repr(output)).replace('CACHED',repr(state['jobs']))
    packet=json.loads(c.ssh(pod,script,timeout=240))
    if not packet['ready']:return
    archive=c.SCRATCH/(pod['name']+f'-stream-{int(time.time())}.tar.gz')
    begin=time.time();c.copy(pod,archive,'/workspace/cdr214-stream.tar.gz',False)
    assert archive.stat().st_size==packet['bytes'] and c.file_hash(archive)==packet['sha256']
    target=c.runpath(pod)/f'artifacts/attempt-{pod["attempt"]:02d}'/f'{pod["task"]}-seed{pod["seed"]}'
    target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as bundle:bundle.extractall(target,filter='data')
    if packet['metadata']:
        inventory=c.read(target/'worker_inventory.json')
        for name,entry in inventory.items():
            path=target/name;assert path.resolve().is_relative_to(target.resolve())
            assert path.stat().st_size==entry['bytes'] and c.file_hash(path)==entry['sha256'],name
        actual={p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file() and p.name!='worker_inventory.json'}
        assert actual==set(inventory)
        state['done']=True
        receipt=dict(utc=c.stamp(),pod_id=pod['id'],task=pod['task'],seed=pod['seed'],run_id=pod.get('run_id','214'),
            verified=True,scientific_status=c.read(target/'worker_manifest.json')['status'],
            files=len(inventory),bytes=sum(x['bytes'] for x in inventory.values()),
            path=target.relative_to(c.ROOT).as_posix(),inventory_sha256=c.file_hash(target/'worker_inventory.json'))
        c.save(HERE/('stream-retrieval-'+pod['name']+'.json'),receipt)
    else:
        for name in packet['selected']:
            inventory_path=target/name/'inventory.json'
            # A bounded stop can retain an unfinished recovery directory.
            # Its files are certified by the terminal worker inventory below.
            if inventory_path.exists():
                inventory=c.read(inventory_path)
                for relative,entry in inventory.items():
                    path=target/name/relative;assert path.resolve().is_relative_to((target/name).resolve())
                    assert path.stat().st_size==entry['bytes'] and c.file_hash(path)==entry['sha256'],relative
            else:assert packet['terminal'] and not (target/name/'summary.json').exists()
            state['jobs'].append(name)
    state['chunks'].append(dict(utc=c.stamp(),selected=packet['selected'],metadata=packet['metadata'],
        bytes=packet['bytes'],sha256=packet['sha256'],transfer_verify_seconds=time.time()-begin))
    c.save(progress,state)
    archive.unlink()  # Verified extracted raw files and hash receipt are retained.
    print(c.stamp(),pod['name'],'retrieved',packet['selected'],'terminal',state['done'],flush=True)
    return packet['terminal']


def main():
    pending={};next_poll={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        while not (c.SCRATCH/'closed.json').exists():
            state=c.lease()
            for pod in state['pods']:
                if pod.get('deleted_epoch') or not pod.get('production_pid') or pod.get('defer_stream'):continue
                progress=c.SCRATCH/(pod['name']+'-stream.json')
                if progress.exists() and c.read(progress)['done']:continue
                if pod['id'] not in pending and time.time()>=next_poll.get(pod['id'],0):pending[pod['id']]=pool.submit(tick,dict(pod))
            for key,future in list(pending.items()):
                if not future.done():continue
                terminal=False
                try:terminal=bool(future.result())
                except Exception as error:print(c.stamp(),key,repr(error),flush=True)
                del pending[key];next_poll[key]=time.time()+(5 if terminal else 30)
            time.sleep(3)


if __name__=='__main__':main()
