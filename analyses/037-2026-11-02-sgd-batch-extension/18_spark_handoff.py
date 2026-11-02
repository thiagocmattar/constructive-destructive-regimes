"""Hash-check Spark forks and atomically hand them to the still-earlier cloud queue."""
import importlib.util
import json
from pathlib import Path
import tarfile
import time

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('spark037',HERE/'16_spark_helper.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);c=t.c
OUT=t.REMOTE+'/'+t.REL+'/artifacts/spark-assist/resnet-seed2'
LOCAL=c.RUN/'artifacts/spark-assist/resnet-seed2'


def reference(pod,state):
    receipt=HERE/f'spark-reference-state-{state:04d}.json'
    if receipt.exists():return
    remote=c.outpath(pod)+f'/reference/state-{state:04d}.pt'
    result=c.ssh(pod,f'if test -f {remote}; then sha256sum {remote}; fi\n').strip()
    if not result:return
    digest=result.split()[0];local=c.SCRATCH/f'spark-state-{state:04d}.pt'
    c.copy(pod,local,remote,False);assert c.file_hash(local)==digest
    target=t.REMOTE+f'/cloud-reference/state-{state:04d}.pt'
    t.copy(local,target)
    assert t.ssh(f'sha256sum {target}\n').split()[0]==digest
    c.save(receipt,dict(utc=c.stamp(),source_pod=pod['id'],state=state,sha256=digest,
        bytes=local.stat().st_size,verified=True))


def tick(pod,state):
    reference(pod,2000)
    script=f'''python3 - <<'PY'
import json,hashlib,subprocess
from pathlib import Path
root=Path({OUT!r});cached=set({state['downloaded']!r})
ready=sorted(p.name for p in root.glob('*') if p.is_dir() and p.name not in cached and (p/'summary.json').exists() and (p/'inventory.json').exists())[:2]
if not ready:print(json.dumps(dict(ready=False)));raise SystemExit
archive=Path({(t.REMOTE+'/handoff.tar.gz')!r})
subprocess.run(['tar','-I','gzip -1','-cf',str(archive),'-C',str(root),*ready],check=True)
with archive.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
print(json.dumps(dict(ready=True,jobs=ready,sha256=digest,bytes=archive.stat().st_size)))
PY
'''
    packet=json.loads(t.ssh(script,timeout=120))
    if not packet['ready']:return
    archive=c.SCRATCH/'spark-handoff.tar.gz';t.copy(archive,t.REMOTE+'/handoff.tar.gz',False)
    assert archive.stat().st_size==packet['bytes'] and c.file_hash(archive)==packet['sha256']
    LOCAL.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as bundle:bundle.extractall(LOCAL,filter='data')
    ordinary=[]
    for name in packet['jobs']:
        folder=LOCAL/name;inventory=c.read(folder/'inventory.json')
        for relative,entry in inventory.items():
            path=folder/relative;assert path.resolve().is_relative_to(folder.resolve())
            assert path.stat().st_size==entry['bytes'] and c.file_hash(path)==entry['sha256']
        summary=c.read(folder/'summary.json');identity=summary['identity']
        assert summary['status']=='complete' and summary['updates']==200 and summary['valid_cosines']==199
        assert identity['task']=='resnet' and identity['seed']==2 and not identity['mechanics_only']
        assert identity['config_sha256']==c.file_hash(c.RUN/'config.json')
        assert identity['input_manifest_sha256']==c.file_hash(c.RUN/'input_manifest.json')
        assert identity['source']['sha256']==c.read(HERE/f'spark-reference-state-{identity["state"]:04d}.json')['sha256']
        history=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        assert [r['update'] for r in history]==list(range(1,201)) and history[0]['cosine'] is None
        np.testing.assert_allclose(np.mean([r['cosine'] for r in history[1:]]),summary['mean_rho'],rtol=1e-12)
        np.testing.assert_allclose(history[-1]['B']/sum(r['norm'] for r in history),summary['R'],rtol=1e-12)
        if identity['batch']!=50:
            rng=np.random.default_rng(np.random.SeedSequence([20261102,1,2,identity['state'],identity['repetition']]))
            indices=np.fromfile(folder/'indices.u32',dtype='<u4').reshape(200,identity['batch'])
            for row in indices:np.testing.assert_array_equal(row,rng.permutation(50)[:identity['batch']])
            ordinary.append(name)
        state['downloaded'].append(name)
    if ordinary:
        # The parent must still be at least three jobs before the imported arm.
        # Refuse an occupied destination; never overwrite a running or finished fork.
        transfer=c.SCRATCH/'spark-to-cloud.tar.gz'
        with tarfile.open(transfer,'w:gz',compresslevel=1) as bundle:
            for name in ordinary:bundle.add(LOCAL/name,arcname=name)
        remote_archive=c.CONTROL+'/spark-handoff.tar.gz';c.copy(pod,transfer,remote_archive)
        script=f'''set -e
echo '{c.file_hash(transfer)}  {remote_archive}' | sha256sum -c - > /dev/null
python3 - <<'PY'
from pathlib import Path
import json,tarfile,os,time,hashlib
root=Path({c.outpath(pod)!r});staging=Path({(c.CONTROL+'/spark-incoming')!r});staging.mkdir(exist_ok=True)
with tarfile.open({remote_archive!r}) as bundle:bundle.extractall(staging,filter='data')
jobs=json.loads((root/'worker_manifest.json').read_text())['identity']['jobs'];current=json.loads((root/'worker_progress.json').read_text())['job_index']
accepted=[]
for name in {ordinary!r}:
 index=next(i for i,j in enumerate(jobs) if j['name']==name)
 target=root/name;source=staging/name
 if (target/'summary.json').exists():
  with (target/'summary.json').open('rb') as stream:existing=hashlib.file_digest(stream,'sha256').hexdigest()
  with (source/'summary.json').open('rb') as stream:incoming=hashlib.file_digest(stream,'sha256').hexdigest()
  assert existing==incoming,'Destination contains a different result: '+name
  accepted.append(dict(name=name,summary_sha256=existing,already_imported=True));continue
 if target.exists() or index-current<3:raise RuntimeError('Destination occupied or too close to running queue: '+name)
 inventory=json.loads((source/'inventory.json').read_text())
 for relative,entry in inventory.items():
  path=source/relative
  with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  assert path.stat().st_size==entry['bytes'] and digest==entry['sha256']
 os.rename(source,target)
 with (target/'summary.json').open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 accepted.append(dict(name=name,summary_sha256=digest))
print(json.dumps(dict(accepted=accepted,parent_job_index=current,epoch=time.time())))
PY
'''
        acknowledgment=json.loads(c.ssh(pod,script,timeout=120))
        for item in acknowledgment['accepted']:
            assert item['summary_sha256']==c.file_hash(LOCAL/item['name']/'summary.json')
        state['handoffs'].append(dict(utc=c.stamp(),destination_pod=pod['id'],**acknowledgment))
    c.save(c.SCRATCH/'spark-handoff-state.json',state)
    c.save(HERE/'spark-handoff.json',dict(status='in_progress',verified=True,source_host=t.HOST,
        output=LOCAL.relative_to(c.ROOT).as_posix(),downloaded=state['downloaded'],handoffs=state['handoffs']))
    print(c.stamp(),'verified Spark jobs',packet['jobs'],'handed off',ordinary,flush=True)


def main():
    while time.time()<c.lease()['scientific_deadline_epoch'] and not (c.SCRATCH/'closed.json').exists():
        state=c.read(c.SCRATCH/'spark-handoff-state.json') if (c.SCRATCH/'spark-handoff-state.json').exists() else dict(downloaded=[],handoffs=[])
        if len(state['downloaded'])==33:
            remote=t.REMOTE+'/helper-final.tar.gz'
            digest=t.ssh(f"set -e\ntar -I 'gzip -1' -cf {remote} -C {OUT} .\nsha256sum {remote}\n",timeout=120).split()[0]
            archive=c.SCRATCH/'spark-helper-final.tar.gz';t.copy(archive,remote,False)
            assert c.file_hash(archive)==digest
            with tarfile.open(archive) as bundle:bundle.extractall(LOCAL,filter='data')
            inventory=c.read(LOCAL/'helper_inventory.json')
            for name,entry in inventory.items():
                path=LOCAL/name;assert path.resolve().is_relative_to(LOCAL.resolve())
                assert path.stat().st_size==entry['bytes'] and c.file_hash(path)==entry['sha256']
            assert set(inventory)=={p.relative_to(LOCAL).as_posix() for p in LOCAL.rglob('*') if p.is_file() and p.name!='helper_inventory.json'}
            assert c.read(LOCAL/'helper_manifest.json')['status']=='complete'
            retained=int(t.ssh(f'du -sb {t.REMOTE}\n').split()[0])
            c.save(HERE/'spark-retrieval.json',dict(status='passed',verified=True,utc=c.stamp(),
                source_host=t.HOST,path=LOCAL.relative_to(c.ROOT).as_posix(),
                files=len(inventory),bytes=sum(x['bytes'] for x in inventory.values()),
                inventory_sha256=c.file_hash(LOCAL/'helper_inventory.json'),archive_sha256=digest,
                retained_remote_project_bytes=retained,cloud_storage_cost_usd=0))
            receipt=c.read(HERE/'spark-handoff.json');receipt['status']='complete';c.save(HERE/'spark-handoff.json',receipt)
            print(c.stamp(),'Spark raw inventory fully retrieved and verified',flush=True)
            return
        pod=next(p for p in c.lease()['pods'] if p['id']=='92xz66q1cvuby0')
        if pod.get('deleted_epoch'):return
        try:tick(pod,state)
        except Exception as error:print(c.stamp(),repr(error),flush=True)
        time.sleep(15)


if __name__=='__main__':main()
