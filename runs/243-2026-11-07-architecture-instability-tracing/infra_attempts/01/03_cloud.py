"""One owned Run243 cloud worker, bounded lease and verified checkpoint retrieval."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];S=HERE/'scratch';S.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('transport',ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/03_cloud.py')
x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x);x.SCRATCH=S
REL=HERE.relative_to(ROOT).as_posix();REMOTE='/workspace/cdr243';OUT=REMOTE+'/'+REL+'/artifacts/attempt-01'

def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',newline='\n')
def log(**v):print(json.dumps(dict(utc=x.stamp(),**v)),flush=True)
def read(p,path):return json.loads(x.ssh(p,'cat '+shlex.quote(path)+'\n'))

def build():
    files=json.loads((HERE/'input_manifest.json').read_text())['files'];packet=S/'cloud-inputs.tar.gz'
    with tarfile.open(packet,'w:gz',compresslevel=1) as t:
        for f in files:
            p=ROOT/f['path']
            if not p.exists() and '/inputs/' in f['path']:
                case=Path(f['path']).parent.name;cfg=json.loads((HERE/'config.json').read_text())['cases'][case]
                base=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts'/cfg['parent_worker']/'payload/worker'
                p=base/('prefix/end.pt' if f['path'].endswith('prefix.pt') else 'control/boundaries.jsonl')
            assert p.stat().st_size==f['bytes'] and x.file_hash(p)==f['sha256']
            t.add(p,arcname=f['path'])
        t.add(HERE/'input_manifest.json',arcname=REL+'/input_manifest.json')
    return packet

def main():
    assert not (HERE/'cloud_lease.json').exists()
    packet=build();catalog=x.cli('gpu','list');g=next(g for g in catalog if g['displayName']=='RTX 5090' and g['available'])
    assert g['securePricePerHr']<=1.19
    image=json.loads((ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/scratch/image-identity.json').read_text())['image']
    public=(Path.home()/'.ssh/id_ed25519_runpod.pub').read_text().strip()
    bootstrap='set -e; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server; mkdir -p /run/sshd /root/.ssh; printf "%s\\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; sleep infinity'
    created=time.time();actual=x.create_wait('pod','create','--name','cdr243-architecture-trace','--image',image,'--gpu-id',g['gpuId'],'--gpu-count','1','--cloud-type','SECURE','--min-cuda-version','13.0','--container-disk-in-gb','30','--volume-in-gb','250','--ports','22/tcp','--ssh','--env',json.dumps({'PUBLIC_KEY':public}),'--docker-args','bash -lc '+shlex.quote(bootstrap),'--wait','--wait-timeout','20s')
    pod=dict(id=actual['id'],created_epoch=created,price_per_hour=actual.get('costPerHr',g['securePricePerHr']),deadline_epoch=created+9000,ceiling_usd=5,image=image)
    write(HERE/'cloud_lease.json',pod);write(S/'pod-creation.json',actual);log(event='created',**pod)
    try:
        deadline=time.time()+600
        while time.time()<deadline:
            a=x.cli('pod','get',pod['id']);info=a.get('ssh') or x.cli('ssh','info',pod['id'])
            if info.get('ip') and info.get('port'):
                pod.update(host=info['ip'],port=info['port']);write(HERE/'cloud_lease.json',pod)
                try:x.ssh(pod,'echo ready\n',timeout=20);break
                except Exception:pass
            time.sleep(10)
        else:raise RuntimeError('SSH unavailable after10minutes')
        x.copy(pod,packet,'/workspace/input243.tar.gz')
        setup=f'''set -e
mkdir -p {REMOTE}
echo '{x.file_hash(packet)}  /workspace/input243.tar.gz' | sha256sum -c -
tar -xzf /workspace/input243.tar.gz -C {REMOTE}
python -m pip install --no-cache-dir numpy==2.5.0 scipy==1.18.1 threadpoolctl==3.6.0
python -c "import torch; assert torch.__version__.split('+')[0]=='2.11.0'; print(torch.__version__,torch.cuda.get_device_name())"
cd {REMOTE}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
python {REL}/01_worker.py --case resnet
python {REL}/01_worker.py --case transformer
'''
        f=S/'cloud-worker.sh';f.write_text(setup,newline='\n');x.copy(pod,f,'/workspace/worker.sh')
        x.ssh(pod,'setsid bash -c "timeout --signal=TERM --kill-after=30 7500 bash /workspace/worker.sh > /workspace/worker.log 2>&1; echo \\$? > /workspace/worker.exit" </dev/null >/workspace/driver.log 2>&1 &\necho $!\n')
        while True:
            assert time.time()<pod['deadline_epoch']-600,'Retrieval reserve reached'
            text=x.ssh(pod,'if test -f /workspace/worker.exit; then cat /workspace/worker.exit; else echo WAIT; fi\n')
            status=x.ssh(pod,'tail -c 1600 /workspace/worker.log\n');log(event='progress',tail=status)
            write(HERE/'cloud_progress.json',dict(utc=x.stamp(),exit=text.strip(),tail=status,elapsed=time.time()-created))
            if text.strip()!='WAIT':
                assert text.strip()=='0',status
                break
            time.sleep(30)
        # Keep replayable model checkpoints and complete raw gradients. Dense
        # activation arrays are deterministic derived caches; hash inventory
        # retained, but their regeneration source checkpoints are downloaded.
        script=f'''python - <<'PY'
from pathlib import Path
import hashlib,json,tarfile
root=Path({OUT!r}); files=[]
for p in sorted(root.rglob('*')):
 if p.is_file():
  keep=('raw' not in p.parts or p.name in ['fixed_inputs.npy','fixed_targets.npy'])
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  files.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=h,download=keep))
(root/'inventory.json').write_text(json.dumps(dict(files=files)))
with tarfile.open('/workspace/results243.tar.gz','w:gz',compresslevel=1) as t:
 t.add(root/'inventory.json',arcname='inventory.json')
 for f in files:
  if f['download']:t.add(root/f['path'],arcname=f['path'])
p=Path('/workspace/results243.tar.gz')
with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(bytes=p.stat().st_size,sha256=h,all_raw_bytes=sum(f['bytes'] for f in files))))
PY
'''
        receipt=json.loads(x.ssh(pod,script,timeout=600));target=HERE/'artifacts/cloud';target.mkdir(parents=True,exist_ok=True);archive=target/'results.tar.gz'
        x.copy(pod,archive,'/workspace/results243.tar.gz',False)
        assert archive.stat().st_size==receipt['bytes'] and x.file_hash(archive)==receipt['sha256']
        with tarfile.open(archive) as t:t.extractall(target,filter='data')
        files=json.loads((target/'inventory.json').read_text())['files']
        for f in files:
            if f['download']:
                p=target/f['path'];assert p.stat().st_size==f['bytes'] and x.file_hash(p)==f['sha256']
        receipt.update(verified=True,downloaded_files=sum(f['download'] for f in files),retained_checkpoint_reproduction=True)
        write(HERE/'cloud_retrieval.json',receipt);pod['verified_retrieval']=True;log(event='retrieved',**receipt)
    except BaseException as e:
        pod['error']=repr(e);write(HERE/'cloud_lease.json',pod);raise
    finally:
        # This pod and its ephemeral disk belong solely to this task. The
        # controller retains failed logs; no unrelated resource is touched.
        if pod.get('host'):
            try:(S/'final-worker.log').write_text(x.ssh(pod,'tail -c 30000 /workspace/worker.log\n'))
            except Exception:pass
        x.cli('pod','delete',pod['id']);pod['deleted_epoch']=time.time();pod['estimated_compute_usd']=(pod['deleted_epoch']-created)/3600*float(pod['price_per_hour']);write(HERE/'cloud_lease.json',pod)
        log(event='deleted',id=pod['id'])

if __name__=='__main__':main()
