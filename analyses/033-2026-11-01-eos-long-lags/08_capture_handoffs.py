"""Capture stable measured checkpoints for faster, unchanged-metric handoffs."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('cloud',HERE/'01_cloud.py')
cloud=importlib.util.module_from_spec(spec);spec.loader.exec_module(cloud)


def one(specification):
    target,worker,rid,anchor,baseline,minimum=specification
    pod=next(p for p in cloud.loadlease()['pods'] if p['worker']==worker and not p.get('deleted_epoch'))
    rel=cloud.run_rel(pod)
    source=f'{cloud.REMOTE}/{rel}/artifacts/attempt-{pod.get("attempt",1):02d}/worker{worker}'
    if baseline:source+='/baseline'
    source+=f'/{rid}-{anchor}'
    cfg=json.loads((cloud.RUN/'config.json').read_text())
    job=next(j for j in cfg['jobs'] if j['parent_run_id']==rid and j['anchor']==anchor)
    output=f'/tmp/handoff-{rid}-{anchor}'
    script=f'''cd {cloud.REMOTE}
/opt/analysis033-venv/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,shutil,subprocess,time
import numpy as np
import torch
def cpu_products(anchor,gradient):
 aa=bb=ab=0.0
 for start in range(0,len(anchor),65536):
  a=np.asarray(anchor[start:start+65536],dtype=np.float64);b=np.asarray(gradient[start:start+65536],dtype=np.float64)
  aa+=float(np.dot(a,a));bb+=float(np.dot(b,b));ab+=float(np.dot(a,b))
 return dict(cosine=ab/(aa*bb)**.5,norm=bb**.5)
source=Path({json.dumps(source)});out=Path({json.dumps(output)});out.mkdir(exist_ok=True)
for trial in range(4):
 d=json.loads((source/'result.json').read_text())
 assert d['last_lag']>={minimum}
 for name in ['end.pt','audit-current.npy','audit-previous.npy','audit-first.npy','result.json']:shutil.copyfile(source/name,out/name)
 state=torch.load(out/'end.pt',map_location='cpu',weights_only=False)
 saved=json.loads((out/'result.json').read_text())
 if state['step']!=saved['last_state'] or saved['last_lag']!=d['last_lag']:time.sleep(1);continue
 anchor=np.load({json.dumps(cloud.REMOTE+'/'+job['anchor_gradient'])});g=np.load(out/'audit-current.npy')
 check=cpu_products(anchor,g)
 if abs(check['cosine']-saved['rows'][-1]['cosine'])>1e-10:time.sleep(1);continue
 assert abs(check['norm']/saved['rows'][-1]['norm']-1)<1e-10
 assert abs(cpu_products(anchor,np.load(out/'audit-previous.npy'))['cosine']-saved['rows'][-2]['cosine'])<1e-10
 break
else:raise RuntimeError('Checkpoint changed during capture')
files=[]
for p in out.iterdir():
 if p.is_file() and p.name!='capture.json':
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  files.append(dict(path=p.name,bytes=p.stat().st_size,sha256=h))
(out/'capture.json').write_text(json.dumps(dict(parent_run_id={json.dumps(rid)},anchor={anchor},last_lag=saved['last_lag'],files=files),indent=2))
archive=str(out)+'.tar.gz'
subprocess.run(['tar','-I','gzip -1','-cf',archive,'-C',str(out),'.'],check=True)
with open(archive,'rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(path=archive,sha256=h,lag=saved['last_lag'])))
PY
'''
    result=json.loads(cloud.ssh(pod,script,timeout=90))
    local=cloud.ROOT/f'cache/analysis033-handoff-{rid}-{anchor}.tar.gz'
    cloud.copy(pod,result['path'],local)
    assert cloud.sha(local)==result['sha256']
    destination=cloud.ROOT/f'cache/analysis033-handoffs/{rid}-{anchor}';destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(local) as tar:tar.extractall(destination,filter='data')
    capture=json.loads((destination/'capture.json').read_text())
    for item in capture['files']:
        p=destination/item['path'];assert p.stat().st_size==item['bytes'] and cloud.sha(p)==item['sha256']
    return dict(worker=target,source_worker=worker,parent_run_id=rid,anchor=anchor,last_lag=capture['last_lag'],
        snapshot=destination.relative_to(cloud.ROOT).as_posix(),archive_sha256=result['sha256'],
        source_artifact=source.removeprefix(cloud.REMOTE+'/'),verified=True)


def main():
    definitions=[(15,5,'125',500,False,512),(17,12,'124',3000,True,256),
                 (18,13,'120',3000,True,256),(16,14,'123',3000,True,128)]
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(one,definitions))
    cloud.save(HERE/'handoff_captures.json',dict(program_sha256=cloud.sha(Path(__file__)),captures=records))
    print(json.dumps(records))


if __name__=='__main__':main()
