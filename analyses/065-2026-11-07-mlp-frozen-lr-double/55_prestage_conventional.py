"""Pre-transfer immutable fork/first-audit tensors without touching ongoing science."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shlex

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud',HERE/'48_cloud_conventional.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
KEY=c.SPARKROOT+'/analyses/065-2026-11-07-mlp-frozen-lr-double/scratch/h050-retention-key'


def prestage(name):
    p=c.get(name);assert not p.get('deleted_epoch') and len(p['arms'])==1
    marker=c.S/('prestage-active-'+name+'.json')
    c.dump(marker,dict(pid=os.getpid(),utc=c.x.stamp()))
    try:
        public=c.spark('cat '+shlex.quote(KEY+'.pub')+'\n').strip()
        c.x.ssh(p,'printf "%s\\n" '+shlex.quote(public)+' >> /root/.ssh/authorized_keys\n')
        files=['fork.pt',p['arms'][0]+'/audit-0544.pt']
        inventory=json.loads(c.x.ssh(p,f'''python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path({(c.REMOTE+'/'+p['output'])!r});rows=[]
for name in {files!r}:
 path=root/name
 with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 rows.append(dict(path=name,bytes=path.stat().st_size,sha256=digest))
print(json.dumps(rows))
PY
''',timeout=60))
        paths=' '.join(shlex.quote(p['output']+'/'+f) for f in files)
        command=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15',
            '-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+KEY+'.hosts','-i',KEY,
            '-p',str(p['port']),'root@'+p['host'],
            'bash -o pipefail -c '+shlex.quote(f'tar -C {shlex.quote(c.REMOTE)} -cf - {paths} | gzip -1')]
        report=json.loads(c.spark(f'''python3 - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
root=Path({c.SPARKROOT!r});out=root/{p['output']!r};out.mkdir(parents=True,exist_ok=True)
producer=subprocess.Popen({command!r},stdout=subprocess.PIPE,stderr=subprocess.PIPE)
consumer=subprocess.Popen(['tar','-C',str(root),'-xzf','-'],stdin=producer.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
producer.stdout.close();_,error=consumer.communicate(timeout=1200);producer.wait(timeout=60)
assert producer.returncode==consumer.returncode==0,error.decode()
for item in {inventory!r}:
 path=out/item['path']
 with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 assert path.stat().st_size==item['bytes'] and digest==item['sha256'],str(path)
print(json.dumps(dict(status='passed',files={inventory!r},root=str(out))))
PY
''',timeout=1280))
        c.dump(c.ROOT/p['run']/(name+'_immutable_prestage.json'),dict(worker=name,utc=c.x.stamp(),**report))
        c.emit(event='immutable_prestage_verified',name=name,bytes=sum(r['bytes'] for r in inventory))
    except Exception as error:
        c.emit(event='immutable_prestage_error',name=name,error=str(error))
        raise
    finally:
        marker.unlink(missing_ok=True)


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(prestage,['trans-control','trans-b95','trans-b90','trans-b50','trans-clip05']))
