"""Task-owned Runpod transport, bounded monitoring and independent deadline guard."""
from pathlib import Path
import argparse
import json
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
CLI=Path.home()/'.local/bin/runpodctl.exe'
REMOTE='/workspace/constructive'
PRODUCTION='runs/186-2026-10-30-mlp-minibatch-noise/artifacts/attempt-01/production'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n');temp.replace(path)


def cli(*args,timeout=60):
    r=subprocess.run([str(CLI),*args],capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stderr[:3000])
    return json.loads(r.stdout)


def lease():return json.loads((HERE/'lease.json').read_text())


def transport(scp=False):
    l=lease()
    return ['scp' if scp else 'ssh','-i',l['ssh_key'],'-o','BatchMode=yes','-o','IdentitiesOnly=yes',
            '-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',
            '-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(ROOT/'cache/runpod/known_hosts'),
            '-P' if scp else '-p',str(l['port'])]


def ssh(script,timeout=60):
    l=lease()
    r=subprocess.run(transport()+[f"root@{l['host']}",'bash -s'],input=script.encode(),capture_output=True,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-3000:]+r.stdout.decode(errors='replace')[-3000:])
    return r.stdout.decode(errors='replace')


def copy(source,destination,to_remote=False):
    l=lease();remote=f"root@{l['host']}:"
    args=[str(source),remote+str(destination)] if to_remote else [remote+str(source),str(destination)]
    r=subprocess.run(transport(True)+['-q','-X','nrequests=256','-X','buffer=131072',*args],capture_output=True,timeout=5400)
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-3000:])


def status():
    return json.loads(ssh(f'''python3 - <<'PY'
import json,time,shutil
from pathlib import Path
root=Path('{REMOTE}');out=root/'{PRODUCTION}'
r={{'observed_epoch':time.time(),'disk_free_gb':shutil.disk_usage('/workspace').free/1e9}}
for name in ('progress','manifest'):
 p=out/(name+'.json')
 if p.exists():r[name]=json.loads(p.read_text())
for name in ('setup','calibration','production'):
 p=root/'worker-control'/(name+'.exit')
 if p.exists():r[name+'_exit']=p.read_text().strip()
print(json.dumps(r))
PY
'''))


def guard():
    l=lease()
    while time.time()<l['stop_epoch']:
        if (HERE/'closeout.json').exists():return
        time.sleep(min(30,max(1,l['stop_epoch']-time.time())))
    state=cli('pod','get',l['pod_id'])
    if state.get('name')!=l['pod_name']:raise RuntimeError('Task ownership mismatch')
    result=cli('pod','stop',l['pod_id'])
    save(HERE/'deadline_stop.json',dict(at_epoch=time.time(),pod_id=l['pod_id'],result=result,
         volume_retained=True,reason='Four-hour task compute limit; preserve durable evidence'))


def monitor():
    while True:
        try:
            r=status();save(HERE/'latest-status.json',r)
            with (HERE/'monitoring.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            print(json.dumps(r),flush=True)
            if r.get('manifest',{}).get('status') in ('complete','failed'):return
        except Exception as error:print(json.dumps({'poll_error':repr(error),'epoch':time.time()}),flush=True)
        if time.time()>lease()['stop_epoch']:return
        time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['guard','status','monitor']);a=p.parse_args()
    if a.action=='guard':guard()
    elif a.action=='status':print(json.dumps(status(),indent=2))
    else:monitor()
