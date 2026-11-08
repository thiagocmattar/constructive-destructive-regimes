"""Measured-runtime adjustment: remove early shell timeout, retain worker bound."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import shlex
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud',HERE/'48_cloud_conventional.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
NAMES=['trans-control','trans-b95','trans-b90','trans-b50','trans-clip05']


def extend(name):
    p=c.get(name);assert not p.get('runtime_margin_added')
    discovery=c.x.ssh(p,'''python3 - <<'PY'
import subprocess,json
rows=[line.split(None,2) for line in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()[1:]]
workers=[r for r in rows if r[2].startswith('/opt/h050/bin/python -m src.eos_tasks.conventional_continuation ')]
assert len(workers)==1,workers
w=workers[0];parent=next(r for r in rows if r[0]==w[1]);assert parent[2].startswith('timeout --signal=TERM '),parent
print(json.dumps(dict(worker_pid=int(w[0]),timeout_pid=int(parent[0]),worker_command=w[2])))
PY
''')
    ids=json.loads(discovery)
    # Scientific worker still enforces frozen maximum_worker_seconds=6500.
    p.update(deadline_epoch=p['deadline_epoch']+1200,runtime_margin_added=True,
        runtime_adjustment=dict(reason='Fresh Transformer timing ~16s/update; permit completion and verified retrieval within USD20',
            additional_lease_seconds=1200,scientific_worker_limit_unchanged=6500,**ids))
    c.record(p)
    program=f'''import os,signal,time,json,subprocess
from pathlib import Path
worker={ids['worker_pid']};parent={ids['timeout_pid']}
os.kill(parent,signal.SIGKILL)
while True:
 try:
  state=Path('/proc/'+str(worker)+'/status').read_text()
  if '\\nState:\\tZ' in state:break
 except FileNotFoundError:break
 time.sleep(5)
root=Path({c.REMOTE!r});out=root/{p['output']!r}
status=json.loads((out/'manifest.json').read_text())
code=0 if status['status']=='completed' else 1
Path('/workspace/worker.exit').write_text(str(code))
if code==0:
 with Path('/workspace/verification.log').open('w') as log:
  result=subprocess.run(['/opt/h050/bin/python',str(root/'analyses/065-2026-11-07-mlp-frozen-lr-double/47_verify_conventional.py'),str(out)],stdout=log,stderr=subprocess.STDOUT)
 Path('/workspace/verification.exit').write_text(str(result.returncode))
'''
    local=c.S/(name+'-supervisor.py');local.write_text(program,encoding='utf-8',newline='\n')
    c.x.copy(p,local,'/workspace/extended-supervisor.py')
    pid=c.x.ssh(p,'setsid python3 /workspace/extended-supervisor.py </dev/null >/workspace/extended-supervisor.log 2>&1 &\necho $!\n').strip()
    c.dump(c.ROOT/p['run']/(name+'_runtime_adjustment.json'),dict(**p['runtime_adjustment'],supervisor_pid=pid,utc=c.x.stamp()))
    return dict(name=name,worker_pid=ids['worker_pid'],supervisor_pid=pid,deadline=p['deadline_epoch'])


def main():
    bound=0.
    for f in list(c.S.glob('lease-*.json'))+list(c.S.glob('aborted-*.json')):
        p=json.loads(f.read_text());seconds=p.get('deleted_epoch',p['deadline_epoch'])-p['created_epoch']
        if p['name'] in NAMES and f.name.startswith('lease-') and not p.get('runtime_margin_added'):seconds+=1200
        bound+=seconds/3600*p['bounded_hourly_usd']
    assert bound<20,bound
    with ThreadPoolExecutor(max_workers=5) as pool:
        for result in pool.map(extend,NAMES):print(json.dumps(result),flush=True)
    print(json.dumps(dict(maximum_committed_usd=bound,task_ceiling_usd=20)))


if __name__=='__main__':main()
