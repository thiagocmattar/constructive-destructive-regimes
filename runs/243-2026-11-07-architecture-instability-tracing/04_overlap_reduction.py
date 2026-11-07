"""Infrastructure only: overlap Transformer training with completed ResNet capture reduction.

Suspend the sequential parent shell (not the scientific worker) so it cannot
launch a duplicate Transformer. Signal completion to the existing lease owner
only after both scientific manifests are complete. The paused shell is removed
with the task pod after verified retrieval, before its existing timeout expires.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--shell-pid',type=int,required=True);p.add_argument('--resnet-pid',type=int,required=True);a=p.parse_args()
run=a.root/'runs/243-2026-11-07-architecture-instability-tracing';out=run/'artifacts/attempt-01'
assert Path(f'/proc/{a.shell_pid}/cmdline').read_bytes().replace(b'\0',b' ').strip()==b'bash /workspace/worker.sh'
assert b'--case\0resnet' in Path(f'/proc/{a.resnet_pid}/cmdline').read_bytes()
assert json.loads((out/'resnet/progress.json').read_text())['state']==311
assert (out/'resnet/reduction_progress.json').exists() and not (out/'transformer').exists()
assert not Path('/workspace/worker.exit').exists()
os.kill(a.shell_pid,signal.SIGSTOP)
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
with Path('/workspace/transformer-parallel.log').open('xb') as log:
    worker=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30','5500','/opt/cdr243-venv/bin/python',str(run/'01_worker.py'),'--case','transformer'],cwd=a.root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(status='running',transformer_pid=worker.pid,suspended_parent_shell=a.shell_pid,resnet_pid=a.resnet_pid,started_epoch=time.time(),scientific_sources_unchanged=True)
path=Path('/workspace/overlap_orchestration.json');path.write_text(json.dumps(record,indent=2))
while True:
    time.sleep(10)
    assert time.time()-record['started_epoch']<6500
    states={case:json.loads((out/case/'manifest.json').read_text())['status'] if (out/case/'manifest.json').exists() else 'starting' for case in ['resnet','transformer']}
    if 'failed' in states.values() or (worker.poll() not in (None,0)):
        record.update(status='failed',states=states);code=1;break
    if all(s=='completed' for s in states.values()):
        assert worker.wait(timeout=30)==0
        record.update(status='completed',states=states);code=0;break
record['finished_epoch']=time.time();path.write_text(json.dumps(record,indent=2))
Path('/workspace/worker.exit.tmp').write_text(str(code)+'\n');os.replace('/workspace/worker.exit.tmp','/workspace/worker.exit')
