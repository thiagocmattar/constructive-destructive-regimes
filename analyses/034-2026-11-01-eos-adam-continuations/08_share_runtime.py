"""Replace stalled CDN setup with the identical verified runtime from a ready Pod."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('cloud034',HERE/'01_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
ARCHIVE='/tmp/analysis034-runtime.tar.gz'
LOCAL=ROOT/'cache/analysis034-runtime.tar.gz'


def pod_for(worker):
    pod=next(p for p in reversed(c.lease()['pods']) if p['worker']==worker and not p.get('deleted_epoch'))
    if not pod.get('host'):
        info=c.cli('pod','get',pod['id'])['ssh'];pod.update(host=info['ip'],port=info['port'])
    return pod


def produce():
    pod=pod_for(18)
    assert c.receipt_path(pod,'209').exists(),'Source calibration must be complete and verified'
    script=f'''#!/bin/bash
set -euo pipefail
tar -I 'gzip -1' -cf {ARCHIVE} -C /opt analysis034-venv
sha256sum {ARCHIVE} > /workspace/analysis034-runtime.sha256
'''
    local=HERE/'scratch/runtime-bundle.sh';local.write_text(script,encoding='utf-8',newline='\n')
    c.copy(pod,local,'/tmp/analysis034-runtime-bundle.sh',upload=True)
    pid=c.ssh(pod,'setsid bash /tmp/analysis034-runtime-bundle.sh > /workspace/analysis034-runtime-bundle.log 2>&1 < /dev/null &\necho $!\n').strip()
    c.save(HERE/'runtime_bundle_source.json',dict(pod_id=pod['id'],worker=18,pid=pid,epoch=time.time(),method='lossless copy of installed environment; no version change'))
    print('Runtime bundle building detached')


def download():
    pod=pod_for(18)
    digest=c.ssh(pod,'cat /workspace/analysis034-runtime.sha256\n').split()[0]
    if not LOCAL.exists() or c.sha(LOCAL)!=digest:c.copy(pod,ARCHIVE,LOCAL)
    assert c.sha(LOCAL)==digest
    c.save(HERE/'runtime_bundle.json',dict(path=LOCAL.relative_to(ROOT).as_posix(),sha256=digest,bytes=LOCAL.stat().st_size,source_pod_id=pod['id']))
    print(json.dumps(dict(bytes=LOCAL.stat().st_size,sha256=digest)))


def stage(worker):
    pod=pod_for(worker);ctrl=c.control('209')
    state=c.status(pod,'209')
    if state.get('setup_exit')=='0' or state.get('manifest'):return dict(worker=worker,status='already_ready')
    c.copy(pod,LOCAL,ARCHIVE,upload=True)
    metadata=json.loads((HERE/'runtime_bundle.json').read_text())
    original=json.loads((HERE/f'setup-209-worker{worker}-attempt{pod["attempt"]}.json').read_text())
    # Kill only the exact task setup shell and its descendants, without running its failure trap.
    # The worker is never touched; ready/started workers are checked again atomically below.
    script=f'''python3 - <<'PY'
from pathlib import Path
import os,signal,json
c=Path('{ctrl}')
if (c/'worker.sh').exists() or ((c/'setup.exit').exists() and (c/'setup.exit').read_text().strip()=='0'):
 print('already-ready');raise SystemExit(0)
pid=int('{original['pid']}')
p=Path('/proc')/str(pid)
if not p.exists():
 print('setup-process-gone');raise SystemExit(1)
assert (c/'setup.sh').as_posix().encode() in (p/'cmdline').read_bytes()
parents={{}}
for q in Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:
  fields=(q/'stat').read_text().split(') ',1)[1].split();parents[int(q.name)]=int(fields[1])
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
desc={{pid}}
while True:
 new={{q for q,parent in parents.items() if parent in desc}}-desc
 if not new:break
 desc.update(new)
for q in [pid]+sorted(desc-{{pid}},reverse=True):
 try:os.kill(q,signal.SIGKILL)
 except ProcessLookupError:pass
(c/'runtime-repair.json').write_text(json.dumps(dict(replaced_setup_pids=sorted(desc),reason='stalled CDN download; identical ready runtime')))
print('replace-runtime')
PY
'''
    result=c.ssh(pod,script,timeout=30).strip()
    if result!='replace-runtime':return dict(worker=worker,status=result)
    repair=f'''#!/bin/bash
set -euo pipefail
trap 'echo $? > {ctrl}/setup.exit' EXIT
printf '%s  %s\\n' '{metadata['sha256']}' {ARCHIVE} | sha256sum -c -
tar -xzf {ARCHIVE} -C /opt
/opt/analysis034-venv/bin/python -c 'import torch,numpy,scipy; assert torch.__version__=="2.11.0+cu130"; assert numpy.__version__=="2.5.0"; assert scipy.__version__=="1.18.1"; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name())'
'''
    local=HERE/f'scratch/runtime-repair-{worker}.sh';local.write_text(repair,encoding='utf-8',newline='\n')
    c.copy(pod,local,ctrl+'/runtime-repair.sh',upload=True)
    pid=c.ssh(pod,f'setsid bash {ctrl}/runtime-repair.sh > {ctrl}/runtime-repair.log 2>&1 < /dev/null &\necho $!\n').strip()
    record=dict(worker=worker,pod_id=pod['id'],epoch=time.time(),pid=pid,sha256=metadata['sha256'],status='repair_detached')
    c.save(HERE/f'runtime-repair-worker{worker}-attempt{pod["attempt"]}.json',record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['produce','download','stage']);p.add_argument('--workers',default='2,6,7,8,1');args=p.parse_args()
    if args.action=='produce':produce()
    elif args.action=='download':download()
    else:
        with ThreadPoolExecutor(max_workers=5) as pool:
            for result in pool.map(stage,[int(x) for x in args.workers.split(',')]):print(json.dumps(result),flush=True)
