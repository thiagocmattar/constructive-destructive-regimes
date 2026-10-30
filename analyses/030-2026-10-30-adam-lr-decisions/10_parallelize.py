"""Add two qualified L40S workers without restarting the active continuation."""
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('launch_h024',HERE/'05_launch.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
control=launch.control
LOCK=threading.Lock()


def background(script, *args, label):
    log=(HERE/'scratch'/f'{label}.log').open('w')
    p=subprocess.Popen([sys.executable,str(HERE/script),*args],cwd=ROOT,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        creationflags=control.NO_WINDOW)
    control.save(HERE/'scratch'/f'{label}-process.json',dict(pid=p.pid,started_at=time.time()))
    return p.pid


def main():
    lease_path=HERE/'scratch/lease.json'
    lease=json.loads(lease_path.read_text())
    if '--adopt-staging' in sys.argv:
        adopt_staging(lease)
        return
    resume='--resume' in sys.argv
    assert 1<=len(lease['pods'])<=3 and lease['owner']=='Analysis030-current-task'
    if not resume:assert not (HERE/'scratch/parallel-launch-started.json').exists()
    primary=lease['pods'][0];control.assert_owned(primary)
    quote=next(g for g in launch.cli('gpu','list','--include-unavailable') if g['gpuId']=='NVIDIA L40S')
    assert quote['securePricePerHr']<=1.09
    placements=[x['dataCenterId'] for x in quote['dataCenterAvailability'] if x['stockStatus'].lower()!='none']
    assert placements
    if not resume:
        started=time.time()
        control.save(HERE/'scratch/parallel-launch-started.json',dict(epoch=started,quote=quote))
        control.save(HERE/'parallel_live_price.json',dict(epoch=started,quote=quote,source='live runpodctl3.0.1 gpu list'))
        control.save(HERE/'scratch/lease-before-parallel.json',lease)
        lease.update(hard_deadline_epoch=started+5*3600,training_deadline_epoch=started+4*3600,
                     maximum_hourly_price=3*1.09+(400*.10+90*.10)/730,
                     storage_hourly_price=(400*.10+90*.10)/730)
        control.save(lease_path,lease)
        control.save(HERE/'scratch/authorization.json',dict(approved=True,
            source='User explicitly authorized parallel execution and increased cost',
            agent_execution_cap=True,maximum_total_usd=25,maximum_hours=5,maximum_gpus=3))
        background('04_control.py','guard',label='parallel-guard')

    # Install only the queue override and guard; the frozen packet stays intact.
    def install_overrides(pod):
        for name in ['09_parallel_worker.py','08_remote_guard.py','parallel_assignment.json']:
            path=HERE/name;remote=f'/workspace/constructive/analyses/{HERE.name}/{name}'
            launch.upload(pod,lease,path,remote)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            launch.ssh(pod,lease,f"echo '{digest}  {remote}' | sha256sum -c\n")

    base=f'/workspace/constructive/analyses/{HERE.name}'
    if not resume:
        install_overrides(primary)
        # Identify and retire only the old queue parent. Its active child continues.
        transition=launch.ssh(primary,lease,"""python - <<'PY'
from pathlib import Path
import json,os,signal,shutil
root=Path('/workspace/constructive/worker-control')
state=json.loads((root/'worker-0.json').read_text())
assert state['status']=='running' and not state['completed']
assert '/187-' in state['output']
pid=int((root/'worker-0.lock').read_text())
assert b'/02_worker.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
children=[int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
assert len(children)==1
child=children[0];cmd=Path(f'/proc/{child}/cmdline').read_bytes()
assert b'baseline.lr_decisions' in cmd and state['output'].encode() in cmd
shutil.copyfile(root/'worker-0.json',root/'worker-0-before-parallel.json')
os.kill(pid,signal.SIGTERM)
assert Path(f'/proc/{child}').exists()
print(json.dumps({'retired_queue_pid':pid,'adopt_pid':child,'adopt_run':Path(state['output']).parts[-4]}))
PY
""")
        adopted=json.loads(transition)
        assert adopted['adopt_run'].startswith('187-')
        control.save(HERE/'parallel_handover.json',dict(**adopted,epoch=time.time(),scientific_child_preserved=True))
        command=f'setsid /opt/h024-env/bin/python -u {base}/09_parallel_worker.py --worker 0 --deadline {lease["training_deadline_epoch"]} --adopt-run {adopted["adopt_run"]} --adopt-pid {adopted["adopt_pid"]} > /workspace/constructive/worker-control/parallel-worker-0.log 2>&1 < /dev/null &\necho $!\n'
        control.save(HERE/'scratch/parallel-worker0-launch.json',dict(output=launch.ssh(primary,lease,command)))

    def arm(pod,replace=False):
        if replace:
            launch.ssh(pod,lease,"""python - <<'PY'
from pathlib import Path
import os,signal
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:cmd=(p/'cmdline').read_bytes().split(b'\\0')
 except OSError:continue
 if b'/workspace/constructive/worker-control/deadline_guard.py' in cmd:
  os.kill(int(p.name),signal.SIGTERM)
PY
""")
        script=f'setsid python -u {base}/08_remote_guard.py --pod {pod["id"]} --deadline {lease["hard_deadline_epoch"]} > /workspace/constructive/worker-control/remote-guard-parallel.log 2>&1 < /dev/null &\necho $!\n'
        launch.ssh(pod,lease,script)

    if not resume:arm(primary,True)
    pub=launch.ssh(primary,lease,"test -f /root/.ssh/analysis030-peer || ssh-keygen -q -t ed25519 -N '' -f /root/.ssh/analysis030-peer\ncat /root/.ssh/analysis030-peer.pub\n").strip()
    assert pub.startswith('ssh-ed25519 ') and '\n' not in pub
    if not (HERE/'scratch/parallel-monitor-process.json').exists():
        background('04_control.py','monitor',label='parallel-monitor')
    for worker in [1,2]:
        if any(p['worker']==worker for p in lease['pods']):continue
        # Persist each allocated ID before any readiness or transfer work.
        candidates=[dc for dc in ['US-MO-1','US-NC-1','US-TX-4','EU-NL-1'] if dc in placements]
        candidates += [dc for dc in placements if dc not in candidates]
        for dc in candidates:
            try:
                result=launch.cli('pod','create','--name',f'cdr-analysis030-worker{worker}',
                    '--template-id','runpod-torch-v280','--gpu-id','NVIDIA L40S','--cloud-type','SECURE',
                    '--gpu-count','1','--data-center-ids',dc,'--container-disk-in-gb','30',
                    '--volume-in-gb','100','--ports','22/tcp','--min-cuda-version','12.8')
                break
            except RuntimeError as exc:
                existing=[p for p in launch.cli('pod','list','--all') if p['name']==f'cdr-analysis030-worker{worker}']
                if existing:
                    assert len(existing)==1;result=existing[0];break
                control.save(HERE/'scratch'/f'placement-{worker}-{dc}.json',dict(error=str(exc),epoch=time.time()))
        else:raise RuntimeError('No additional L40S placement; current workers remain guarded')
        pod=dict(id=result['id'],name=f'cdr-analysis030-worker{worker}',worker=worker,
                 hourly_price=1.09,created_at_epoch=time.time(),volume_gb=100,data_center=dc)
        lease['pods'].append(pod);control.save(lease_path,lease)
    source=json.loads((HERE/'source_packet.json').read_text())
    inputs=json.loads((HERE/'input_packet.json').read_text())

    def prepare(pod):
        i=pod['worker'];state=dict(worker=i,pod=pod['id'],status='waiting_for_ssh')
        state_path=HERE/'scratch'/f'parallel-staging-{i}.json'
        def progress(status):state.update(status=status,epoch=time.time());control.save(state_path,state)
        try:
            progress('waiting_for_ssh');deadline=time.time()+600
            while time.time()<deadline:
                details=control.assert_owned(pod);route=details.get('ssh',{})
                assert details['costPerHr']<=1.09
                if route.get('ip') and route.get('port'):
                    pod.update(host=route['ip'],port=route['port'],user='root')
                    with LOCK:control.save(lease_path,lease)
                    try:
                        launch.ssh(pod,lease,'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader\n')
                        break
                    except (RuntimeError,subprocess.TimeoutExpired):pass
                time.sleep(10)
            else:raise TimeoutError('Additional worker readiness deadline')
            progress('staging_source')
            launch.ssh(pod,lease,'mkdir -p /workspace/constructive/worker-control\n')
            launch.upload(pod,lease,ROOT/source['path'],'/workspace/analysis030-source.tar.gz')
            launch.ssh(pod,lease,f"echo '{source['sha256']}  /workspace/analysis030-source.tar.gz' | sha256sum -c && tar -xzf /workspace/analysis030-source.tar.gz -C /workspace/constructive\n")
            install_overrides(pod);arm(pod)
            launch.ssh(pod,lease,f"printf '%s\\n' {shlex.quote(pub)} >> /root/.ssh/authorized_keys\nchmod 600 /root/.ssh/authorized_keys\n")
            setup=f'cd /workspace/constructive\nbash analyses/{HERE.name}/setup.sh {i} > worker-control/setup-{i}.log 2>&1\n'
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                future=pool.submit(launch.ssh,pod,lease,setup,1800)
                progress('copying_verified_inputs_from_worker0')
                peer=f"scp -q -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/root/.ssh/analysis030-peer-known-hosts -i /root/.ssh/analysis030-peer -P {pod['port']} /workspace/analysis030-inputs.tar root@{pod['host']}:/workspace/analysis030-inputs.tar\n"
                launch.ssh(primary,lease,peer,1800)
                progress('verifying_and_extracting_inputs')
                launch.ssh(pod,lease,f"echo '{inputs['sha256']}  /workspace/analysis030-inputs.tar' | sha256sum -c && tar -xf /workspace/analysis030-inputs.tar -C /workspace/constructive\n",1800)
                future.result()
            progress('launching_replay_qualification')
            script=f'set -e\ntest ! -e /workspace/constructive/worker-control/parallel-worker-{i}.lock\nsetsid /opt/h024-env/bin/python -u {base}/09_parallel_worker.py --worker {i} --deadline {lease["training_deadline_epoch"]} > /workspace/constructive/worker-control/worker-{i}.log 2>&1 < /dev/null &\necho $!\n'
            state['launch_output']=launch.ssh(pod,lease,script);progress('qualification_launched')
        except Exception as exc:
            state.update(error=repr(exc));progress('failed')
            # Preserve the pod volume; only this failed additional worker stops.
            try:control.assert_owned(pod);control.cli('pod','stop',pod['id'])
            except Exception:pass
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(prepare,pod) for pod in lease['pods'][1:]]
        for future in futures:future.result()
    control.save(HERE/'parallel_deployment.json',dict(status='three queues launched; new replay qualification pending',
        pods=[dict(id=p['id'],worker=p['worker']) for p in lease['pods']],
        hard_deadline_epoch=lease['hard_deadline_epoch'],training_deadline_epoch=lease['training_deadline_epoch'],
        completed_at_epoch=time.time(),maximum_total_task_usd=25,
        infrastructure_source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['09_parallel_worker.py','10_parallelize.py','parallel_assignment.json']}))


def adopt_staging(lease):
    """Finish existing peer copies after replacing a local console-spawning parent.

    Requires already installed/tested environments and armed remote guards. This
    mode allocates nothing and preserves the original transfer SSH processes.
    """
    assert lease['owner']=='Analysis030-current-task' and len(lease['pods'])==3
    packet=json.loads((HERE/'input_packet.json').read_text())
    expected=(ROOT/packet['path']).stat().st_size
    base=f'/workspace/constructive/analyses/{HERE.name}'
    def finish(pod):
        i=pod['worker'];control.assert_owned(pod)
        state_path=HERE/'scratch'/f'parallel-staging-{i}.json'
        state=json.loads(state_path.read_text())
        assert state['status']=='copying_verified_inputs_from_worker0'
        launch.ssh(pod,lease,f"set -e\ngrep -q '3 passed' /workspace/constructive/worker-control/setup-{i}.log\ntest -s /workspace/constructive/worker-control/remote-guard.jsonl\n")
        deadline=min(time.time()+900,lease['training_deadline_epoch'])
        while time.time()<deadline:
            size=int(launch.ssh(pod,lease,'stat -c %s /workspace/analysis030-inputs.tar\n').strip())
            if size==expected:break
            assert size<expected
            time.sleep(30)
        else:raise TimeoutError('Existing input transfer did not finish')
        state.update(status='verifying_and_extracting_inputs',epoch=time.time());control.save(state_path,state)
        launch.ssh(pod,lease,f"set -e\necho '{packet['sha256']}  /workspace/analysis030-inputs.tar' | sha256sum -c\ntar -xf /workspace/analysis030-inputs.tar -C /workspace/constructive\n",1800)
        script=f'set -e\ntest ! -e /workspace/constructive/worker-control/parallel-worker-{i}.lock\nsetsid /opt/h024-env/bin/python -u {base}/09_parallel_worker.py --worker {i} --deadline {lease["training_deadline_epoch"]} > /workspace/constructive/worker-control/worker-{i}.log 2>&1 < /dev/null &\necho $!\n'
        state.update(launch_output=launch.ssh(pod,lease,script),status='qualification_launched',epoch=time.time())
        control.save(state_path,state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(finish,p) for p in lease['pods'][1:]]:future.result()
    control.save(HERE/'parallel_deployment.json',dict(status='three queues launched; new replay qualification pending',
        pods=[dict(id=p['id'],worker=p['worker']) for p in lease['pods']],
        hard_deadline_epoch=lease['hard_deadline_epoch'],training_deadline_epoch=lease['training_deadline_epoch'],
        completed_at_epoch=time.time(),maximum_total_task_usd=25,
        local_supervisor_replaced_to_suppress_console_windows=True,
        infrastructure_source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['09_parallel_worker.py','10_parallelize.py','parallel_assignment.json']}))


if __name__=='__main__':main()
