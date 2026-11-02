"""Run213-only detached calibration, bounded billing, verified retrieval and cleanup."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCRATCH = HERE/'scratch'
RUN = ROOT/'runs/213-2026-11-02-eos-adam-from-init-calibration'
REMOTE = '/workspace/constructive'
CONTROL = '/workspace/cdr213-control'
PREFIX = 'cdr-analysis036-cal-'
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json


def cli(*args, timeout=75):
    result = subprocess.run(['runpodctl', *args], capture_output=True, text=True,
        timeout=timeout, creationflags=NO_WINDOW)
    if result.returncode:
        raise RuntimeError(result.stderr[:2500])
    return json.loads(result.stdout)


def lease():
    return json.loads((SCRATCH/'calibration-lease.json').read_text())


def transport(pod, scp=False):
    return ['scp' if scp else 'ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
        '-o', 'ConnectTimeout=12', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
        '-o', 'StrictHostKeyChecking=accept-new', '-o', 'UserKnownHostsFile='+str(SCRATCH/'known-hosts'),
        '-i', str(Path.home()/'.ssh/id_ed25519_runpod'), '-P' if scp else '-p', str(pod['port'])]


def ssh(pod, script, timeout=60):
    result = subprocess.run(transport(pod)+['root@'+pod['host'], 'bash -s'], input=script.encode(),
        capture_output=True, timeout=timeout, creationflags=NO_WINDOW)
    if result.returncode:
        raise RuntimeError((result.stderr+result.stdout).decode(errors='replace')[-2000:])
    return result.stdout.decode()


def copy(pod, local, remote, upload=True):
    endpoint = 'root@'+pod['host']+':'+remote
    source, target = (str(local), endpoint) if upload else (endpoint, str(local))
    result = subprocess.run(transport(pod, True)+['-q', source, target], capture_output=True,
        text=True, timeout=900, creationflags=NO_WINDOW)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])


def outpath(pod):
    return REMOTE+'/'+RUN.relative_to(ROOT).as_posix()+'/artifacts/attempt-01/worker'+str(pod['worker'])


def status(pod):
    script = """python3 - <<'PY'
from pathlib import Path
import json,time
c=Path('/workspace/cdr213-control'); o=Path(OUTPUT)
r={'epoch':time.time()}
for name in ['setup','worker']:
 p=c/(name+'.exit')
 if p.exists():r[name+'_exit']=p.read_text().strip()
 p=c/(name+'.log')
 if p.exists():r[name+'_tail']=p.read_text()[-1000:]
r['arms']=[]
if o.exists():
 for p in sorted(o.glob('lr-*/progress.json')):
  r['arms'].append({'arm':p.parent.name,'progress':json.loads(p.read_text())})
print(json.dumps(r))
PY
""".replace('OUTPUT', repr(outpath(pod)))
    return json.loads(ssh(pod, script))


def packet(task):
    manifest = json.loads((RUN/'input_manifest.json').read_text())
    paths = {item['path'] for item in manifest['files'] if task in item['tasks']}
    paths.add((RUN/'input_manifest.json').relative_to(ROOT).as_posix())
    target = ROOT/'cache'/('analysis036-'+task+'-inputs.tar.gz')
    with tarfile.open(target, 'w:gz', compresslevel=1) as archive:
        for relative in sorted(paths):
            archive.add(ROOT/relative, arcname=relative)
    return dict(path=target.relative_to(ROOT).as_posix(), bytes=target.stat().st_size,
        sha256=file_hash(target))


def setup_and_launch(pod, archive):
    until = time.time()+480
    while time.time() < until:
        try:
            state = cli('pod', 'get', pod['id'])
            info = state.get('ssh') or cli('ssh', 'info', pod['id'])
            if info.get('ip') and info.get('port'):
                pod.update(host=info['ip'], port=info['port'])
                ssh(pod, 'echo connection-ready\n', timeout=25)
                break
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        time.sleep(8)
    else:
        raise RuntimeError('SSH startup exceeded eight minutes.')
    ssh(pod, 'mkdir -p '+CONTROL+'\n')
    remote_hash = ssh(pod, 'if test -f /workspace/cdr213-inputs.tar.gz; then sha256sum /workspace/cdr213-inputs.tar.gz; else echo missing; fi\n').split()[0]
    if remote_hash != archive['sha256']:
        copy(pod, ROOT/archive['path'], '/workspace/cdr213-inputs.tar.gz')
    relative = RUN.relative_to(ROOT).as_posix()
    commands = []
    for rate in pod['rates']:
        output = outpath(pod)+'/lr-'+format(rate, 'g')
        commands.append('timeout --signal=TERM 3600 python3 '+relative+'/01_train.py --task '+pod['task']+
            ' --lr '+str(rate)+' --output '+output+' || code=$?')
    setup = f'''set -euo pipefail
trap 'echo $? > {CONTROL}/setup.exit' EXIT
echo '{archive['sha256']}  /workspace/cdr213-inputs.tar.gz' | sha256sum -c -
mkdir -p {REMOTE}
tar -xzf /workspace/cdr213-inputs.tar.gz -C {REMOTE}
python3 -m venv --system-site-packages --without-pip /opt/cdr213-venv
python3 -m pip --python /opt/cdr213-venv install --no-cache-dir --disable-pip-version-check numpy==2.5.0 scipy==1.18.1
/opt/cdr213-venv/bin/python -c "import torch,numpy,scipy; assert torch.__version__.split('+')[0]=='2.11.0'; assert torch.cuda.is_available(); print(torch.__version__,numpy.__version__,scipy.__version__,torch.cuda.get_device_name())"
'''
    worker = f'''set -uo pipefail
cd {REMOTE}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PATH=/opt/cdr213-venv/bin:$PATH
code=0
{chr(10).join(commands)}
echo "$code" > {CONTROL}/worker.exit
'''
    driver = f'''set -uo pipefail
bash {CONTROL}/setup.sh > {CONTROL}/setup.log 2>&1
if test "$(cat {CONTROL}/setup.exit)" = 0; then
 bash {CONTROL}/worker.sh > {CONTROL}/worker.log 2>&1
else
 echo 125 > {CONTROL}/worker.exit
fi
'''
    for name, body in [('setup', setup), ('worker', worker), ('driver', driver)]:
        local = SCRATCH/('worker'+str(pod['worker'])+'-'+name+'.sh')
        local.write_text(body, encoding='utf-8', newline='\n')
        copy(pod, local, CONTROL+'/'+name+'.sh')
    ssh(pod, f'setsid bash {CONTROL}/driver.sh > {CONTROL}/driver.log 2>&1 < /dev/null &\necho $!\n')
    return pod


def retrieve_delete(pod):
    output = outpath(pod)
    script = f'''set -e
mkdir -p {output}/control
cp -a {CONTROL}/. {output}/control/
python3 - <<'PY'
from pathlib import Path
import hashlib,json
p=Path({output!r}); rows=[]
for f in sorted(p.rglob('*')):
 if f.is_file() and f.name!='inventory.json':
  with f.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
  rows.append(dict(path=f.relative_to(p).as_posix(),bytes=f.stat().st_size,sha256=h))
(p/'inventory.json').write_text(json.dumps(dict(files=rows),indent=2)+'\\n')
PY
tar -I 'gzip -1' -cf /workspace/cdr213-output.tar.gz -C {output} .
sha256sum /workspace/cdr213-output.tar.gz
'''
    digest = ssh(pod, script, timeout=600).split()[0]
    archive = ROOT/'cache'/('analysis036-worker'+str(pod['worker'])+'-outputs.tar.gz')
    copy(pod, archive, '/workspace/cdr213-output.tar.gz', upload=False)
    assert file_hash(archive) == digest
    destination = RUN/'artifacts/attempt-01'/('worker'+str(pod['worker']))
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination, filter='data')
    inventory = json.loads((destination/'inventory.json').read_text())
    for item in inventory['files']:
        file = (destination/item['path']).resolve()
        assert file.is_relative_to(destination.resolve())
        assert file.stat().st_size == item['bytes'] and file_hash(file) == item['sha256']
    receipt = dict(worker=pod['worker'], pod_id=pod['id'], verified=True, utc=stamp(),
        archive_sha256=digest, files=len(inventory['files']), bytes=sum(r['bytes'] for r in inventory['files']),
        artifact_root=destination.relative_to(ROOT).as_posix())
    write_json(HERE/('calibration-retrieval-worker'+str(pod['worker'])+'.json'), receipt)
    actual = cli('pod', 'get', pod['id'])
    assert actual['name'] == pod['name'] and actual['name'].startswith(PREFIX)
    cli('pod', 'delete', pod['id'])
    pod['deleted_epoch'] = time.time()
    try:
        cli('pod', 'get', pod['id'])
        raise AssertionError('Deleted pod still exists.')
    except RuntimeError as error:
        assert 'not_found' in str(error)
    return receipt


def spend(state):
    return sum((p.get('deleted_epoch', time.time())-p['created_epoch'])/3600*
        (p['price_per_gpu_hour']+80*.10/720) for p in state['pods'])


def guard():
    while not (SCRATCH/'calibration-closed.json').exists():
        state = lease()
        total = spend(state)
        stop = total >= 14 or time.time() >= state['deadline_epoch']-900
        if stop:
            write_json(SCRATCH/'budget-stop.json', dict(utc=stamp(), estimated_usd=total))
            for pod in state['pods']:
                if not pod.get('deleted_epoch') and pod.get('host'):
                    try:
                        ssh(pod, 'touch '+REMOTE+'/'+RUN.relative_to(ROOT).as_posix()+'/STOP\n', timeout=20)
                    except Exception:
                        pass
        if total >= 18.5 or time.time() >= state['deadline_epoch']:
            for pod in state['pods']:
                if not pod.get('deleted_epoch'):
                    actual = cli('pod', 'get', pod['id'])
                    assert actual['name'] == pod['name'] and actual['name'].startswith(PREFIX)
                    cli('pod', 'stop', pod['id'])
            write_json(SCRATCH/'guard-stopped.json', dict(utc=stamp(), estimated_usd=total,
                note='Compute stopped; task volumes retained for verified recovery.'))
            return
        time.sleep(20)


def main(resume=False):
    statefile = SCRATCH/'calibration-lease.json'
    if resume:
        return resume_lease()
    assert not statefile.exists(), 'Existing lease must be resumed, not overwritten.'
    cfg = json.loads((RUN/'config.json').read_text())
    packets = {task: packet(task) for task in cfg['tasks']}
    write_json(HERE/'calibration-packets.json', packets)
    catalog = cli('gpu', 'list', '--include-unavailable')
    existing = cli('pod', 'list', '--all')
    assert not any(p['name'].startswith(PREFIX) for p in existing)
    image = json.loads((SCRATCH/'image-identity.json').read_text())['image']
    public = (Path.home()/'.ssh/id_ed25519_runpod.pub').read_text().strip()
    bootstrap = 'set -e; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server; mkdir -p /run/sshd /root/.ssh; printf "%s\\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; sleep infinity'
    state = dict(created_utc=stamp(), deadline_epoch=time.time()+5400, ceiling_usd=20,
        image=image, run_id='213', configuration_sha256=file_hash(RUN/'config.json'), pods=[])
    write_json(statefile, state)
    # The guard is a separate hidden process so a controller failure cannot disable it.
    log = (SCRATCH/'guard.log').open('w')
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'guard'], stdout=log,
        stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
    log.close()
    worker = 0
    for task in ['transformer', 'resnet', 'mlp']:
        made = []
        for gpu_name in ['RTX PRO 5000', 'RTX 5090', 'RTX 4090', 'RTX PRO 4000', 'L4']:
            gpu = next((r for r in catalog if r['displayName'] == gpu_name), None)
            if gpu is None or not gpu['available'] or gpu.get('securePricePerHr') is None or gpu['securePricePerHr'] > 1.19:
                continue
            for index in range(3):
                name = PREFIX+task+'-'+str(index)
                try:
                    actual = cli('pod', 'create', '--name', name, '--image', image, '--gpu-id', gpu['gpuId'],
                        '--gpu-count', '1', '--cloud-type', 'SECURE', '--min-cuda-version', '13.0',
                        '--container-disk-in-gb', '60', '--volume-in-gb', '20', '--ports', '22/tcp', '--ssh',
                        '--env', json.dumps(dict(PUBLIC_KEY=public)), '--docker-args', 'bash -lc '+shlex.quote(bootstrap))
                except RuntimeError as error:
                    write_json(SCRATCH/('capacity-'+task+'-'+str(index)+'-'+gpu_name.replace(' ', '')+'.json'), dict(error=str(error)))
                    break
                pod = dict(id=actual['id'], name=name, worker=worker, task=task, gpu=gpu_name,
                    price_per_gpu_hour=gpu['securePricePerHr'], created_epoch=time.time())
                write_json(SCRATCH/('create-worker'+str(worker)+'.json'), actual)
                made.append(pod); state['pods'].append(pod); worker += 1
                write_json(statefile, state)
                print(json.dumps(dict(phase='created', task=task, worker=pod['worker'], gpu=gpu_name)), flush=True)
            if made:
                break
        if not made:
            write_json(SCRATCH/('unavailable-'+task+'.json'), dict(utc=stamp(), task=task))
            continue
        for pod in made:
            pod['rates'] = []
        for i, rate in enumerate(cfg['learning_rates_by_task'][task]):
            made[i % len(made)]['rates'].append(rate)
        write_json(statefile, state)
    if not state['pods']:
        write_json(SCRATCH/'calibration-closed.json', dict(status='no_capacity', utc=stamp(), spend_usd=0))
        return
    with ThreadPoolExecutor(max_workers=9) as pool:
        futures = {pool.submit(setup_and_launch, pod, packets[pod['task']]): pod for pod in state['pods']}
        for future in as_completed(futures):
            pod = futures[future]
            try:
                future.result(); pod['launched_epoch'] = time.time()
            except Exception as error:
                pod['setup_error'] = repr(error)
            write_json(statefile, state)
            print(json.dumps(dict(phase='setup', worker=pod['worker'], error=pod.get('setup_error'))), flush=True)
    while any(not pod.get('deleted_epoch') for pod in state['pods']):
        for pod in state['pods']:
            if pod.get('deleted_epoch'):
                continue
            try:
                current = status(pod)
                write_json(SCRATCH/('status-worker'+str(pod['worker'])+'.json'), current)
                if current.get('worker_exit') is not None:
                    receipt = retrieve_delete(pod)
                    write_json(statefile, state)
                    print(json.dumps(dict(phase='retrieved_deleted', worker=pod['worker'], files=receipt['files'], bytes=receipt['bytes'])), flush=True)
            except Exception as error:
                pod['last_monitor_error'] = repr(error)
                write_json(statefile, state)
        print(json.dumps(dict(phase='monitor', estimated_usd=spend(state),
            active=sum(not p.get('deleted_epoch') for p in state['pods']))), flush=True)
        if any(not p.get('deleted_epoch') for p in state['pods']):
            time.sleep(30)
    close = dict(status='retrieved_and_deleted', utc=stamp(), compute_storage_estimate_usd=spend(state),
        pods=len(state['pods']), retained_cloud_storage_gb=0, billing='provisional')
    write_json(HERE/'calibration-closeout.json', close)
    write_json(SCRATCH/'calibration-closed.json', close)


def resume_lease(setup=True):
    state = lease()
    assert not (SCRATCH/'calibration-closed.json').exists()
    packets = json.loads((HERE/'calibration-packets.json').read_text())
    if setup:
        log = (SCRATCH/'guard-resumed.log').open('w')
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'guard'], stdout=log,
            stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        log.close()
    with ThreadPoolExecutor(max_workers=9) as pool:
        futures = {pool.submit(setup_and_launch, pod, packets[pod['task']]): pod
            for pod in state['pods'] if setup and not pod.get('deleted_epoch')}
        for future in as_completed(futures):
            pod = futures[future]
            try:
                future.result(); pod['launched_epoch'] = time.time(); pod.pop('setup_error', None)
            except Exception as error:
                pod['setup_error'] = repr(error)
            write_json(SCRATCH/'calibration-lease.json', state)
            print(json.dumps(dict(phase='setup_resumed', worker=pod['worker'], error=pod.get('setup_error'))), flush=True)
    while any(not pod.get('deleted_epoch') for pod in state['pods']):
        for pod in state['pods']:
            if pod.get('deleted_epoch'):
                continue
            try:
                current = status(pod)
                write_json(SCRATCH/('status-worker'+str(pod['worker'])+'.json'), current)
                if current.get('worker_exit') is not None:
                    receipt = retrieve_delete(pod)
                    write_json(SCRATCH/'calibration-lease.json', state)
                    print(json.dumps(dict(phase='retrieved_deleted', worker=pod['worker'], files=receipt['files'], bytes=receipt['bytes'])), flush=True)
            except Exception as error:
                pod['last_monitor_error'] = repr(error)
                write_json(SCRATCH/'calibration-lease.json', state)
        print(json.dumps(dict(phase='monitor', estimated_usd=spend(state),
            active=sum(not p.get('deleted_epoch') for p in state['pods']))), flush=True)
        if any(not p.get('deleted_epoch') for p in state['pods']):
            time.sleep(30)
    close = dict(status='retrieved_and_deleted', utc=stamp(), compute_storage_estimate_usd=spend(state),
        pods=len(state['pods']), retained_cloud_storage_gb=0, billing='provisional')
    write_json(HERE/'calibration-closeout.json', close)
    write_json(SCRATCH/'calibration-closed.json', close)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'guard':
        guard()
    elif len(sys.argv) > 1 and sys.argv[1] == 'resume':
        main(resume=True)
    elif len(sys.argv) > 1 and sys.argv[1] == 'monitor':
        resume_lease(setup=False)
    else:
        main()
