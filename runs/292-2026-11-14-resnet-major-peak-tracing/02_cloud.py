"""Task-owned bounded Runpod replay, authenticated transfer, and cleanup."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT).as_posix()
REMOTE = '/workspace/cdr292'
OUT = REMOTE + '/' + REL + '/artifacts/attempt-01'
S = HERE / 'scratch'
S.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('transport', ROOT / 'runs/215-2026-11-02-controlled-full-batch-adam/03_cloud.py')
x = importlib.util.module_from_spec(spec); spec.loader.exec_module(x); x.SCRATCH = S


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


def build():
    assert not (HERE / 'cloud_lease.json').exists()
    old = ROOT / 'runs/273-2026-11-07-resnet-distributed-displacement'
    files = {}
    def add(p, target=None, pinned=None):
        name = target or p.relative_to(ROOT).as_posix()
        digest = x.file_hash(p)
        if pinned: assert digest == pinned['sha256'] and p.stat().st_size == pinned['bytes'], name
        files[name] = dict(path=name, bytes=p.stat().st_size, sha256=digest, local_path=p.relative_to(ROOT).as_posix())
    for row in json.loads((old / 'input_manifest.json').read_text())['files']:
        if row['path'].startswith(('cache/', 'src/', 'runs/223-')):
            add(ROOT / row['path'], pinned=row)
    for name in ['01_worker.py', 'config.json', 'input_manifest.json']:
        add(ROOT / 'runs/223-2026-11-04-multispike-interventions' / name)
    add(ROOT / 'runs/243-2026-11-07-architecture-instability-tracing/diagnostics.py')
    worker = old / 'artifacts/resnet-a/resnet-distributed'
    for local, target, inventory_dir in [('shared/fork.pt', 'fork.pt', 'shared'), ('control/boundaries.jsonl', 'boundaries.jsonl', 'control'), ('control/audit_296.pt', 'audit_296.pt', 'control')]:
        inventory = json.loads((worker / inventory_dir / 'transfer_inventory.json').read_text())['files']
        pinned = next(v for v in inventory if v['path'] == Path(local).name)
        add(worker / local, REL + '/inputs/' + target, pinned)
    for name in ['01_worker.py', 'config.json']: add(HERE / name)
    dump(HERE / 'input_manifest.json', dict(files=list(files.values())))
    archive = S / 'inputs.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for row in files.values(): tar.add(ROOT / row['local_path'], arcname=row['path'])
        tar.add(HERE / 'input_manifest.json', arcname=REL + '/input_manifest.json')
    dump(HERE / 'packet.json', dict(files=len(files), bytes=archive.stat().st_size, sha256=x.file_hash(archive)))
    print(json.dumps(dict(prepared=True, files=len(files), archive_bytes=archive.stat().st_size)), flush=True)


def collect(pod):
    receipt = json.loads(x.ssh(pod, f'''python3 - <<'PY'
import hashlib,json,tarfile
from pathlib import Path
root=Path({OUT!r}); files=[]
for p in sorted(root.rglob('*')):
 if p.is_file():
  keep=('raw' not in p.parts or p.name in ['fixed_inputs.npy','fixed_targets.npy','raw_gradient.npy'])
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  files.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=h,download=keep))
(root/'inventory.json').write_text(json.dumps(dict(files=files)))
with tarfile.open('/workspace/results292.tar.gz','w:gz',compresslevel=1) as t:
 t.add(root/'inventory.json',arcname='inventory.json')
 for f in files:
  if f['download']:t.add(root/f['path'],arcname=f['path'])
p=Path('/workspace/results292.tar.gz')
with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(bytes=p.stat().st_size,sha256=h,all_raw_bytes=sum(f['bytes'] for f in files))))
PY
''', timeout=600))
    target = HERE / 'artifacts/attempt-01'; target.mkdir(parents=True, exist_ok=True)
    archive = HERE / 'artifacts/results.tar.gz'
    if (HERE / 'http_transfer.json').exists():
        import requests
        http = json.loads((HERE / 'http_transfer.json').read_text())
        token = json.loads((S / 'http-config.json').read_text())['token']
        with requests.get(http['url'] + '/result', headers={'Authorization': 'Bearer ' + token}, stream=True, timeout=(20,180)) as response:
            response.raise_for_status()
            with archive.open('wb') as stream:
                for block in response.iter_content(1024*1024): stream.write(block)
    else:
        x.copy(pod, archive, '/workspace/results292.tar.gz', False)
    assert archive.stat().st_size == receipt['bytes'] and x.file_hash(archive) == receipt['sha256']
    with tarfile.open(archive) as tar: tar.extractall(target, filter='data')
    files = json.loads((target / 'inventory.json').read_text())['files']
    for f in files:
        if f['download']:
            p = target / f['path']; assert p.stat().st_size == f['bytes'] and x.file_hash(p) == f['sha256']
    receipt.update(verified=True, downloaded_files=sum(f['download'] for f in files), retained_bytes=sum(f['bytes'] for f in files if f['download']))
    dump(HERE / 'retrieval.json', receipt)
    pod['verified_retrieval'] = True


def launch(resume=False):
    assert resume or not (HERE / 'cloud_lease.json').exists()
    packet = S / 'inputs.tar.gz'; pin = json.loads((HERE / 'packet.json').read_text())
    assert x.file_hash(packet) == pin['sha256']
    if resume:
        pod = json.loads((HERE / 'cloud_lease.json').read_text())
        assert not pod.get('deleted_epoch') and time.time() < pod['deadline_epoch']-1200
        pod.pop('needs_retrieval_and_cleanup', None)
        created = pod['created_epoch']
    gpu = next(g for g in x.cli('gpu', 'list', '--include-unavailable') if g['displayName'] == 'RTX 4090')
    assert gpu['securePricePerHr'] <= .89
    dump(HERE / 'live_gpu_quote.json', dict(utc=x.stamp(), gpu=gpu, maximum_total_usd=2, maximum_hours=2))
    image = 'pytorch/pytorch@sha256:bfbb4a2b4fdba0fefdb428ea737e626d61bb3daf74a16e1ff935bdb03aa7c3f0'
    public = (Path.home() / '.ssh/id_ed25519_runpod.pub').read_text().strip()
    bootstrap = 'set -e; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server; mkdir -p /run/sshd /root/.ssh; printf "%s\\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; sleep infinity'
    if not resume:
        created = time.time()
        actual = x.create_wait('pod', 'create', '--name', 'cdr292-resnet-major-peak', '--image', image, '--gpu-id', gpu['gpuId'], '--gpu-count', '1', '--cloud-type', 'SECURE', '--min-cuda-version', '13.0', '--container-disk-in-gb', '30', '--volume-in-gb', '40', '--ports', '22/tcp', '--ssh', '--env', json.dumps({'PUBLIC_KEY': public}), '--docker-args', 'bash -lc ' + shlex.quote(bootstrap), '--wait', '--wait-timeout', '20s')
        pod = dict(id=actual['id'], created_epoch=created, price_per_hour=actual.get('costPerHr', gpu['securePricePerHr']), maximum_total_usd=2, deadline_epoch=created+7200, image=image, container_gb=30, volume_gb=40)
    dump(HERE / 'cloud_lease.json', pod)
    print(json.dumps(dict(event='created', **pod)), flush=True)
    try:
        readiness_deadline = min(time.time()+600,pod['deadline_epoch']-1200)
        while time.time() < readiness_deadline:
            a = x.cli('pod', 'get', pod['id']); info = a.get('ssh') or x.cli('ssh', 'info', pod['id'])
            if info.get('ip') and info.get('port'):
                pod.update(host=info['ip'], port=info['port']); dump(HERE / 'cloud_lease.json', pod)
                try: x.ssh(pod, 'echo ready\n', timeout=20); break
                except Exception: pass
            time.sleep(10)
        else: raise RuntimeError('SSH startup exceeded ten minutes')
        if (HERE / 'http_transfer.json').exists():
            receipt = json.loads((HERE / 'http_transfer.json').read_text())
            assert receipt['sha256'] == pin['sha256']
            x.ssh(pod, 'mv /workspace/inputs292-http.tar.gz /workspace/inputs292.tar.gz\n')
        else:
            x.copy(pod, packet, '/workspace/inputs292.tar.gz')
        script = f'''set -e
mkdir -p {REMOTE}
echo '{pin['sha256']}  /workspace/inputs292.tar.gz' | sha256sum -c -
tar -xzf /workspace/inputs292.tar.gz -C {REMOTE}
python3 -m venv --system-site-packages --without-pip /opt/cdr292
python3 -m pip --python /opt/cdr292 install --no-cache-dir numpy==2.5.0 scipy==1.18.1 threadpoolctl==3.6.0
cd {REMOTE}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
/opt/cdr292/bin/python {REL}/01_worker.py
'''
        shell = S / 'worker.sh'; shell.write_text(script, newline='\n'); x.copy(pod, shell, '/workspace/worker292.sh')
        x.ssh(pod, 'setsid bash -c "timeout --signal=TERM --kill-after=30 5400 bash /workspace/worker292.sh > /workspace/worker292.log 2>&1; echo \\$? > /workspace/worker292.exit" </dev/null >/workspace/driver292.log 2>&1 &\necho $!\n')
        while time.time() < pod['deadline_epoch']-600:
            progress = x.ssh(pod, f"if test -f /workspace/worker292.exit; then cat /workspace/worker292.exit; else echo WAIT; fi\nif test -f {OUT}/progress.json; then cat {OUT}/progress.json; else tail -c 600 /workspace/worker292.log; fi\n")
            dump(HERE / 'scratch/progress.json', dict(utc=x.stamp(), elapsed=time.time()-created, output=progress))
            print(json.dumps(dict(event='progress', output=progress)), flush=True)
            if progress.splitlines()[0] != 'WAIT': break
            time.sleep(30)
        else: raise RuntimeError('Retrieval reserve reached')
        collect(pod)
        manifest = json.loads((HERE / 'artifacts/attempt-01/manifest.json').read_text())
        assert manifest['status'] == 'completed', manifest
    except BaseException as error:
        pod['error'] = repr(error); dump(HERE / 'cloud_lease.json', pod)
        raise
    finally:
        if pod.get('host'):
            try: (S / 'final-worker.log').write_text(x.ssh(pod, 'tail -c 20000 /workspace/worker292.log\n'), encoding='utf-8')
            except Exception: pass
        if pod.get('verified_retrieval') or not pod.get('host'):
            x.cli('pod', 'delete', pod['id'])
            pod.update(deleted_epoch=time.time(), estimated_compute_usd=(time.time()-created)/3600*float(pod['price_per_hour']))
            assert not [p for p in x.cli('pod', 'list', '--all') if p['id'] == pod['id']]
            pod['deletion_verified'] = True
        else:
            # Preserve recoverable outputs if transfer fails; root agent handles
            # retrieval and deletion within the recorded lease deadline.
            pod['needs_retrieval_and_cleanup'] = True
        dump(HERE / 'cloud_lease.json', pod)
        print(json.dumps(dict(event='closeout', **pod)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'launch', 'resume'])
    action = parser.parse_args().action
    build() if action == 'prepare' else launch(resume=action=='resume')
