"""Wait for the frozen upload, verify/extract it once, and start both workers."""
from pathlib import Path
import importlib.util
import json
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('control', HERE / '04_control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def ssh(pod, lease, program, timeout=45):
    result = subprocess.run(control.transport(pod, lease) +
        [f"{pod['user']}@{pod['host']}", 'bash -s'], input=program.encode(),
        capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace') + result.stdout.decode(errors='replace'))
    return result.stdout.decode()


def main():
    lease = json.loads((HERE / 'lease.json').read_text())
    assert lease['owner'] == 'Analysis025-current-task'
    auth = json.loads((HERE / 'authorization.json').read_text())
    assert auth['approved'] and auth['maximum_total_usd'] <= 25
    packet = json.loads((HERE / 'input_packet.json').read_text())
    first = lease['pods'][0]
    control.save(HERE / 'staging-status.json', dict(status='waiting_for_upload', at_epoch=time.time()))
    while time.time() < lease['training_deadline_epoch']:
        size = int(ssh(first, lease, "stat -c %s /workspace/analysis025-inputs.tar 2>/dev/null || echo 0\n").strip())
        print(json.dumps(dict(at_epoch=time.time(), uploaded_bytes=size, expected_bytes=packet['bytes'])), flush=True)
        if size == packet['bytes']:
            break
        if size > packet['bytes']:
            raise ValueError('Input packet size exceeds the frozen archive')
        time.sleep(30)
    else:
        raise TimeoutError('Input staging missed the training deadline')
    program = f'''set -euo pipefail
printf '%s  %s\\n' {shlex.quote(packet['sha256'])} /workspace/analysis025-inputs.tar | sha256sum -c
tar -xf /workspace/analysis025-inputs.tar -C /workspace/constructive
'''
    print(ssh(first, lease, program, timeout=1800), flush=True)
    control.save(HERE / 'staging-status.json', dict(status='input_archive_verified_and_extracted', at_epoch=time.time(),
        bytes=packet['bytes'], sha256=packet['sha256']))
    for pod in lease['pods']:
        worker = pod['worker']
        root = '/workspace/constructive'
        worker_script = root + '/analyses/' + HERE.name + '/02_worker.py'
        program = f'''set -euo pipefail
cd {root}
test "$(cat worker-control/setup-{worker}.exit)" = 0
test ! -e worker-control/worker-{worker}.lock
test ! -e worker-control/worker-{worker}.json
setsid /opt/h021-env/bin/python -u {shlex.quote(worker_script)} --worker {worker} --deadline-epoch {lease['training_deadline_epoch']} > worker-control/worker-{worker}.log 2>&1 < /dev/null &
echo $!
'''
        pid = ssh(pod, lease, program).strip()
        print(json.dumps(dict(worker=worker, detached_pid=pid, at_epoch=time.time())), flush=True)
    control.save(HERE / 'staging-status.json', dict(status='workers_launched', at_epoch=time.time()))
    subprocess.run([sys.executable, str(HERE / '04_control.py'), 'monitor'], cwd=ROOT, check=True)
    if (HERE / 'closeout.json').exists():
        subprocess.run([sys.executable, str(HERE / '08_analyze.py')], cwd=ROOT, check=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        control.save(HERE / 'staging-error.json', dict(at_epoch=time.time(), error=repr(error)))
        if not (HERE / 'closeout.json').exists():
            lease = json.loads((HERE / 'lease.json').read_text())
            for pod in lease['pods']:
                try:
                    state = control.assert_owned(pod)
                    if state.get('runtimeStatus') not in {'stopped', 'terminated'}:
                        control.cli('pod', 'stop', pod['id'])
                except Exception as stop_error:
                    print(repr(stop_error), file=sys.stderr, flush=True)
        raise
