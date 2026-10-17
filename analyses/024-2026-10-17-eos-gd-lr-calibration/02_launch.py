"""Verify/upload the frozen packet and detach the bounded owned-Spark queue."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json

HOST = 'tcml-spark@spark-2277.local'
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def remote(python, code):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST,
        shlex.join([python, '-c', code])], text=True, timeout=60, creationflags=FLAGS)


def main():
    assert not (HERE / 'launch.json').exists(), 'Already launched.'
    c = json.loads((HERE / 'cohort.json').read_text())
    p = json.loads((HERE / 'packet.json').read_text())
    packet = ROOT / p['path']
    assert file_hash(packet) == p['sha256']
    base, python = c['remote_root'], c['remote_python']
    remote(python, f'import pathlib,shutil; p=pathlib.Path({base!r}); assert shutil.disk_usage(p.parent).free>{c["additional_storage_allowance_bytes"]}; p.mkdir(exist_ok=False)')
    subprocess.run(['scp', '-q', str(packet), HOST + ':' + base + '/sources.tar'], check=True, creationflags=FLAGS)
    remote(python, f'import pathlib,hashlib,tarfile; r=pathlib.Path({base!r}); p=r/"sources.tar"; '
        f'assert p.stat().st_size=={p["bytes"]}; assert hashlib.file_digest(p.open("rb"),"sha256").hexdigest()=={p["sha256"]!r}; '
        'tarfile.open(p).extractall(r,filter="data")')
    command = [python, base + '/analyses/' + HERE.name + '/01_queue.py']
    remote(python, f'import os,subprocess; e=dict(os.environ,PYTHONPATH={base + ":" + c["remote_dependencies"]!r}); '
        f'subprocess.run({command + ["--verify-only"]!r},env=e,cwd={base!r},check=True)')
    receipt = json.loads(remote(python, f'import pathlib; print(pathlib.Path({base + "/analyses/" + HERE.name + "/remote_preflight.json"!r}).read_text())'))
    assert receipt['status'] == 'passed'
    write_json(HERE / 'remote_preflight.json', receipt)
    pid = int(remote(python, f'import pathlib,subprocess,os; r=pathlib.Path({base!r}); '
        f'e=dict(os.environ,PYTHONPATH={base + ":" + c["remote_dependencies"]!r}); '
        f'p=subprocess.Popen({command!r},cwd=r,env=e,stdin=subprocess.DEVNULL,stdout=(r/"queue.log").open("x"),'
        'stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)').strip())
    write_json(HERE / 'launch.json', dict(status='launched', utc=stamp(), pid=pid, remote_root=base,
        expected_hours=c['estimated_hours'], qualified_hours=c['qualified_hours'], maximum_hours=120,
        packet_sha256=p['sha256'], remote_packet_sha256=p['sha256'], rental_cost=0, resume_predecessor=False))
    with (HERE / 'controller.log').open('x') as log:
        monitor = subprocess.Popen([sys.executable, str(HERE / '04_monitor.py')], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=FLAGS)
    write_json(HERE / 'controller_launch.json', dict(status='running', pid=monitor.pid, utc=stamp(),
        poll_seconds=1800, window='hidden', remote_training_detached=True))
    print((HERE / 'launch.json').read_text())


if __name__ == '__main__': main()
