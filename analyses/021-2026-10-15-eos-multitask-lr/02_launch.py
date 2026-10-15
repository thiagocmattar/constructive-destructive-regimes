"""Upload a hashed source packet and start the detached, predecessor-gated queue."""
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


def main():
    assert not (HERE / 'launch.json').exists(), 'Queue already launched.'
    cohort = json.loads((HERE / 'cohort.json').read_text())
    assert json.loads((HERE / 'preflight.json').read_text())['status'] == 'passed'
    remote, python = cohort['remote_root'], cohort['remote_python']
    packet = json.loads((HERE / 'packet.json').read_text())
    source = ROOT / packet['path']
    assert file_hash(source) == packet['sha256']
    def call(code):
        return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', HOST,
            shlex.join([python, '-c', code])], text=True, timeout=40)
    code = f'import pathlib; p=pathlib.Path({remote!r}); p.mkdir(exist_ok=False)'
    call(code)
    subprocess.run(['scp', '-q', str(source), HOST + ':' + remote + '/sources.tar'], check=True)
    code = ('import pathlib,hashlib,tarfile; ' + f'r=pathlib.Path({remote!r}); p=r/"sources.tar"; '
        + f'assert p.stat().st_size=={packet["bytes"]}; '
        + 'digest=hashlib.file_digest(p.open("rb"),"sha256").hexdigest(); '
        + f'assert digest=={packet["sha256"]!r}; '
        + 'tarfile.open(p).extractall(r,filter="data"); print(digest)')
    remote_hash = call(code).strip()
    args = [python, remote + '/analyses/' + HERE.name + '/01_queue.py']
    code = ('import os,pathlib,subprocess,json; ' + f'r=pathlib.Path({remote!r}); '
        + f'env=dict(os.environ,PYTHONPATH={remote + ":" + cohort["remote_dependencies"]!r}); '
        + 'log=(r/"queue.log").open("x"); '
        + f'p=subprocess.Popen({args!r},cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); '
        + '(r/"process.json").write_text(json.dumps({"pid":p.pid,"command":p.args})); print(p.pid)')
    pid = int(call(code).strip())
    write_json(HERE / 'launch.json', dict(status='launched_waiting_for_predecessor', utc=stamp(),
        pid=pid, remote_root=remote, packet_sha256=packet['sha256'], remote_packet_sha256=remote_hash,
        expected_compute_hours=cohort['estimated_hours'], qualified_compute_hours=cohort['qualified_hours'],
        maximum_compute_hours=72, maximum_predecessor_wait_hours=72, rental_cost=0))
    print(json.dumps(json.loads((HERE / 'launch.json').read_text()), indent=2))


if __name__ == '__main__':
    main()
