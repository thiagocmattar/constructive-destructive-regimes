"""Isolated, user-authorized Spark qualification for unchanged ResNet continuations."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud037',HERE/'05_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
HOST='tcml-spark@spark-2277.local'
REMOTE='/home/tcml-spark/experiments/constructive-destructive-regimes/analysis037-sgd-helper-2026-11-02'
BASE='/home/tcml-spark/experiments/constructive-destructive-regimes/spark-calibration-2026-10-08/venv'
REL='runs/214-2026-11-02-sgd-batch-extension'


def ssh(script,timeout=60):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,'bash -s'],
        input=script.encode(),capture_output=True,timeout=timeout,creationflags=c.NO_WINDOW)
    if p.returncode:raise RuntimeError((p.stdout+p.stderr).decode(errors='replace')[-2500:])
    return p.stdout.decode()


def copy(local,remote,upload=True):
    pair=[str(local),HOST+':'+remote] if upload else [HOST+':'+remote,str(local)]
    p=subprocess.run(['scp','-q',*pair],capture_output=True,timeout=600,creationflags=c.NO_WINDOW)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-2000:])


def stage():
    packet=next(p for p in c.read(HERE/'transfer_packets.json')['packets'] if p['task']=='resnet')
    ssh(f'set -e\ntest ! -e {REMOTE}/calibration.pid\nmkdir -p {REMOTE}\n')
    copy(c.ROOT/packet['path'],REMOTE+'/inputs.tar.gz')
    script=f'''set -euo pipefail
trap 'echo $? > {REMOTE}/calibration.exit' EXIT
cd {REMOTE}
echo '{packet['sha256']}  inputs.tar.gz' | sha256sum -c -
tar -xf inputs.tar.gz
python3 -m venv --system-site-packages --without-pip venv
{BASE}/bin/python -m pip --python {REMOTE}/venv install --disable-pip-version-check numpy==2.5.0 scipy==1.18.1
export PYTHONPATH={BASE}/lib/python3.12/site-packages
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
venv/bin/python -m pip freeze > environment-lock.txt
venv/bin/python {REL}/00_profile.py --task resnet --full --output profile.json
venv/bin/python {REL}/02_worker.py --task resnet --seed 2 --smoke --output short-worker --deadline-epoch {min(c.lease()['scientific_deadline_epoch'],__import__('time').time()+900)}
'''
    path=c.SCRATCH/'spark-calibration.sh';path.write_text(script,encoding='utf-8',newline='\n')
    copy(path,REMOTE+'/calibration.sh')
    pid=ssh(f'set -e\nsetsid bash {REMOTE}/calibration.sh > {REMOTE}/calibration.log 2>&1 < /dev/null &\necho $! | tee {REMOTE}/calibration.pid\n').strip()
    receipt=dict(utc=c.stamp(),status='calibrating',host=HOST,remote=REMOTE,pid=pid,
        input_packet_sha256=packet['sha256'],unrelated_suspended_queue_preserved=True,cloud_cost_usd=0)
    c.save(HERE/'spark-qualification.json',receipt)
    print(json.dumps(receipt),flush=True)


def status():
    print(ssh(f'cat {REMOTE}/calibration.exit 2>/dev/null || true\ntail -c 1800 {REMOTE}/calibration.log\n'))


if __name__=='__main__':
    {'stage':stage,'status':status}[sys.argv[1]]()
