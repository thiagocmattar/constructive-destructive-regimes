"""Read-only Runpod preflight and per-case transfer packets; never allocates."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'runs/214-2026-11-02-sgd-batch-extension'
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,stamp,write_json


def cli(*args):
    return json.loads(subprocess.check_output(['runpodctl',*args],text=True))


def main():
    catalog=cli('gpu','list','--include-unavailable')
    templates=cli('template','list','--type','official')
    template=next(t for t in templates if t['id']=='runpod-torch-v280')
    pods=cli('pod','list')
    volumes=cli('network-volume','list')
    cfg=json.loads((RUN/'config.json').read_text())
    inputs=json.loads((RUN/'input_manifest.json').read_text())
    packets=[]
    scratch=HERE/'scratch';scratch.mkdir(exist_ok=True)
    for task in cfg['tasks']:
        packet=scratch/f'run214-{task}-inputs.tar'
        selected=[x for x in inputs['files'] if task in x['tasks']]
        for item in selected:
            path=ROOT/item['path']
            assert path.stat().st_size==item['bytes'] and file_hash(path)==item['sha256']
        names=[x['path'] for x in selected]+[(RUN/'input_manifest.json').relative_to(ROOT).as_posix()]
        with tarfile.open(packet,'w') as archive:
            for name in names:archive.add(ROOT/name,arcname=name,recursive=False)
        packets.append(dict(task=task,path=packet.relative_to(ROOT).as_posix(),bytes=packet.stat().st_size,
                            sha256=file_hash(packet),files=names))
    receipt=dict(utc=stamp(),epoch=time.time(),cli_version=subprocess.check_output(['runpodctl','version'],text=True).strip(),
        status='prepared_awaiting_spending_envelope',config_sha256=file_hash(RUN/'config.json'),
        input_manifest_sha256=file_hash(RUN/'input_manifest.json'),catalog=catalog,template=template,
        baseline_pods=[dict(id=p['id'],name=p['name']) for p in pods],
        baseline_volumes=[{k:v[k] for k in ('id','name','size','dataCenterId')} for v in volumes],
        packets=packets,production_forks=348,continuation_updates=69600,reference_updates=12000,
        workers=[dict(task=task,seed=seed) for task in cfg['tasks'] for seed in cfg['seeds']],
        runtime_policy='One pinned image/environment and GPU model per case; all initial tensors loaded by hash; no historical trajectory mixing.',
        cloud_etc='Unmeasured. Paid representative calibration is part of the requested envelope; accept production only when its conservative ETC fits the remaining deadline.',
        proposed_envelope=dict(max_total_usd=40,max_wall_hours=4,max_concurrent_gpus=6,
            max_gpu_usd_per_hour=1.19,retrieval_margin_minutes=30,calibration_max_minutes=20,
            includes='Setup, calibration, all workers/retries, running/stopped storage, retrieval and verification.'),
        storage=dict(container_gb_per_worker=30,pod_volume_gb_per_worker=35,
            running_usd_per_gb_month=.10,stopped_pod_volume_usd_per_gb_month=.20,
            source='https://docs.runpod.io/pods/pricing',network_volumes_created=0,
            retention='Pod volume retains recovery and evidence. Retrieve and verify before deleting task Pods. Preserve unrelated volumes.'),
        monitoring=dict(interval_seconds=60,fields=['updates','sampled/full loss','throughput','ETC','GPU memory','event age','disk','aggregate estimated cost'],
            warnings=['nonfinite or undefined diagnostics','GPU identity/runtime mismatch','ETC exceeds remaining time','progress stale for five full updates','cost reaches reserve threshold'],
            stop='Stop scientific work 30 minutes before the wall limit or at USD32 estimated spend; save recovery and inventory. Stop task compute at the hard deadline. Retained disk must fit USD40 aggregate ceiling.'),
        lifecycle='Detached worker and independent local guard; incremental verified retrieval; terminal inventory and per-worker acceptance before deletion. No push.')
    write_json(HERE/'runpod_preflight.json',receipt)
    path=HERE/'runpod_preflight.json'
    path.write_text(path.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    print(json.dumps(dict(status=receipt['status'],packets=[{k:x[k] for k in ('task','bytes','sha256')} for x in packets],workers=6)))


if __name__=='__main__':main()
