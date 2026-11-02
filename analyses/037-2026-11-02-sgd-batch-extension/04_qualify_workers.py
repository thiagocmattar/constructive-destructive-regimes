"""Qualify short real-data workers, local retrieval, and corruption rejection."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import torch
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'runs/214-2026-11-02-sgd-batch-extension'
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,stamp,write_json
spec=importlib.util.spec_from_file_location('verification037',HERE/'02_verify_results.py')
verification=importlib.util.module_from_spec(spec);spec.loader.exec_module(verification)


def main():
    receipt=dict(utc=stamp(),status='passed',mechanics_only=True,workers=[])
    scratch=HERE/'scratch/retrieval-qualification-v3'
    assert not scratch.exists(),'Preserve qualification attempts.'
    scratch.mkdir(parents=True)
    manifest=RUN/'scratch/qualified-worker-input-manifest.json'
    receipt['mechanics_input_manifest']=dict(path=manifest.relative_to(ROOT).as_posix(),sha256=file_hash(manifest))
    assert json.loads(manifest.read_text())==json.loads((RUN/'input_manifest.json').read_text())
    receipt['input_manifest_formatting']='Mechanics used CRLF; production manifest has identical parsed content with LF. Both byte identities are retained.'
    for task in ('resnet','transformer'):
        source=RUN/'scratch'/f'qualified-{task}-worker'
        verification.verify_worker(source,mechanics=True,input_manifest=manifest)
        target=scratch/task
        shutil.copytree(source,target)
        forks=verification.verify_worker(target,mechanics=True,input_manifest=manifest)
        victim=target/'reference/summary.json';original=victim.read_bytes()
        victim.write_bytes(original+b' ')
        try:
            verification.verify_worker(target,mechanics=True,input_manifest=manifest)
        except AssertionError:
            rejected=True
        else:
            raise AssertionError('Corruption was not detected.')
        finally:
            victim.write_bytes(original)
        verification.verify_worker(target,mechanics=True,input_manifest=manifest)
        inventory=json.loads((target/'worker_inventory.json').read_text())
        full_batch=50 if task=='resnet' else 59680
        a=torch.load(target/'reference/state-0003.pt',map_location='cpu',weights_only=False)['model']
        b=torch.load(target/f'state-0-b{full_batch}-rep00/final.pt',map_location='cpu',weights_only=False)['model']
        error2=sum(float((a[k].double()-b[k].double()).square().sum()) for k in a)
        scale2=sum(float(a[k].double().square().sum()) for k in a)
        maxabs=max(float((a[k]-b[k]).abs().max()) for k in a)
        receipt['workers'].append(dict(task=task,source=source.relative_to(ROOT).as_posix(),
            inventory_sha256=file_hash(source/'worker_inventory.json'),files=len(inventory),
            bytes=sum(x['bytes'] for x in inventory.values()),forks=len(forks),
            full_batch_replay=dict(updates=3,max_absolute_parameter_difference=maxabs,
                relative_parameter_l2_error=(error2/scale2)**.5,bitwise_identical=error2==0,
                parameter_tolerance='rtol1e-6/atol1e-7, plus relative total L2 <=1e-6',
                geometry_tolerance='rtol1e-6/atol1e-7',
                initial_strict_test='ResNet bitwise equality failed; small cuDNN FP32 drift recorded explicitly. Transformer exact.'),
            sampled_fork='three real-data updates with exact saved-index replay',
            audit_vectors='CPU NumPy norms, prefixes and adjacent cosine',
            retrieval='copied and all inventory sizes/SHA256 independently verified',
            corruption_rejected=rejected))
    receipt['sources']={p.relative_to(ROOT).as_posix():file_hash(p)
        for p in list(RUN.glob('*.py'))+list(HERE.glob('*.py'))+[RUN/'config.json',RUN/'input_manifest.json']}
    receipt['limits']=['Mechanics only; no new paper evidence or production updates.',
        'Cloud placement, sustained throughput, GPU/runtime parity and actual SSH/SCP transfer require the funded calibration.',
        'Full 348-fork cohort and manuscript integration remain pending.']
    write_json(HERE/'worker_qualification.json',receipt)
    path=HERE/'worker_qualification.json'
    path.write_text(path.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='passed',workers=2,mechanics_only=True)))


if __name__=='__main__':main()
