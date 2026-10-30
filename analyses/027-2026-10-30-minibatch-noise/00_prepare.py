"""Prepare the approved H023 inputs without accessing Spark or launching compute."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
RUN=ROOT/'runs/186-2026-10-30-mlp-minibatch-noise'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    inputs=json.loads((ROOT/'cache/eos-multitask/inputs.json').read_text())
    original=json.loads((ROOT/'cache/cifar10-eos/inputs.json').read_text())
    assert sha(ROOT/'cache/cifar10-eos/data.pt')==original['prepared_files']['data.pt']['sha256']
    for c in inputs['initial']['mlp']:
        assert sha(ROOT/'cache/eos-multitask'/c['path'])==c['sha256']
    cache=ROOT/'cache/minibatch-noise-preparation';cache.mkdir(exist_ok=True)
    history=ROOT/'runs/062-2026-10-13-cifar10-eos-gd-lr-2e-2/artifacts/attempt-01-spark/production/metrics.jsonl'
    baseline=[]
    for line in history.open():
        r=json.loads(line)
        if r['update']>2000:break
        baseline.append(dict(update=r['update'],loss=r['train_loss'],norm=r['streams']['raw']['norm'],cosine=r['streams']['raw']['cosine']))
    assert len(baseline)==2000
    save(cache/'run062-prefix.json',{'original_metrics_sha256':sha(history),'rows':baseline})
    paths=['cache/cifar10-eos/data.pt','cache/cifar10-eos/inputs.json','cache/eos-multitask/inputs.json',
           'cache/minibatch-noise-preparation/run062-prefix.json']+[f'cache/eos-multitask/mlp_initial_seed{s}.pt' for s in range(3)]
    config=dict(run_id='186',hypothesis='H023',analysis='027',purpose='Approved trained-state frozen controls and matched moving SGD continuations',
        eta=.02,seeds=[0,1,2],reference_updates=2000,checkpoints=[0,1000,2000],batches=[32,128,512,2048,5000],
        frozen_repetitions=32,frozen_draws=128,moving_repetitions=8,continuation_updates=200,hvp_predecessor_gradients=[1,51,101,151],
        recovery_every=50,physical_batch_size=1000,frozen_sampling_seed=2026103001,moving_sampling_seed=2026103002,
        data_path='cache/cifar10-eos/data.pt',initial_paths={str(s):f'cache/eos-multitask/mlp_initial_seed{s}.pt' for s in range(3)},
        initial_tensor_hashes={str(s):inputs['initial']['mlp'][s]['initial_tensor_sha256'] for s in range(3)},
        input_files_sha256={p:sha(ROOT/p) for p in paths},
        optimizer='plain constant-step SGD; no clipping, momentum, decay or scheduler',
        objective='mean half squared error summed over ten outputs; complete first5000 CIFAR10',
        precision='FP32 parameters/gradients; FP64 diagnostic reductions and accumulation; TF32 disabled',
        sampling='fresh independent uniform subsets without replacement within each update; paired permutation prefixes across batch sizes',
        covariance_normalization='population 1/N',zero_denominator='null with reason, no epsilon replacement',
        reproduction_tolerances={'loss_absolute':1e-4,'loss_relative':1e-3,'norm_absolute':1e-5,'norm_relative':.01,'cosine_absolute':.02},
        reproduction_failure_policy='new matched reference trajectory; retain discrepancies rather than claim historical checkpoint reuse',
        smoothing={'cosine_trailing_pairs':20,'require_complete_window':True,'loss_and_R':'none'},
        inference='conditional sampling uncertainty and three independent initialization blocks; no stability or generalization inference')
    configfile=RUN/'config.json'
    if configfile.exists():assert json.loads(configfile.read_text())==config,'Do not overwrite changed scientific inputs'
    else:save(configfile,config)
    source=[p for p in [ROOT/'src/__init__.py',ROOT/'src/cifar_eos/__init__.py',ROOT/'src/cifar_eos/core.py',
        ROOT/'src/eos_tasks/__init__.py',ROOT/'src/eos_tasks/task.py',ROOT/'src/eos_tasks/noise.py'] if p.exists()]
    source += list(RUN.glob('*.py'))+[RUN/'config.json']
    source += list((ROOT/'runs/185-2026-10-30-minibatch-noise-calibration').glob('*.py'))
    source += [ROOT/'tools/artifact_inventory.py']
    files=source+[ROOT/p for p in paths]
    inventory={str(p.relative_to(ROOT)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
    save(RUN/'input_manifest.json',inventory)
    packet=cache/'inputs.tar.gz'
    with tarfile.open(packet,'w:gz',compresslevel=1) as archive:
        for p in files:archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        archive.add(RUN/'input_manifest.json',arcname=str((RUN/'input_manifest.json').relative_to(ROOT)))
    save(HERE/'packet.json',dict(path=str(packet.relative_to(ROOT)),bytes=packet.stat().st_size,sha256=sha(packet),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=inventory))
    print(json.dumps({'packet_bytes':packet.stat().st_size,'files':len(files),'config_sha256':sha(configfile)}))


if __name__=='__main__':main()
