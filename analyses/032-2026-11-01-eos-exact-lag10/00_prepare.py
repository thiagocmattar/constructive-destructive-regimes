"""Verify original inventories, extract minimal exact states and freeze Run205."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
RUN=ROOT/'runs/205-2026-11-01-eos-exact-lag10'
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import tensor_hash
from src.eos_tasks.task import model_for


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    assert not (RUN/'config.json').exists(),'Scientific inputs already frozen'
    taskcache=ROOT/'cache/analysis032-inputs';taskcache.mkdir(exist_ok=True)
    scratch=HERE/'scratch';scratch.mkdir(parents=True,exist_ok=True)
    for name in ['research/INDEX.md','research/HYPOTHESES.md','analyses/INDEX.md','runs/INDEX.md','research/findings/README.md']:
        target=scratch/'original-worktree'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    old=json.loads((ROOT/'analyses/024-2026-10-17-eos-gd-lr-calibration/cohort.json').read_text())
    shared=json.loads((ROOT/'cache/eos-multitask/inputs.json').read_text())
    original=json.loads((ROOT/'cache/cifar10-eos/inputs.json').read_text())
    files=[ROOT/'cache/eos-multitask/inputs.json',ROOT/'cache/cifar10-eos/inputs.json',ROOT/'cache/cifar10-eos/data.pt']
    assert sha(files[-1])==original['prepared_files']['data.pt']['sha256']
    for task in ['resnet','transformer']:
        path=ROOT/f'cache/eos-multitask/{task}_data.pt'
        assert sha(path)==shared['prepared_files'][path.name]['sha256'];files.append(path)
    for name,item in shared['reference_sources'].items():
        path=ROOT/'cache/eos-multitask/reference'/name
        assert sha(path)==item['sha256'];files.append(path)
    files.append(ROOT/'cache/eos-multitask/mlp_initial_seed0.pt')
    assert sha(files[-1])==shared['initial']['mlp'][0]['sha256']
    frozen=ROOT/'runs/123-2026-10-17-eos-resnet-gd-lr-0.025-3k/artifacts/attempt-01-spark/source'
    for name in ['src/cifar_eos/core.py','src/eos_tasks/task.py','src/eos_tasks/production.py']:
        assert sha(ROOT/name)==sha(frozen/name),name
    conditions=[];provenance=[]
    for task in ['mlp','resnet','transformer']:
        selected=sorted([r for r in old['runs'] if r['task']==task],key=lambda r:r['lr'])
        for worker,parent in enumerate(selected):
            run=ROOT/'runs'/parent['run'];prod=run/'artifacts/attempt-01-spark/production'
            rid=parent['run_id'];legacy=rid in ['059','062']
            if legacy:
                manifest=json.loads((prod/'manifest.json').read_text())
                expected={k:v for k,v in manifest['files'].items()}
                inventory_path=prod/'manifest.json'
            else:
                inventory_path=prod.parent/'transfer_inventory.json'
                expected={Path(r['path']).name:r for r in json.loads(inventory_path.read_text())['files'] if r['path'].startswith('production/')}
            assert sha(run/'config.json')==parent['config_sha256']
            metrics=prod/'metrics.jsonl';assert sha(metrics)==expected['metrics.jsonl']['sha256']
            rows=[]
            for line in metrics.open():
                r=json.loads(line);s=r['gradient_state_step']
                if s>3010:break
                if legacy or any(a<=s<=a+10 for a in [500,2000]):
                    rows.append(dict(state=s,loss=r['train_loss'] if legacy else r['full_objective_loss'],
                                     norm=r['streams']['raw']['norm'],cosine=r['streams']['raw']['cosine']))
            summary=json.loads((prod/'summary.json').read_text())
            reference=taskcache/'references'/f'{rid}.json'
            save(reference,dict(rows=rows,final_loss=summary['final_train']['loss'],source_metrics=metrics.relative_to(ROOT).as_posix(),
                                source_metrics_sha256=sha(metrics)))
            files.append(reference)
            cond=dict(parent_run_id=rid,parent_run=parent['run'],task=task,lr=parent['lr'],worker=worker,seed=0,
                physical=64 if task=='transformer' else 1000,reconstructed=legacy,
                data='cache/cifar10-eos/data.pt' if task=='mlp' else f'cache/eos-multitask/{task}_data.pt',
                reference=reference.relative_to(ROOT).as_posix(),parents=[])
            if legacy:
                cond.update(initial='cache/eos-multitask/mlp_initial_seed0.pt',initial_tensor_sha256=shared['initial']['mlp'][0]['initial_tensor_sha256'])
            else:
                model=model_for(task,ROOT/'cache/eos-multitask',shared['wikitext']['vocabulary_size'])
                for anchor in [500,2000,3000]:
                    path=prod/f'state-{anchor:06d}.pt';assert sha(path)==expected[path.name]['sha256']
                    state=torch.load(path,map_location='cpu',weights_only=False)
                    assert state['step']==anchor and state['config_sha256']==parent['config_sha256']
                    assert not state['optimizer']['state']
                    assert state['optimizer']['param_groups'][0]['lr']==parent['lr']
                    thin={k:state[k] for k in ['step','model','optimizer','torch_rng','cuda_rng']}
                    destination=taskcache/'parents'/rid/path.name;destination.parent.mkdir(parents=True,exist_ok=True)
                    torch.save(thin,destination)
                    model.load_state_dict(thin['model'])
                    digest=tensor_hash(model.parameters())
                    record=dict(state=anchor,path=destination.relative_to(ROOT).as_posix(),tensor_sha256=digest,
                        original=path.relative_to(ROOT).as_posix(),original_bytes=path.stat().st_size,
                        original_sha256=sha(path),original_inventory=inventory_path.relative_to(ROOT).as_posix())
                    cond['parents'].append(record);provenance.append(record);files.append(destination)
                    del state,thin
                del model
            conditions.append(cond)
    config=dict(run_id='205',analysis='032',hypothesis='H020',purpose='Exact anchored lag1..10 geometry near Analysis024 saved states',
        anchors=[500,2000,3000],max_lag=10,conditions=conditions,task_cache='cache/eos-multitask',
        inputs_manifest='runs/'+RUN.name+'/input_manifest.json',original_inputs='cache/eos-multitask/inputs.json',
        metric='cos(gradient(theta_anchor),gradient(theta_anchor+lag)); one pair per lag; no averaging or compression',
        objective=dict(mlp='complete first5000 images; half squared error summed over outputs and mean over examples',
            resnet='complete50000 images; cross entropy; same50 fixed1000-image train-mode BN groups; no running-stat mutation',
            transformer='all59680 fixed sequences; NLL summed then divided by2088600 target tokens'),
        optimizer='constant plain SGD, original LR; no momentum, weight decay, clipping, schedule, augmentation or dropout',
        precision='FP32 parameters/accumulated gradients; TF32 disabled; exact FP64 diagnostic dot products and norms',
        parameter_scope='all trainable parameters, original model order, no compression',zero_denominator='null with undefined flag; no epsilon',
        original_history_limit=3000,late_window='new10-update continuation beyond Analysis024; never append to original runs',
        qualification_tolerances=dict(loss_absolute=1e-4,loss_relative=1e-3,norm_absolute=1e-5,norm_relative=.01,cosine_absolute=.02),
        qualification_policy='Saved-model continuations are valid new forks; historical agreement reported separately. Reconstructed MLP windows failing any preceding check are excluded from primary profiles and retained separately.',
        fixed_repeat_relative_tolerance=1e-5,recovery='parent inputs retained; each11-gradient window keeps start/end model, optimizer and RNG states',
        validation='none; complete training objective only; no new held-out/generalization claims',
        authorization='User: Use runpod. No need to compress. Use the checkpoints for faster run. Produce an analysis folder at the end.',
        resource_limits=dict(workers=3,max_total_usd=6,max_lease_seconds=7200,production_timeout_seconds=3600,
                             container_gb=30,pod_volume_gb=15,network_volume=False),
        inference='One initialization per model;27 anchored profiles; one dependent pair per lag. No seed uncertainty, stationarity, stability or loss-benefit inference.')
    save(RUN/'config.json',config)
    source=[ROOT/'src/__init__.py',ROOT/'src/cifar_eos/__init__.py',ROOT/'src/cifar_eos/core.py',ROOT/'src/eos_tasks/__init__.py',ROOT/'src/eos_tasks/task.py',ROOT/'src/eos_tasks/production.py']
    source=[p for p in source if p.exists()]+list(RUN.glob('*.py'))+[RUN/'config.json']
    files=sorted(set(files+source))
    manifest=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    save(RUN/'input_manifest.json',manifest);save(HERE/'checkpoint_provenance.json',dict(parents=provenance))
    packet=ROOT/'cache/analysis032-inputs.tar.gz'
    with tarfile.open(packet,'w:gz',compresslevel=1) as archive:
        for path in files+[RUN/'input_manifest.json']:archive.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    save(HERE/'packet.json',dict(path=packet.relative_to(ROOT).as_posix(),bytes=packet.stat().st_size,sha256=sha(packet),files=len(files)+1,config_sha256=sha(RUN/'config.json')))
    print(json.dumps(dict(status='prepared',conditions=len(conditions),saved_states=len(provenance),packet_bytes=packet.stat().st_size)))


if __name__=='__main__':main()
