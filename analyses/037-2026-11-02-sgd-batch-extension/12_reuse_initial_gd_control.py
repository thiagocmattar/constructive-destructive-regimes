"""Derive the initial full-GD control from its exact measured reference prefix.

No update is skipped in the reference or any stochastic continuation. The first
200 reference updates already are a 200-update full-GD trajectory from state0.
Retain its checkpoint, raw metrics and complete endpoint loss; avoid executing
that identical condition twice. This only applies before any outcome is selected.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True)
    p.add_argument('--worker',type=Path,required=True);p.add_argument('--max-seconds',type=float,default=1200)
    a=p.parse_args();sys.path.insert(0,str(a.root))
    from src.cifar_eos.core import file_hash,stamp,tensor_hash
    from src.eos_tasks.production import ProductionTask
    from src.eos_tasks.task import configure,model_for
    s=importlib.util.spec_from_file_location('frozen_measurement',a.root/'runs/214-2026-11-02-sgd-batch-extension/01_measure.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    configure();run=a.root/'runs'/a.run;cfg=json.loads((run/'config.json').read_text())
    reference=a.worker/'reference';until=time.time()+a.max_seconds
    while time.time()<until:
        progress=reference/'progress.json'
        if progress.exists():
            index=json.loads(progress.read_text())['update']
            if index>200:raise RuntimeError('Missed exact state200; keep the independent full control.')
            if index==200:break
        time.sleep(2)
    else:raise RuntimeError('Reference prefix not ready within bound.')
    source=m.torch.load(reference/'recovery.pt',map_location='cpu',weights_only=False)
    assert source['completed']==200 and source['identity']['mode']=='reference'
    identity=dict(source['identity']);task=identity['task'];seed=identity['seed'];batch=cfg['batches'][task][-1]
    assert identity['config_sha256']==file_hash(run/'config.json')
    target=a.worker/f'state-0-b{batch}-rep00';assert not target.exists()
    target.mkdir()
    captured=target/'reference-prefix-recovery.pt'
    # Atomic saves replace recovery.pt; verify the copied checkpoint still is state200.
    shutil.copyfile(reference/'recovery.pt',captured)
    copied=torch.load(captured,map_location='cpu',weights_only=False)
    assert copied['completed']==200
    source=copied
    lines=(reference/'metrics.jsonl').read_bytes().splitlines(keepends=True)[:200]
    (target/'metrics.jsonl').write_bytes(b''.join(lines))
    history=[json.loads(x) for x in lines]
    assert [r['update'] for r in history]==list(range(1,201))
    assert len([r for r in history if r['cosine'] is not None])==199
    cache=a.root/cfg['cache'];shared=json.loads((cache/'inputs.json').read_text())
    data={k:v.cuda() for k,v in torch.load(cache/(task+'_data.pt'),weights_only=True).items() if k in ('x_train','y_train')}
    model=model_for(task,cache,shared['wikitext']['vocabulary_size']).cuda();model.load_state_dict(source['model'])
    objective=ProductionTask(task,model,data,cfg['physical'][task]);evaluation=objective.evaluate()
    initial=torch.load(reference/'state-0000.pt',map_location='cpu',weights_only=False)
    delta2=sum(float((value.double()-initial['model'][key].double()).square().sum()) for key,value in source['model'].items() if value.is_floating_point())
    np.testing.assert_allclose(delta2**.5,history[-1]['actual_displacement'],rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(float(source['raw']['total'].norm())/source['raw']['A'],history[-1]['B_over_A'],rtol=1e-12)
    identity.update(mode='fork',state=0,repetition=0,updates=200,
        source=dict(path=(reference/'state-0000.pt').as_posix(),sha256=file_hash(reference/'state-0000.pt')))
    summary=dict(identity=identity,**m.summarize(history),initial_loss=source['baseline'],final_loss=evaluation['loss'],
        loss_change=evaluation['loss']-source['baseline'],elapsed_seconds=source['elapsed_seconds'],status='complete',
        derived_from_reference_prefix=True)
    final=dict(model=source['model'],task=task,seed=seed,state=200,mechanics_only=False,
        config_sha256=identity['config_sha256'],tensor_sha256=tensor_hash(model.parameters()),
        torch_rng=source['torch_rng'],cuda_rng=source['cuda_rng'],rng=source['rng'])
    torch.save(final,target/'final.pt');(target/'indices.u32').write_bytes(b'')
    if seed==0:
        shutil.copyfile(reference/'audit-0000.pt',target/'audit-0000.pt')
        torch.save(dict(gradient=source['raw']['previous'].float(),raw_prefix=source['raw']['total'],
            previous_gradient=None,state=199,ids=None,
            coverage_note='Terminal gradient/prefix from exact checkpoint; preceding gradient not retained. Adjacent scalar comes from the measured reference stream.'),target/'audit-0199.pt')
    def write(name,value):(target/name).write_text(json.dumps(value,indent=2)+'\n')
    write('manifest.json',dict(identity=identity,status='complete',created_utc=stamp(),completed_updates=200,
        derived_from_reference_prefix=True,objective_units=objective.n,loss_units=objective.units,
        starting_tensor_sha256=initial['tensor_sha256']))
    write('derivation.json',dict(created_utc=stamp(),reference=reference.as_posix(),updates=[1,200],
        recovery_sha256=file_hash(captured),metrics_prefix_sha256=file_hash(target/'metrics.jsonl'),
        endpoint_evaluation=evaluation,training_updates_saved=200,
        audit_limit='Terminal adjacent pair is inherited from the reference scalar stream; terminal gradient/prefix and parameter displacement are independently checked.'))
    write('summary.json',summary)
    write('inventory.json',{f.name:dict(bytes=f.stat().st_size,sha256=file_hash(f)) for f in target.iterdir() if f.is_file() and f.name!='inventory.json'})
    print(json.dumps(dict(status='complete',task=task,seed=seed,updates=200,training_updates_saved=200)),flush=True)


if __name__=='__main__':main()
