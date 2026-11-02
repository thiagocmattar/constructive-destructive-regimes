"""Verify complete production arms and reduce full histories without changing training."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,flat,tensor_hash,write_json
from src.eos_tasks.task import model_for


def rows(file):return [json.loads(line) for line in file.read_text().splitlines()]


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    cfg=json.loads((HERE/'config.json').read_text());target=cfg['updates']
    shared=json.loads((ROOT/cfg['cache']/'inputs.json').read_text());cache=ROOT/cfg['cache']
    expected={(t,lr) for t,rates in cfg['learning_rates_by_task'].items() for lr in rates}
    gdfile=ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/gd_loss_milestones.json'
    gd=json.loads(gdfile.read_text())
    conditions=[];incomplete=[]
    for file in sorted((HERE/'artifacts/attempt-01').rglob('manifest.json')):
        m=json.loads(file.read_text());folder=file.parent
        if m.get('mechanics_only'):continue
        assert m['run_id']=='215' and m['config_sha256']==file_hash(HERE/'config.json')
        assert m['input_manifest_sha256']==file_hash(HERE/'input_manifest.json')
        assert (m['task'],m['lr']) in expected
        if m['status']!='complete':
            incomplete.append(dict(task=m['task'],lr=m['lr'],status=m['status'],source=folder.relative_to(ROOT).as_posix()))
            continue
        assert m['completed_updates']==target
        for name,identity in m['files'].items():
            p=folder/name;assert p.stat().st_size==identity['bytes'] and file_hash(p)==identity['sha256']
        for name,digest in m['source_sha256'].items():assert file_hash(ROOT/name)==digest
        b=rows(folder/'boundaries.jsonl');metrics=rows(folder/'metrics.jsonl');e=rows(folder/'evaluation.jsonl')
        assert [r['state'] for r in b]==list(range(target+1))
        assert [r['update'] for r in metrics]==list(range(1,target+1))
        assert [r['state'] for r in e]==list(range(0,target+1,cfg['evaluate_every']))
        units={'mlp':{'test':10000},'resnet':{'test':10000},'transformer':{'valid':217630,'test':245550}}
        assert all({s:v['loss_units'] for s,v in r['splits'].items()}==units[m['task']] for r in e)
        final=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
        assert all(int(s['step'])==target for s in final['optimizer']['state'].values())
        for stream in ['raw','moment','adaptive','actual']:
            values=[r['streams'][stream] for r in metrics];saved=final['diagnostics'][stream]
            norms=np.array([r['norm'] for r in values])
            np.testing.assert_allclose([r['A'] for r in values],norms.cumsum(),rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose([r['energy'] for r in values],(norms**2).cumsum(),rtol=2e-12,atol=2e-12)
            assert saved['count']==target
            np.testing.assert_allclose(float(saved['total'].norm()),values[-1]['B'],rtol=1e-12,atol=1e-10)
            assert all(0<=r['B_over_A']<=1+1e-10 for r in values)
        model=model_for(m['task'],cache,shared['wikitext']['vocabulary_size'])
        initial=shared['initial'][m['task']][cfg['seed']]
        model.load_state_dict(torch.load(cache/initial['path'],weights_only=True))
        assert tensor_hash(model.parameters())==m['initial_tensor_sha256']==initial['initial_tensor_sha256']
        assert torch.equal(flat(model.parameters()).double(),final['initial_parameters'])
        model.load_state_dict(final['model']);assert tensor_hash(model.parameters())==m['final_tensor_sha256']
        error=float((flat(model.parameters()).double()-final['initial_parameters']-final['diagnostics']['actual']['total']).norm())
        assert error<1e-9 and max(r['actual_displacement_identity_residual_norm'] for r in metrics)<1e-9
        first=torch.load(folder/'audit-gradient-0.pt',map_location='cpu',weights_only=True)['raw_gradient'].double()
        second=torch.load(folder/'audit-gradient-1.pt',map_location='cpu',weights_only=True)['raw_gradient'].double()
        terminal=torch.load(folder/('audit-gradient-'+str(target)+'.pt'),map_location='cpu',weights_only=True)['raw_gradient'].double()
        for i,before,after in [(1,first,second),(target,final['diagnostics']['raw']['previous'],terminal)]:
            expression=float(before.dot(before-after))/float(before.dot(before))
            assert abs(expression-b[i]['projected_gradient_change'])<2e-10
        losses=[r['loss'] for r in b];matches=[]
        for condition in gd['conditions']:
            if condition['task']!=m['task']:continue
            for state,loss in condition['targets_by_parameter_state'].items():
                matches.append(dict(gd_run=condition['run_id'],gd_state=int(state),target_loss=loss,
                    first_adam_state=next((i for i,value in enumerate(losses) if value<=loss),None),
                    first_trailing64_state=next((i for i in range(63,len(losses)) if np.mean(losses[i-63:i+1])<=loss),None)))
        conditions.append(dict(task=m['task'],lr=m['lr'],updates=target,status='complete',initial_loss=losses[0],
            final_loss=losses[-1],final_evaluations=e[-1]['splits'],
            tail64_cosines={s:float(np.median([r['streams'][s]['cosine'] for r in metrics[-64:]])) for s in ['raw','moment','adaptive','actual']},
            final_R_t=metrics[-1]['streams']['actual']['B_over_A'],parameter_telescoping_error=error,
            elapsed_seconds=m['elapsed_seconds'],environment=m['environment'],gd_loss_matches=matches,
            source=folder.relative_to(ROOT).as_posix(),manifest_sha256=file_hash(file)))
        del final,model,first,second,terminal
    keys=[(r['task'],r['lr']) for r in conditions]
    assert len(keys)==len(set(keys))
    complete=set(keys)==expected
    result=dict(status='complete_verified' if complete else 'partial_verified',run_id='215',updates_per_arm=target,
        expected_arms=9,completed_arms=len(keys),conditions=conditions,incomplete=incomplete,
        missing=sorted(expected-set(keys)),source_gd_milestones_sha256=file_hash(gdfile),
        interpretation='One initialization; descriptive matched fixed-LR trajectories. No Hessian, stability or generalization claim.')
    write_json(HERE/'results.json',result)
    print(json.dumps(dict(status=result['status'],completed_arms=len(keys),missing=result['missing'])))
    if args.require_complete:assert complete and not incomplete


if __name__=='__main__':main()
