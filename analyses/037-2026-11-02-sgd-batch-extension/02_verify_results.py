"""Verify retrieved worker inventories and independently reduce every new fork."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'runs/214-2026-11-02-sgd-batch-extension'
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,write_json


def read(path):return json.loads(path.read_text())


def verify_worker(folder,mechanics=False,input_manifest=RUN/'input_manifest.json'):
    folder=folder.resolve()
    inventory=read(folder/'worker_inventory.json')
    for relative,entry in inventory.items():
        path=folder/relative
        assert path.resolve().is_relative_to(folder.resolve())
        assert path.stat().st_size==entry['bytes'] and file_hash(path)==entry['sha256'],relative
    assert set(inventory)=={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name!='worker_inventory.json'}
    manifest=read(folder/'worker_manifest.json')
    assert manifest['status']==('smoke_complete' if mechanics else 'complete')
    identity=manifest['identity']
    assert identity['mechanics_only']==mechanics
    assert identity['config_sha256']==file_hash(RUN/'config.json')
    assert identity['input_manifest_sha256']==file_hash(input_manifest)
    cfg=read(RUN/'config.json');task=identity['task'];seed=identity['seed']
    expected_jobs=[dict(name='reference',mode='reference')]
    for state in cfg['states']:
        for batch in reversed(cfg['batches'][task]):
            for rep in range(1 if batch==cfg['batches'][task][-1] else cfg['stochastic_repetitions']):
                expected_jobs.append(dict(name=f'state-{state}-b{batch}-rep{rep:02d}',mode='fork',state=state,batch=batch,rep=rep))
    if mechanics:
        expected_jobs=[expected_jobs[0]]+[j for j in expected_jobs[1:] if j['state']==0 and j['rep']==0 and j['batch'] in (cfg['batches'][task][0],cfg['batches'][task][-1])]
    assert identity['jobs']==expected_jobs
    assert manifest['complete_jobs']==manifest['job_count']==len(expected_jobs)
    results=[]
    reference=read(folder/'reference/summary.json')
    assert reference['updates']==(3 if mechanics else 2000)
    for job in identity['jobs']:
        target=folder/job['name']; summary=read(target/'summary.json')
        assert summary['identity']['config_sha256']==identity['config_sha256']
        assert summary['status']==('smoke_complete' if mechanics else 'complete')
        assert summary['identity']['mechanics_only']==mechanics
        assert summary['identity']['task']==task and summary['identity']['seed']==seed
        history=[json.loads(line) for line in (target/'metrics.jsonl').read_text().splitlines()]
        horizon=3 if mechanics else (2000 if job['mode']=='reference' else 200)
        assert len(history)==horizon and summary['updates']==horizon
        assert [r['update'] for r in history]==list(range(1,horizon+1))
        cosines=[r['cosine'] for r in history[1:]]
        assert all(x is not None and np.isfinite(x) and abs(x)<=1+1e-10 for x in cosines)
        assert history[0]['cosine'] is None
        norms=np.array([r['norm'] for r in history]); end=history[-1]
        np.testing.assert_allclose(norms.sum(),end['A'],rtol=1e-12)
        np.testing.assert_allclose(np.mean(cosines),summary['mean_rho'],rtol=1e-12)
        np.testing.assert_allclose(end['B']/norms.sum(),summary['R'],rtol=1e-12)
        np.testing.assert_allclose(summary['final_loss']-summary['initial_loss'],summary['loss_change'],atol=1e-12)
        assert summary['valid_cosines']==horizon-1
        assert all(0<=r['B_over_A']<=1+1e-10 and r['norm']>0 for r in history)
        assert all(np.isfinite(summary[k]) for k in ('initial_loss','final_loss','R','actual_R'))
        for audit_file in target.glob('audit-*.pt'):
            audit=torch.load(audit_file,map_location='cpu',weights_only=False)
            row=history[audit['state']];g=audit['gradient'].double().numpy();prefix=audit['raw_prefix'].numpy()
            np.testing.assert_allclose(np.linalg.norm(g),row['norm'],rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(prefix),row['B'],rtol=1e-10,atol=1e-12)
            previous=audit['previous_gradient']
            if previous is not None:
                previous=previous.numpy()
                rho=np.dot(g,previous)/(np.linalg.norm(g)*np.linalg.norm(previous))
                np.testing.assert_allclose(rho,row['cosine'],rtol=1e-10,atol=1e-12)
        if job['mode']=='reference':continue
        assert summary['identity']['batch']==job['batch'] and summary['identity']['state']==job['state']
        # Terminal actual displacement, using independently loaded parameter tensors.
        start=torch.load(folder/'reference'/f'state-{job["state"]:04d}.pt',map_location='cpu',weights_only=False)
        finish=torch.load(target/'final.pt',map_location='cpu',weights_only=False)
        assert finish['state']==job['state']+horizon
        # Model buffers are absent/constant in these objectives. Use parameter count to check scope.
        distance=np.sqrt(sum(float((finish['model'][key].double()-value.double()).square().sum())
            for key,value in start['model'].items() if value.is_floating_point()))
        np.testing.assert_allclose(distance,summary['absolute_displacement'],rtol=1e-8,atol=1e-9)
        actual_length=sum(r['actual']['norm'] for r in history)
        np.testing.assert_allclose(distance/actual_length,summary['actual_R'],rtol=1e-8,atol=1e-9)
        assert summary['max_actual_prefix_error']<1e-10
        if job['batch']!=read(RUN/'config.json')['batches'][identity['task']][-1]:
            indices=np.fromfile(target/'indices.u32',dtype='<u4').reshape(horizon,job['batch'])
            assert all(len(np.unique(row))==job['batch'] for row in indices)
            rng=np.random.default_rng(np.random.SeedSequence([cfg['sampling_seed'],{'resnet':1,'transformer':2}[task],seed,job['state'],job['rep']]))
            for row in indices:np.testing.assert_array_equal(row,rng.permutation(cfg['batches'][task][-1])[:job['batch']])
        elif mechanics:
            # FP32 cuDNN reductions need not be bitwise deterministic. Qualify the
            # short control by explicit small-error bounds; retain measured drift.
            end_reference=torch.load(folder/'reference/state-0003.pt',map_location='cpu',weights_only=False)
            error2=scale2=0.
            for key,value in finish['model'].items():
                expected=end_reference['model'][key]
                torch.testing.assert_close(value,expected,rtol=1e-6,atol=1e-7)
                error2+=float((value.double()-expected.double()).square().sum())
                scale2+=float(expected.double().square().sum())
            assert np.sqrt(error2/scale2)<=1e-6
            for metric in ('mean_rho','R'):
                np.testing.assert_allclose(summary[metric],reference[metric],rtol=1e-6,atol=1e-7)
        results.append(dict(task=identity['task'],seed=identity['seed'],**job,
                            mean_rho=summary['mean_rho'],R=summary['R'],loss_change=summary['loss_change'],
                            source=target.relative_to(ROOT).as_posix(),summary_sha256=file_hash(target/'summary.json')))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('workers',type=Path,nargs='+')
    p.add_argument('--partial',action='store_true',help='Verify supplied complete workers, never certify the cohort.')
    p.add_argument('--mechanics',action='store_true',help='Qualify short workers and replay, without production acceptance.')
    p.add_argument('--input-manifest',type=Path,default=RUN/'input_manifest.json',help='Explicit preserved input identity for historical mechanics qualification.')
    args=p.parse_args(); cfg=read(RUN/'config.json')
    results=[]
    for worker in args.workers:results.extend(verify_worker(worker,args.mechanics,args.input_manifest))
    if args.mechanics:
        print(json.dumps(dict(status='mechanics_passed',workers=len(args.workers),forks=len(results))))
        return
    keys=[(r['task'],r['seed'],r['state'],r['batch'],r['rep']) for r in results]
    assert len(keys)==len(set(keys))
    expected=[(task,seed,state,batch,rep) for task in cfg['tasks'] for seed in cfg['seeds']
        for state in cfg['states'] for batch in cfg['batches'][task]
        for rep in range(1 if batch==cfg['batches'][task][-1] else cfg['stochastic_repetitions'])]
    if not args.partial:assert set(keys)==set(expected)
    assert set(keys)<=set(expected)
    cells=[]
    for task in cfg['tasks']:
        for state in cfg['states']:
            for batch in cfg['batches'][task]:
                block=[r for r in results if (r['task'],r['state'],r['batch'])==(task,state,batch)]
                if not block:continue
                means={metric:[float(np.mean([r[metric] for r in block if r['seed']==seed]))
                    for seed in cfg['seeds'] if any(r['seed']==seed for r in block)]
                    for metric in ['mean_rho','R','loss_change']}
                cells.append(dict(task=task,state=state,batch=batch,seed_means=means,
                    mean={k:float(np.mean(v)) for k,v in means.items()},
                    minimum={k:min(v) for k,v in means.items()},maximum={k:max(v) for k,v in means.items()}))
    output=HERE/('partial_verification.json' if args.partial else 'results.json')
    write_json(output,dict(status='partial' if args.partial else 'passed',forks=results,cells=cells,
        expected_forks=len(expected),verified_forks=len(results)))
    print(json.dumps(dict(status='partial' if args.partial else 'passed',forks=len(results),cells=len(cells))))


if __name__=='__main__':main()
