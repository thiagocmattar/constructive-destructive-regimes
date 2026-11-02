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


def verify_worker(folder):
    inventory=read(folder/'worker_inventory.json')
    for relative,entry in inventory.items():
        path=folder/relative
        assert path.resolve().is_relative_to(folder.resolve())
        assert path.stat().st_size==entry['bytes'] and file_hash(path)==entry['sha256'],relative
    manifest=read(folder/'worker_manifest.json')
    assert manifest['status']=='complete'
    identity=manifest['identity']
    assert identity['config_sha256']==file_hash(RUN/'config.json')
    assert identity['input_manifest_sha256']==file_hash(RUN/'input_manifest.json')
    results=[]
    reference=read(folder/'reference/summary.json')
    assert reference['updates']==2000 and reference['status']=='complete'
    for job in identity['jobs']:
        target=folder/job['name']; summary=read(target/'summary.json')
        assert summary['identity']['config_sha256']==identity['config_sha256']
        history=[json.loads(line) for line in (target/'metrics.jsonl').read_text().splitlines()]
        horizon=2000 if job['mode']=='reference' else 200
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
        if job['batch']!=read(RUN/'config.json')['batches'][identity['task']][-1]:
            indices=np.fromfile(target/'indices.u32',dtype='<u4').reshape(horizon,job['batch'])
            assert all(len(np.unique(row))==job['batch'] for row in indices)
        results.append(dict(task=identity['task'],seed=identity['seed'],**job,
                            mean_rho=summary['mean_rho'],R=summary['R'],loss_change=summary['loss_change'],
                            source=target.relative_to(ROOT).as_posix(),summary_sha256=file_hash(target/'summary.json')))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('workers',type=Path,nargs='+')
    p.add_argument('--partial',action='store_true',help='Verify supplied complete workers, never certify the cohort.')
    args=p.parse_args(); cfg=read(RUN/'config.json')
    results=[]
    for worker in args.workers:results.extend(verify_worker(worker))
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
