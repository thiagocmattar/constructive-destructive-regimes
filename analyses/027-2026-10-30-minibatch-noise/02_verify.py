"""Independent coverage, scalar, checkpoint and transfer audit for Run186."""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,tensor_hash
from tools.artifact_inventory import verify
RUN=ROOT/'runs/186-2026-10-30-mlp-minibatch-noise'


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def rows(p):return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def close(a,b,atol=1e-9,rtol=1e-8):
    assert math.isclose(a,b,rel_tol=rtol,abs_tol=atol),(a,b)


def sampling(path,b,count,stream_seed,seed,cp,rep,largest):
    if b==5000:
        assert path.stat().st_size==0
        return
    ids=np.fromfile(path,dtype='<u4').reshape(count,b)
    assert ids.min()>=0 and ids.max()<5000
    assert np.all(np.diff(np.sort(ids,axis=1).astype(np.int64),axis=1)>0)
    rng=np.random.default_rng(np.random.SeedSequence([stream_seed,seed,cp,rep]))
    for k in range(3):np.testing.assert_array_equal(ids[k],rng.permutation(5000)[:b])
    if b!=2048:
        other=np.fromfile(largest,dtype='<u4').reshape(count,2048)
        np.testing.assert_array_equal(ids,other[:,:b])


def main():
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=RUN/'artifacts/attempt-01/production')
    p.add_argument('--inventory',action='store_true');args=p.parse_args();out=args.output
    cfg=json.loads((RUN/'config.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['status']=='complete'
    assert manifest['config_sha256']==file_hash(RUN/'config.json')
    assert manifest['experiment_sha256']==file_hash(RUN/'01_experiment.py')
    counts=dict(reference_updates=0,frozen_draws=0,continuation_updates=0)
    max_identity=max_update=0.;moving_summaries=[];frozen_summaries=[];reconstruction={};full_control_replays=[]
    for seed in cfg['seeds']:
        ref=out/f'reference-seed{seed}'
        records=rows(ref/'metrics.jsonl');assert len(records)==2000
        assert [r['update'] for r in records]==list(range(1,2001))
        counts['reference_updates']+=len(records)
        for cp in cfg['checkpoints']:assert (ref/f'state-{cp:04d}.pt').is_file()
        initial=torch.load(ref/'state-0000.pt',weights_only=False,map_location='cpu')['model']
        assert tensor_hash(initial.values())==cfg['initial_tensor_hashes'][str(seed)]
        if seed==0:
            old=json.loads((ROOT/'cache/minibatch-noise-preparation/run062-prefix.json').read_text())['rows']
            tolerances=cfg['reproduction_tolerances']
            for field,atol,rtol in [('loss',tolerances['loss_absolute'],tolerances['loss_relative']),
                                  ('norm',tolerances['norm_absolute'],tolerances['norm_relative']),
                                  ('cosine',tolerances['cosine_absolute'],0)]:
                pairs=[(a['update'],a[field],b[field]) for a,b in zip(records,old) if a[field] is not None and b[field] is not None]
                error=np.abs(np.array([a-b for _,a,b in pairs]));allowed=np.array([atol+rtol*abs(b) for _,a,b in pairs])
                failed=[pairs[j][0] for j in np.flatnonzero(error>allowed)]
                reconstruction[field]=dict(max_absolute_error=float(error.max()),failed_states=len(failed),count=len(pairs),
                    first_failed_state=None if not failed else failed[0],absolute_tolerance=atol,relative_tolerance=rtol)
            reconstruction['qualified']=all(v['failed_states']==0 for v in reconstruction.values())
        for cp in cfg['checkpoints']:
            folder=out/f'frozen-seed{seed}-state{cp}'
            population=json.loads((folder/'population.json').read_text());assert population['measured_units']==5000
            for b in cfg['batches']:
                V=(5000-b)/(b*4999)*population['trace_population_covariance']
                for rep in range(32):
                    d=folder/f'b{b}/rep{rep:02d}';r=rows(d/'metrics.jsonl');s=json.loads((d/'summary.json').read_text())
                    assert len(r)==128 and [v['draw'] for v in r]==list(range(1,129))
                    assert (d/'indices.u32').stat().st_size==(0 if b==5000 else 128*b*4)
                    sampling(d/'indices.u32',b,128,cfg['frozen_sampling_seed'],seed,cp,rep,
                             folder/f'b2048/rep{rep:02d}/indices.u32')
                    counts['frozen_draws']+=128
                    close(s['predicted_variance'],V)
                    close(s['mean_noise_energy'],np.mean([v['noise_energy'] for v in r]))
                    close(s['mean_rho'],np.mean([v['cosine'] for v in r if v['cosine'] is not None]))
                    close(s['R'],r[-1]['B_over_A'])
                    for v in r:
                        close(v['A']*v['A'],v['A_squared']);close(v['B']*v['B'],v['B_squared'])
                        close(v['all_pair'],v['B_squared']-v['energy'])
                        if v['q'] is not None:
                            close(v['q']+v['epsilon'],0)
                            max_identity=max(max_identity,abs(v['correction_identity_residual']))
                    if b==5000:
                        close(s['R'],1);close(s['mean_rho'],1);close(s['mean_noise_energy'],0)
                    frozen_summaries.append(s)
                for rep in range(1 if b==5000 else 8):
                    d=out/f'moving-seed{seed}-state{cp}-b{b}-rep{rep:02d}'
                    r=rows(d/'metrics.jsonl');s=json.loads((d/'summary.json').read_text())
                    assert len(r)==200 and [v['update'] for v in r]==list(range(1,201))
                    counts['continuation_updates']+=200
                    assert (d/'indices.u32').stat().st_size==(0 if b==5000 else 200*b*4)
                    sampling(d/'indices.u32',b,200,cfg['moving_sampling_seed'],seed,cp,rep,
                             out/f'moving-seed{seed}-state{cp}-b2048-rep{rep:02d}/indices.u32')
                    norms=np.array([v['norm'] for v in r]);np.testing.assert_allclose(np.cumsum(norms),[v['A'] for v in r],rtol=1e-12)
                    adjacent=np.array([r[j]['cosine']*norms[j-1]*norms[j] for j in range(1,200)])
                    close(r[-1]['lag_one'],2*adjacent.sum(),atol=1e-7)
                    for j,v in enumerate(r):
                        assert 0<=v['B_over_A']<=1+1e-10
                        close(v['all_pair'],v['B']**2-v['energy'])
                        close(v['actual_displacement'],v['actual']['B'],atol=1e-10)
                        max_update=max(max_update,v['displacement_identity_relative_error'])
                        if j:
                            close(v['q'],1-v['cosine']*norms[j]/norms[j-1])
                            close(v['epsilon'],v['noise_projection']/norms[j-1]**2)
                            close(v['corrected'],v['q']+v['epsilon'])
                            max_identity=max(max_identity,abs(v['correction_identity_residual']))
                        if v['taylor_residual'] is not None:close(v['taylor_residual'],v['corrected']-v['previous_hvp_eta_kappa'])
                    assert sum(v['taylor_residual'] is not None for v in r)==4
                    for j in cfg['hvp_predecessor_gradients']:assert (d/f'probe-before-{j:03d}.pt').is_file()
                    recovery=torch.load(d/'recovery.pt',map_location='cpu',weights_only=False)
                    final=torch.load(d/'final.pt',map_location='cpu',weights_only=False)
                    assert tensor_hash(recovery['model'].values())==tensor_hash(final['model'].values())
                    origin=torch.load(ref/f'state-{cp:04d}.pt',map_location='cpu',weights_only=False)['model']
                    displacement=torch.cat([(final['model'][k].double()-v.double()).flatten() for k,v in origin.items()])
                    close(float(displacement.norm()),s['absolute_displacement'])
                    close(float(recovery['raw_stream']['total'].norm())/recovery['raw_stream']['A'],s['R'])
                    assert float((displacement-recovery['update_stream']['total']).abs().max())<1e-10
                    close(s['mean_rho'],np.mean([v['cosine'] for v in r[1:]]))
                    close(s['mean_full_rho'],np.mean([v['rho_full'] for v in r[1:]]))
                    close(s['loss_change'],s['final_loss']-r[0]['full_loss'])
                    if b==5000:
                        for v in r:
                            close(v['noise_energy'],0)
                            if v['cosine'] is not None:close(v['cosine'],v['rho_full'])
                        if cp+200<=2000:
                            reference=records[cp:cp+200]
                            errors={field:max(abs(a[field]-z[target]) for a,z in zip(r,reference)
                                if a[field] is not None and z[target] is not None)
                                for field,target in [('full_loss','loss'),('norm','norm'),('cosine','cosine')]}
                            full_control_replays.append(dict(seed=seed,checkpoint=cp,maximum_absolute_errors=errors))
                            assert max(errors.values())<1e-6,full_control_replays[-1]
                    moving_summaries.append(s)
    assert counts==dict(reference_updates=6000,frozen_draws=184320,continuation_updates=59400)
    assert manifest['completed']==counts
    assert len(moving_summaries)==297 and len(frozen_summaries)==1440
    receipt=verify(out,out/'inventory.json',strict=True) if args.inventory else None
    if receipt:assert receipt['verified'],receipt
    write_json(HERE/'results.json',dict(frozen=frozen_summaries,moving=moving_summaries))
    write_json(HERE/'verification.json',dict(status='passed',counts=counts,frozen_sequences=len(frozen_summaries),
         analysis_numpy=np.__version__,analysis_torch=torch.__version__,
         moving_forks=len(moving_summaries),source_config_sha256=file_hash(RUN/'config.json'),
         results_sha256=file_hash(HERE/'results.json'),
         production_inventory_sha256=file_hash(out/'inventory.json') if (out/'inventory.json').exists() else None,
         reconstruction=reconstruction,maximum_correction_identity_error=max_identity,
         full_control_replays=full_control_replays,
         maximum_update_roundoff_relative_error=max_update,transfer=receipt,
         note='Qualification of reconstruction is reported separately; approximation errors are results, not acceptance conditions.'))
    print(json.dumps(dict(status='passed',counts=counts,reconstruction=reconstruction,max_update_roundoff=max_update)))


if __name__=='__main__':main()
