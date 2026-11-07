"""Fixed SVDs of raw gradients and consecutive output/hidden displacements.

User correction: gradients are NOT differenced. All fits have rows448..800.
One extra row447 is included in Gram matrices solely for adjacent cosines.
"""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT/'runs/240-2026-11-07-mlp-hidden-change-svd'
RAW = RUN/'artifacts/spark/payload/worker'
ART = HERE/'artifacts/three-space-svd'
STREAMS = ['gradient', 'output', 'hidden', 'visible', 'null']
DIMENSIONS = dict(gradient=656810, output=50000, hidden=1000000, visible=50000, null=1000000)


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def cosine_from_gram(gram):
    norm = np.sqrt(np.maximum(np.diag(gram), 0))
    den = norm[1:]*norm[:-1]
    return np.divide(np.diag(gram, k=-1), den, out=np.full(len(den), np.nan), where=den>0)


def decompose(gram):
    eigen, left = torch.linalg.eigh(torch.from_numpy(gram[1:, 1:].copy()))
    eigen = eigen.flip(0); left = left.flip(1)
    assert float(eigen.min()) > -1e-11*float(eigen.max())
    eigen = eigen.clamp_min(0)
    assert float(eigen[9]) > 1e-14*float(eigen[0])
    return eigen.numpy(), left[:, :10], eigen[:10].sqrt()


class Data:
    def __init__(self):
        self.g = np.load(RAW/'gradients.npy', mmap_mode='r')
        self.f = np.load(RAW/'outputs.npy', mmap_mode='r')
        self.h = np.load(RAW/'hidden.npy', mmap_mode='r')
        self.heads = torch.load(RAW/'heads.pt', weights_only=False)
        assert self.g.shape == (355, 656810)
        assert self.f.shape == (355, 5000, 10)
        assert self.h.shape == (355, 5000, 200)
        assert self.heads['states'] == list(range(447,802))
        self.w = self.heads['weights'].double()
        self.b = self.heads['biases'].double()
        _, s, self.q = torch.linalg.svd(self.w[1], full_matrices=False)
        assert bool((s > s[0]*1e-12).all())
        self.p = self.q.T@self.q

    def blocks(self, stream):
        if stream in ['gradient', 'output']:
            raw = self.g if stream == 'gradient' else self.f.reshape(355,-1)
            for start in range(0, raw.shape[1], 25000):
                if stream == 'gradient':
                    block = torch.from_numpy(np.array(raw[:354,start:start+25000], dtype=np.float64))
                else:
                    level = torch.from_numpy(np.array(raw[:,start:start+25000], dtype=np.float64))
                    block = level[1:]-level[:-1]
                yield start, block
        else:
            width = 10 if stream == 'visible' else 200
            for start in range(0,5000,125):
                level = torch.from_numpy(np.array(self.h[:,start:start+125], dtype=np.float64))
                block = level[1:]-level[:-1]
                if stream == 'visible':
                    block = block@self.q.T
                elif stream == 'null':
                    block = block-(block@self.q.T)@self.q
                yield start*width, block.flatten(1)


def fit(data, name, gram):
    eigen, left, singular = decompose(gram)
    coefficients = (left*singular).numpy()
    basis = np.lib.format.open_memmap(ART/(name+'_basis.npy'), mode='w+', dtype=np.float64,
                                     shape=(10,DIMENSIONS[name]))
    projected = torch.zeros((353,10),dtype=torch.float64)
    orthogonal = torch.zeros((10,10),dtype=torch.float64)
    for start, block in data.blocks(name):
        pattern = (left.T@block[1:])/singular[:,None]
        basis[:,start:start+block.shape[1]] = pattern.numpy()
        projected += block[1:]@pattern.T
        orthogonal += pattern@pattern.T
    for k in range(10):
        if basis[k,np.argmax(np.abs(basis[k]))] < 0:
            basis[k] *= -1; coefficients[:,k] *= -1; projected[:,k] *= -1
    basis.flush()
    error = float(np.max(abs(projected.numpy()-coefficients)))
    assert error < 1e-9*max(1,float(np.max(abs(coefficients))))
    np.testing.assert_allclose(orthogonal.numpy(),np.eye(10),rtol=0,atol=1e-9)
    fractions = eigen/eigen.sum()
    reversals = coefficients[1:,0]*coefficients[:-1,0] < 0
    result = dict(shape=[353,DIMENSIONS[name]], singular_values=np.sqrt(eigen).tolist(),
                  energy_fractions=fractions.tolist(), total_energy=float(eigen.sum()),
                  top1=float(fractions[0]),top3=float(fractions[:3].sum()),top10=float(fractions[:10].sum()),
                  rank90=int(np.searchsorted(fractions.cumsum(),.90)+1),
                  rank95=int(np.searchsorted(fractions.cumsum(),.95)+1),
                  rank99=int(np.searchsorted(fractions.cumsum(),.99)+1),
                  participation_ratio=float(1/(fractions**2).sum()),
                  leading_sign_reversals=int(reversals.sum()),adjacent_coefficient_pairs=352,
                  projection_max_error=error,orthogonality_max_error=float((orthogonal-torch.eye(10)).abs().max()),
                  basis_sha256=sha(ART/(name+'_basis.npy')))
    return coefficients,result


def tests():
    # Known orthogonal visible/null directions, unequal energies, exact reversal.
    q=torch.tensor([[1.,0.,0.]],dtype=torch.float64)
    x=torch.tensor([[2.,1.,0.],[-2.,-1.,0.],[0.,0.,3.]],dtype=torch.float64)
    visible=(x@q.T)@q; null=x-visible
    torch.testing.assert_close(x@x.T,visible@visible.T+null@null.T)
    assert abs(float((null@q.T).abs().max()))<1e-14
    np.testing.assert_allclose(cosine_from_gram((x@x.T).numpy()),[-1.,0.],rtol=0,atol=1e-14)
    g=np.diag([0.,9.,4.,1.]);vals=np.linalg.eigvalsh(g[1:,1:])[::-1]
    np.testing.assert_array_equal(vals,[9.,4.,1.])
    # Exact finite head decomposition, with nonzero bias and interaction.
    h=torch.tensor([[1.,2.,3.]],dtype=torch.float64);dh=x[:1]
    w=q;dw=torch.tensor([[.5,.3,.2]],dtype=torch.float64);db=torch.tensor([.7])
    torch.testing.assert_close((h+dh)@(w+dw).T+db-h@w.T,
                               dh@w.T+h@dw.T+dh@dw.T+db)
    return ['orthogonal split and Gram identity','adjacent cosine indexing','known singular energies','finite head decomposition']


def main():
    torch.set_num_threads(4);start=time.monotonic();ART.mkdir(parents=True,exist_ok=True)
    checks=tests()
    receipt=json.loads((RUN/'retrieval.json').read_text());assert receipt['verified']
    inventory=json.loads((RAW.parent/'transfer_inventory.json').read_text())['files']
    for item in inventory:
        path=RAW.parent/item['path']
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
    data=Data();grams={}
    for name in STREAMS[:-1]:
        gram=torch.zeros((354,354),dtype=torch.float64)
        for _,block in data.blocks(name):gram+=block@block.T
        grams[name]=gram.numpy()
        print(json.dumps(dict(phase='gram',stream=name,elapsed=time.monotonic()-start)),flush=True)
    grams['null']=grams['hidden']-grams['visible']
    # Null Gram is also independently accumulated from its actual vectors in verification.
    coefficients={};results={}
    for name in STREAMS:
        coefficients[name],results[name]=fit(data,name,grams[name])
        print(json.dumps(dict(phase='svd',stream=name,top1=results[name]['top1'],top10=results[name]['top10'],elapsed=time.monotonic()-start)),flush=True)
    np.savez(ART/'grams.npz',**grams)
    prior=torch.load(HERE/'artifacts/svd-output-changes/patterns.pt',weights_only=False)
    np.testing.assert_allclose(coefficients['output'],prior['coefficients'].numpy(),rtol=1e-9,atol=1e-9)
    prior_output=ROOT/'runs/239-2026-11-07-mlp-output-change-alignment/artifacts/spark/payload/worker/outputs.pt'
    expected=torch.load(prior_output,weights_only=False)['outputs'].numpy()
    np.testing.assert_array_equal(data.f,expected)
    del expected
    with (HERE/'control_coherence_448_800.csv').open() as f:old=list(csv.DictReader(f))
    cosines={name:cosine_from_gram(grams[name]) for name in STREAMS}
    np.testing.assert_allclose(cosines['gradient'],[float(r['rho']) for r in old],rtol=0,atol=1e-12)
    np.testing.assert_allclose(cosines['output'],[float(r['output_change_cosine']) for r in old],rtol=0,atol=1e-12)
    boundaries={r['state']:r for r in map(json.loads,(RAW/'boundaries.jsonl').read_text().splitlines())}
    v1=prior['patterns'][0].double();metrics=[];identity_max=0.
    for j,t in enumerate(range(448,801)):
        i=j+1;h=torch.from_numpy(np.array(data.h[i],dtype=np.float64))
        dh=torch.from_numpy(np.array(data.h[i+1],dtype=np.float64))-h
        w=data.w[i];dw=data.w[i+1]-w;db=data.b[i+1]-data.b[i]
        terms=[dh@w.T,h@dw.T,dh@dw.T,db.expand(5000,10)]
        df=torch.from_numpy(np.array(data.f[i+1],dtype=np.float64))-torch.from_numpy(np.array(data.f[i],dtype=np.float64))
        residual=df-sum(terms);identity_max=max(identity_max,float(residual.abs().max()))
        projection=[float((term*v1).sum()) for term in terms]
        total_h=float(h.square().sum());visible_h=float((h@data.q.T).square().sum())
        _,_,qt=torch.linalg.svd(w,full_matrices=False)
        total_change=float(grams['hidden'][i,i]);visible_change=float(grams['visible'][i,i])
        row=dict(state=t,loss=boundaries[t]['loss'],hidden_activation_rms=(total_h/1000000)**.5,
                 hidden_activation_visible_share=visible_h/total_h,
                 hidden_change_visible_share=visible_change/total_change,
                 hidden_change_instant_head_visible_share=float((dh@qt.T).square().sum())/total_change,
                 reference_head_relative_drift=float((w-data.w[1]).norm()/data.w[1].norm()),
                 reference_null_head_leakage=float((w-w@data.p).norm()/w.norm()),
                 alpha_hidden=projection[0],alpha_head=projection[1],alpha_interaction=projection[2],alpha_bias=projection[3],
                 alpha_roundoff=float((residual*v1).sum()))
        for label,term in zip(['hidden','head','interaction','bias'],terms):
            row['output_term_'+label+'_squared_norm']=float(term.square().sum())
            row['output_term_'+label+'_dot_output']=float((term*df).sum())
        fixed_null_output=(dh-(dh@data.q.T)@data.q)@w.T
        row['output_fixed_null_squared_norm']=float(fixed_null_output.square().sum())
        row['output_fixed_null_dot_output']=float((fixed_null_output*df).sum())
        row['output_roundoff_dot_output']=float((residual*df).sum())
        for name in STREAMS:
            row[name+'_cosine']=float(cosines[name][j])
            row[name+'_norm']=float(np.sqrt(grams[name][i,i]))
            for k in range(10):row[name+f'_alpha_{k+1:02d}']=float(coefficients[name][j,k])
        metrics.append(row)
    with (HERE/'three_space_svd_values.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]),lineterminator='\n');writer.writeheader();writer.writerows(metrics)
    visible_total=np.trace(grams['visible'][1:,1:])/np.trace(grams['hidden'][1:,1:])
    correlations={f'{a}_{b}':float(np.corrcoef(cosines[a],cosines[b])[0,1]) for a,b in
                  [('gradient','output'),('gradient','hidden'),('output','hidden'),('output','visible'),('output','null')]}
    output_energy=sum(r['output_norm']**2 for r in metrics)
    alpha_energy=sum(r['output_alpha_01']**2 for r in metrics)
    attribution={key:dict(
        full_output_signed_projection_share=sum(r['output_term_'+key+'_dot_output'] for r in metrics)/output_energy,
        full_output_relative_rms=(sum(r['output_term_'+key+'_squared_norm'] for r in metrics)/output_energy)**.5,
        leading_coefficient_signed_projection_share=sum(r['alpha_'+key]*r['output_alpha_01'] for r in metrics)/alpha_energy)
        for key in ['hidden','head','interaction','bias']}
    summary=dict(status='numeric_checks_passed',corrected_gradient_definition='raw g_t, not gradient changes',
                 tests=checks,source_run=240,states=[448,800],fixed_head_state=448,projector_rank=10,
                 streams=results,correlations=correlations,hidden_change_visible_energy_share=float(visible_total),
                 activation_visible_share_range=[min(r['hidden_activation_visible_share'] for r in metrics),max(r['hidden_activation_visible_share'] for r in metrics)],
                 hidden_activation_rms_range=[min(r['hidden_activation_rms'] for r in metrics),max(r['hidden_activation_rms'] for r in metrics)],
                 max_reference_head_relative_drift=max(r['reference_head_relative_drift'] for r in metrics),
                 max_reference_null_head_leakage=max(r['reference_null_head_leakage'] for r in metrics),
                 decomposition_max_coordinate_roundoff=identity_max,
                 decomposition_max_alpha_roundoff=max(abs(r['alpha_roundoff']) for r in metrics),
                 attribution=attribution,
                 output_roundoff_signed_projection_share=sum(r['output_roundoff_dot_output'] for r in metrics)/output_energy,
                 fixed_null_through_current_head_relative_rms=(sum(r['output_fixed_null_squared_norm'] for r in metrics)/output_energy)**.5,
                 fixed_null_through_current_head_signed_projection_share=sum(r['output_fixed_null_dot_output'] for r in metrics)/output_energy,
                 checkpoints={str(t):metrics[t-448] for t in [465,473,474,485,512,521,522,529,683]},
                 sources={p.name:sha(p) for p in [RAW/'gradients.npy',RAW/'hidden.npy',RAW/'outputs.npy',RAW/'heads.pt',RAW/'boundaries.jsonl']},
                 generator_sha256=sha(Path(__file__)),csv_sha256=sha(HERE/'three_space_svd_values.csv'),elapsed_seconds=time.monotonic()-start)
    dump(HERE/'three_space_svd_results.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['streams','checkpoints','sources']},indent=2),flush=True)


if __name__=='__main__':main()
