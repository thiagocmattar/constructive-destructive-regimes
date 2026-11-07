"""Independent NumPy checks of projections, energy, reconstruction and attribution."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('spaces',HERE/'09_three_space_svd.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def main():
    m.torch.set_num_threads(4)
    result=json.loads((HERE/'three_space_svd_results.json').read_text())
    assert result['generator_sha256']==m.sha(HERE/'09_three_space_svd.py')
    assert result['csv_sha256']==m.sha(HERE/'three_space_svd_values.csv')
    for name,h in result['sources'].items():assert m.sha(m.RAW/name)==h
    with (HERE/'three_space_svd_values.csv').open() as f:rows=list(csv.DictReader(f))
    assert [int(r['state']) for r in rows]==list(range(448,801))
    data=m.Data();q=data.q.numpy();p=data.p.numpy()
    np.testing.assert_allclose(p@p,p,rtol=0,atol=1e-12)
    np.testing.assert_allclose(p,p.T,rtol=0,atol=1e-12)
    np.testing.assert_allclose(data.w[1].numpy()@(np.eye(200)-p),0,rtol=0,atol=1e-12)
    grams=np.load(m.ART/'grams.npz');checks={}
    selected=np.array([0,17,25,26,64,73,74,235,352])
    with threadpool_limits(limits=4):
        for name in m.STREAMS:
            basis=np.load(m.ART/(name+'_basis.npy'),mmap_mode='r')
            assert m.sha(m.ART/(name+'_basis.npy'))==result['streams'][name]['basis_sha256']
            coefficients=np.array([[float(r[name+f'_alpha_{k+1:02d}']) for k in range(10)] for r in rows])
            direct=np.zeros((353,10));orthogonal=np.zeros((10,10));energy=0.;residual=np.zeros(len(selected))
            adjacent=np.zeros(353);norms=np.zeros(354);selected_gram=np.zeros((len(selected),len(selected)))
            # Separate NumPy arithmetic, with direct reconstruction of selected full vectors.
            for start,block_tensor in data.blocks(name):
                block=block_tensor.numpy();v=np.asarray(basis[:,start:start+block.shape[1]])
                direct+=block[1:]@v.T;orthogonal+=v@v.T
                energy+=float(np.sum(block[1:]**2))
                norms+=np.sum(block**2,axis=1)
                adjacent+=np.sum(block[1:]*block[:-1],axis=1)
                residual+=np.sum((block[1:][selected]-coefficients[selected]@v)**2,axis=1)
                selected_gram+=block[1:][selected]@block[1:][selected].T
            projection_error=float(np.max(abs(direct-coefficients)))
            np.testing.assert_allclose(direct,coefficients,rtol=1e-8,atol=1e-8)
            np.testing.assert_allclose(orthogonal,np.eye(10),rtol=0,atol=1e-9)
            np.testing.assert_allclose(selected_gram,grams[name][1:,1:][np.ix_(selected,selected)],rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(norms,np.diag(grams[name]),rtol=1e-10,atol=1e-9)
            np.testing.assert_allclose(adjacent/np.sqrt(norms[1:]*norms[:-1]),[float(r[name+'_cosine']) for r in rows],rtol=0,atol=1e-10)
            np.testing.assert_allclose(residual,norms[1:][selected]-np.sum(coefficients[selected]**2,axis=1),rtol=1e-8,atol=1e-8)
            np.testing.assert_allclose(np.sum(coefficients**2,axis=0)/energy,result['streams'][name]['energy_fractions'][:10],rtol=1e-10,atol=1e-12)
            assert all(basis[k,np.argmax(abs(basis[k]))]>0 for k in range(10))
            checks[name]=dict(coefficients_verified=3530,cosines_verified=353,
                selected_full_reconstructions=len(selected),projection_max_error=projection_error,
                independent_top10_energy=float(np.sum(coefficients**2)/energy))
            print(json.dumps(dict(stream=name,**checks[name])),flush=True)
    native={r['state']:r for r in map(json.loads,(m.RAW/'boundaries.jsonl').read_text().splitlines())}
    np.testing.assert_allclose(np.sqrt(np.diag(grams['gradient'])[1:]),[native[t]['raw_gradient_norm'] for t in range(448,801)],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(grams['hidden'],grams['visible']+grams['null'],rtol=1e-12,atol=1e-10)
    alpha=np.array([float(r['output_alpha_01']) for r in rows])
    terms=np.array([[float(r[k]) for k in ['alpha_hidden','alpha_head','alpha_interaction','alpha_bias','alpha_roundoff']] for r in rows])
    np.testing.assert_allclose(terms.sum(axis=1),alpha,rtol=0,atol=1e-9)
    assert result['decomposition_max_coordinate_roundoff']<2e-6
    for j in selected:
        i=j+1;h=np.array(data.h[i],dtype=np.float64);dh=np.array(data.h[i+1],dtype=np.float64)-h
        w=data.w[i].numpy();dw=data.w[i+1].numpy()-w;db=data.b[i+1].numpy()-data.b[i].numpy()
        df=np.array(data.f[i+1],dtype=np.float64)-np.array(data.f[i],dtype=np.float64)
        for label,term in zip(['hidden','head','interaction','bias'],
                             [dh@w.T,h@dw.T,dh@dw.T,np.broadcast_to(db,(5000,10))]):
            np.testing.assert_allclose(np.sum(term*df),float(rows[j]['output_term_'+label+'_dot_output']),rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(np.sum(term**2),float(rows[j]['output_term_'+label+'_squared_norm']),rtol=1e-10,atol=1e-10)
    shares=sum(v['full_output_signed_projection_share'] for v in result['attribution'].values())
    np.testing.assert_allclose(shares+result['output_roundoff_signed_projection_share'],1,rtol=0,atol=1e-12)
    record=dict(status='passed',streams=checks,all_source_hashes_checked=True,
                projector_and_energy_split=True,raw_gradient_norms_all353=True,
                output_attribution_with_forward_roundoff=True,full_attribution_selected_states=len(selected),verifier_sha256=m.sha(Path(__file__)))
    m.dump(HERE/'three_space_svd_independent_verification.json',record)
    print('All independent checks passed',flush=True)


if __name__=='__main__':main()
