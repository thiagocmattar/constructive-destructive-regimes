"""Independent NumPy checks of the earlier-layer SVD and finite transmission."""
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('early_hidden',HERE/'12_first_hidden_svd.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


def main():
    e.torch.set_num_threads(4)
    result=json.loads((HERE/'first_hidden_svd_results.json').read_text())
    assert result['generator_sha256']==e.m.sha(HERE/'12_first_hidden_svd.py')
    for path,h in result['sources'].items():assert e.m.sha(e.ROOT/path)==h
    assert result['csv_sha256']==e.m.sha(HERE/'first_hidden_svd_values.csv')
    with (HERE/'first_hidden_svd_values.csv').open() as f:rows=list(csv.DictReader(f))
    assert [int(r['state']) for r in rows]==list(range(448,801))
    data=e.Data();grams=np.load(e.ART/'grams.npz');checks={};selected=np.array([0,17,25,26,64,73,74,235,352])
    with threadpool_limits(limits=4):
        for name,raw in [('hidden1',data.h1),('preactivation2',data.z2)]:
            basis=np.load(e.ART/(name+'_basis.npy'),mmap_mode='r')
            assert e.m.sha(e.ART/(name+'_basis.npy'))==result['streams'][name]['basis_sha256']
            a=np.array([[float(r[name+f'_alpha_{k:02d}']) for k in range(1,11)] for r in rows])
            projected=np.zeros((353,10));ortho=np.zeros((10,10));norms=np.zeros(354);dots=np.zeros(353)
            residual=np.zeros(len(selected));selected_gram=np.zeros((len(selected),len(selected)))
            for start in range(0,5000,125):
                level=np.array(raw[:,start:start+125],dtype=np.float64)
                block=(level[1:]-level[:-1]).reshape(354,-1)
                v=np.asarray(basis[:,start*200:(start+125)*200])
                projected+=block[1:]@v.T;ortho+=v@v.T
                norms+=np.sum(block**2,axis=1);dots+=np.sum(block[1:]*block[:-1],axis=1)
                x=block[1:][selected];selected_gram+=x@x.T
                residual+=np.sum((x-a[selected]@v)**2,axis=1)
            np.testing.assert_allclose(projected,a,rtol=1e-8,atol=1e-8)
            np.testing.assert_allclose(ortho,np.eye(10),rtol=0,atol=1e-9)
            np.testing.assert_allclose(norms,np.diag(grams[name]),rtol=1e-10,atol=1e-9)
            np.testing.assert_allclose(dots/np.sqrt(norms[1:]*norms[:-1]),[float(r[name+'_cosine']) for r in rows],rtol=0,atol=1e-10)
            np.testing.assert_allclose(selected_gram,grams[name][1:,1:][np.ix_(selected,selected)],rtol=1e-10,atol=1e-8)
            np.testing.assert_allclose(residual,norms[1:][selected]-np.sum(a[selected]**2,axis=1),rtol=1e-8,atol=1e-8)
            np.testing.assert_allclose(np.sum(a*a,axis=0)/norms[1:].sum(),result['streams'][name]['energy_fractions'][:10],rtol=1e-10,atol=1e-12)
            assert all(basis[k,np.argmax(abs(basis[k]))]>0 for k in range(10))
            checks[name]=dict(coefficients=3530,cosines=353,reconstructions=9,max_projection_error=float(np.max(abs(projected-a))))
            print(json.dumps(dict(stream=name,**checks[name])),flush=True)
        for j in selected:
            i=j+1;h=np.array(data.h1[i],dtype=np.float64);dh=np.array(data.h1[i+1],dtype=np.float64)-h
            z=np.array(data.z2[i],dtype=np.float64);zn=np.array(data.z2[i+1],dtype=np.float64);dz=zn-z
            h2=np.array(data.h2[i],dtype=np.float64);dh2=np.array(data.h2[i+1],dtype=np.float64)-h2
            w=data.w[i].numpy();dw=data.w[i+1].numpy()-w;db=data.b[i+1].numpy()-data.b[i].numpy()
            slope=np.divide(np.tanh(zn)-np.tanh(z),dz,out=1-np.tanh(z)**2,where=dz!=0)
            terms=[dh@w.T,h@dw.T,dh@dw.T,np.broadcast_to(db,z.shape)]
            np.testing.assert_allclose(slope*dz,np.tanh(zn)-np.tanh(z),rtol=1e-12,atol=1e-14)
            for label,term in zip(['upstream','weights','interaction','bias'],terms):
                for prefix,part,delta in [('z',term,dz),('h',slope*term,dh2)]:
                    np.testing.assert_allclose(np.sum(part*delta),float(rows[j][prefix+'_'+label+'_dot']),rtol=1e-8,atol=1e-8)
                    np.testing.assert_allclose(np.sum(part**2),float(rows[j][prefix+'_'+label+'_squared_norm']),rtol=1e-8,atol=1e-8)
            assert np.max(abs(dz-sum(terms)))<1e-5
            assert np.max(abs(dh2-sum(slope*t for t in terms)))<1e-5
            assert np.max(abs(h2-np.tanh(z)))<1e-6
        # Independent temporal-subspace calculation; no mode sign matching needed.
        u={name:np.linalg.eigh(gram[1:,1:])[1][:,-10:] for name,gram in grams.items()}
        for a,b in [('hidden1','preactivation2'),('hidden1','hidden2'),('hidden1','output'),('preactivation2','hidden2')]:
            s=np.linalg.svd(u[a].T@u[b],compute_uv=False)
            np.testing.assert_allclose(s,result['temporal_subspaces'][a+'_'+b]['principal_cosines'],rtol=0,atol=1e-10)
            assert np.max(s)<1+1e-12
    for name,terms in result['attribution'].items():
        total=sum(terms[k]['signed_projection_share'] for k in ['upstream','weights','interaction','bias'])+terms['roundoff_signed_projection_share']
        np.testing.assert_allclose(total,1,rtol=0,atol=1e-11)
    for q in rows:
        assert 0<=float(q['tanh_displacement_gain'])<=1+1e-5
    audit=dict(status='passed',streams=checks,source_hashes_verified=True,selected_transmission_states=9,
               temporal_subspaces_verified=4,all_353_tanh_displacements_contract=True,
               attribution_adds_to_one=True,verifier_sha256=e.m.sha(Path(__file__)))
    e.m.dump(HERE/'first_hidden_independent_verification.json',audit)
    print('Independent earlier-hidden checks passed',flush=True)


if __name__=='__main__':main()
