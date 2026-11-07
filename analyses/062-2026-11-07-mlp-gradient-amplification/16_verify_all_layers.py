"""Independent verification of fixed inputs, first projection and actual steps."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('all_layers',HERE/'15_all_layer_svd.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


def main():
    e.torch.set_num_threads(4)
    r=json.loads((HERE/'all_layer_svd_results.json').read_text())
    assert r['generator_sha256']==e.m.sha(HERE/'15_all_layer_svd.py')
    for path,h in r['sources'].items():assert e.m.sha(e.ROOT/path)==h
    assert r['csv_sha256']==e.m.sha(HERE/'all_layer_svd_values.csv')
    with (HERE/'all_layer_svd_values.csv').open() as f:rows=list(csv.DictReader(f))
    assert [int(q['state']) for q in rows]==list(range(448,801))
    data=e.Data();input_audit=json.loads((e.RAW/'input_verification.json').read_text())
    assert hashlib.sha256(np.asarray(data.x).tobytes()).hexdigest()==input_audit['input_sha256']
    cache=e.torch.load(e.ROOT/'cache/cifar10-eos/data.pt',map_location='cpu',weights_only=True)
    np.testing.assert_array_equal(data.x,cache['x_train'].flatten(1).numpy());del cache
    assert input_audit['states_checked']==355 and input_audit['max_input_change']==0
    assert all(float(q['input_change_norm'])==0 and q['input_change_cosine']=='' for q in rows)
    assert r['input']['change_rank']==0 and r['input']['normalized_singular_energy_fractions'] is None
    grams=np.load(e.ART/'grams.npz');selected=np.array([0,17,25,26,64,73,74,235,352]);checks={}
    with threadpool_limits(limits=4):
        for name in ['preactivation1','parameter_step']:
            basis=np.load(e.ART/(name+'_basis.npy'),mmap_mode='r')
            assert e.m.sha(e.ART/(name+'_basis.npy'))==r['streams'][name]['basis_sha256']
            a=np.array([[float(q[name+f'_alpha_{k:02d}']) for k in range(1,11)] for q in rows])
            direct=np.zeros((353,10));ortho=np.zeros((10,10));norms=np.zeros(354);dots=np.zeros(353);residual=np.zeros(len(selected))
            sources=[(0,data.z.reshape(355,-1))] if name=='preactivation1' else [(0,data.w.reshape(355,-1)),(614400,data.b)]
            for offset,raw in sources:
                for start in range(0,raw.shape[1],25000):
                    level=np.array(raw[:,start:start+25000],dtype=np.float64);block=level[1:]-level[:-1]
                    v=np.asarray(basis[:,offset+start:offset+start+block.shape[1]])
                    direct+=block[1:]@v.T;ortho+=v@v.T;norms+=np.sum(block**2,axis=1)
                    dots+=np.sum(block[1:]*block[:-1],axis=1)
                    residual+=np.sum((block[1:][selected]-a[selected]@v)**2,axis=1)
            np.testing.assert_allclose(direct,a,rtol=1e-8,atol=1e-8)
            np.testing.assert_allclose(ortho,np.eye(10),rtol=0,atol=1e-9)
            np.testing.assert_allclose(norms,np.diag(grams[name]),rtol=1e-10,atol=1e-8)
            np.testing.assert_allclose(dots/np.sqrt(norms[1:]*norms[:-1]),[float(q[name+'_cosine']) for q in rows],rtol=0,atol=1e-10)
            np.testing.assert_allclose(residual,norms[1:][selected]-np.sum(a[selected]**2,axis=1),rtol=1e-8,atol=1e-8)
            np.testing.assert_allclose(np.sum(a*a,axis=0)/norms[1:].sum(),r['streams'][name]['energy_fractions'][:10],rtol=1e-10,atol=1e-12)
            assert all(basis[k,np.argmax(abs(basis[k]))]>0 for k in range(10))
            checks[name]=dict(coefficients=3530,cosines=353,reconstructions=9,max_projection_error=float(np.max(abs(direct-a))))
            print(json.dumps(dict(stream=name,**checks[name])),flush=True)
        x=np.array(data.x,dtype=np.float64)
        for j in selected:
            i=j+1;w=np.array(data.w[i],dtype=np.float64);dw=np.array(data.w[i+1],dtype=np.float64)-w
            b=np.array(data.b[i],dtype=np.float64);db=np.array(data.b[i+1],dtype=np.float64)-b
            z=np.array(data.z[i],dtype=np.float64);zn=np.array(data.z[i+1],dtype=np.float64);dz=zn-z
            h=np.array(data.h[i],dtype=np.float64);dh=np.array(data.h[i+1],dtype=np.float64)-h
            slope=np.divide(np.tanh(zn)-np.tanh(z),dz,out=1-np.tanh(z)**2,where=dz!=0)
            wt=x@dw.T;bt=np.broadcast_to(db,z.shape)
            for label,term in [('weights',wt),('bias',bt)]:
                for prefix,delta,part in [('z',dz,term),('h',dh,slope*term)]:
                    np.testing.assert_allclose(np.sum(delta*part),float(rows[j][prefix+'_'+label+'_dot']),rtol=1e-8,atol=1e-8)
                    np.testing.assert_allclose(np.sum(part**2),float(rows[j][prefix+'_'+label+'_squared_norm']),rtol=1e-8,atol=1e-8)
            assert np.max(abs(dz-wt-bt))<3e-4
            assert np.max(abs(dh-slope*(wt+bt)))<3e-4
            assert np.max(abs(h-np.tanh(z)))<1e-6
            np.testing.assert_allclose(np.mean(slope>.5),float(rows[j]['secant_gt_half_coordinate_fraction']),rtol=0,atol=1e-6)
            np.testing.assert_allclose(np.sum(dh[slope>.5]**2)/np.sum(dh**2),float(rows[j]['h1_change_energy_on_secant_gt_half']),rtol=0,atol=1e-6)
        u={name:np.linalg.eigh(g[1:,1:])[1][:,-10:] for name,g in grams.items()}
        for a,b in [('parameter_step','preactivation1'),('preactivation1','hidden1'),('preactivation1','output')]:
            s=np.linalg.svd(u[a].T@u[b],compute_uv=False)
            np.testing.assert_allclose(s,r['temporal_subspaces'][a+'_'+b]['principal_cosines'],rtol=0,atol=1e-10)
    for q in rows:
        np.testing.assert_allclose(float(q['parameter_step_norm'])**2,float(q['first_weight_increment_norm'])**2+float(q['first_bias_increment_norm'])**2,rtol=1e-10,atol=1e-12)
        assert 0<=float(q['first_tanh_displacement_gain'])<=1+1e-5
    for values in r['attribution'].values():
        np.testing.assert_allclose(values['weights']['signed_projection_share']+values['bias']['signed_projection_share']+values['roundoff_signed_projection_share'],1,rtol=0,atol=1e-10)
    audit=dict(status='passed',streams=checks,input_matches_pinned_cache=True,input_checks=355,
        input_zero_cosine_undefined=True,selected_transmission_states=9,temporal_subspaces_verified=3,
        actual_increment_norms_all353=True,attribution_adds_to_one=True,verifier_sha256=e.m.sha(Path(__file__)))
    e.m.dump(HERE/'all_layer_independent_verification.json',audit)
    print('All independent all-layer checks passed',flush=True)


if __name__=='__main__':main()
