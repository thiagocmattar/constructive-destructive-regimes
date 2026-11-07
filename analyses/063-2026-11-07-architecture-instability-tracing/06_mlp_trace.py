"""Adapt the retained full-data MLP Grams to the common architecture report."""
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=HERE.parent/'062-2026-11-07-mlp-gradient-amplification'
spec=importlib.util.spec_from_file_location('common',HERE/'01_reduce.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))

def main():
    source_paths=[OLD/'all_layer_svd_results.json',OLD/'all_layer_svd_values.csv',
        OLD/'three_space_svd_results.json',OLD/'three_space_svd_values.csv',
        OLD/'artifacts/all-layer-svd/grams.npz',OLD/'artifacts/three-space-svd/grams.npz',
        OLD/'all_layer_independent_verification.json',HERE/'mlp_reference_verification.json',HERE/'01_reduce.py']
    previous=json.loads(source_paths[0].read_text());gradient_previous=json.loads(source_paths[2].read_text())
    audited=json.loads((HERE/'mlp_reference_verification.json').read_text())
    assert audited['status']=='passed'
    for path,digest in audited['source_hashes'].items():assert r.sha(Path(path))==digest
    assert r.sha(source_paths[1])==previous['csv_sha256']
    assert r.sha(source_paths[3])==gradient_previous['csv_sha256']
    assert json.loads(source_paths[6].read_text())['status']=='passed'
    assert previous['input']['exactly_constant'] and previous['input']['states_checked']==355
    values=read_csv(source_paths[1]);gvalues=read_csv(source_paths[3])
    times=np.array([int(q['state']) for q in values]);np.testing.assert_array_equal(times,np.arange(448,801))
    np.testing.assert_array_equal(times,[int(q['state']) for q in gvalues])
    np.testing.assert_array_equal([float(q['loss']) for q in values],[float(q['loss']) for q in gvalues])
    names=['preactivation1','hidden1','preactivation2','hidden2','output']
    all_grams=np.load(source_paths[4]);gradient_grams=np.load(source_paths[5])
    grams={n:all_grams[n] for n in names};grams['raw_gradient']=gradient_grams['gradient']
    np.testing.assert_array_equal(grams['output'],gradient_grams['output'])
    with threadpool_limits(limits=4):fits={n:r.decompose(g) for n,g in grams.items()}
    output=fits['output'];out_rho=r.cosine(grams['output']);records=[];curves={};checks={}
    for i,(name,g) in enumerate(grams.items()):
        raw=name=='raw_gradient';old_name='gradient' if raw else name
        old_rows=gvalues if raw else values
        reference=(gradient_previous if raw else previous)['streams'][old_name]
        fit=fits[name];rho=r.cosine(g);norm=np.sqrt(np.diag(g)[1:]);alpha=fit['coeff'][:,0]
        shape=[656810] if raw else [5000,10 if name=='output' else 200];p=int(np.prod(shape))
        np.testing.assert_allclose(rho,[float(q[old_name+'_cosine']) for q in old_rows],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(norm,[float(q[old_name+'_norm']) for q in old_rows],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(fit['fractions'],reference['energy_fractions'],rtol=1e-5,atol=1e-12)
        old_alpha=np.array([float(q[old_name+'_alpha_01']) for q in old_rows])
        sign=1 if alpha@old_alpha>=0 else -1
        np.testing.assert_allclose(alpha,sign*old_alpha,rtol=1e-7,atol=1e-8)
        rms=norm/np.sqrt(p);k=min(10,fit['rank'],output['rank'])
        overlap=float(np.sum((fit['u'][:,:k].T@output['u'][:,:k])**2)/k)
        record=dict(index=i,name=name,shape=shape,coordinates=p,negative=int(np.sum(rho<0)),defined=len(times),
            top1=float(fit['fractions'][0]),top10=float(fit['fractions'][:10].sum()),
            rank95=int(np.searchsorted(np.cumsum(fit['fractions']),.95)+1),
            rank99=int(np.searchsorted(np.cumsum(fit['fractions']),.99)+1),
            leading_sign_flips=int(np.sum(alpha[1:]*alpha[:-1]<0)),
            correlation_with_output=float(np.corrcoef(rho,out_rho)[0,1]),temporal_top10_overlap=overlap,
            change_rms_min=float(rms.min()),change_rms_max=float(rms.max()),
            vector_kind='raw gradient' if raw else 'representation displacement')
        records.append(record);curves[name]=dict(rho=rho.tolist(),change_rms=rms.tolist(),alpha=alpha.tolist(),spectrum=fit['fractions'].tolist())
        checks[name]=dict(cosines=len(rho),norms=len(norm),singular_energies=len(fit['fractions']),
            leading_coefficients=len(alpha),sign_relative_to_original=sign,
            alpha_max_error=float(np.max(abs(alpha-sign*old_alpha))))
    np.testing.assert_allclose(records[0]['temporal_top10_overlap'],previous['temporal_subspaces']['preactivation1_output']['overlap'],rtol=1e-10)
    result=dict(case='mlp',states=times.tolist(),stages=records,curves=curves,
        boundaries=[dict(state=int(q['state']),loss=float(q['loss']),probe_loss=float(q['loss'])) for q in values],
        input=previous['input'],coverage=dict(training_images=5000,diagnostic_images=5000,diagnostic_set_equals_objective=True),
        contract=dict(output='raw linear outputs',loss='mean per-image half squared error against one-hot labels',
            gradient='full-objective raw g_t, not differences',architecture='3072-200-200-10, tanh after first two linear layers',
            svd='uncentered, no row normalization, fixed full window; largest absolute temporal coefficient positive',
            precision='retained native FP32 captures; exact FP64 temporal Grams and eigendecomposition'),
        source_runs=[240,241,242],historical_exact=True,
        sources={p.relative_to(ROOT).as_posix():r.sha(p) for p in source_paths},generator_sha256=r.sha(Path(__file__)),
        verification=dict(status='passed',streams=checks,fixed_inputs=True,full_objective_loss_matches=True,
            output_grams_identical=True,first_layer_output_overlap_matches=True))
    with (HERE/'mlp_stages.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    r.dump(HERE/'mlp_results.json',result)
    print(json.dumps(dict(status='passed',states=len(times),streams=len(records),checks=checks),indent=2))

if __name__=='__main__':main()
