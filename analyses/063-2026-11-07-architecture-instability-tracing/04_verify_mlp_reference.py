"""Verify the standardized method reproduces the previously audited MLP trace."""
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'062-2026-11-07-mlp-gradient-amplification'
s=importlib.util.spec_from_file_location('reduce',HERE/'01_reduce.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
reference=json.loads((OLD/'all_layer_svd_results.json').read_text())
grams=np.load(OLD/'artifacts/all-layer-svd/grams.npz')
with (OLD/'all_layer_svd_values.csv').open() as f:values=list(csv.DictReader(f))
checks={}
with threadpool_limits(limits=4):
    for name in ['preactivation1','hidden1','preactivation2','hidden2','output']:
        fit=r.decompose(grams[name]);rho=r.cosine(grams[name]);ref=reference['streams'][name]
        np.testing.assert_allclose(fit['fractions'],ref['energy_fractions'],rtol=1e-5,atol=1e-12)
        old=np.array([float(x[name+'_cosine']) for x in values]);np.testing.assert_allclose(rho,old,rtol=1e-10,atol=1e-10)
        assert int(np.sum(fit['coeff'][1:,0]*fit['coeff'][:-1,0]<0))==ref['leading_sign_reversals']
        checks[name]=dict(cosines=len(rho),singular_energies=len(fit['fractions']),max_cosine_error=float(np.max(abs(rho-old))))
report=dict(status='passed',checks=checks,script_sha256=r.sha(Path(__file__)),method_generator_sha256=r.sha(HERE/'01_reduce.py'),
    source_hashes={str(p):r.sha(p) for p in [OLD/'all_layer_svd_results.json',OLD/'artifacts/all-layer-svd/grams.npz',OLD/'all_layer_svd_values.csv']})
r.dump(HERE/'mlp_reference_verification.json',report)
print(json.dumps(report['checks'],indent=2))
