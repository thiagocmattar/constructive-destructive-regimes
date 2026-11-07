"""Mathematical and observer mechanics checks; not scientific evidence."""
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import diagnostics as d

torch.manual_seed(7)
model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.ReLU(inplace=True),torch.nn.Linear(4,2))
x=torch.randn(5,3); expected=model(x).detach().clone(); snapshots={}
with d.Capture(model,'resnet',lambda name,a:snapshots.setdefault(name,a)):
    actual=model(x)
assert torch.equal(expected,actual)
assert np.any(snapshots['1.call0.input']<0) and np.all(snapshots['1.call0.output']>=0)
with tempfile.TemporaryDirectory() as folder:
    p=Path(folder)/'data.npy'; a=np.random.default_rng(7).normal(size=(8,5,4)).astype(np.float32);np.save(p,a)
    for raw in [False,True]:
        gram,levels=d.gram_reduce(p,raw); b=a.astype(np.float64).reshape(8,-1)
        v=b if raw else np.diff(b,axis=0)
        np.testing.assert_allclose(gram,v@v.T)
        np.testing.assert_allclose(levels,np.sum(b*b,axis=1))
        u,s,_=np.linalg.svd(v[1:],full_matrices=False)
        eig,vec=np.linalg.eigh(gram[1:,1:]); order=np.argsort(eig)[::-1]
        np.testing.assert_allclose(eig[order][:len(s)],s*s,atol=1e-12)
        assert abs(np.trace(vec[:,order[:2]].T@(u[:,:2]@u[:,:2].T)@vec[:,order[:2]])/2-1)<1e-12
    np.save(p,np.zeros_like(a));g,_=d.gram_reduce(p);assert np.count_nonzero(g)==0
print(json.dumps(dict(status='passed',checks=['observer_forward_parity','inplace_snapshot','full_gram','raw_gradient_not_differenced','svd_equivalence','temporal_subspace','zero_motion'])))
