import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('temporal_analysis',Path(__file__).resolve().parents[1]/'analyses/005-2026-10-04-temporal-lag-profiles/03_analyze.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_uncentered_signed_lags_pair_counts_and_energy_identity():
    # Four identical maps; alternating trajectory and a shared mean component.
    g=np.array([[1,1],[-1,1],[1,1],[-1,1]],dtype=float)
    gram=np.repeat((g@g.T)[None],4,axis=0)
    cosine,cs,weighted,counts=module.profiles(gram,np.linalg.norm(g,axis=1))
    assert counts==[3,2,1]
    np.testing.assert_allclose(cs,np.tile([0,1,0],(4,1)),atol=1e-15)
    np.testing.assert_allclose(2*weighted.sum(1),np.sum(g.sum(0)**2)-np.sum(g*g))
    # Excluding the final row changes coverage and must recompute the denominator.
    _,short,_,counts=module.profiles(gram[:,:-1,:-1],np.linalg.norm(g[:-1],axis=1))
    assert counts==[2,1]
    np.testing.assert_allclose(short,np.tile([0,1],(4,1)),atol=1e-15)

def test_common_origins_keep_coverage_when_window_behavior_changes():
    g=np.array([[1,0]]*4+[[-1,0]]*4,dtype=float)
    gram=np.repeat((g@g.T)[None],4,axis=0)
    cosine,all_pairs,_,_=module.profiles(gram,np.linalg.norm(g,axis=1))
    fixed,counts=module.common_origins(cosine,4)
    assert counts==[4]*4
    np.testing.assert_allclose(fixed,np.tile([.5,0,-.5,-1],(4,1)))
    np.testing.assert_allclose(all_pairs[:,0],5/7)
    np.testing.assert_allclose(fixed[:,-1],all_pairs[:,3])
