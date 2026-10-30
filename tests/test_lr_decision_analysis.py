"""Independent ridge reference and held-fold transform isolation."""
import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'analyses/030-2026-10-30-adam-lr-decisions/06_analyze.py'
spec=importlib.util.spec_from_file_location('h024_analysis',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_ridge_matches_augmented_least_squares_and_ignores_held_targets():
    rng=np.random.default_rng(24);x=rng.normal(size=(18,9));y=rng.normal(size=18)
    train=np.arange(12);test=np.arange(12,18)
    actual=module.ridge_predict(x,y,train,test)
    mean=x[train].mean(0);sd=x[train].std(0)
    a=(x[train]-mean)/sd;b=(x[test]-mean)/sd
    augmented=np.vstack([a,np.sqrt(10)*np.eye(9)])
    target=np.r_[y[train]-y[train].mean(),np.zeros(9)]
    beta=np.linalg.lstsq(augmented,target,rcond=None)[0]
    np.testing.assert_allclose(actual,b@beta+y[train].mean(),atol=1e-12)
    y[test]+=1000
    np.testing.assert_array_equal(actual,module.ridge_predict(x,y,train,test))
