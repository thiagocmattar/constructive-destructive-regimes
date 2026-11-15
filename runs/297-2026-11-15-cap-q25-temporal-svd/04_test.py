"""Check spectral energy and cross-arm right-subspace projection independently."""
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('reducer',Path(__file__).with_name('02_reduce.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
rng=np.random.default_rng(42)
A=rng.normal(size=(16,37));B=rng.normal(size=(16,37))
result=r.compare(A@A.T,B@B.T,A@B.T,37)
u,s,v=np.linalg.svd(A,full_matrices=False);k=result['control']['rank95']
assert np.allclose(result['control']['eigenvalues'],s*s)
assert np.isclose(result['comparison']['cap_energy_inside_baseline95'],np.linalg.norm(B@v[:k].T)**2)
same=r.compare(A@A.T,4*A@A.T,2*A@A.T,37)
assert same['comparison']['rank95_change']==0
assert np.isclose(same['comparison']['energy_ratio'],4)
assert np.isclose(same['comparison']['spatial_subspace_overlap'],1)
# Attenuating a dominant direction raises effective rank without adding a direction.
A=np.diag([10.,1.,1.]);B=np.eye(3)
change=r.compare(A@A.T,B@B.T,A@B.T,3)
assert change['comparison']['rank95_change']>0
assert change['comparison']['energy_ratio']<1
zero,_,_=r.spectrum(np.zeros((3,3)),5);assert zero['rank95'] is None
print('PASS: direct SVD, spatial projection, scale invariance, redistribution, zero energy')
