"""Independent synthetic checks for all-row inactivity and reductions."""
import importlib.util,json
from pathlib import Path
import numpy as np
import torch
p=Path(__file__).with_name('01_measure.py');spec=importlib.util.spec_from_file_location('measure',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3,3),torch.nn.Tanh(),torch.nn.Linear(3,3),torch.nn.Tanh(),torch.nn.Linear(3,2)).cuda()
c=m.Collector(model,'mlp')
x=np.array([[0,1e-4,-3],[0,-1e-4,0],[0,0,1]],dtype=np.float32)
for block in [x[:2],x[2:]]:c.capture('2',torch.from_numpy(block).cuda())
rows,a=c.export();r=rows[0]
np.testing.assert_array_equal(a['0_maximum'],np.abs(x).max(0))
np.testing.assert_allclose(a['0_rms'],np.sqrt(np.square(x.astype(float)).mean(0)),rtol=1e-14)
assert r['dead_zero_fraction']==1/3 and r['zero_entry_fraction']==5/9
assert r['dead_absolute_fractions'][3]==2/3
assert r['threshold_entry_fractions']==[(np.abs(x)<=e).mean() for e in m.CFG['absolute_thresholds']]
assert r['top10pct_energy']>.99 and r['effective_unit_fraction']<.334
c.reset();c.capture('2',torch.zeros((7,3),device='cuda'));r,_=c.export()
assert r[0]['dead_zero_fraction']==1 and r[0]['effective_unit_fraction'] is None
c.close();print(json.dumps({'status':'passed','checks':['max vs entry sparsity','signed values','threshold equality','streamed moments','energy concentration','zero denominators']}))
