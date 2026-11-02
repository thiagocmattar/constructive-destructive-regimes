"""Check fixed-endpoint indexing, orientation, zero norms and source matching."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/212-2026-11-02-eos-backward-anchors'
spec=importlib.util.spec_from_file_location('backward212',RUN/'01_worker.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
h=np.array([[1,0],[0,1],[-1,-1],[1,1]],dtype=np.float32)
r=m.backward_rows(h,100,torch.device('cpu'))
assert [x['reference_state'] for x in r]==[103,103,103]
assert [x['compared_state'] for x in r]==[102,101,100]
np.testing.assert_allclose([x['cosine'] for x in r],[-1,2**-.5,2**-.5],atol=1e-15)
z=m.backward_rows(np.array([[0,0],[1,0]],dtype=np.float32),7,torch.device('cpu'))
assert z[0]['cosine'] is None and not z[0]['defined']
z=m.backward_rows(np.array([[1,0],[0,0]],dtype=np.float32),7,torch.device('cpu'))
assert z[0]['cosine'] is None and not z[0]['defined']
result=dict(status='passed',checks=['fixed endpoint while earlier index decreases','hand-computed nonparallel/opposite cosines',
    'zero earlier gradient undefined','zero endpoint gradient undefined'])
(Path(__file__).parent/'local_tests.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(result))
