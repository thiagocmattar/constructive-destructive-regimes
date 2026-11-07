import json
import torch
from diagnostics import output_alignment

x=torch.tensor([[[0.,0.]],[[1.,2.]],[[0.,0.]],[[0.,0.]]])
r=output_alignment(x,[447,448,449,450])
assert [q['state'] for q in r]==[448,449]
assert abs(r[0]['output_change_cosine']+1)<1e-14
assert r[1]['output_change_cosine'] is None
# Test full concatenation, rather than averaging per-example cosine.
x=torch.tensor([[[0.,0.],[0.,0.]],[[1.,0.],[0.,2.]],[[2.,0.],[0.,1.]]])
r=output_alignment(x,[0,1,2])[0]
assert abs(r['output_change_cosine']-(-1/(10**.5)))<1e-14
print(json.dumps(dict(status='passed',checks=['incoming/outgoing state indexing','exact reversal','zero displacement null','global concatenated cosine with unequal sample magnitudes'])))
