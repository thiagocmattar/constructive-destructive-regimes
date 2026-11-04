"""Independent CPU read-back of terminal checkpoint and exact replay."""
import json,torch,numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
base=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
p=HERE/'artifacts/spark/payload/worker'
def eq(a,b):
 if isinstance(a,torch.Tensor): assert torch.equal(a,b)
 elif isinstance(a,np.ndarray): assert np.array_equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a: eq(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):eq(x,y)
 else:assert a==b
if __name__=='__main__':
 a=torch.load(base/'qualification/direct/end.pt',map_location='cpu',weights_only=False)
 b=torch.load(p/'qualification2/end.pt',map_location='cpu',weights_only=False)
 for k in ['model','optimizer','previous_gradient','rng','current_gradient']:eq(a[k],b[k])
 end=torch.load(p/'global_clip1/end.pt',map_location='cpu',weights_only=False)
 assert end['completed_updates']==800
 clocks=sorted({float(v['step']) for v in end['optimizer']['state'].values()});assert clocks==[800.]
 last=json.loads((p/'global_clip1/boundaries.jsonl').read_text().splitlines()[-1]);assert abs(float(end['current_gradient'].double().norm())-last['raw_gradient_norm'])<1e-12
 result=dict(status='passed',independent_cpu_exact_replay=True,final_state=end['completed_updates'],optimizer_clocks=clocks,final_raw_gradient_norm_matches=True)
 (HERE/'verification_final.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
