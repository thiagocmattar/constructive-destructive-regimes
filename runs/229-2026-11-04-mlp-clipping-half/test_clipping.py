"""Focused CPU check of the exact reused clipping function and Adam placement."""
import ast,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'runs/223-2026-11-04-multispike-interventions/01_worker.py';tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='modify_gradients_and_decay');ns={'torch':torch,'flat':lambda xs:torch.cat([x.reshape(-1) for x in xs])};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns);clip=ns[node.name]
model=torch.nn.Linear(1,1);opt=torch.optim.Adam(model.parameters(),lr=.001,foreach=False,fused=False)
# Warm existing moments, then clip across separate weight and bias tensors.
for p in model.parameters():p.grad=torch.ones_like(p)*.2
opt.step();old={p:{k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in opt.state[p].items()} for p in model.parameters()}
for p,v in zip(model.parameters(),[3.,4.]):p.grad=torch.full_like(p,v)
raw=ns['flat']([p.grad for p in model.parameters()]).clone();factor,before,after=clip(model,raw,{'kind':'clip','factor':1.},.5,.001)
assert factor==.1 and torch.equal(raw,torch.tensor([3.,4.])) and torch.equal(before,after)
torch.testing.assert_close(ns['flat']([p.grad for p in model.parameters()]),torch.tensor([.3,.4]));assert abs(float(ns['flat']([p.grad for p in model.parameters()]).norm())-.5)<1e-7
expected={p:(.9*old[p]['exp_avg']+.1*p.grad,.999*old[p]['exp_avg_sq']+.001*p.grad.square()) for p in model.parameters()};opt.step()
for p,(mm,vv) in expected.items():torch.testing.assert_close(opt.state[p]['exp_avg'],mm);torch.testing.assert_close(opt.state[p]['exp_avg_sq'],vv);assert int(opt.state[p]['step'])==2
for value in [.1,0.]:
 for p in model.parameters():p.grad=torch.full_like(p,value)
 raw=ns['flat']([p.grad for p in model.parameters()]).clone();factor,_,_=clip(model,raw,{'kind':'clip','factor':1.},.5,.001);assert factor==1 and torch.equal(ns['flat']([p.grad for p in model.parameters()]),raw)
print(json.dumps(dict(status='passed',checks=['whole-model common factor','raw vector preserved','no parameter decay','warm moments consume clipped input','clock retained','inactive and zero gradient cases'])))
