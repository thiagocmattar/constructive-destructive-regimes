"""Focused native-Adam clock and replay tests; no experiment data required."""
import importlib.util,tempfile
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('w',Path(__file__).with_name('01_worker.py'));w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
class Task:
    def __init__(self):
        self.model=torch.nn.Linear(2,1,bias=False);self.model.weight.data.copy_(torch.tensor([[.7,-.3]]));self.data={}
    def gradient(self):
        self.model.zero_grad();loss=(self.model.weight**2*torch.tensor([[1.,3.]])).sum()/2;loss.backward();return float(loss),self.model.weight.grad.flatten().detach().clone()
def context():
    t=Task();o=torch.optim.Adam(t.model.parameters(),lr=.01,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    return dict(task=t,opt=o,case='test',base=.01,state=0,fork=5,previous_gradient=None,threshold=0,identity={})
def test_native_clocks():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        for arm in [dict(id='half',kind='fixed',ratio=.5),dict(id='cos',kind='cosine',ratio=.5,duration=32)]:
            c=context();d=context()
            control=dict(id='unchanged',kind='fixed',ratio=1.)
            w.run(c,control,root/(arm['id']+'prefix'),5,lambda:False)
            for i in range(5):d['task'].gradient();d['opt'].step()
            w.run(c,arm,root/arm['id'],40,lambda:False)
            for i in range(5,40):
                d['task'].gradient();d['opt'].param_groups[0]['lr']=w.lr_at(arm,i-5,.01);d['opt'].step()
            assert torch.equal(c['task'].model.weight,d['task'].model.weight)
            rs=w.rows(root/arm['id']/'boundaries.jsonl')
            assert len(rs)==36 and rs[0]['state']==5 and rs[-1]['state']==40 and rs[-1]['applied_lr'] is None
            assert rs[32]['applied_lr']==.005
            assert rs[0]['applied_lr']==(.005 if arm['kind']=='fixed' else .01)
            assert int(c['opt'].state[c['task'].model.weight]['step'])==40
            assert c['opt'].param_groups[0]['lr']==.01
if __name__=='__main__':test_native_clocks();print('Native Adam clocks and floor hold: passed')
