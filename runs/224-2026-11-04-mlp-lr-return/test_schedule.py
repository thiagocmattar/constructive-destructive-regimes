import ast,math,unittest
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent

def selected_function(path,name,scope):
    node=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]

class ScheduleTest(unittest.TestCase):
    def test_schedule_and_adam_clock(self):
        old=selected_function(HERE.parent/'223-2026-11-04-multispike-interventions/01_worker.py','lr_at',{'math':math})
        new=selected_function(HERE/'01_worker.py','lr_at',{'cfg':dict(return_state=650,start=512,return_ratio=.75),'original_lr':old})
        for arm in [dict(kind='fixed',ratio=.5),dict(kind='cosine',ratio=.5,duration=32)]:
            p=torch.nn.Parameter(torch.tensor([1.,-2.]));q=torch.nn.Parameter(p.detach().clone())
            a=torch.optim.Adam([p],lr=.001,foreach=False,fused=False);b=torch.optim.Adam([q],lr=.001,foreach=False,fused=False)
            for state in range(800):
                lr=.001 if state<512 else new(arm,state-512,.001)
                expected=.001 if state<512 else .00075 if state>=650 else old(arm,state-512,.001)
                self.assertEqual(lr,expected)
                for opt,v,l in [(a,p,lr),(b,q,expected)]:
                    v.grad=2*v.detach();opt.param_groups[0]['lr']=l;opt.step();opt.param_groups[0]['lr']=.001
                self.assertTrue(torch.equal(p,q));self.assertEqual(int(a.state[p]['step']),state+1)
            self.assertEqual(new(arm,137,.001),.0005);self.assertEqual(new(arm,138,.001),.00075)
if __name__=='__main__':unittest.main()
