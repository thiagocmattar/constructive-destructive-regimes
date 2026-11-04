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
        new=selected_function(HERE/'01_worker.py','lr_at',{'cfg':dict(return_state=650,start=512,return_ratio=.75,rise_duration=70),'original_lr':old,'math':math})
        for arm in [dict(kind='cosine',ratio=.5,duration=32)]:
            p=torch.nn.Parameter(torch.tensor([1.,-2.]));q=torch.nn.Parameter(p.detach().clone())
            a=torch.optim.Adam([p],lr=.001,foreach=False,fused=False);b=torch.optim.Adam([q],lr=.001,foreach=False,fused=False)
            for state in range(800):
                lr=.001 if state<512 else new(arm,state-512,.001)
                expected=.001 if state<512 else old(arm,state-512,.001) if state<650 else .001*(.625-.125*math.cos(math.pi*min(state-650,70)/70))
                self.assertAlmostEqual(lr,expected,places=18)
                for opt,v,l in [(a,p,lr),(b,q,expected)]:
                    v.grad=2*v.detach();opt.param_groups[0]['lr']=l;opt.step();opt.param_groups[0]['lr']=.001
                self.assertTrue(torch.allclose(p,q,rtol=0,atol=1e-7));self.assertEqual(int(a.state[p]['step']),state+1)
            self.assertEqual(new(arm,137,.001),.0005);self.assertEqual(new(arm,138,.001),.0005)
            self.assertEqual(new(arm,173,.001),.000625);self.assertEqual(new(arm,208,.001),.00075)
            self.assertTrue(all(new(arm,k+1,.001)>=new(arm,k,.001) for k in range(138,288)))
if __name__=='__main__':unittest.main()
