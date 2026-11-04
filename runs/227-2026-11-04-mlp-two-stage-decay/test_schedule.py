import ast,json,math,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
def function(path,name,scope):
    node=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope);return scope[name]
class ScheduleTest(unittest.TestCase):
    def test_two_cosines(self):
        cfg=json.loads((HERE/'config.json').read_text());arm=cfg['arms'][0]
        original=function(HERE.parent/'223-2026-11-04-multispike-interventions/01_worker.py','lr_at',{'math':math})
        actual=function(HERE/'01_worker.py','lr_at',{'math':math,'cfg':cfg,'original_lr':original})
        values=[actual(arm,t-512,.001) for t in range(512,800)]
        for t,v in zip(range(512,800),values):
            expected=.001*(1-.125*(1-math.cos(math.pi*min(t-512,32)/32))) if t<650 else .001*(.75-.125*(1-math.cos(math.pi*min(t-650,32)/32)))
            self.assertAlmostEqual(v,expected,places=18)
        for t,v in [(512,.001),(528,.000875),(544,.00075),(650,.00075),(666,.000625),(682,.0005),(799,.0005)]:self.assertEqual(values[t-512],v)
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
if __name__=='__main__':unittest.main()
