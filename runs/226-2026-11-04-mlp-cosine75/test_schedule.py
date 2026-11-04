import ast,json,math,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class ScheduleTest(unittest.TestCase):
    def test_decay_and_hold(self):
        cfg=json.loads((HERE/'config.json').read_text());arm=cfg['arms'][0]
        source=HERE.parent/'223-2026-11-04-multispike-interventions/01_worker.py'
        node=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='lr_at')
        scope={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        values=[scope['lr_at'](arm,state-cfg['start'],cfg['base_lr']) for state in range(512,800)]
        for j,v in enumerate(values):self.assertAlmostEqual(v,.001*(1-.125*(1-math.cos(math.pi*min(j,32)/32))),places=18)
        self.assertEqual(values[0],.001);self.assertEqual(values[16],.000875)
        self.assertTrue(all(v==.00075 for v in values[32:]));self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
if __name__=='__main__':unittest.main()
