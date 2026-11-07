"""Replay three recorded updates from recovery544 to audit failure547.

The original worker evidence is never overwritten; no new treatment is run.
"""
import importlib.util
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('remote234',HERE/'02_remote.py')
r=importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
def remote(code):
    try:
        return r.remote(code)
    except subprocess.CalledProcessError as error:
        print(error.stderr)
        raise

result=remote(f'''
import os,subprocess,json
env=dict(os.environ,PYTHONPATH={r.PYTHONPATH!r})
code={'''
import hashlib,importlib.util,json,math
from pathlib import Path
root=Path.cwd()
parent=root/'runs/223-2026-11-04-multispike-interventions'
run=root/'runs/234-2026-11-07-mlp-gd-learning-rates'
spec=importlib.util.spec_from_file_location('w223',parent/'01_worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
for item in json.loads((run/'input_manifest.json').read_text())['files']:
 p=root/item['path'];assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256']
w.configure();w.torch.use_deterministic_algorithms(True);w.torch.backends.cudnn.deterministic=True
c=w.prepare('mlp',json.loads((parent/'config.json').read_text()))
checkpoint=run/'artifacts/attempt-01/worker/gd_lr0p08/recovery.pt'
p=w.torch.load(checkpoint,map_location='cpu',weights_only=False)
assert p['completed_updates']==544 and p['current_gradient'] is None
c['task'].model.load_state_dict(p['model']);w.restore_rng(p['rng'])
optimizer=w.torch.optim.SGD(c['task'].model.parameters(),lr=.08,momentum=0,weight_decay=0,foreach=False)
optimizer.load_state_dict(p['optimizer'])
records={r['state']:r for r in w.rows(checkpoint.parent/'boundaries.jsonl')}
for state in range(544,547):
 loss,g=c['task'].gradient()
 assert loss==records[state]['loss'] and float(g.double().norm())==records[state]['raw_gradient_norm']
 optimizer.step()
loss,g=c['task'].gradient()
parameters=w.flat(c['task'].model.parameters()).detach()
result=dict(status='verified_nonfinite_boundary',state=547,
 checkpoint_sha256=w.sha(checkpoint),loss=loss if math.isfinite(loss) else str(loss),
 gradient_nan_count=int(w.torch.isnan(g).sum()),gradient_inf_count=int(w.torch.isinf(g).sum()),
 gradient_elements=g.numel(),all_parameters_finite=bool(w.torch.isfinite(parameters).all()),
 maximum_parameter_magnitude=float(parameters.abs().max()),replayed_updates=3,
 audit='Exact recorded loss/norm replay544..546 from recovery544, then reevaluate failed boundary547; original artifacts unchanged.',utc=w.stamp())
assert not math.isfinite(loss) or not bool(w.torch.isfinite(g).all())
w.write(run/'failure_audit.json',result)
print(json.dumps(result))
'''!r}
p=subprocess.run([{r.PYTHON!r},'-c',code],cwd={r.REMOTE!r},env=env,capture_output=True,text=True,timeout=45)
assert p.returncode==0,p.stderr
print(p.stdout)
''')
result['audit_script_sha256']=r.sha(Path(__file__))
r.dump(HERE/'failure_audit.json',result)
print(json.dumps(result,indent=2))
