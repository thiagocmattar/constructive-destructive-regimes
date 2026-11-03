"""Warm-parent identity/gradient bridge only; zero parameter updates."""
import importlib.util,json
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('worker221',HERE/'01_worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
cfg=json.loads((HERE/'config.json').read_text());parents=json.loads((HERE/'parent_manifest.json').read_text())
w.configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
results={}
for case in cfg['cases']:
 c,parent=w.prepare(case,cfg,parents,dict(mechanics_only=True));loss,g=c['task'].gradient();ref=parent['current_gradient_reference'].to('cuda:0')
 history=json.loads((w.ROOT/parents[case]['history']).read_text());old_loss=history[-1]['loss']
 error=float((g.double()-ref.double()).norm()/ref.double().norm());loss_error=abs(loss-old_loss)/max(abs(old_loss),1e-30)
 for key,value in parent['model'].items():assert torch.equal(c['task'].model.state_dict()[key].cpu(),value)
 for saved,current in zip(parent['optimizer']['state'].values(),c['opt'].state.values()):
  for key in ['step','exp_avg','exp_avg_sq']:assert torch.equal(saved[key].cpu(),current[key].cpu())
 results[case]=dict(status='passed' if error<=.001 and loss_error<=.0001 else 'outside_gradient_gate',gradient_relative_error=error,loss_relative_error=loss_error,
  model_moments_exact=True,loss=loss,original_loss=old_loss,gradient_norm=float(g.double().norm()),updates=0)
 print(json.dumps(dict(case=case,**results[case])),flush=True)
 del c,parent,g,ref;torch.cuda.empty_cache()
w.write(HERE/'artifacts/warm_parent_qualification.json',dict(status='passed' if all(x['status']=='passed' for x in results.values()) else 'gate_failed',results=results,updates=0))
