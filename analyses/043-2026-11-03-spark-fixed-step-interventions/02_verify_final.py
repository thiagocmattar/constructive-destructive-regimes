"""Verify terminal Run222 transfer, frozen inputs, checkpoint clocks and exact replay."""
import hashlib,json,math,os
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'runs/222-2026-11-03-spark-matched-interventions'

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def equal(a,b):
 if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and torch.equal(a,b)
 elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b

def finite(value):
 if isinstance(value,torch.Tensor):assert bool(torch.isfinite(value).all())
 elif isinstance(value,dict):
  for v in value.values():finite(v)
 elif isinstance(value,(tuple,list)):
  for v in value:finite(v)
 elif isinstance(value,float):assert math.isfinite(value)

def checkpoint(folder,state,identity,case,arm,base):
 assert sha(folder/'end.pt')==sha(folder/'recovery.pt')
 p=torch.load(folder/'end.pt',map_location='cpu',weights_only=False)
 assert p['schema']==1 and p['completed_updates']==state and p['identity']==identity
 assert p['case']==case and p['arm']==arm
 finite(p)
 assert p['previous_gradient'].dtype==torch.float32
 total=0
 for group in p['optimizer']['param_groups']:
  assert group['lr']==base and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
  assert group['weight_decay']==0 and group['foreach'] is False and group['fused'] is False
  for key in group['params']:
   s=p['optimizer']['state'][key]
   assert int(s['step'])==state and s['exp_avg'].dtype==s['exp_avg_sq'].dtype==torch.float32
   assert s['exp_avg'].shape==s['exp_avg_sq'].shape and bool((s['exp_avg_sq']>=0).all())
   total+=s['exp_avg'].numel()
 assert p['previous_gradient'].numel()==total
 assert set(p['rng'])=={'torch','cuda','numpy','python'}
 return p

def main():
 torch.set_num_threads(4)
 receipt=read(RUN/'retrieval.json');assert receipt['verified'] and receipt['final'] and receipt['status']=='completed'
 snapshot=ROOT/receipt['payload_root']
 if os.name=='nt':snapshot=Path('\\\\?\\'+str(snapshot.resolve()))
 raw=snapshot/'runs/222-2026-11-03-spark-matched-interventions/artifacts/attempts/001-spark'
 transfer=read(snapshot/'transfer_inventory.json');inventory=read(raw/'inventory.json');manifest=read(raw/'manifest.json');cfg=read(RUN/'config.json')
 assert transfer['final'] and inventory['status']==manifest['status']=='completed'
 transferred={x['path']:x for x in transfer['files']}
 for item in inventory['files']:
  assert transferred[item['path']]==item
  p=snapshot/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 inputs=read(raw/'executed_input_manifest.json');assert inputs==read(RUN/'input_manifest.json')
 assert sha(raw/'executed_input_manifest.json')==manifest['identity']['manifest_sha256']
 assert sha(RUN/'config.json')==manifest['identity']['config_sha256']
 copies=0
 for item in inputs['files']:
  p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
  copy=raw/'executed_inputs'/item['path']
  if copy.exists():assert sha(copy)==item['sha256'];copies+=1
 arms={x['id']:x for x in cfg['arms']};expected={c+'/'+a for c in cfg['cases'] for a in arms}
 assert set(manifest['results'])==expected and not manifest['skipped_optional']
 main_updates=0;gradients=0;checkpoints=0;inactive=[];coverage={}
 for case,spec in cfg['cases'].items():
  q=manifest['qualification'][case]
  assert q['status']=='passed' and q['same_parent_Spark_observations_exact'] and q['direct2_vs1plus1_exact']
  assert q['updates']==4 and q['full_gradient_calls']==7
  qual=raw/'qualification'/case
  a=checkpoint(qual/'direct2',1026,manifest['identity'],case,arms['unchanged'],spec['lr'])
  b=checkpoint(qual/'resume2',1026,manifest['identity'],case,arms['unchanged'],spec['lr'])
  for key in ['model','optimizer','previous_gradient','rng']:equal(a[key],b[key])
  del a,b
  p=checkpoint(qual/'prefix1',1025,manifest['identity'],case,arms['unchanged'],spec['lr']);del p;checkpoints+=3
  a=rows(qual/'direct2/boundaries.jsonl')[0];b=rows(qual/'prefix1/boundaries.jsonl')[0]
  for k in ['loss','raw_gradient_norm','rho']:assert a[k]==b[k]
  control=checkpoint(raw/'conditions'/case/'unchanged',1088,manifest['identity'],case,arms['unchanged'],spec['lr'])
  fork=rows(raw/'conditions'/case/'unchanged/boundaries.jsonl')[0]
  for name,arm in arms.items():
   folder=raw/'conditions'/case/name;status=read(folder/'status.json');bound=rows(folder/'boundaries.jsonl');metrics=rows(folder/'metrics.jsonl')
   assert status==manifest['results'][case+'/'+name] and status['status']=='completed' and status['updates']==64
   assert status['start_state']==1024 and status['completed_state']==status['end_state']==1088
   assert [x['state'] for x in bound]==list(range(1024,1089)) and [x['update'] for x in metrics]==list(range(1025,1089))
   assert len(rows(folder/'events.jsonl'))==64
   assert all(x['manuscript_gradient_index']==x['state']+1 for x in bound)
   for k in ['loss','rho','raw_gradient_norm','eta_kappa_hat','parameter_norm']:assert bound[0][k]==fork[k]
   finite(bound);finite(metrics);finite(status['evaluation'])
   p=control if name=='unchanged' else checkpoint(folder,1088,manifest['identity'],case,arm,spec['lr']);checkpoints+=1
   previous_norm=float(p['previous_gradient'].double().norm())
   assert math.isclose(previous_norm,bound[-2]['raw_gradient_norm'],rel_tol=1e-11)
   moment=torch.cat([p['optimizer']['state'][k]['exp_avg'].reshape(-1) for g in p['optimizer']['param_groups'] for k in g['params']]).double()
   assert math.isclose(float(moment.norm()),metrics[-1]['moment_norm'],rel_tol=1e-11)
   if arm['kind']=='clip' and all(x['clip_factor']==1 for x in metrics):
    for k in ['model','optimizer','previous_gradient','rng']:equal(p[k],control[k])
    inactive.append(case+'/'+name)
   if name!='unchanged':del p
   main_updates+=len(metrics);gradients+=len(bound)
   coverage[case+'/'+name]=dict(state=1088,updates=64,gradient_boundaries=65,finite=True)
  del control
 gradients+=sum(len(rows(p)) for p in raw.glob('qualification/*/*/boundaries.jsonl'))
 assert main_updates==2304 and gradients==2361 and checkpoints==45
 deadline=datetime.fromisoformat(cfg['compute_deadline_utc'])
 primary_end=max(datetime.fromisoformat(v['finished_utc']) for k,v in manifest['results'].items() if arms[k.split('/')[1]]['primary'])
 optional={}
 for name,arm in arms.items():
  if arm['primary']:continue
  start=datetime.fromisoformat(manifest['results']['mlp/'+name]['started_utc'])
  required=65*sum(cfg['runtime_profile_seconds'].values())*1.25+300
  assert start>=primary_end and (deadline-start).total_seconds()>required
  optional[name]=dict(start_utc=start.isoformat(),required_seconds=required,available_seconds=(deadline-start).total_seconds())
 assert datetime.fromisoformat(manifest['finished_utc'])<deadline
 prior=read(ROOT/'runs/221-2026-11-03-spark-fixed-step-interventions/warm_parent_qualification.json')
 assert prior['status']=='gate_failed' and prior['results']['resnet']['gradient_relative_error']>.001
 result=dict(status='passed',source_sha256=sha(__file__),snapshot=receipt,terminal_inventory_files=len(inventory['files']),transferred_files=len(transfer['files']),frozen_inputs_verified=len(inputs['files']),executed_source_copies_verified=copies,main_updates=main_updates,qualification_updates=12,full_gradient_calls=gradients,checkpoints_validated=checkpoints,same_Spark_replay_exact=True,inactive_clipping_checkpoints_exact=inactive,coverage=coverage,optional_admission_verified=optional,finished_utc=manifest['finished_utc'],elapsed_worker_seconds=manifest['elapsed_worker_seconds'],retained_failed_ResNet_gradient_relative_error=prior['results']['resnet']['gradient_relative_error'],no_historical_bitwise_replay_claim=True,local_verification_torch=torch.__version__)
 (HERE/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ['coverage','snapshot']},indent=2))
if __name__=='__main__':main()
