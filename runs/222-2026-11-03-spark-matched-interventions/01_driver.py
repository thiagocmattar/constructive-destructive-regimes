"""Run222: matched Spark interventions after retained cross-hardware qualification."""
import importlib.util,json,os,shutil,signal,time,traceback
from datetime import datetime
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PRIOR=ROOT/'runs/221-2026-11-03-spark-fixed-step-interventions'
spec=importlib.util.spec_from_file_location('frozen_worker221',PRIOR/'01_worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

def main():
 cfg=json.loads((HERE/'config.json').read_text());parents=json.loads((PRIOR/'parent_manifest.json').read_text());manifest=json.loads((HERE/'input_manifest.json').read_text())
 for item in manifest['files']:assert (ROOT/item['path']).stat().st_size==item['bytes'] and w.sha(ROOT/item['path'])==item['sha256'],item['path']
 w.configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
 deadline=datetime.fromisoformat(cfg['compute_deadline_utc']).timestamp();stopped=[False]
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
 stop=lambda:stopped[0] or time.time()>=deadline or (HERE/'STOP').exists()
 out=HERE/'artifacts/attempts/001-spark';out.mkdir(parents=True,exist_ok=False)
 identity=dict(run_id='222',config_sha256=w.sha(HERE/'config.json'),manifest_sha256=w.sha(HERE/'input_manifest.json'),torch=torch.__version__,cuda=torch.version.cuda,
  deterministic=True,tf32=False,physical={k:v['physical'] for k,v in cfg['cases'].items()})
 packet=dict(status='running',run_id='222',started_utc=w.stamp(),identity=identity,qualification={},results={},skipped_optional=[])
 w.write(out/'manifest.json',packet);began=time.perf_counter()
 frozen=out/'executed_inputs'
 for item in manifest['files']:
  p=ROOT/item['path']
  if p.suffix in ('.py','.json') and '/artifacts/' not in item['path']:
   target=frozen/item['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 shutil.copyfile(HERE/'input_manifest.json',out/'executed_input_manifest.json')
 try:
  for case in cfg['cases']:
   arm=cfg['arms'][0]
   for label,end,resume in [('direct2',1026,None),('prefix1',1025,None),('resume2',1026,'prefix1')]:
    c,parent=w.prepare(case,cfg,parents,identity)
    if resume:w.load_branch(c,out/'qualification'/case/resume/'recovery.pt',arm)
    w.run(c,arm,out/'qualification'/case/label,end,stop)
    del c,parent;torch.cuda.empty_cache()
   a=torch.load(out/'qualification'/case/'direct2/end.pt',map_location='cpu',weights_only=False)
   b=torch.load(out/'qualification'/case/'resume2/end.pt',map_location='cpu',weights_only=False)
   for key in ['model','optimizer','previous_gradient','rng']:w.equal(a[key],b[key])
   rows=[json.loads(x) for x in (out/'qualification'/case/'direct2/boundaries.jsonl').read_text().splitlines()]
   pfx=[json.loads(x) for x in (out/'qualification'/case/'prefix1/boundaries.jsonl').read_text().splitlines()]
   assert (rows[0]['loss'],rows[0]['raw_gradient_norm'],rows[0]['rho'])==(pfx[0]['loss'],pfx[0]['raw_gradient_norm'],pfx[0]['rho'])
   packet['qualification'][case]=dict(status='passed',same_parent_Spark_observations_exact=True,direct2_vs1plus1_exact=True,updates=4,full_gradient_calls=7,
    fork_loss=rows[0]['loss'],fork_raw_gradient_norm=rows[0]['raw_gradient_norm'])
   w.write(out/'manifest.json',packet);print(w.stamp(),'QUALIFIED',case,packet['qualification'][case],flush=True);del a,b
  for arm in cfg['arms']:
   estimate=(cfg['updates']+1)*sum(cfg['runtime_profile_seconds'].values())*1.25+300
   if not arm['primary'] and deadline-time.time()<estimate:
    packet['skipped_optional'].append(arm['id']);w.write(out/'manifest.json',packet);continue
   for case in cfg['cases']:
    if stop():raise TimeoutError('Overnight deadline/STOP reached.')
    c,parent=w.prepare(case,cfg,parents,identity);del parent
    def progress(event):
     event.update(completed_conditions=len(packet['results']),elapsed_worker_seconds=time.perf_counter()-began)
     w.write(out/'progress.json',event)
    key=case+'/'+arm['id'];print(w.stamp(),'START',key,flush=True)
    try:packet['results'][key]=w.run(c,arm,out/'conditions'/case/arm['id'],cfg['end_state'],stop,progress)
    except Exception as error:packet['results'][key]=json.loads((out/'conditions'/case/arm['id']/'status.json').read_text())
    w.write(out/'manifest.json',packet);print(w.stamp(),'END',key,packet['results'][key]['status'],flush=True)
    del c;torch.cuda.empty_cache()
  packet['status']='completed' if all(x['status']=='completed' for x in packet['results'].values()) else 'completed_with_failures'
 except BaseException as error:
  packet.update(status='interrupted' if isinstance(error,TimeoutError) else 'failed',exception=repr(error),traceback=traceback.format_exc());raise
 finally:
  packet.update(finished_utc=w.stamp(),elapsed_worker_seconds=time.perf_counter()-began)
  w.write(out/'manifest.json',packet);w.write(HERE/'status.json',packet)
  inventory=[]
  for p in sorted((HERE/'artifacts').rglob('*')):
   if p.is_file() and p.name!='inventory.json':inventory.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=w.sha(p)))
  w.write(out/'inventory.json',dict(status=packet['status'],files=inventory))

if __name__=='__main__':main()
