"""Accept mechanics, measured memory/time and initial identity before full replay."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,shlex,sys
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('operate',HERE/'02_operate.py')
operate=importlib.util.module_from_spec(spec);spec.loader.exec_module(operate)
def main():
 rows=json.loads((HERE/'cohort.json').read_text())
 ids=sys.argv[1:]
 for row in rows:
  if ids and row['run'][:3] not in ids:continue
  run=ROOT/'runs'/row['run']; pre=run/'prelaunch'; lease=json.loads((pre/'lease.json').read_text()); attempt=lease['attempt']
  directory='/workspace/constructive/runs/'+row['run']+'/artifacts/'+attempt+'/calibration'
  files=['manifest.json','summary.json','environment.json','input_identity.json','temporal/calibration.json','events.jsonl']
  code='from pathlib import Path; import json; r=Path('+repr(directory)+'); files='+repr(files)+'; print(json.dumps({f:(r/f).read_text() for f in files}))'
  raw=json.loads(operate.native(row,'command','python3 -c '+shlex.quote(code)))
  for name,text in raw.items():(pre/('calibration-'+attempt+'-'+name.replace('/','-'))).write_text(text)
  manifest=json.loads(raw['manifest.json']); summary=json.loads(raw['summary.json']); environment=json.loads(raw['environment.json']); identity=json.loads(raw['input_identity.json']); cal=json.loads(raw['temporal/calibration.json'])
  assert manifest['status']=='completed' and summary['completed_updates']==4
  assert cal['qualified'] and cal['completed_updates']==4
  cfg=json.loads((run/'config.json').read_text())
  assert environment['initial_parameter_sha256']==cfg['model']['initial_parameter_sha256']
  assert environment['gpu']=='NVIDIA GeForce RTX 4090'
  parent=ROOT/'runs'/row['reference']/'artifacts/attempt-01/full-pass'
  assert identity['order_sha256']==json.loads((parent/'input_identity.json').read_text())['order_sha256']
  events=[e for line in raw['events.jsonl'].splitlines() if (e:=json.loads(line))['event']=='update']
  reference=[e for line in (parent/'events.jsonl').read_text().splitlines() if (e:=json.loads(line))['event']=='update'][:4]
  maximum_cosine_difference=max(abs(e['adjacent_cosine']-f['adjacent_cosine']) for e,f in zip(events[1:],reference[1:]))
  assert maximum_cosine_difference<.001
  assert all(e['input_tokens']==f['input_tokens'] and e['learning_rate']==f['learning_rate'] for e,f in zip(events,reference))
  peak=max(e['peak_cuda_allocated_bytes'] for e in events)
  projected_peak=peak+5*cfg['model']['parameter_count']*4
  assert projected_peak<24e9*.9
  step_seconds=float(np.median(summary['optimizer_step_seconds']))
  training_seconds=712*step_seconds
  now=datetime.now(timezone.utc)
  remaining=lease['deadline_epoch']-now.timestamp()
  extra=3*summary['checkpoint_seconds']+sum(summary['full_validation_seconds'])+180+1200
  qualified_etc=training_seconds+extra
  assert qualified_etc<remaining, f'ETC {qualified_etc} exceeds remaining lease {remaining}'
  report={'qualified':True,'attempt':attempt,'pod_id':lease['pod_id'],'gpu':environment['gpu'],
   'initial_hash_matches':True,'order_hash_matches':True,'maximum_first_four_cosine_difference':maximum_cosine_difference,
   'projection_calibration':cal,'measured_peak_bytes':peak,'conservative_peak_with_five_anchors':projected_peak,
   'measured_median_update_seconds':step_seconds,'training_seconds':training_seconds,
   'qualified_etc_including_saves_validation_setup_retrieval':qualified_etc,
   'remaining_lease_seconds':remaining,'live_hourly_price':lease['price_per_hour'],'checked_at':now.isoformat(),
   'calibration_is_scientific_evidence':False}
  (pre/'calibrated-launch.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({'run':row['run'][:3],**report},indent=2))
if __name__=='__main__':main()
