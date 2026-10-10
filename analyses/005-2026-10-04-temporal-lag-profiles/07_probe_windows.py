"""Read completed, immutable window prefixes while the replay continues."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('operate',HERE/'02_operate.py')
operate=importlib.util.module_from_spec(spec);spec.loader.exec_module(operate)
spec=importlib.util.spec_from_file_location('analysis',HERE/'03_analyze.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)

def main():
 row=next(r for r in json.loads((HERE/'cohort.json').read_text()) if r['run'][:3]==sys.argv[1])
 window=sys.argv[2]; run=ROOT/'runs'/row['run']; pre=run/'prelaunch'
 lease=json.loads((pre/'lease.json').read_text()); cfg=json.loads((run/'config.json').read_text())
 first,last=cfg['temporal_sketch']['windows'][window]
 base='/workspace/constructive/runs/'+row['run']+'/artifacts/'+lease['attempt']+'/full-pass'
 remote='/opt/probe-'+row['run'][:3]+'-'+window+'.npy'
 code='''from pathlib import Path
import hashlib,json,numpy as np
b=Path(BASE); first,last=BOUNDS
progress=json.loads((b/'temporal/sketch_progress.json').read_text())
assert progress['completed_updates']>=last
events=[json.loads(s) for s in (b/'events.jsonl').read_text().splitlines()]
u=[e for e in events if e['event']=='update']
assert len(u)>=last
z=np.load(b/'temporal/raw_sketches.npy',mmap_mode='r')
np.save(REMOTE,np.array(z[first-1:last]))
with open(REMOTE,'rb') as s: digest=hashlib.file_digest(s,'sha256').hexdigest()
pairs=[p for s in (b/'temporal/exact_pairs.jsonl').read_text().splitlines() if first<=(p:=json.loads(s))['source_update'] and p['target_update']<=last]
records=[json.loads(s) for s in (b/'temporal/exact_windows.jsonl').read_text().splitlines()]
print(json.dumps({'sha256':digest,'bytes':Path(REMOTE).stat().st_size,'updates':u[first-1:last],'pairs':pairs,'exact_window':next(r for r in records if r['window']==WINDOW)}))
'''.replace('BASE',repr(base)).replace('BOUNDS',repr((first,last))).replace('REMOTE',repr(remote)).replace('WINDOW',repr(window))
 raw=json.loads(operate.native(row,'command','/opt/baseline-env/bin/python -c '+shlex.quote(code)))
 scratch=HERE/'scratch'; scratch.mkdir(exist_ok=True); destination=scratch/Path(remote).name
 t=json.loads((pre/'transport.json').read_text())
 subprocess.run(['scp','-q','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15',
   '-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(ROOT/'cache/runpod/known_hosts'),
   '-i','C:/Users/thima/.ssh/id_ed25519_runpod','-P',str(t['port']),
   t['user']+'@'+t['host']+':'+remote,str(destination)],check=True,timeout=300)
 with destination.open('rb') as s:assert hashlib.file_digest(s,'sha256').hexdigest()==raw['sha256']
 assert destination.stat().st_size==raw['bytes']
 z=np.array(np.load(destination),dtype=np.float64).transpose(1,0,2)
 n=np.array([e['gradient_norm'] for e in raw['updates']]); gram=z@z.transpose(0,2,1)
 _,cs,weighted,counts=analysis.profiles(gram,n)
 errors=np.array([np.mean(p['projected_cosines'])-p['exact_cosine'] for p in raw['pairs']])
 report={'run':row['run'],'window':window,'provisional_snapshot':True,
   'lag_cosines_1_to_8':cs.mean(0)[:8].tolist(),'mean_cosine_lags32_64':float(cs.mean(0)[31:64].mean()),
   'exact_lag_one_mean':float(np.mean([e['adjacent_cosine'] for e in raw['updates'][1:]])),
   'exact_window':raw['exact_window'],'calibration_pairs':len(errors),
   'calibration_rmse':float(np.sqrt(np.mean(errors**2))),'calibration_max_abs':float(abs(errors).max()),
   'projection_profiles_1_to_8':cs[:,:8].tolist(),'counts':counts}
 (scratch/f'probe-{row["run"][:3]}-{window}.json').write_text(json.dumps({'snapshot':raw,'report':report},indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['counts','projection_profiles_1_to_8']},indent=2))

if __name__=='__main__':main()
