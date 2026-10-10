"""Operate only the six recorded seed workers, using native SSH/SCP in parallel."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
from datetime import datetime,timezone
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('action',choices=['setup','probe','setup-status','resolve-init',
    'upload-freeze','verify-source','audit','calibrate','calibration-status','full',
    'status','logs','archive','archive-status'])
parser.add_argument('--ids',nargs='*')
args=parser.parse_args()
rows=json.loads((ROOT/'cache/runpod/seed-cohort.json').read_text())
if args.ids: rows=[r for r in rows if r['run'][:3] in args.ids]
assert rows

def work(row):
    run=row['run']
    prelaunch=ROOT/'runs'/run/'prelaunch'
    if args.action in ('calibrate','full'):
        launch_receipt=prelaunch/('calibration-start.json' if args.action=='calibrate' else 'full-pass-start.json')
        if launch_receipt.exists(): raise RuntimeError('A launch receipt already exists; inspect the recorded worker.')
        freeze=json.loads((prelaunch/'scientific-freeze.json').read_text())
        assert hashlib.sha256((ROOT/'runs'/run/'config.json').read_bytes()).hexdigest()==freeze['config_sha256']
        if args.action=='full': assert json.loads((prelaunch/'calibrated-launch.json').read_text())['qualified']
    actions=['probe','upload-source','bootstrap','upload-data','stage'] if args.action=='setup' else [args.action]
    for action in actions:
        result=subprocess.run([sys.executable,str(ROOT/'tools/scale_worker.py'),run,action],
            capture_output=True,text=True,encoding='utf-8')
        shown=result.stdout[-5000:]
        if action in ('status','calibration-status') and result.returncode==0:
            snapshot=json.loads(result.stdout)
            m=snapshot.get('manifest',{}); e=snapshot.get('latest_update',{})
            shown={'status':m.get('status'),'updates':m.get('completed_updates'),
                'tokens':m.get('completed_input_tokens'),'loss':m.get('latest_loss'),
                'tokens_per_second':e.get('tokens_per_second'),'remaining_training_seconds':m.get('remaining_training_seconds'),
                'overflow_retries':e.get('overflow_retries'),'verified':snapshot.get('verification',{}).get('verified')}
        print(json.dumps({'run':run[:3],'action':action,'exit_code':result.returncode,
            'output':shown,'error':result.stderr[-500:]}),flush=True)
        if result.returncode: return False
        if action in ('calibrate','full'):
            launch_receipt.write_text(json.dumps({'started_at':datetime.now(timezone.utc).isoformat(),
                'pid':int(result.stdout.strip()),'phase':action,'pod_id':json.loads((prelaunch/'transport.json').read_text())['pod_id']},indent=2)+'\n')
        if action=='probe':
            (prelaunch/'hardware-probe.txt').write_text(result.stdout,encoding='utf-8')
        elif action=='resolve-init':
            identity=json.loads(result.stdout)
            config_path=ROOT/'runs'/run/'config.json'
            config=json.loads(config_path.read_text())
            assert identity['optimizer_steps']==0 and identity['parameters']==config['model']['parameter_count']
            config['model']['initial_parameter_sha256']=identity['initial_parameter_sha256']
            config_path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
            (prelaunch/'initialization-execution.json').write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
        elif action=='audit':
            (prelaunch/'gpu-audit.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
            checks=json.loads((prelaunch/'local-checks.json').read_text())
            checks['cuda_fused_adam_audit_passed']=True
            checks['cuda_fused_adam_audit_source']='gpu-audit.txt'
            (prelaunch/'local-checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    return True

with ThreadPoolExecutor(max_workers=6) as pool:
    complete=list(pool.map(work,rows))
sys.exit(0 if all(complete) else 1)
