"""Bounded native SSH/SCP operations for this attributed three-worker cohort."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,subprocess,sys,shlex
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')
def native(row,action,command=None):
    args=[sys.executable,str(ROOT/'tools/scale_worker.py'),row['run'],action]
    if command: args+=['--command',command]
    result=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',timeout=2800)
    if result.returncode: raise RuntimeError(row['run']+': '+result.stderr[-900:]+result.stdout[-900:])
    return result.stdout
def work(row,action):
    pre=ROOT/'runs'/row['run']/'prelaunch'
    attempt=json.loads((pre/'lease.json').read_text()).get('attempt','attempt-01')
    if action=='setup':
        for stage in ['probe','upload-source','bootstrap','upload-data','stage']:
            print(json.dumps({'run':row['run'][:3],'stage':stage}),flush=True)
            result=native(row,stage)
        return
    if action in ['calibrate','full']:
        receipt=pre/(action+'-start.json' if attempt=='attempt-01' else action+'-'+attempt+'-start.json')
        assert not receipt.exists(),'Duplicate launch blocked'
        if action=='full': assert json.loads((pre/'calibrated-launch.json').read_text())['qualified']
        if attempt!='attempt-01':
            rd='/workspace/constructive/runs/'+row['run']
            phase='calibrate' if action=='calibrate' else 'full-pass'
            logphase='calibration' if action=='calibrate' else 'full-pass'
            command='cd '+rd+'; setsid bash prelaunch/run-'+attempt+'.sh '+phase+' > /workspace/constructive/'+logphase+'.log 2>&1 < /dev/null & echo $!'
            result=native(row,'command',command)
        else: result=native(row,action)
        receipt.write_text(json.dumps({'pid':int(result.strip()),'utc':datetime.now(timezone.utc).isoformat()}))
    elif action=='archive' and attempt!='attempt-01':
        rd='/workspace/constructive/runs/'+row['run']
        result=native(row,'command',"trap '' HUP; nohup setsid --fork bash "+rd+'/prelaunch/archive-'+attempt+'.sh > /workspace/constructive/archive.log 2>&1 < /dev/null & echo $!')
    elif action=='archive-status' and attempt!='attempt-01':
        stem='/opt/run'+row['run'][:3]+'-'+attempt+'.tar.gz'
        result=native(row,'command','tail -n 3 /workspace/constructive/archive.log; cat '+stem+'.sha256 '+stem+'.bytes 2>/dev/null || true')
    elif action=='download' and attempt!='attempt-01':
        t=json.loads((pre/'transport.json').read_text()); stem='/opt/run'+row['run'][:3]+'-'+attempt+'.tar.gz'
        command=['scp','-q','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15',
            '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','StrictHostKeyChecking=accept-new',
            '-o','UserKnownHostsFile='+str(ROOT/'cache/runpod/known_hosts'),'-i','C:/Users/thima/.ssh/id_ed25519_runpod',
            '-P',str(t['port'])]
        for suffix in ['', '.sha256','.bytes']:command.append(t['user']+'@'+t['host']+':'+stem+suffix)
        command.append(str(ROOT/'cache/runpod'));subprocess.run(command,check=True,timeout=2700);result='downloaded'
    else: result=native(row,action)
    if action in ['status','calibration-status']:
        raw=json.loads(result); m=raw.get('manifest',{}); e=raw.get('latest_update',{})
        result={'status':m.get('status'),'update':m.get('completed_updates'),
            'loss':m.get('latest_loss'),'tokens':m.get('completed_input_tokens'),
            'tokens_per_second':e.get('tokens_per_second'),'memory_gb':e.get('peak_cuda_allocated_bytes',0)/1e9,
            'remaining_seconds':m.get('remaining_training_seconds'),'verified':raw.get('verification',{}).get('verified')}
    print(json.dumps({'run':row['run'][:3],'action':action,'result':result}),flush=True)
def main():
    p=argparse.ArgumentParser(); p.add_argument('action'); p.add_argument('--ids',nargs='*')
    args=p.parse_args(); rows=json.loads((HERE/'cohort.json').read_text())
    if args.ids: rows=[r for r in rows if r['run'][:3] in args.ids]
    with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(lambda r:work(r,args.action),rows))
if __name__=='__main__':main()
