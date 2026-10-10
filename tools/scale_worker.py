"""Native SSH/SCP operations for one explicitly attributed scale worker."""
from pathlib import Path
import argparse
import json
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('action', choices=['probe','upload-source','upload-data','bootstrap','stage',
        'setup-status','calibrate','calibration-status','full','status','logs','audit','archive',
        'archive-status','download','command','resolve-init','upload-freeze','verify-source'])
    p.add_argument('--command')
    args = p.parse_args()
    run = ROOT / 'runs' / args.run
    t = json.loads((run / 'prelaunch/transport.json').read_text())
    assert t['run_id'] == args.run[:3]
    rd = '/workspace/constructive/runs/' + args.run
    common = ['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15',
              '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',
              '-o','StrictHostKeyChecking=accept-new',
              '-o','UserKnownHostsFile=' + str(ROOT / 'cache/runpod/known_hosts'),
              '-i','C:/Users/thima/.ssh/id_ed25519_runpod']
    host = t['user'] + '@' + t['host']
    if args.action in ['upload-source','upload-data','download','upload-freeze']:
        base = ['scp',*common,'-P',str(t['port'])]
        if args.action == 'upload-source':
            cmd = base + [str(ROOT / f'cache/runpod/run{args.run[:3]}-source.tar.gz'),host+':/workspace/source.tar.gz']
        elif args.action == 'upload-data':
            cmd = base + [str(ROOT / 'cache/runpod/train-tokens.gz'),host+':/workspace/minipile-cache/train/tokens.gz']
        elif args.action == 'upload-freeze':
            result = subprocess.run(base + [str(run/'config.json'),host+':'+rd+'/config.json'],timeout=120)
            if result.returncode: sys.exit(result.returncode)
            cmd = base + [str(run/'prelaunch/source-inventory.json'),host+':'+rd+'/prelaunch/source-inventory.json']
        else:
            stem = f'run{args.run[:3]}-attempt-01.tar.gz'
            cmd = base + [host+':/opt/'+stem,host+':/opt/'+stem+'.sha256',
                          host+':/opt/'+stem+'.bytes',str(ROOT / 'cache/runpod')]
        result = subprocess.run(cmd, timeout=2700)
        sys.exit(result.returncode)
    verify_code = "from pathlib import Path; import hashlib,json; root=Path('/workspace/constructive'); inv=json.loads((root/'runs/"+args.run+"/prelaunch/source-inventory.json').read_text()); assert all(hashlib.sha256((root/e['path']).read_bytes()).hexdigest()==e['sha256'] for e in inv['files']); print('ALL_SOURCE_HASHES_MATCH')"
    commands = {
      'probe':'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /opt /workspace',
      'bootstrap':f'mkdir -p /workspace/constructive; tar -xzf /workspace/source.tar.gz -C /workspace/constructive; mv /workspace/constructive/input-cache /workspace/minipile-cache; cd {rd}; setsid bash 00_setup_remote.sh > /workspace/constructive/setup.log 2>&1 < /dev/null & echo $!',
      'stage':f'setsid python3 {rd}/prelaunch/accept_local_cache.py > /workspace/constructive/transfer.log 2>&1 < /dev/null & echo $!',
      'setup-status':'tail -n 3 /workspace/constructive/setup.log; cat /workspace/constructive/transfer.log 2>/dev/null || true',
      'calibrate':f'cd {rd}; setsid bash prelaunch/run.sh calibrate > /workspace/constructive/calibration.log 2>&1 < /dev/null & echo $!',
      'calibration-status':f'python3 {rd}/prelaunch/remote_status.py calibration',
      'full':f'cd {rd}; setsid bash prelaunch/run.sh full-pass > /workspace/constructive/full-pass.log 2>&1 < /dev/null & echo $!',
      'status':f'python3 {rd}/prelaunch/remote_status.py full-pass',
      'logs':'tail -n 4 /workspace/constructive/full-pass.log 2>/dev/null; tail -n 4 /workspace/constructive/calibration.log 2>/dev/null',
      'audit':f'cd /workspace/constructive; /opt/baseline-env/bin/python {rd}/prelaunch/audit_gpu_adam.py',
      'archive':f"trap '' HUP; nohup setsid --fork bash {rd}/prelaunch/archive.sh > /workspace/constructive/archive.log 2>&1 < /dev/null & echo $!",
      'archive-status':f'tail -n 3 /workspace/constructive/archive.log; cat /opt/run{args.run[:3]}-attempt-01.tar.gz.sha256 /opt/run{args.run[:3]}-attempt-01.tar.gz.bytes 2>/dev/null || true',
      'command':args.command,
      'resolve-init':f'/opt/baseline-env/bin/python /workspace/constructive/tools/resolve_scale_initialization.py {args.run}',
      'verify-source':'python3 -c '+shlex.quote(verify_code),
    }
    result = subprocess.run(['ssh',*common,'-p',str(t['port']),host,commands[args.action]],
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=75)
    print(result.stdout,end='')
    print(result.stderr,end='',file=sys.stderr)
    if args.action in ['status','calibration-status'] and result.returncode == 0:
        snapshot = json.loads(result.stdout)
        with (run / 'prelaunch/monitoring.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'pod_id':t['pod_id'],**snapshot})+'\n')
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
