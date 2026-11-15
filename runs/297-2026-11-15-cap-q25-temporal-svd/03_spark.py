"""Task-specific SSH transport; no cloud resources or unrelated processes."""
import argparse,hashlib,json,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
L=json.loads((ROOT/'analyses/066-2026-11-15-activation-inactivity/launch.json').read_text())
HOST='tcml-spark@spark-2277.local';REMOTE=L['root'];REL=HERE.relative_to(ROOT).as_posix()
def ssh(script):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',HOST,'bash -s'],input=script.encode(),capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError((p.stdout+p.stderr).decode())
    return p.stdout.decode()
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','status','upload']);p.add_argument('--case',default='mlp',choices=['mlp','resnet','transformer']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    tag=('smoke-' if a.smoke else 'production-')+a.case
    if a.action in ['launch','upload']:
        print(ssh('mkdir -p '+shlex.quote(REMOTE+'/'+REL+'/scratch')))
        for f in HERE.iterdir():
            if f.is_file() and f.suffix in ['.py','.json','.md']:
                subprocess.run(['scp','-q','-o','BatchMode=yes',str(f),HOST+':'+REMOTE+'/'+REL+'/'+f.name],check=True,timeout=60)
        if a.action=='upload':return
        output=REL+'/artifacts/'+('smoke' if a.smoke else 'attempt-01')
        script='#!/bin/bash\nset -e\ncd '+shlex.quote(REMOTE)+'\n'
        script+='export PYTHONPATH='+shlex.quote(REMOTE+':'+L['pythonpath'])+' CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n'
        command=[L['python'],REL+'/01_capture.py','--case',a.case,'--source-root',REMOTE,'--data-root',L['data_root'],'--out',output]
        if a.smoke:command+=['--smoke']
        script+=' '.join(map(shlex.quote,command))+'\n'
        if not a.smoke:script+=' '.join(map(shlex.quote,[L['python'],REL+'/02_reduce.py','--case-root',output+'/'+a.case]))+'\n'
        script+='echo 0 > '+REL+'/scratch/'+tag+'.exit\n'
        scratch=HERE/'scratch';scratch.mkdir(exist_ok=True);path=scratch/(tag+'.sh');path.write_text(script,encoding='utf8',newline='\n')
        subprocess.run(['scp','-q',str(path),HOST+':'+REMOTE+'/'+REL+'/scratch/'+path.name],check=True,timeout=60)
        pid=ssh('cd '+shlex.quote(REMOTE)+'\nsetsid bash '+REL+'/scratch/'+path.name+' </dev/null >'+REL+'/scratch/'+tag+'.log 2>&1 &\necho $!\n').strip()
        receipt=dict(pid=pid,root=REMOTE,output=output,case=a.case,smoke=a.smoke,files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.iterdir() if f.is_file() and f.suffix in ['.py','.json','.md']})
        (HERE/(tag+'-launch.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
    else:
        print(ssh('cd '+shlex.quote(REMOTE)+'\ntail -n 6 '+REL+'/scratch/'+tag+'.log\nif test -f '+REL+'/scratch/'+tag+'.exit; then cat '+REL+'/scratch/'+tag+'.exit; fi\n'))
if __name__=='__main__':main()
