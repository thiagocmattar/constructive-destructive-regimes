"""Queue four-thread gzip for the largest archive without changing frozen inputs."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,importlib.util,json,shlex

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('operate',HERE/'02_operate.py')
operate=importlib.util.module_from_spec(spec);spec.loader.exec_module(operate)

def main():
 row=next(r for r in json.loads((HERE/'cohort.json').read_text()) if r['size']=='70M')
 run=ROOT/'runs'/row['run']; pre=run/'prelaunch'; lease=json.loads((pre/'lease.json').read_text())
 attempt=lease['attempt']; start=pre/f'archive-{attempt}-start.json'
 assert not start.exists(),'Never compete with an already started archive'
 command="timeout 55s bash -c 'apt-get update -qq && apt-get install -y --no-install-recommends --no-upgrade pigz' > /opt/pigz-install.log 2>&1 || true; command -v pigz && pigz --version"
 try: installed=operate.native(row,'command',command)
 except RuntimeError:
  print('Parallel compression unavailable; original archive workflow retained.');return
 (pre/'parallel-compression-setup.txt').write_text(installed)
 assert 'pigz' in installed
 original=pre/f'archive-{attempt}.sh'; text=original.read_text()
 assert text.count('gzip -1')==1
 text=text.replace('gzip -1','pigz -1 -p 4')
 text=text.replace('nvidia-smi --query','cp "$RUN_DIR/prelaunch/archive-parallel.sh" "$ATTEMPT/source/prelaunch/archive-runtime.sh"\nnvidia-smi --query')
 rd='/workspace/constructive/runs/'+row['run']
 verify=rd+'/artifacts/'+attempt+'/full-pass/verification.json'
 check='from pathlib import Path; import json; assert json.loads(Path('+repr(verify)+').read_text())["verified"]'
 waiter='#!/usr/bin/env bash\nset -euo pipefail\nuntil /opt/baseline-env/bin/python -c '+shlex.quote(check)+' >/dev/null 2>&1; do\n'
 waiter+=f'  [ "$(date +%s)" -lt {lease["deadline_epoch"]-1200} ] || exit 1\n  sleep 5\ndone\nexec bash '+rd+'/prelaunch/archive-parallel.sh\n'
 files={'archive-parallel.sh':text,'archive-wait.sh':waiter}
 for name,body in files.items():(pre/name).write_text(body,encoding='utf-8',newline='\n')
 code='from pathlib import Path; import base64; b=Path('+repr(rd+'/prelaunch')+'); '
 code+='; '.join('(b/'+repr(name)+').write_bytes(base64.b64decode('+repr(base64.b64encode(body.encode()).decode())+'))' for name,body in files.items())
 operate.native(row,'command','python3 -c '+shlex.quote(code))
 receipt={'pod_id':lease['pod_id'],'queued_at':datetime.now(timezone.utc).isoformat(),
  'requires_terminal_verification':True,'original_archive_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
  'runtime_sources':{name:hashlib.sha256(body.encode()).hexdigest() for name,body in files.items()},
  'compression':'pigz -1 -p 4','scientific_inputs_unchanged':True}
 start.write_text(json.dumps(receipt,indent=2)+'\n')
 result=operate.native(row,'command',"trap '' HUP; nohup setsid --fork bash "+rd+'/prelaunch/archive-wait.sh > /workspace/constructive/archive.log 2>&1 < /dev/null & echo $!')
 receipt['pid']=int(result.strip()); start.write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({'run':'019','stage':'parallel archive queued after verification','pid':receipt['pid']}))

if __name__=='__main__':main()
