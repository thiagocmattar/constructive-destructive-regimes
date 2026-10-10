"""Transfer four fixed ranges via native SSH, then verify the complete archive."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'runs'/sys.argv[1]
rid=run.name[:3]
t=json.loads((run/'prelaunch/transport.json').read_text())
lease=json.loads((run/'prelaunch/lease.json').read_text())
stem=f'run{rid}-attempt-01.tar.gz'
remote='/opt/'+stem
common=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15',
    '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','StrictHostKeyChecking=accept-new',
    '-o','UserKnownHostsFile='+str(ROOT/'cache/runpod/known_hosts'),
    '-i','C:/Users/thima/.ssh/id_ed25519_runpod','-p',str(t['port']),t['user']+'@'+t['host']]
info=subprocess.run(common+['cat '+shlex.quote(remote+'.sha256')+' '+shlex.quote(remote+'.bytes')],
    capture_output=True,text=True,timeout=60,check=True).stdout.splitlines()
digest=info[0].split()[0];size=int(info[1])
assert len(digest)==64 and size>0
cache=ROOT/'cache/runpod'
target=cache/stem
target.with_name(stem+'.sha256').write_text(info[0]+'\n')
target.with_name(stem+'.bytes').write_text(str(size)+'\n')
parts=cache/f'run{rid}-transfer-parts'
parts.mkdir(exist_ok=True)
start=time.monotonic()
segments=[]
block=math.ceil(size/4)
for i in range(4):
    offset=i*block;count=min(block,size-offset)
    segments.append((i,offset,count))
def fetch(segment):
    i,offset,count=segment
    path=parts/f'{i:02d}.part'
    command='dd if='+shlex.quote(remote)+f' bs=4M iflag=skip_bytes,count_bytes skip={offset} count={count} status=none'
    remaining=int(lease['deadline_epoch']-time.time()-600)
    assert remaining>60
    # Consume binary stdout explicitly; never send archive bytes to the UI.
    process=subprocess.Popen(common+['-n',command],stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    end=time.monotonic()+remaining
    with path.open('wb') as output:
        while chunk:=process.stdout.read(1024*1024):
            output.write(chunk)
            if time.monotonic()>end:
                process.kill();raise TimeoutError('Archive segment exceeded retrieval deadline')
    error=process.stderr.read()
    code=process.wait(timeout=60)
    assert code==0, error.decode(errors='replace')[-500:]
    assert path.stat().st_size==count
    print(json.dumps({'segment':i,'bytes':count,'seconds':time.monotonic()-start}),flush=True)
    return path
with ThreadPoolExecutor(max_workers=4) as pool:
    files=list(pool.map(fetch,segments))
combined=target.with_name(stem+'.assembling')
with combined.open('wb') as output:
    for path in files:
        with path.open('rb') as source:shutil.copyfileobj(source,output,8*1024*1024)
assert combined.stat().st_size==size
with combined.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==digest
os.replace(combined,target)
receipt={'archive_bytes':size,'archive_sha256':actual,'verified':True,
    'method':'four native SSH byte ranges; exact concatenation and whole-archive SHA',
    'seconds':time.monotonic()-start,'pod_id':t['pod_id'],
    'checked_at':datetime.now(timezone.utc).isoformat()}
(run/'prelaunch/transfer.json').write_text(json.dumps(receipt,indent=2)+'\n')
for p in files:p.unlink()
print(json.dumps(receipt),flush=True)
