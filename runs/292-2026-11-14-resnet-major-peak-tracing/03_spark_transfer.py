"""Optional task-local Spark relay for slow Windows-to-Pod SCP transfers."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud',HERE/'02_cloud.py')
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
HOST='tcml-spark@spark-2277.local'
STAGE='/home/tcml-spark/experiments/constructive-destructive-regimes/multispike-2026-11-04/runs/292-2026-11-14-resnet-major-peak-tracing/scratch/relay'


def remote(code,timeout=60):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,'python3','-'],input=code,text=True,capture_output=True,timeout=timeout)
    if p.returncode: raise RuntimeError(p.stderr[-1500:])
    return json.loads(p.stdout)


def main():
    pod=json.loads((HERE/'cloud_lease.json').read_text())
    public=remote(f'''import json,subprocess
from pathlib import Path
root=Path({STAGE!r});root.mkdir(parents=True,exist_ok=True);key=root/'key'
if not key.exists():subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-f',str(key)],check=True)
print(json.dumps(dict(public=(root/'key.pub').read_text().strip())))
''')['public']
    c.x.ssh(pod,'printf "%s\\n" '+shlex.quote(public)+' >> /root/.ssh/authorized_keys\n')
    archive=HERE/'scratch/inputs.tar.gz'
    subprocess.run(['scp','-q',str(archive),HOST+':'+STAGE+'/inputs.tar.gz'],check=True,timeout=240)
    result=remote(f'''import json,hashlib,subprocess
from pathlib import Path
root=Path({STAGE!r});source=root/'inputs.tar.gz'
with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={c.x.file_hash(archive)!r}
command=['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(root/'known-hosts'),'-i',str(root/'key'),'-P',{str(pod['port'])!r},str(source),{'root@'+pod['host']+':/workspace/inputs292-fast.tar.gz'!r}]
with (root/'transfer.log').open('w') as log:
 p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(json.dumps(dict(pid=p.pid,relay_root=str(root),bytes=source.stat().st_size)))
''')
    c.dump(HERE/'relay.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
