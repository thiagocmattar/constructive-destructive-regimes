"""Spark-side parallel, resumable-by-file, hash-verified immutable transfer."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid


def digest(p):
    with p.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('request',type=Path);args=parser.parse_args()
    request=json.loads(args.request.read_text());root=Path(request['destination_root']).resolve()
    start=time.monotonic()
    def copy(item):
        path=(root/item['path']).resolve();assert path.is_relative_to(root)
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.stat().st_size==item['bytes'] and digest(path)==item['sha256']:
            return dict(bytes=item['bytes'],copied=False)
        temp=path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
        command=['scp','-q','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15',
            '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=accept-new',
            '-o','UserKnownHostsFile='+request['key']+'.hosts','-i',request['key'],'-P',str(request['port']),
            'root@'+request['host']+':'+request['source_root']+'/'+item['path'],str(temp)]
        try:
            p=subprocess.run(command,capture_output=True,timeout=1500)
            assert p.returncode==0,p.stderr.decode(errors='replace')[-1500:]
            assert temp.stat().st_size==item['bytes'] and digest(temp)==item['sha256'],str(path)
            temp.replace(path)
        finally:temp.unlink(missing_ok=True)
        return dict(bytes=item['bytes'],copied=True)
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(copy,request['files']))
    result=dict(status='passed',root=str(root),files=len(rows),bytes=sum(v['bytes'] for v in rows),
        copied_files=sum(v['copied'] for v in rows),copied_bytes=sum(v['bytes'] for v in rows if v['copied']),elapsed_seconds=time.monotonic()-start)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
