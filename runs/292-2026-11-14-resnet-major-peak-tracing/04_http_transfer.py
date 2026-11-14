"""Authenticated chunked HTTPS transfer to this task's temporary Pod only."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import secrets

import requests

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud',HERE/'02_cloud.py')
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)

SERVER=r'''
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
import hashlib,json,hmac
root=Path('/workspace/cdr292-transfer'); cfg=json.loads((root/'config.json').read_text())
class Handler(BaseHTTPRequestHandler):
 def authorized(self):
  if hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+cfg['token']): return True
  self.send_error(403);return False
 def do_POST(self):
  if not self.authorized():return
  if self.path.startswith('/part/'):
   try:part=int(self.path.rsplit('/',1)[-1])
   except ValueError:self.send_error(400);return
   length=int(self.headers.get('Content-Length','0'))
   if not 0<=part<cfg['parts'] or not 0<length<=cfg['chunk_size']:self.send_error(400);return
   p=root/f'part-{part}';remaining=length
   with p.open('wb') as f:
    while remaining:
     block=self.rfile.read(min(1024*1024,remaining))
     if not block:raise IOError('Short upload')
     f.write(block);remaining-=len(block)
   reply={'part':part,'bytes':p.stat().st_size}
  elif self.path=='/assemble':
   target=Path('/workspace/inputs292-http.tar.gz');h=hashlib.sha256()
   with target.open('wb') as f:
    for part in range(cfg['parts']):
     with (root/f'part-{part}').open('rb') as source:
      while block:=source.read(1024*1024):f.write(block);h.update(block)
   assert h.hexdigest()==cfg['sha256'] and target.stat().st_size==cfg['bytes']
   reply={'sha256':h.hexdigest(),'bytes':target.stat().st_size}
  else:self.send_error(404);return
  body=json.dumps(reply).encode();self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def do_GET(self):
  if not self.authorized():return
  if self.path!='/result':self.send_error(404);return
  p=Path('/workspace/results292.tar.gz')
  if not p.exists():self.send_error(404);return
  self.send_response(200);self.send_header('Content-Length',str(p.stat().st_size));self.end_headers()
  with p.open('rb') as f:
   while block:=f.read(1024*1024):self.wfile.write(block)
ThreadingHTTPServer(('0.0.0.0',8000),Handler).serve_forever()
'''


def main():
    pod=json.loads((HERE/'cloud_lease.json').read_text());archive=HERE/'scratch/inputs.tar.gz'
    chunk_size=32*1024*1024
    cfg=dict(token=secrets.token_urlsafe(32),chunk_size=chunk_size,parts=(archive.stat().st_size+chunk_size-1)//chunk_size,
             bytes=archive.stat().st_size,sha256=c.x.file_hash(archive))
    secret=HERE/'scratch/http-config.json'; secret.write_text(json.dumps(cfg))
    code="python3 - <<'PY'\nfrom pathlib import Path\nimport os,subprocess,json\n"
    code+="root=Path('/workspace/cdr292-transfer');root.mkdir(exist_ok=True)\n"
    code+=f"(root/'config.json').write_text({json.dumps(cfg)!r});os.chmod(root/'config.json',0o600)\n"
    code+=f"(root/'server.py').write_text({SERVER!r})\n"
    code+="with (root/'server.log').open('w') as log:\n p=subprocess.Popen(['python3',str(root/'server.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid)))\nPY\n"
    server=json.loads(c.x.ssh(pod,code));url=f"https://{pod['id']}-8000.proxy.runpod.net"
    headers={'Authorization':'Bearer '+cfg['token']}
    def upload(part):
        with archive.open('rb') as f:f.seek(part*chunk_size);data=f.read(chunk_size)
        response=requests.post(url+f'/part/{part}',headers=headers,data=data,timeout=(20,180));response.raise_for_status()
        result=response.json();assert result['bytes']==len(data)
        print(json.dumps(result),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:parts=list(pool.map(upload,range(cfg['parts'])))
    response=requests.post(url+'/assemble',headers=headers,timeout=(20,180));response.raise_for_status();receipt=response.json()
    assert receipt['sha256']==cfg['sha256'] and receipt['bytes']==cfg['bytes']
    c.dump(HERE/'http_transfer.json',dict(status='verified',url=url,server_pid=server['pid'],parts=parts,**receipt))
    print(json.dumps(dict(status='verified',**receipt)),flush=True)


if __name__=='__main__':main()
