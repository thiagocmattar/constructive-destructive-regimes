"""Hide transfer latency behind training, using only finished immutable files."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('fleet',HERE/'58_matched_fleet.py');f=importlib.util.module_from_spec(s);s.loader.exec_module(f)


def prestage(name):
    p=f.get(name)
    if p.get('deleted_epoch') or p['phase']!='arms':return
    rel=p['output'];f.authorize_spark(p)
    result=f.x.ssh(p,f'''python3 - <<'PY'
import hashlib,json
from pathlib import Path
root=Path({(f.REMOTE+'/'+rel)!r});files=[]
progress=root/'progress.json'
progress=json.loads(progress.read_text()) if progress.exists() else {{}}
for folder in root.iterdir():
 if not folder.is_dir():continue
 status=folder/'status.json'
 if not status.exists():continue
 status=json.loads(status.read_text())
 for path in folder.iterdir():
  if not path.is_file():continue
  safe=status['status']=='completed'
  if path.name.startswith('audit-') and path.suffix=='.pt':
   state=int(path.stem.split('-')[1])
   safe=safe or (progress.get('arm')==folder.name and progress.get('state',-1)>state)
  if not safe:continue
  with path.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
  files.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,sha256=digest))
print(json.dumps(files))
PY
''',timeout=120)
    inventory=json.loads(result)
    if not inventory:return
    receipt=f.copy_inventory(p,rel,inventory,'prestage')
    f.dump(f.ROOT/p['run']/(name+'_immutable_prestage.json'),dict(utc=f.x.stamp(),inventory=inventory,**receipt))
    f.emit(event='immutable_artifacts_prestaged',name=name,files=len(inventory),copied_bytes=receipt['copied_bytes'])


def main():
    running={}
    with ThreadPoolExecutor(max_workers=8) as pool:
        while not (f.S/'STOP_PRESTAGE').exists():
            for path in f.S.glob('lease-*.json'):
                p=json.loads(path.read_text());name=p['name']
                if name in running:
                    if not running[name].done():continue
                    try:running[name].result()
                    except Exception as e:f.emit(event='prestage_retry_later',name=name,error=repr(e))
                    del running[name]
                if p['phase']=='arms' and not p.get('deleted_epoch'):running[name]=pool.submit(prestage,name)
            time.sleep(90)


if __name__=='__main__':main()
