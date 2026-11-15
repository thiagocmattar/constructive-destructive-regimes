"""Retrieve and independently verify compact Run297 spectral evidence."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'runs/297-2026-11-15-cap-q25-temporal-svd'
L=json.loads((ROOT/'analyses/066-2026-11-15-activation-inactivity/launch.json').read_text())
HOST='tcml-spark@spark-2277.local';REL=RUN.relative_to(ROOT).as_posix()+'/artifacts/attempt-01'
def ssh(script):
    p=subprocess.run(['ssh','-o','BatchMode=yes',HOST,'bash -s'],input=script.encode(),capture_output=True,timeout=300)
    if p.returncode:raise RuntimeError((p.stdout+p.stderr).decode())
    return p.stdout.decode()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('case',choices=['mlp','resnet','transformer']);a=p.parse_args()
    root=HERE/'svd_q25';raw=root/'raw';raw.mkdir(parents=True,exist_ok=True)
    remote=L['root']+'/'+REL+'/'+a.case
    script=f"""set -e
cd {remote}
test -f reduction/inventory.json
find . -type f \\( -name '*.json' -o -name '*.jsonl' -o -name '*.npz' \\) -print0 | tar --null -T - -czf ../{a.case}-compact.tar.gz
sha256sum ../{a.case}-compact.tar.gz
"""
    digest=ssh(script).split()[0];archive=raw/(a.case+'-compact.tar.gz')
    subprocess.run(['scp','-q',HOST+':'+L['root']+'/'+REL+'/'+a.case+'-compact.tar.gz',str(archive)],check=True,timeout=300)
    assert sha(archive)==digest
    dest=raw/a.case;dest.mkdir(exist_ok=True)
    with tarfile.open(archive) as t:t.extractall(dest,filter='data')
    inv=json.loads((dest/'reduction/inventory.json').read_text());count=0
    for rel,item in inv.items():
        f=dest/rel
        if f.exists():assert f.stat().st_size==item['bytes'] and sha(f)==item['sha256'];count+=1
    receipt=dict(case=a.case,archive_sha256=digest,verified_files=count,remote=remote,retained_bytes=sum(x['bytes'] for x in inv.values()))
    (root/(a.case+'-retrieval.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
