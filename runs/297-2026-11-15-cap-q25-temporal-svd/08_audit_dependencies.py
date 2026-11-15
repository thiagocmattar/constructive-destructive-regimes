"""Read-only comparison of current source dependencies with original run hashes."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];checks=[];manifests={}
for number in [293,294,295]:
    run=next((root/'runs').glob(str(number)+'-*'));manifest=run/'input_manifest.json'
    manifests[str(number)]=hashlib.sha256(manifest.read_bytes()).hexdigest()
    for item in json.loads(manifest.read_text())['files']:
        if not item['path'].startswith('src/'):continue
        path=root/item['path'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        checks.append(dict(source_run=number,path=item['path'],expected=item['sha256'],actual=digest,match=digest==item['sha256']))
result=dict(status='passed' if all(x['match'] for x in checks) else 'mismatch',manifest_hashes=manifests,checks=checks)
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],checks=len(checks),mismatches=[x for x in checks if not x['match']])))
