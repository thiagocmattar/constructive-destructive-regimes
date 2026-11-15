"""Retrieve compact evidence from owned Spark and verify every byte."""
from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent
l=json.loads((HERE/'launch.json').read_text());host='tcml-spark@spark-2277.local'
(HERE/'raw').mkdir(exist_ok=True)
for case in ['mlp','resnet','transformer']:
    remote=l['output']+'/raw/'+case
    subprocess.run(['ssh',host,'test -f '+remote+'/checksums.json'],check=True)
    subprocess.run(['scp','-r',host+':'+remote,str(HERE/'raw')],check=True)
    checks=json.loads((HERE/'raw'/case/'checksums.json').read_text())
    for name,r in checks.items():
        p=HERE/'raw'/case/name
        assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],p
    print(case,len(checks),'verified')
receipt={c:json.loads((HERE/'raw'/c/'checksums.json').read_text()) for c in ['mlp','resnet','transformer']}
(HERE/'retrieval.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')
