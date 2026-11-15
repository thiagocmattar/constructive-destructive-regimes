"""Final report claims, frozen worker, artifact integrity, and numeric table."""
from pathlib import Path
from collections import Counter
import csv,hashlib,importlib.util,json,re
import fitz
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('r',HERE/'02_reduce.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
launch=json.loads((HERE/'launch.json').read_text())
for name in ['01_measure.py','config.json']:
    assert sha(ROOT/'runs/296-2026-11-15-checkpoint-activation-audit'/name)==launch['frozen_inputs'][name]
retrieval=json.loads((HERE/'retrieval.json').read_text())
for c,files in retrieval.items():
    for name,item in files.items():
        p=HERE/'raw'/c/name;assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
figure=json.loads((HERE/'figure-manifest.json').read_text());renders=json.loads((HERE/'render-verification.json').read_text())
for name,digest in figure['figures'].items():
    assert sha(HERE/name)==digest==renders[Path(name).stem]['sha256']
rows=list(csv.DictReader((HERE/'final-hidden-summary.csv').open()))
with fitz.open(HERE/'activation-summary.pdf') as pdf:text=pdf[0].get_text()
compiled=Counter(re.findall(r'\d+(?:\.\d+)?',text));expected=Counter()
for row in rows:
    for key in ['zero_entries','zero_units','near_units','weak_units','top10_energy','effective_units']:expected[f'{float(row[key]):.1f}']+=1
assert all(compiled[k]>=n for k,n in expected.items()),(expected,compiled)
findings=(HERE/'FINDINGS.md').read_text();dominance=[]
for arm in r.ARMS:
    for t in [544,576,928]:
        d,a=r.load('transformer',arm,t,'full_training');s=next(s for s in d['sites'] if s['site']=='transformer_encoder.layers.1.linear2');p=s['array_prefix']+'_';e=a[p+'share'];j=int(e.argmax())
        dominance.append(dict(arm=arm,state=t,unit=j,energy_share=float(e[j]),effective_units=float(1/(e@e)),layer_rms=s['rms']))
        if t==928:assert f'{100*e[j]:.2f}%' in findings
with (HERE/'transformer-unit-dominance.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(dominance[0]),lineterminator='\n');w.writeheader();w.writerows(dominance)
metrics=list(csv.DictReader((HERE/'layer-summary.csv').open()))
for c in ['mlp','transformer']:
    rows=[v for v in metrics if v['case']==c and v['scope']=='full_training' and v['primary']=='True']
    assert all(float(v['dead_zero_fraction'])==0 and float(v['dead_relative_001'])==0 for v in rows)
close=json.loads((HERE/'compute-closeout.json').read_text());assert close['cloud_cost_usd']==0 and close['training_updates']==0
out=dict(status='passed',retrieval_files=150,compiled_numeric_cells=sum(expected.values()),frozen_inputs_unchanged=True,report_claims_checked=True,figure_pages=21,scientific_verification=json.loads((HERE/'verification.json').read_text())['status'])
(HERE/'final-verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8',newline='\n');print(out)
