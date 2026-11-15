"""Exact window retention from retained, untruncated temporal Gram matrices."""
from pathlib import Path
import csv, hashlib, json
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
paths = {
 'mlp': ROOT/'analyses/062-2026-11-07-mlp-gradient-amplification/artifacts/all-layer-svd/grams.npz',
 'resnet': ROOT/'analyses/063-2026-11-07-architecture-instability-tracing/resnet_390_450_results.json',
 'transformer': ROOT/'runs/243-2026-11-07-architecture-instability-tracing/artifacts/cloud/transformer/grams.npz'}
windows = {'mlp': (451,501), 'resnet': (390,450), 'transformer': (553,601)}
rows, checks = [], []
for case,path in paths.items():
    data = json.loads(path.read_text())['temporal_grams'] if case=='resnet' else np.load(path)
    for site in data:
        if site in ('raw_gradient','parameter_step'): continue
        K = np.asarray(data[site],dtype=float)
        K = K[3:54,3:54] if case=='mlp' else K[1:,1:]
        a,b=windows[case]
        assert K.shape==(b-a+1,b-a+1)
        np.testing.assert_allclose(K,K.T,rtol=1e-12,atol=1e-12)
        eigen,U=np.linalg.eigh(K)
        assert eigen.min()>=-1e-10*max(eigen.max(),1)
        net2=float(K.sum()); path_length=float(np.sqrt(np.maximum(np.diag(K),0)).sum())
        other=float(np.sum(np.maximum(eigen,0)*(U.T@np.ones(len(K)))**2))
        np.testing.assert_allclose(net2,other,rtol=1e-9,atol=1e-8)
        R=np.sqrt(max(net2,0))/path_length if path_length else None
        assert R is None or 0<=R<=1+1e-12
        rows.append(dict(case=case,site=site,first_step=a,last_step=b,changes=len(K),
                         net_displacement=np.sqrt(max(net2,0)),path_length=path_length,retention=R))
        checks.append(dict(case=case,site=site,min_eigenvalue=float(eigen.min()),eigen_net_squared=other,
                           direct_net_squared=net2))
with (HERE/'representation-retention.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
manifest=dict(definition='sqrt(sum(K))/sum(sqrt(diag(K))); incoming changes a:b, no mode truncation',
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
    checks=checks,rows=rows)
(HERE/'retention-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
print(json.dumps([r for r in rows if r['site'] in ['output','output.log_probabilities','encoder.call0.output']]))
