"""Independent NumPy projections and reconstruction from retained raw outputs."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    torch.set_num_threads(4)
    audit=json.loads((HERE/'output_heatmap_verification.json').read_text())
    raw_path=ROOT/'runs/239-2026-11-07-mlp-output-change-alignment/artifacts/spark/payload/worker/outputs.pt'
    assert sha(raw_path)==audit['sources'][raw_path.relative_to(ROOT).as_posix()]
    raw=torch.load(raw_path,weights_only=False)['outputs'].numpy()
    artifact=HERE/audit['patterns']['path'];assert sha(artifact)==audit['patterns']['sha256']
    saved=torch.load(artifact,weights_only=False)
    basis=saved['patterns'].numpy().reshape(10,50000)
    csv_path=HERE/audit['values_csv']['path'];assert sha(csv_path)==audit['values_csv']['sha256']
    with csv_path.open() as f:rows=list(csv.DictReader(f))
    states=np.array([int(r['state']) for r in rows]);assert np.array_equal(states,np.arange(448,801))
    plotted=np.array([[float(r[f'mode_{k+1:02d}_coefficient_over_sqrtN']) for k in range(10)] for r in rows])
    coefficients=plotted*np.sqrt(5000)
    np.testing.assert_allclose(basis@basis.T,np.eye(10),rtol=0,atol=1e-10)
    assert all(basis[k,np.argmax(np.abs(basis[k]))]>0 for k in range(10))
    total=residual=0.;projection_error=0.
    for j in range(0,353,32):
        end=min(j+32,353)
        delta=(raw[j+2:end+2].astype(np.float64)-raw[j+1:end+1].astype(np.float64)).reshape(end-j,50000)
        projected=delta@basis.T
        projection_error=max(projection_error,float(np.max(np.abs(projected-coefficients[j:end]))))
        np.testing.assert_allclose(projected,coefficients[j:end],rtol=1e-10,atol=1e-10)
        total+=float(np.sum(delta**2));residual+=float(np.sum((delta-projected@basis)**2))
    fractions=np.sum(coefficients**2,axis=0)/total
    np.testing.assert_allclose(fractions,audit['energy_fractions'][:10],rtol=1e-12,atol=1e-14)
    assert abs(residual/total-audit['discarded_energy_fraction'])<1e-12
    result=dict(status='passed',states=353,patterns=10,coordinates_per_pattern=50000,coefficient_checks=3530,
        maximum_projection_absolute_difference=projection_error,retained_energy_fraction=float(fractions.sum()),
        independently_reconstructed_relative_squared_error=residual/total,
        orthogonality_max_absolute_error=float(np.max(np.abs(basis@basis.T-np.eye(10)))),
        sign_rule_verified=True,generator_sha256=sha(Path(__file__)))
    (HERE/'svd_heatmap_independent_verification.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print(json.dumps(result))


if __name__=='__main__':main()
