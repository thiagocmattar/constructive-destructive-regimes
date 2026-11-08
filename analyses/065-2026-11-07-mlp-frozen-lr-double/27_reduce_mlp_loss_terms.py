"""Reduce the exact output-space squared-loss identity from complete scalar histories."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN_IDS=[269,272,275,276,277,278,279,282,283,284]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    replay=json.loads((HERE/'artifacts/mlp_loss_terms_audit.json').read_text());assert replay['status']=='passed'
    assert replay['source_sha256']==sha(HERE/'26_audit_mlp_loss_terms.py')
    results=[]
    for number in RUN_IDS:
        run=next((ROOT/'runs').glob(str(number)+'-*'));worker=run/'artifacts/spark/scientific/worker';cfg=json.loads((run/'config.json').read_text())
        manifest=json.loads((worker/'manifest.json').read_text());assert manifest['status']=='completed'
        for arm in manifest['arms']:
            folder=worker/arm;b=[json.loads(l) for l in (folder/'boundaries.jsonl').read_text().splitlines()];u=[json.loads(l) for l in (folder/'updates.jsonl').read_text().splitlines()]
            loss=np.array([r['loss'] for r in b]);change=np.diff(loss);rms=np.array([r['representations']['output']['change_rms'] for r in b[1:]])
            q=5*rms*rms;linear=change-q;parameter_linear=np.array([r['raw_gradient_dot_actual'] for r in u]);assert len(change)==len(u)
            # The identity is exact for FP64 losses; stored loss reductions add a small known rounding error.
            checks=[r for r in replay['checks'] if r['run']==number and r['arm']==arm];assert len(checks)==3
            for r in checks:
                i=r['state']-cfg['fork'];np.testing.assert_allclose(q[i],r['quadratic'],rtol=2e-8,atol=2e-11)
                assert abs(linear[i]-r['linear'])<=2*r['loss_reduction_rounding_error']+2e-12
            increases=change>1e-8;overshoot=increases&(linear < -1e-8)
            results.append(dict(run=number,seed=cfg['seed'],arm=arm,states=[r['state'] for r in u],loss_change=change.tolist(),functional_linear=linear.tolist(),functional_quadratic=q.tolist(),parameter_linear=parameter_linear.tolist(),
                summary=dict(updates=len(u),loss_increases=int(increases.sum()),functional_overshoots=int(overshoot.sum()),
                    fraction_increases_with_functional_descent=float(overshoot.sum()/increases.sum()) if increases.any() else None,
                    fraction_increases_with_parameter_descent=float(np.mean(parameter_linear[increases]<0)) if increases.any() else None,
                    mean_quadratic=float(q.mean()),mean_functional_linear=float(linear.mean()))))
    (HERE/'mlp_loss_terms.json').write_text(json.dumps(dict(status='verified',generator_sha256=sha(Path(__file__)),audit_sha256=sha(HERE/'artifacts/mlp_loss_terms_audit.json'),loss_increase_tolerance=1e-8,results=results),indent=2)+'\n')
    print(json.dumps([dict(run=r['run'],arm=r['arm'],**r['summary']) for r in results],indent=2))


if __name__=='__main__':main()
