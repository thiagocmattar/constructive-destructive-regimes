"""Verify the adaptive stopping decision and identical full fork states locally."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUNS=[ROOT/'runs'/name for name in [
 '247-2026-11-07-mlp-frozen-lr-0p01','248-2026-11-07-mlp-frozen-lr-0p005',
 '249-2026-11-07-mlp-frozen-lr-0p0025','250-2026-11-07-mlp-frozen-lr-0p00125',
 '251-2026-11-07-mlp-frozen-lr-0p000625']]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    plan=json.loads((HERE/'halving_plan.json').read_text());assert plan['stop_rule']['absolute_tolerance']==1e-8
    spec=importlib.util.spec_from_file_location('checks',RUNS[0]/'03_verify.py')
    checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
    reference=ROOT/'runs/245-2026-11-07-mlp-frozen-lr-double/artifacts/spark/payload/worker/fork500.pt'
    fork=torch.load(reference,map_location='cpu',weights_only=False)
    records=[];sources={reference.relative_to(ROOT).as_posix():sha(reference)}
    for i,run in enumerate(RUNS):
        raw=run/'artifacts/spark/payload/worker'
        assert json.loads((run/'verification.json').read_text())['status']=='passed'
        assert json.loads((run/'retrieval.json').read_text())['verified']
        q=json.loads((run/'halving_assessment.json').read_text());assert q['lr']==.01/2**i
        assert q['boundaries_sha256']==sha(raw/'boundaries.jsonl')
        rows=[json.loads(s) for s in (raw/'boundaries.jsonl').read_text().splitlines()]
        selected=[r for r in rows if 500<=r['state']<=800]
        assert [r['state'] for r in selected]==list(range(500,801))
        loss=np.array([r['loss'] for r in selected]);change=np.diff(loss)
        passed=bool(np.isfinite(loss).all() and change.max()<=1e-8 and loss.max()<=loss[0]+1e-8)
        assert passed==(q['status']=='passed')
        np.testing.assert_allclose([q['loss_max'],q['loss_final'],q['maximum_adjacent_increase']],
                                  [loss.max(),loss[-1],change.max()],rtol=0,atol=0)
        assert q['increases_above_tolerance']==int((change>1e-8).sum())
        current=torch.load(raw/'fork500.pt',map_location='cpu',weights_only=False)
        for key in ['completed_updates','model','optimizer','rng','current_gradient','previous_gradient']:
            checks.equal(fork[key],current[key])
        records.append({k:v for k,v in q.items() if k!='manifest'})
        for p in [run/'config.json',run/'halving_assessment.json',run/'verification.json',run/'retrieval.json',raw/'fork500.pt']:
            sources[p.relative_to(ROOT).as_posix()]=sha(p)
    assert all(q['status']=='peak_present' for q in records[:-1]) and records[-1]['status']=='passed'
    summary=dict(status='verified',stop_lr=records[-1]['lr'],stop_run=records[-1]['run'],
        criterion=plan['stop_rule'],records=records,all_fork500_model_optimizer_rng_gradients_exactly_identical=True,
        first_pass_in_requested_halving_sequence=True,source_hashes=sources,generator_sha256=sha(Path(__file__)))
    (HERE/'halving_summary.json').write_text(json.dumps(summary,indent=2)+'\n',newline='\n')
    plan.update(status='completed',next_lr=None,stop_lr=records[-1]['lr'],tested_rates=[q['lr'] for q in records],
        stopping_decision_locally_verified=True,user_confirmed_stop=True)
    (HERE/'halving_plan.json').write_text(json.dumps(plan,indent=2)+'\n',newline='\n')
    print(json.dumps(dict(status=summary['status'],stop_lr=summary['stop_lr'],records=records),indent=2))


if __name__=='__main__':main()
