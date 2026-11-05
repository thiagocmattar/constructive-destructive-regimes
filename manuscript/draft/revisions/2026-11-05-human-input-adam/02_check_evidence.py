"""Audit displayed Adam claims and every new figure against retained data."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
AN=ROOT/'analyses/057-2026-11-05-adam-response-memory'
PRIOR=ROOT/'analyses/055-2026-11-05-adam-directional-amplification'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8')))

def main():
    incidence=rows(AN/'event_incidence.csv');phase=rows(AN/'phase_summary.csv')
    features=rows(AN/'event_features.csv');fits=json.loads((AN/'regressions.json').read_text())
    assert len(features)==127
    for threshold,n in [(.05,206),(.1,129),(.2,74)]:
        es=[r for r in incidence if float(r['threshold'])==threshold]
        assert sum(int(r['recovered']) for r in es)==n
        assert sum(int(r['censored']) for r in es)==2
    es=[r for r in incidence if float(r['threshold'])==.1]
    assert sum(int(r['negative_prerise']) for r in es)==117
    assert sum(int(r['defined_prerise']) for r in es)==127
    assert sum(int(r['defined_prerise']) for r in es if r['arm'] in ['mlp:0.0003','mlp:0.001'])==84
    for phase_name,total,negative in [('rise',794,794),('recovery',1376,1374)]:
        ps=[r for r in phase if r['phase']==phase_name]
        assert sum(int(r['updates']) for r in ps)==total
        assert sum(int(r['locally_descending']) for r in ps)==negative
    cos=[float(r['mean_gradient_direction']) for r in phase if r['phase']=='rise']
    assert round(min(cos),3)==.071 and round(max(cos),3)==.767
    improvements=[];worse={'logA':0,'A':0}
    for arm in dict.fromkeys(r['arm'] for r in fits):
        for scope in ['in_sample','held_out_rate']:
            fs={r['model']:r for r in fits if r['arm']==arm and r['scope']==scope}
            if scope=='in_sample':improvements.append(fs['rho+logA']['r2']-fs['logA']['r2'])
            else:
                for family in worse:worse[family]+=fs['rho+'+family]['mse']>fs[family]['mse']
    assert worse=={'logA':7,'A':5}
    assert round(min(improvements),3)==.006 and round(max(improvements),3)==.221
    coeffs=[f for f in fits if f['scope']=='in_sample' and f['model']=='rho+logA']
    assert {f['arm'] for f in coeffs if f['coefficients'][1]>0}=={'mlp:0.0001','transformer:0.001'}
    # Independent recreation of the Analysis055 quiet-state controls from raw logs.
    run=json.loads((ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/results.json').read_text())
    controls=[]
    for task,expected in [('resnet',(953,726,186,0)),('transformer',(953,764,215,0))]:
        c=next(c for c in run['conditions'] if c['task']==task and c['lr']==.0003)
        folder=ROOT/c['source'];bs=[json.loads(s) for s in (folder/'boundaries.jsonl').read_text().splitlines()]
        ms=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
        loss=[b['loss'] for b in bs];eligible=[]
        for s in range(64,1017):
            if loss[s] < 1.05*min(loss[max(0,s-16):s+1]):eligible.append(s)
        actual=(len(eligible),sum(ms[s]['streams']['raw']['cosine']<0 for s in eligible),
                sum(ms[s]['projected_gradient_change']>2 for s in eligible),
                sum(max(loss[s+1:s+9])/loss[s]-1>=.1 for s in eligible))
        assert actual==expected
        saved=[r for r in rows(PRIOR/'derived/states.csv') if r['task']==task and float(r['lr'])==.0003 and r['eligible']=='1']
        assert [int(r['state']) for r in saved]==eligible
        controls.append(dict(task=task,eligible=actual[0],negative_cosine=actual[1],z_above_two=actual[2],future_rises=actual[3]))
    fm=json.loads((AN/'figure_manifest.json').read_text());points=0
    for curve in fm['curves']:
        if curve['tag'].startswith('ablation:'):
            _,task,model=curve['tag'].split(':');lrs=sorted({float(e['lr']) for e in features if e['task']==task})
            want=[next(f['r2'] for f in fits if f['task']==task and f['lr']==lr and f['model']==model and f['scope']=='in_sample') for lr in lrs]
            np.testing.assert_array_equal(want,curve['y'])
        else:
            task,lr,kind=curve['tag'].split(':');lr=float(lr)
            es=[r for r in features if r['task']==task and float(r['lr'])==lr]
            x=np.array([float(r['rho'])*float(r['A']) for r in es]);y=np.array([float(r['log_loss_ratio']) for r in es])
            if kind=='points':
                np.testing.assert_array_equal(x,curve['x']);np.testing.assert_allclose(np.exp(y),curve['y'],rtol=1e-14)
            else:
                assert min(curve['x'])==min(x) and max(curve['x'])==max(x)
                slope=np.cov(x,y,ddof=0)[0,1]/np.var(x);intercept=y.mean()-slope*x.mean()
                np.testing.assert_allclose(np.exp(intercept+slope*np.array(curve['x'])),curve['y'],rtol=1e-12)
        points+=len(curve['x'])
    # Numeric table text must match each retained row, including missing-value dashes.
    table=(ROOT/fm['table']['path']).read_text()
    for e in [r for r in incidence if float(r['threshold'])==.1]:
        lr=float(e['lr']);rs={float(r['threshold']):r for r in incidence if r['task']==e['task'] and float(r['lr'])==lr}
        part=f"& {lr:g} & {rs[.05]['recovered']} & {e['recovered']} & {rs[.2]['recovered']} & {e['censored']} & {100*float(e['occupancy']):.1f}"
        assert part in table
    report=dict(status='passed',controls=controls,figure_curves=len(fm['curves']),figure_points=points,
        counts_verified=True,phase_counts_verified=True,regression_claims_verified=True,
        table_rows=9,results_sha256=sha(AN/'results.json'),figure_manifest_sha256=sha(AN/'figure_manifest.json'),
        prior_states_sha256=sha(PRIOR/'derived/states.csv'),checker_sha256=sha(HERE/'02_check_evidence.py'))
    (HERE/'evidence_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report))

if __name__=='__main__':main()
