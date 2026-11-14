"""Independent endpoint reconciliation and hand-calculated censoring fixtures."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
s=importlib.util.spec_from_file_location('reducer',HERE/'60_reduce_matched.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=HERE/'matched');a=p.parse_args()
    # The largest event peak is before the fork; retain its original reference,
    # but calculate the magnitude actually observed within the treated window.
    values=np.array([5.,4.,6.,5.,4.,3.,4.,3.])
    events=r.event_rows(values,2,7,'fixture','control')
    assert len(events)==2
    assert events[0]['before_state']==1 and events[0]['left_truncated']
    assert events[0]['peak_state']==2 and events[0]['window_peak_state']==3
    assert events[0]['window_height_percent']==25 and events[0]['recovery_state']==4
    assert not events[1]['left_truncated'] and events[1]['recovery_state']==7
    censored=r.event_rows(np.array([4.,3.,5.,5.,4.]),1,4,'fixture','control')
    assert len(censored)==1 and censored[0]['right_censored'] and censored[0]['peak_state']==2
    assert censored[0]['full_recovery_duration'] is None
    assert not r.event_rows(np.array([3.,2.,1.]),0,2,'fixture','control')
    assert r.relative(0,0) is None and r.relative(3,2)==50
    factors=r.gate_factors(dict(gates={'active':{'factors':[.5,1.]},'disabled':{'disabled_zero_reference':True}}),{'active':2,'disabled':3})
    assert len(factors)==5 and np.mean(np.array(factors)<1)==.2
    report=json.loads((a.input/'results.json').read_text());curves=np.load(a.input/'curves.npz')
    summary=list(csv.DictReader((a.input/'summary.csv').open(encoding='utf-8')))
    stage_rows=list(csv.DictReader((a.input/'stage_metrics.csv').open(encoding='utf-8')))
    costs=list(csv.DictReader((a.input/'costs.csv').open(encoding='utf-8')))
    pools=np.load(a.input/'calibration.npz')
    for item in csv.DictReader((a.input/'calibration.csv').open(encoding='utf-8')):
        values=pools[item['case']+'__'+item['module']]
        assert values.shape==(8,int(item['units']))
        assert float(item['budget'])==float(np.quantile(values,float(item['quantile']),method='linear'))
    assert len(summary)==report['conditions']
    units={'mlp':{'test':10000},'resnet':{'test':10000},'transformer':{'valid':217630,'test':245550}}
    observed_evaluations=0;retained_source_checks=0
    for path,entry in report['sources'].items():
        f=ROOT/path
        assert f.stat().st_size==entry['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==entry['sha256']
        retained_source_checks+=1
    for row in summary:
        key=row['case']+'__'+row['arm']+'__';loss=curves[key+'loss'];steps=curves[key+'state']
        f,e=int(row['fork']),int(row['end']);np.testing.assert_array_equal(steps,np.arange(f,e+1))
        pv=sum(max(float(after)-float(before),0) for before,after in zip(loss[:-1],loss[1:]))
        np.testing.assert_allclose([float(row['positive_variation']),float(row['mean_loss']),float(row['final_loss']),float(row['last32_loss'])],
            [pv,sum(map(float,loss[1:]))/(e-f),float(loss[-1]),sum(map(float,loss[-32:]))/32],rtol=2e-14,atol=2e-14)
        base=next(v for v in summary if v['case']==row['case'] and v['arm']=='control')
        if float(base['positive_variation']):np.testing.assert_allclose(float(row['pv_reduction_percent']),100*(float(base['positive_variation'])-pv)/float(base['positive_variation']),rtol=2e-13,atol=2e-13)
        assert curves[key+'stage_rho'].shape==(len(r.STAGES[row['case']]),e-f+1)
        assert curves[key+'stage_change_rms'].shape==curves[key+'stage_rho'].shape
        if row['case']=='mlp':
            linear=curves[key+'mse_linear'][1:];quadratic=curves[key+'mse_quadratic'][1:]
            delta=curves[key+'mse_delta_loss'][1:];rises=delta>1e-8
            np.testing.assert_allclose(delta,linear+quadratic,rtol=0,atol=1e-10)
            assert int(row['mse_rises_gt_1em8'])==int(rises.sum())
            assert int(row['mse_rises_negative_linear_outweighed'])==int((rises&(linear<0)&(quadratic>-linear)).sum())
        for index,(name,_) in enumerate(r.STAGES[row['case']]):
            st=next(v for v in stage_rows if v['case']==row['case'] and v['arm']==row['arm'] and v['stage']==name)
            rms=curves[key+'stage_change_rms'][index,1:];rho=curves[key+'stage_rho'][index,1:]
            energy=int(st['coordinates'])*sum(float(v)**2 for v in rms)
            np.testing.assert_allclose([float(st['squared_motion_energy']),float(st['mean_rms']),float(st['max_rms']),float(st['reversal_fraction'])],
                [energy,np.mean(rms),np.max(rms),np.mean(rho[np.isfinite(rho)]<0)],rtol=2e-13,atol=2e-13)
        held=[v for v in csv.DictReader((a.input/'heldout.csv').open()) if v['case']==row['case'] and v['arm']==row['arm']]
        assert len(held)==5*len(units[row['case']])
        for v in held:assert int(v['loss_units'])==units[row['case']][v['split']]
        observed_evaluations+=5
        terminal=next(float(v['loss']) for v in held if int(v['state'])==e and v['split']=='test')
        assert terminal==float(row['terminal_test_loss'])
        cost=next(v for v in costs if v['case']==row['case'] and v['arm']==row['arm'])
        folder=r.RUNS[row['case']]/'artifacts/attempt-01'/cost['worker']/row['arm']
        boundaries=[json.loads(v) for v in (folder/'boundaries.jsonl').read_text().splitlines()]
        updates=[json.loads(v) for v in (folder/'updates.jsonl').read_text().splitlines()]
        evaluations=[json.loads(v) for v in (folder/'evaluations.jsonl').read_text().splitlines()]
        status=json.loads((folder/'status.json').read_text())
        assert int(cost['updates'])==len(updates)==e-f
        assert int(cost['full_gradient_passes'])==int(cost['probe_passes'])==len(boundaries)==e-f+1
        for field,source in [('gradient_seconds',boundaries),('probe_seconds',boundaries),
                             ('optimizer_seconds',updates),('controller_seconds',updates)]:
            assert math.isclose(float(cost[field]),math.fsum(v[field] for v in source),rel_tol=2e-13,abs_tol=1e-10)
        assert float(cost['evaluation_seconds'])==sum(v['seconds'] for v in evaluations)
        assert int(cost['process_peak_cuda_bytes_through_arm'])==status['peak_cuda_bytes']
        assert float(cost['wall_seconds'])==status['elapsed_seconds']
        total=sum(float(cost[k]) for k in ['gradient_seconds','probe_seconds','optimizer_seconds',
                                         'controller_seconds','evaluation_seconds','other_seconds'])
        assert math.isclose(total,float(cost['wall_seconds']),rel_tol=2e-13,abs_tol=1e-10)
    assert observed_evaluations==report['heldout_evaluation_states']
    result=dict(status='passed',conditions=len(summary),source_files=retained_source_checks,heldout_states=observed_evaluations,
        component_timing_rows=len(costs),
        fixtures=['carried event with pre-fork maximum','first tied maximum','right censoring','monotone loss','zero denominator'],
        result_sha256=r.sha(a.input/'results.json'),script_sha256=r.sha(Path(__file__)))
    r.write(a.input/'verification.json',result);print(json.dumps(result))


if __name__=='__main__':main()
