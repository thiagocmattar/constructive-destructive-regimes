"""Independent source-clock, forecasting, aggregation and figure checks."""
import csv
import importlib.util
import json
from pathlib import Path

import fitz
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('peak_analysis',HERE/'15_peak_temporal_analysis.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def independent_average_precision(y,p):
    order=np.argsort(-p,kind='stable');ordered=y[order];probability=p[order]
    ends=np.flatnonzero(np.r_[probability[1:]!=probability[:-1],True])
    true=np.cumsum(ordered)[ends];recall=true/sum(y);precision=true/(ends+1)
    return float(sum(np.diff(np.r_[0,recall])*precision))


def main():
    result=analysis.read(HERE/'peak_temporal_results.json')
    figure=analysis.read(HERE/'peak_micro_three_models_manifest.json')
    assert result['generator_sha256']==analysis.sha(ROOT/result['generator'])
    assert figure['generator_sha256']==analysis.sha(ROOT/figure['generator'])
    assert figure['status']=='passed'
    for record in [result,figure]:
        for path,digest in record['sources'].items():assert analysis.sha(ROOT/path)==digest
    for path,digest in result['outputs'].items():assert analysis.sha(ROOT/path)==digest
    # Features at state100 must survive arbitrary changes after state100; the
    # future label must respond. This also checks the eligible-state boundary.
    synthetic=dict(id='mlp:0.001',task='mlp',lr=.001,loss=np.ones(1025),
                   projected=np.ones(1024),gradient=np.ones(1024),momentum=np.ones(1024))
    rows,base,diag=analysis.dataset([synthetic],8,.1)
    index=next(i for i,r in enumerate(rows) if r['state']==100)
    original_base,original_diag=base[index].copy(),diag[index].copy()
    assert rows[index]['target']==0
    synthetic['loss'][101:109]=2
    synthetic['projected'][101:]=3;synthetic['gradient'][101:]=4;synthetic['momentum'][101:]=5
    rows,base,diag=analysis.dataset([synthetic],8,.1)
    index=next(i for i,r in enumerate(rows) if r['state']==100)
    assert rows[index]['target']==1
    np.testing.assert_array_equal(base[index],original_base)
    np.testing.assert_array_equal(diag[index],original_diag)
    # Rebounds before recovery must be merged, and unfinished recovery retained.
    synthetic['loss']=np.ones(1025)
    synthetic['loss'][100:105]=[1.2,1.1,1.4,1.1,.99]
    synthetic['loss'][1024]=2
    events=analysis.episodes(synthetic)
    assert len(events)==2 and events[0]['onset_state']==100 and events[0]['peak_state']==102
    assert events[0]['recovery_state']==104 and events[1]['censored']
    histories={}
    for arm in analysis.read(ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/results.json')['conditions']:
        folder=ROOT/arm['source']
        b=[json.loads(s) for s in (folder/'boundaries.jsonl').read_text().splitlines()]
        m=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
        histories[f'{arm["task"]}:{arm["lr"]:g}']=(b,m)
    predictions=list(csv.DictReader((HERE/'peak_forecasts.csv').open(newline='',encoding='utf-8')))
    assert len(predictions)==result['forecast']['forward_test_rows']==4534
    labels=[];maximum_score_error=0.
    for row in predictions:
        state=int(row['state']);b,m=histories[row['arm']]
        loss=b[state]['loss'];assert state>=64 and state+8<=1024
        assert loss<1.05*min(b[s]['loss'] for s in range(state-16,state+1))
        assert m[state]['gradient_state']==state and m[state]['update']==state+1
        assert float(row['loss'])==loss
        assert float(row['gradient_norm'])==b[state]['gradient_norm']==m[state]['streams']['raw']['norm']
        assert float(row['momentum_norm'])==m[state]['streams']['moment']['norm']
        assert float(row['eta_kappa_raw'])==b[state]['projected_gradient_change']
        future=max(b[s]['loss'] for s in range(state+1,state+9))/loss-1
        assert float(row['future_max_relative_increase'])==future
        assert int(row['target'])==int(future>=.1);labels.append(int(row['target']))
        fold=next(f for f in result['forecast']['folds'] if f['start']==int(row['fold_start']))
        assert fold['start']<=state<=fold['end']
        assert fold['last_training_outcome']<fold['first_test_history_state']
        p=m[state]['projected_gradient_change'];previous=m[state-1]['projected_gradient_change']
        growg=m[state]['streams']['raw']['norm']>1.2*m[state-1]['streams']['raw']['norm']
        growm=m[state]['streams']['moment']['norm']>1.2*m[state-1]['streams']['moment']['norm']
        expected=[p>2,p>2 and previous<=2,growg,growm,p>2 and previous<=2 and growm]
        for key,value in zip(analysis.ALARMS,expected):assert (row[key]=='True')==value
    labels=np.asarray(labels)
    for name in analysis.MODELS:
        probability=np.array([float(row['prob_'+name]) for row in predictions])
        target=result['forecast']['overall'][name]
        ap=independent_average_precision(labels,probability)
        brier=float(np.mean((labels-probability)**2))
        maximum_score_error=max(maximum_score_error,abs(ap-target['average_precision']),abs(brier-target['brier']))
        assert abs(ap-target['average_precision'])<2e-14 and abs(brier-target['brier'])<2e-14
    for key,groups in [('task',result['forecast']['per_task']),('arm',result['forecast']['per_arm'])]:
        for group,models in groups.items():
            selected=[r for r in predictions if r[key]==group]
            y=np.array([int(r['target']) for r in selected])
            for name,expected in models.items():
                p=np.array([float(r['prob_'+name]) for r in selected])
                assert expected['rows']==len(y) and expected['positive']==int(sum(y))
                assert abs(float(np.mean((y-p)**2))-expected['brier'])<2e-14
                if sum(y):assert abs(independent_average_precision(y,p)-expected['average_precision'])<2e-14
    for sensitivity in result['sensitivity']:
        for fold in sensitivity['folds']:
            assert fold['gap']==16+sensitivity['horizon']
            assert fold['last_training_outcome']<fold['first_test_history_state']
    for episode in result['events']:
        b,_=histories[episode['arm']];start=episode['onset_state'];peak=episode['peak_state']
        baseline=min(b[s]['loss'] for s in range(start-16,start))
        assert baseline==episode['baseline_loss'] and b[start]['loss']>=1.1*baseline
        end=episode['recovery_state']
        stop=end if end is not None else 1025
        assert b[peak]['loss']==max(b[s]['loss'] for s in range(start,stop))
        assert all(b[s]['loss']>=baseline for s in range(start+1,stop))
        if end is not None:assert b[end]['loss']<baseline
        m=histories[episode['arm']][1]
        eligible=[s for s in range(max(64,start-8),start)
                  if b[s]['loss']<1.05*min(b[u]['loss'] for u in range(s-16,s+1))]
        assert eligible==episode['eligible_pre_onset_states']
        for name in analysis.ALARMS:
            leads=[]
            for s in eligible:
                above=m[s]['projected_gradient_change']>2
                cross=above and m[s-1]['projected_gradient_change']<=2
                growg=m[s]['streams']['raw']['norm']>1.2*m[s-1]['streams']['raw']['norm']
                growm=m[s]['streams']['moment']['norm']>1.2*m[s-1]['streams']['moment']['norm']
                flags=dict(zip(analysis.ALARMS,[above,cross,growg,growm,cross and growm]))
                if flags[name]:leads.append(start-s)
            assert episode['alarm_leads'][name]==max(leads,default=None)
    original=analysis.read(HERE/'full_unsmoothed_manifest.json')
    original_points=0;difference_points=0
    for curve in figure['curves']:
        assert curve['smoothing_window'] is None
        if curve['row']=='loss_difference':
            loss=next(c for c in figure['curves'] if c['task']==curve['task'] and c['row']=='training_loss_log')
            assert curve['x']==loss['x'][1:]
            assert curve['y']==[v-u for u,v in zip(loss['y'],loss['y'][1:])]
            difference_points+=len(curve['x']);continue
        prior=next(c for c in original['curves'] if c['task']==curve['task'] and c['lr']==curve['lr'] and c['row']==curve['row'])
        shift=0 if curve['row'] in ['training_loss_log','R_t'] else -1
        pairs=[(x+shift,y) for x,y in zip(prior['x'],prior['y']) if curve['x'][0]<=x+shift<=curve['x'][-1]]
        assert curve['x']==[x for x,y in pairs] and curve['y']==[y for x,y in pairs]
        original_points+=len(pairs)
    assert original_points==1408 and difference_points==173
    assert analysis.sha(ROOT/figure['output'])==figure['output_sha256']
    pdf=fitz.open(ROOT/figure['output']);assert len(pdf)==1 and '(aa)' in pdf[0].get_text()
    report=dict(status='passed',generator=Path(__file__).relative_to(ROOT).as_posix(),
        generator_sha256=analysis.sha(__file__),result_sha256=analysis.sha(HERE/'peak_temporal_results.json'),
        figure_manifest_sha256=analysis.sha(HERE/'peak_micro_three_models_manifest.json'),
        all_source_hashes_verified=True,state_clock_and_no_future_increment_verified=True,
        future_perturbation_feature_invariance_verified=True,episode_merging_and_censoring_verified=True,
        all_primary_targets_and_alarm_labels_rebuilt=True,forward_predictions=4534,
        per_task_and_per_arm_scores_verified=True,all_episode_early_alarm_leads_rebuilt=True,
        eligible_positive_states=int(sum(labels)),episodes=len(result['events']),
        independent_score_max_absolute_error=maximum_score_error,
        all_training_outcomes_precede_test_feature_history=True,
        original_figure_points_exact=original_points,loss_differences_exact=difference_points,
        visual_review=figure['visual_review'])
    analysis.dump(HERE/'peak_temporal_verification.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
