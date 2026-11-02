"""Independent source reductions, temporal fits, table and vector-PDF checks."""
import csv
import importlib.util
import json
import math
from pathlib import Path

import fitz
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'paper_evidence'
spec=importlib.util.spec_from_file_location('prior_checks',HERE/'17_verify_peak_temporal.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
read=prior.analysis.read;sha=prior.analysis.sha


def check_scores(rows,expected):
    y=np.array([int(r['target']) for r in rows])
    for name,stats in expected.items():
        p=np.array([float(r['prob_'+name]) for r in rows])
        assert len(y)==stats['rows'] and sum(y)==stats['positive']
        assert abs(float(y.mean())-stats['prevalence'])<1e-15
        if sum(y): assert abs(prior.independent_average_precision(y,p)-stats['average_precision'])<2e-13
        else: assert stats['average_precision'] is None
        assert abs(float(np.mean((y-p)**2))-stats['brier'])<2e-13


def main():
    result=read(OUT/'evidence_results.json');fig=read(OUT/'figure_manifest.json')
    old=read(HERE/'peak_temporal_results.json')
    for record in [result,fig]:
        assert sha(ROOT/record['generator'])==record['generator_sha256']
        for path,digest in record['sources'].items(): assert sha(ROOT/path)==digest,path
        for path,digest in record['outputs'].items(): assert sha(ROOT/path)==digest,path
    raw={}
    arms=read(ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/results.json')['conditions']
    arms=sorted(arms,key=lambda a:(['mlp','resnet','transformer'].index(a['task']),a['lr']))
    for a in arms:
        folder=ROOT/a['source']
        b=[json.loads(x) for x in (folder/'boundaries.jsonl').read_text().splitlines()]
        m=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
        assert all(x['gradient_state']==i and x['update']==i+1 for i,x in enumerate(m))
        raw[f"{a['task']}:{a['lr']:g}"]=dict(loss=np.array([x['loss'] for x in b]),
            projected=np.array([np.nan if x['projected_gradient_change'] is None else x['projected_gradient_change'] for x in b[:-1]]),
            raw_cosine=np.array([np.nan if x['streams']['raw']['cosine'] is None else x['streams']['raw']['cosine'] for x in m]),
            gradient=np.array([x['gradient_norm'] for x in b[:-1]]),
            momentum=np.array([x['streams']['moment']['norm'] for x in m]))
    for r in result['episode_summary']:
        subset=[e for e in old['events'] if e['arm']==r['arm'] and (r['severity']=='all' or e['strong'])]
        leads=[e['alarm_leads'][r['alarm']] for e in subset if e['alarm_leads'][r['alarm']] is not None]
        assert len(subset)==r['episodes'] and len(leads)==r['hits'] and len(subset)-len(leads)==r['misses']
        assert sum(e['censored'] for e in subset)==r['censored_recoveries']
        if leads:
            assert float(np.median(leads))==r['lead_median']
            assert min(leads)==r['lead_min'] and max(leads)==r['lead_max']
    primary=list(csv.DictReader((HERE/'peak_forecasts.csv').open(newline='',encoding='utf-8')))
    for r in result['alarm_summary']:
        subset=[p for p in primary if p['arm']==r['arm']]
        tp=sum(p[r['alarm']]=='True' and p['target']=='1' for p in subset)
        fp=sum(p[r['alarm']]=='True' and p['target']=='0' for p in subset)
        assert (tp,fp,len(subset))==(r['tp'],r['fp'],r['states'])
    for e in result['aligned_events']:
        a=raw[e['arm']];event=next(v for v in old['events'] if v['arm']==e['arm'] and v['onset_state']==e['onset'])
        states=np.arange(e['onset']-8,e['onset']+9)
        for key in ['loss','projected','raw_cosine','gradient','momentum']:
            divisor=event['baseline_loss'] if key=='loss' else a[key][states[0]] if key in ['gradient','momentum'] else 1
            np.testing.assert_array_equal(e[key],a[key][states]/divisor)
    rows=list(csv.DictReader((OUT/'robustness_forecasts.csv').open(newline='',encoding='utf-8')))
    pmap={(r['arm'],r['state']):r for r in primary}
    for r in rows:
        a=raw[r['arm']];t=int(r['state'])
        assert a['loss'][t]<1.05*min(a['loss'][t-16:t+1])
        assert int(r['target'])==int(max(a['loss'][t+1:t+9])/a['loss'][t]-1>=.1)
        for field,key in [('loss','loss'),('eta_kappa_raw','projected'),('gradient_norm','gradient'),('momentum_norm','momentum')]:
            assert float(r[field])==a[key][t]
        if r['group']=='components_and_richer_loss':
            for name in ['loss_history','projected_change']:
                assert abs(float(r['prob_'+name])-float(pmap[r['arm'],r['state']]['prob_'+name]))<1e-12
    for group,stats in result['robustness'].items():
        subset=[r for r in rows if r['group']==group]
        assert len(subset)==4534 and len({(r['arm'],r['state']) for r in subset})==4534
        check_scores(subset,stats['overall'])
        for task,expected in stats['per_task'].items(): check_scores([r for r in subset if r['task']==task],expected)
        for block in stats['blocks']:
            assert block['last_training_outcome']<block['first_test_input']
            selected=[r for r in subset if int(r['fold_start'])==block['start']]
            check_scores(selected,block['overall'])
            for task,expected in block['per_task'].items(): check_scores([r for r in selected if r['task']==task],expected)
    # Rebuild all feature matrices directly, use NumPy training-only scaling,
    # and independently refit every richer/delayed model in every time block.
    features=[];times=[];labels=[];identities=[]
    ids=list(raw)
    for i,arm in enumerate(ids):
        a=raw[arm]
        for t in range(64,1017):
            if a['loss'][t]>=1.05*min(a['loss'][t-16:t+1]): continue
            logloss=np.log(a['loss'])
            rich=[*[float(j==i) for j in range(9)],t/1024,logloss[t],
                  *[logloss[t]-logloss[t-lag] for lag in range(1,17)],
                  logloss[t]-min(logloss[t-16:t+1])]
            features.append([*rich,a['projected'][t],a['projected'][t-16],a['projected'][t-32]])
            times.append(t);labels.append(int(max(a['loss'][t+1:t+9])/a['loss'][t]-1>=.1));identities.append((arm,str(t)))
    x=np.array(features);times=np.array(times);labels=np.array(labels)
    max_fit_error=0.
    for group,stats in result['robustness'].items():
        subset={(r['arm'],r['state']):r for r in rows if r['group']==group}
        for block in stats['blocks']:
            train=times<block['start']-stats['gap'];test=(times>=block['start'])&(times<=block['end'])
            for name,added in [('rich_loss',None),('rich_loss_projected',-3),
                               ('rich_loss_projected_delay16',-2),('rich_loss_projected_delay32',-1)]:
                if name not in stats['models']:continue
                matrix=x[:,:-3] if added is None else np.c_[x[:,:-3],x[:,added]]
                mean=matrix[train].mean(axis=0);scale=matrix[train].std(axis=0);scale[scale==0]=1.
                z=(matrix-mean)/scale
                estimator=LogisticRegression(C=1.,solver='liblinear',max_iter=1000,random_state=0).fit(z[train],labels[train])
                probabilities=estimator.predict_proba(z[test])[:,1]
                expected=[float(subset[identities[i]]['prob_'+name]) for i in np.flatnonzero(test)]
                error=float(np.max(np.abs(probabilities-expected)));max_fit_error=max(max_fit_error,error)
                assert error<1e-10,(group,name,error)
    max_curve_error=0.;points=0
    for c in fig['curves']:
        x=np.array(c['x']);y=np.array(c['y']);key=c['key'];kind=c['figure']
        if kind in ['overview','spikes']:
            a=raw[f"{c['task']}:{c['lr']:g}"];values=a[key]/c['normalization']
            if c['window']:
                expected=np.array([math.fsum(values[int(t)-c['window']+1:int(t)+1])/c['window'] for t in x])
            else: expected=values[x.astype(int)]
            if kind=='spikes': assert c['window'] is None
        elif kind=='events':
            selected=[e for e in result['aligned_events'] if e['task']==c['task']]
            assert x.tolist()==list(range(-8,9))
            expected=np.median([e[key] for e in selected],axis=0) if c['role']=='median' else next(
                e[key] for e in selected if c['role']==f"episode_{e['onset']}")
        else:
            selected=[r for r in primary if r['task']==c['task']]
            yy=np.array([int(r['target']) for r in selected]);p=np.array([float(r['prob_'+key]) for r in selected])
            thresholds=np.unique(p)
            precision=np.array([yy[p>=v].mean() for v in thresholds]+[1.])
            recall=np.array([yy[p>=v].sum()/yy.sum() for v in thresholds]+[0.])
            np.testing.assert_array_equal(x,recall);expected=precision
        np.testing.assert_allclose(y,expected,rtol=3e-14,atol=3e-14)
        max_curve_error=max(max_curve_error,float(np.max(np.abs(y-expected))));points+=len(y)
    for path in fig['outputs']:
        with fitz.open(ROOT/path) as doc:
            assert len(doc)==1 and abs(doc[0].rect.width/72-7.16)<1e-6
            assert not doc[0].get_images()
            for entry in doc[0].get_fonts(full=True): assert doc.extract_font(entry[0])[3]
    assert fig['visual_review'].startswith('passed:')
    # Archived integration versions must still match the unchanged manuscript copies.
    for path,digest in fig['archived_outputs'].items():
        assert sha(ROOT/path)==digest
        assert sha(ROOT/'manuscript/draft/figures/analysis036'/Path(path).name)==digest
    output=dict(status='passed',generator_sha256=sha(__file__),
        evidence_results_sha256=sha(OUT/'evidence_results.json'),figure_manifest_sha256=sha(OUT/'figure_manifest.json'),
        episode_summary_rows=len(result['episode_summary']),alarm_summary_rows=len(result['alarm_summary']),
        robustness_forecast_rows=len(rows),independently_refitted_models=30,
        maximum_refit_probability_error=max_fit_error,curves=len(fig['curves']),points=points,
        maximum_curve_error=max_curve_error,pdfs=4,visual_review=fig['visual_review'],
        manuscript_figure_copies_unchanged=True)
    prior.analysis.dump(OUT/'verification.json',output);print(output)


if __name__=='__main__':main()
