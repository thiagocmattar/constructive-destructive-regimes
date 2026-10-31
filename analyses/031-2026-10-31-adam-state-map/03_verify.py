"""Independent reductions, dummy-variable OLS, temporal boundaries and PDF checks."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import fmean
import numpy as np
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def point(e):
    return {'raw_cosine':e['adjacent_cosine'], 'projected_change':e['projected_gradient_change'],
        'input_cosine':e['incoming_previous_moment_cosine'], 'incoming_ratio':e['incoming_to_retained_moment_ratio'],
        'moment_cosine':e['momentum']['adjacent_cosine'],'adaptive_cosine':e['adaptive']['adjacent_cosine'],
        'update_cosine':e['actual_parameter_update']['adjacent_cosine'],'descent_cosine':-e['raw_actual_update_cosine'],
        'log_raw_norm':math.log(e['gradient_norm']),'log_moment_norm':math.log(e['momentum']['momentum_norm']),
        'log_adaptive_gain':math.log(e['adaptive']['direction_norm']/e['momentum']['momentum_norm']),
        'log_step_norm':math.log(e['actual_parameter_update']['gradient_norm']),
        'first_order_decrease':-e['gradient_norm']*e['actual_parameter_update']['gradient_norm']*e['raw_actual_update_cosine'],
        'clipping_fraction':int(e['clip_coefficient']<1),
        'log_previous_moment_norm':math.log(e['previous_moment_norm'])}


def reduced(events,t):
    early=[e for e in events if 2<=e['local_update']<=t]
    values=[point(e) for e in early]
    out={k:fmean(v[k] for v in values) for k in values[0]}
    last=next(e for e in events if e['local_update']==t)
    for key,stream,num in [('raw_retention',last,'cumulative_gradient_norm'),
        ('moment_retention',last['momentum'],'cumulative_momentum_norm'),
        ('adaptive_retention',last['adaptive'],'cumulative_direction_norm'),
        ('update_retention',last['actual_parameter_update'],'cumulative_gradient_norm')]:
        out[key]=stream[num]/stream['cumulative_norm_sum']
    for key,vals in [('raw_growth',[e['gradient_norm'] for e in early]),
        ('moment_growth',[e['momentum']['momentum_norm'] for e in early])]:
        out[key]=math.log(fmean(vals[-4:])/fmean(vals[:4]))
    return out


def main():
    f=json.loads((HERE/'features.json').read_text());r=json.loads((HERE/'results.json').read_text())
    data={};summaries={};feature_count=0;prediction_count=0;ratios=0
    for s in f['sources']:
        p=ROOT/'runs'/s['run']/'artifacts/attempt-01/full-pass'
        assert sha(p/'events.jsonl')==s['events_sha256'] and sha(p/'summary.json')==s['summary_sha256']
        data[s['run']]=[e for e in map(json.loads,(p/'events.jsonl').read_text().splitlines()) if e['event']=='update']
        summaries[s['run']]=json.loads((p/'summary.json').read_text())
        for e in data[s['run']]:
            for stream,num in [(e,'cumulative_gradient_norm'),(e['momentum'],'cumulative_momentum_norm'),
                (e['adaptive'],'cumulative_direction_norm'),(e['actual_parameter_update'],'cumulative_gradient_norm')]:
                np.testing.assert_allclose(stream['cumulative_ratio'],stream[num]/stream['cumulative_norm_sum'],rtol=1e-12,atol=1e-14);ratios+=1
    for row in f['rows']:
        held=next(run for run in data if row['block'].split('-')[0] in run and f"seed{row['seed']}-lr1p0" in run)
        vals={e['local_update']:e['loss'] for e in summaries[row['run']]['validations']}
        hv={e['local_update']:e['loss'] for e in summaries[held]['validations']}
        np.testing.assert_allclose([row['gain_final'],row['gain_after16']],
            [hv[128]-vals[128],(hv[128]-hv[16])-(vals[128]-vals[16])],rtol=1e-12,atol=1e-14)
        t=row['cutoff']
        if t==0:
            common=point(data[held][0]);expected={k:math.log2(row['multiplier'])*common[k] for k in row['features']}
        else:
            a=reduced(data[row['run']],t);b=reduced(data[held],t)
            expected={k:a[k]-b[k] for k in row['features'] if k!='probe_gain'}
            if 'probe_gain' in row['features']:
                ap=next(e['loss'] for e in summaries[row['run']]['probes'] if e['local_update']==t)
                hp=next(e['loss'] for e in summaries[held]['probes'] if e['local_update']==t)
                expected['probe_gain']=hp-ap
        for k,v in expected.items():
            np.testing.assert_allclose(row['features'][k],v,rtol=1e-10,atol=1e-12);feature_count+=1
    # Independently fit full cell-intercept design, not the residualization code.
    for experiment in r['experiments']:
        rows=[row for row in f['rows'] if row['cutoff']==experiment['cutoff']]
        assert [row['run'] for row in rows]==experiment['run_order']
        models=[experiment['baseline'],*experiment['candidates'].values(),*experiment['conditional_on_probe'].values()]
        groups=sorted({(row['size'],row['multiplier']) for row in rows})
        dummy=np.array([[float((row['size'],row['multiplier'])==g) for g in groups] for row in rows])
        y=np.array([row[experiment['target']] for row in rows])
        for model in models:
            x=np.array([[row['features'][k] for k in model['features']] for row in rows])
            predictions=np.zeros(len(rows))
            for seed in [1234,1235,1236]:
                train=np.array([i for i,row in enumerate(rows) if row['seed']!=seed])
                test=np.array([i for i,row in enumerate(rows) if row['seed']==seed])
                center=x[train].mean(0);scale=x[train].std(0);scale[scale<1e-14]=1
                z=(x-center)/scale
                # A feature constant within every training cell has no identified
                # incremental slope. Constrain it to zero, rather than selecting
                # an arbitrary dummy/feature decomposition with a pseudoinverse.
                residual=z[train]-dummy[train]@np.linalg.lstsq(dummy[train],z[train],rcond=None)[0]
                active=np.sqrt(np.mean(residual*residual,axis=0))>1e-12
                design=np.column_stack([dummy,z[:,active]])
                beta=np.linalg.lstsq(design[train],y[train],rcond=1e-12)[0]
                predictions[test]=design[test]@beta
                fit=next(v for v in model['fits'] if v['seed']==seed)
                assert set(fit['training_runs']).isdisjoint(fit['test_runs'])
            np.testing.assert_allclose(predictions,model['predictions'],rtol=1e-8,atol=1e-10,
                err_msg=f"cutoff={experiment['cutoff']},target={experiment['target']},features={model['features']}")
            np.testing.assert_allclose(np.mean(abs(predictions-y)),model['mae'],rtol=1e-8,atol=1e-10)
            prediction_count+=len(rows)
    plot=json.loads((HERE/'figure_data.json').read_text())['rows'];plot_values=0
    for row in plot:
        held=next(run for run in data if row['block'].split('-')[0] in run and f"seed{row['seed']}-lr1p0" in run)
        for key in ['input_cosine','incoming_ratio','update_retention']:
            if key=='update_retention':
                values=[(ea['actual_parameter_update']['cumulative_gradient_norm']/ea['actual_parameter_update']['cumulative_norm_sum'])-
                        (eb['actual_parameter_update']['cumulative_gradient_norm']/eb['actual_parameter_update']['cumulative_norm_sum'])
                        for ea,eb in zip(data[row['run']],data[held])]
            else:
                delta=[point(ea)[key]-point(eb)[key] for ea,eb in zip(data[row['run']],data[held])]
                values=[fmean(delta[i-7:i+1]) for i in range(7,128)]
            np.testing.assert_allclose(values,row['series'][key]['y'],rtol=1e-10,atol=1e-12);plot_values+=len(values)
        state=reduced(data[row['run']],16)
        for k,v in row['state16'].items():
            np.testing.assert_allclose(v,state[k],rtol=1e-10,atol=1e-12);plot_values+=1
        for kind,key in [('probes','probe_gain'),('validations','validation_gain')]:
            held_losses={v['local_update']:v['loss'] for v in summaries[held][kind]}
            for val in summaries[row['run']][kind]:
                np.testing.assert_allclose(row[key][str(val['local_update'])],held_losses[val['local_update']]-val['loss'],rtol=1e-10,atol=1e-12);plot_values+=1
    spec=importlib.util.spec_from_file_location('analysis',HERE/'01_analyze.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    events=next(iter(data.values()));early=a.summarize(events,16)
    # No event after the declared cutoff may affect a feature.
    truncated=a.summarize(events[:16],16)
    assert early==truncated
    # Held targets must not alter predictions for that held seed.
    rows=[row for row in f['rows'] if row['cutoff']==16]
    base=a.cv(rows,'gain_final',['update_retention'])
    changed=json.loads(json.dumps(rows))
    for row in changed:
        if row['seed']==1234:row['gain_final']+=100
    modified=a.cv(changed,'gain_final',['update_retention'])
    for i,row in enumerate(rows):
        if row['seed']==1234:assert base['predictions'][i]==modified['predictions'][i]
    figs=json.loads((HERE/'figure_checks.json').read_text())
    for figure in figs['figures']:
        p=HERE/figure['path'];assert sha(p)==figure['sha256']
        pdf=PdfReader(p);assert len(pdf.pages)==1 and len(pdf.pages[0].extract_text())>80
    verification=dict(verified=True,source_runs=len(data),updates=sum(map(len,data.values())),
        feature_values_checked=feature_count,ratio_identities_checked=ratios,
        held_seed_predictions_checked=prediction_count,plotted_source_values_checked=plot_values,temporal_boundary_check=True,
        held_target_noninterference=True,pdfs=4,results_sha256=sha(HERE/'results.json'),
        verifier_sha256=sha(__file__))
    (HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
