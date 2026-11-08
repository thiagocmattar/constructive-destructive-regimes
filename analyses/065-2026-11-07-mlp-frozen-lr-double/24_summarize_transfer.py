"""Comparable full/held-forward metrics; do not rank interventions by endpoint alone."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def summarize(worker,begin,end):
    t=np.asarray(worker['states']);mask=(t>=begin)&(t<=end);result=[]
    assert t[mask][0]==begin and t[mask][-1]==end
    for name,c in worker['curves'].items():
        loss=np.asarray(c['loss'])[mask];output=c['representations']['output']
        row=dict(case=worker['case'],seed=worker['seed'],run=worker['run_id'],worker=worker['worker'],arm=name,begin=begin,end=end,
            mean_loss=float(loss.mean()),tail32_mean=float(loss[-32:].mean()),final_loss=float(loss[-1]),max_loss=float(loss.max()),
            positive_variation=float(np.maximum(np.diff(loss),0).sum()),max_positive_jump=float(np.maximum(np.diff(loss),0).max()),
            output_rms_mean=float(np.mean(np.asarray(output['change_rms'])[mask])),output_rms_max=float(np.max(np.asarray(output['change_rms'])[mask])),
            output_reversal_fraction=float(np.mean(np.asarray(output['rho'],dtype=float)[mask]<0)),
            output_cosine_defined_fraction=float(np.mean(np.isfinite(np.asarray(output['rho'],dtype=float)[mask]))),
            raw_gradient_reversal_fraction=float(np.mean(np.asarray(c['rho'],dtype=float)[mask]<0)),
            raw_gradient_cosine_defined_fraction=float(np.mean(np.isfinite(np.asarray(c['rho'],dtype=float)[mask]))),
            actual_update_reversal_fraction=float(np.mean(np.asarray(c['actual_update_rho'],dtype=float)[mask]<0)),
            actual_update_cosine_defined_fraction=float(np.mean(np.isfinite(np.asarray(c['actual_update_rho'],dtype=float)[mask]))),
            raw_gradient_ascent_fraction=float(np.mean(np.asarray(c['raw_gradient_actual_cosine'],dtype=float)[mask]>0)),
            gradient_update_cosine_defined_fraction=float(np.mean(np.isfinite(np.asarray(c['raw_gradient_actual_cosine'],dtype=float)[mask]))),
            actual_update_zero_fraction=float(np.mean(np.asarray(c['actual_norm'])[mask]==0)),
            mean_target_fraction_modified=float(np.mean(np.asarray(c['mean_target_fraction_modified'])[mask])),
            mean_retained_norm=float(np.mean(np.asarray(c['retained_norm_ratio'])[mask])))
        for label,key in [('probe','probe_loss'),('outside_probe','outside_probe_loss')]:
            values=None if c[key] is None else np.asarray(c[key])[mask]
            row[label+'_mean_loss']=None if values is None else float(values.mean())
            row[label+'_tail32_mean']=None if values is None else float(values[-32:].mean())
            row[label+'_positive_variation']=None if values is None else float(np.maximum(np.diff(values),0).sum())
        result.append(row)
    baseline=result[0];assert baseline['arm']=='control'
    for row in result:
        for key in ['mean_loss','tail32_mean','final_loss','max_loss','positive_variation','max_positive_jump','output_rms_mean','output_rms_max',
                    'probe_mean_loss','probe_tail32_mean','probe_positive_variation','outside_probe_mean_loss','outside_probe_tail32_mean','outside_probe_positive_variation']:
            row[key+'_pct']=100*(row[key]/baseline[key]-1) if baseline[key] is not None and baseline[key]!=0 else None
    return result


def main():
    filenames=['transfer_comparison.json']+['transfer_comparison_run'+str(n)+'.json' for n in range(272,286)]
    rows=[];sources={};workers=[];scopes={}
    for filename in filenames:
        path=HERE/filename;data=json.loads(path.read_text());assert data['status']=='verified'
        assert data['generator_sha256']==sha(HERE/'17_reduce_transfer.py')
        sources[filename]=sha(path);workers+=data['workers']
    for worker in workers:
        start,end=worker['states'][0],worker['states'][-1];rows+=summarize(worker,start,end)
        layout=worker['layout'];total=max(g['end'] for g in layout['groups'].values())
        scope=dict(total_parameters=total,probe_loss_units=worker['metrics'][0]['probe_loss_units'],full_loss_units=worker['metrics'][0]['full_loss_units'])
        for name in ['entry','trunk']:
            count=sum(layout['groups'][n]['end']-layout['groups'][n]['start'] for n in layout[name])
            scope[name]=dict(modules=layout[name],parameters=count,parameter_fraction=count/total)
        if worker['case']=='transformer':
            span=layout['groups']['encoder'];count=span['end']-span['start']
            scope['embedding']=dict(modules=['encoder'],parameters=count,parameter_fraction=count/total)
        if worker['case'] in scopes:assert scopes[worker['case']]==scope
        else:scopes[worker['case']]=scope
        if worker['case']=='mlp' and end==1152:
            rows+=summarize(worker,448,800);rows+=summarize(worker,801,1152)
    # Same-platform replay controls must match wherever their windows overlap.
    groups={};control_checks=[];references=[]
    for worker in workers:
        provider=worker['worker'].split('/artifacts/')[1].split('/')[0]
        groups.setdefault((worker['case'],worker['seed'],provider),[]).append(worker)
    for identity,group in groups.items():
        reference=max(group,key=lambda w:len(w['states']))
        references.append(reference)
        for worker in group:
            shared=sorted(set(worker['states'])&set(reference['states']))
            left=[worker['states'].index(t) for t in shared];right=[reference['states'].index(t) for t in shared]
            for key in ['loss','raw_gradient_norm','rho','eta_kappa_hat']:
                np.testing.assert_array_equal(np.asarray(worker['curves']['control'][key])[left],np.asarray(reference['curves']['control'][key])[right])
        control_checks.append(dict(case=identity[0],seed=identity[1],provider=identity[2],workers=len(group)))
    cross_worker=[]
    for i,left in enumerate(references):
        for right in references[i+1:]:
            if (left['case'],left['seed'])!=(right['case'],right['seed']):continue
            shared=sorted(set(left['states'])&set(right['states']))
            a=np.asarray(left['curves']['control']['loss'])[[left['states'].index(t) for t in shared]]
            b=np.asarray(right['curves']['control']['loss'])[[right['states'].index(t) for t in shared]]
            cross_worker.append(dict(case=left['case'],seed=left['seed'],left=left['worker'],right=right['worker'],
                left_environment=left['environment'],right_environment=right['environment'],states=len(shared),
                loss_exact=bool(np.array_equal(a,b)),maximum_absolute_loss_difference=float(np.max(np.abs(a-b))),
                left_mean_loss=float(a.mean()),right_mean_loss=float(b.mean())))
    paired=[]
    for worker in workers:
        if worker['run_id']!=274:continue
        selected=[r for r in rows if r['worker']==worker['worker'] and r['begin']==worker['states'][0] and r['end']==worker['states'][-1]]
        cap=next(r for r in selected if r['arm']=='functional_trunk_q50')
        combined=next(r for r in selected if r['arm']=='embedding_x4_functional_trunk_q50')
        keys=['mean_loss','tail32_mean','max_loss','positive_variation','output_rms_mean','output_rms_max']
        paired.append(dict(worker=worker['worker'],reference=cap['arm'],treatment=combined['arm'],
            interpretation='Matched same-worker effect of adding embedding LR 4x to the distributed q50 cap; not a cross-GPU cap-effect estimate.',
            changes_pct={key:100*(combined[key]/cap[key]-1) if cap[key] else None for key in keys}))
    stage_metrics=[]
    for worker in workers:
        for arm,curve in worker['curves'].items():
            for stage,trace in curve['representations'].items():
                baseline=worker['curves']['control']['representations'][stage]
                rms=np.asarray(trace['change_rms']);reference=np.asarray(baseline['change_rms'])
                cosine=np.asarray(trace['rho'],dtype=float);defined=np.isfinite(cosine)
                stage_metrics.append(dict(case=worker['case'],seed=worker['seed'],run=worker['run_id'],worker=worker['worker'],
                    arm=arm,stage=stage,begin=worker['states'][0],end=worker['states'][-1],
                    mean_change_rms=float(rms.mean()),maximum_change_rms=float(rms.max()),
                    mean_change_rms_pct=100*(float(rms.mean()/reference.mean())-1) if reference.mean() else None,
                    maximum_change_rms_pct=100*(float(rms.max()/reference.max())-1) if reference.max() else None,
                    reversal_fraction_all_updates=float(np.mean(cosine<0)),cosine_defined_fraction=float(defined.mean()),
                    mean_defined_cosine=float(cosine[defined].mean()) if defined.any() else None))
    evaluations=[]
    for worker in workers:
        reference=worker['metrics'][0].get('evaluation',{})
        assert worker['metrics'][0]['arm']=='control'
        for metric in worker['metrics']:
            evaluation=metric.get('evaluation',{})
            if evaluation:
                assert evaluation['state']==worker['states'][-1]==reference['state']
                assert evaluation['splits'].keys()==reference['splits'].keys()
            for split,result in evaluation.get('splits',{}).items():
                baseline=reference['splits'][split]
                assert result['loss_units']==baseline['loss_units'] and result['objective_units']==baseline['objective_units']
                assert np.isfinite(result['loss']) and result['loss_units']>0
                evaluations.append(dict(case=worker['case'],seed=worker['seed'],run=worker['run_id'],worker=worker['worker'],
                    arm=metric['arm'],state=evaluation['state'],split=split,**result,
                    loss_change_pct=100*(result['loss']/baseline['loss']-1)))
    output=dict(status='verified',generator_sha256=sha(Path(__file__)),sources=sources,
        same_platform_controls_exact=control_checks,cross_worker_controls=cross_worker,parameter_scopes=scopes,metrics=rows,
        paired_embedding_comparisons=paired,captured_stage_metrics=stage_metrics,terminal_heldout_evaluations=evaluations,
        evaluation_note='Secondary terminal evaluations on complete retained splits; no new policy selection from these outcomes. Loss units are examples or nonpadding tokens. Missing splits are not inferred.')
    (HERE/'transfer_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    with (HERE/'transfer_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if evaluations:
        with (HERE/'transfer_heldout_evaluations.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(evaluations[0]));writer.writeheader();writer.writerows(evaluations)
    with (HERE/'transfer_captured_stage_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stage_metrics[0]));writer.writeheader();writer.writerows(stage_metrics)
    print(json.dumps(dict(rows=len(rows),workers=len(workers),same_platform_controls_exact=output['same_platform_controls_exact'])))


if __name__=='__main__':main()
