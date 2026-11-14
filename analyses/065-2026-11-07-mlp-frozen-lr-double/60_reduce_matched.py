"""H051 prespecified matched-window reductions and provenance, no outcome selection."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUNS={c:ROOT/f'runs/{n}-2026-11-14-{c}-matched-interventions' for n,c in [(293,'mlp'),(294,'resnet'),(295,'transformer')]}
STAGES={
 'mlp':[('1','Linear 1'),('2','Tanh 1'),('3','Linear 2'),('4','Tanh 2'),('output','Output')],
 'resnet':[('conv1','Conv 1'),('layer1','Stage 1'),('layer2','Stage 2'),('layer3','Stage 3'),('avgpool','Pooling'),('output','Output')],
 'transformer':[('encoder','Embedding'),('transformer_encoder.layers.0.self_attn','Attention 1'),('transformer_encoder.layers.0','Block 1'),('transformer_encoder.layers.1.self_attn','Attention 2'),('transformer_encoder.layers.1','Block 2'),('output','Output')]}
LABELS={'control':'Unchanged Adam','lr_0p75':'LR x0.75','lr_0p5':'LR x0.5','lr_0p1':'LR x0.1',
 'beta2_0p95':'beta2=0.95','beta2_0p9':'beta2=0.9','beta2_0p5':'beta2=0.5',
 'clip_0p5':'Clip 0.5','clip_1p0':'Clip 1.0','cap_q25':'Cap q25','cap_q50':'Cap q50','cap_q75':'Cap q75'}
spec=importlib.util.spec_from_file_location('loss_events',ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py')
events_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(events_module)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
def csvwrite(p,rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=columns,lineterminator='\n');w.writeheader();w.writerows(rows)
def relative(value,baseline):return 100*(value/baseline-1) if baseline else None
def safe_mean(values):
    v=[x for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def gate_factors(update,target_units):
    return [z for module,g in update['gates'].items() for z in g.get('factors',[1.]*target_units[module])]


def event_rows(loss,f,e,case,arm):
    result=[]
    for number,event in enumerate(events_module.excursions(loss)):
        a,z=event['before_state'],event['end_state']
        if a>=e or z<=f:continue
        lo=max(a+1,f+1);hi=min(z,e)
        peak=lo+int(np.argmax(loss[lo:hi+1]));reference=float(loss[a])
        height=relative(float(loss[peak]),reference)
        result.append(dict(case=case,arm=arm,event=number,**event,left_truncated=a<f,
            window_first_state=lo,window_last_state=hi,window_peak_state=peak,
            reference_loss=reference,window_peak_loss=float(loss[peak]),window_height_percent=height,
            observed_window_duration=hi-lo+1,
            full_recovery_duration=None if event['recovery_state'] is None else event['recovery_state']-a))
    return result


def reduce(cases,output):
    output.mkdir(parents=True,exist_ok=True)
    sources={};summary=[];stage_rows=[];all_events=[];heldout=[];cost=[];arrays={};case_records=[];budgets=[];pools={}
    def source(p):
        sources[p.relative_to(ROOT).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
        return p
    def read(p):return json.loads(source(p).read_text(encoding='utf-8'))
    def lines(p):return [json.loads(s) for s in source(p).read_text().splitlines() if s.strip()]
    for case in cases:
        run=RUNS[case];cfg=read(run/'config.json');f,e=cfg['fork'],cfg['end'];root=run/'artifacts/attempt-01'
        retained=read(run/'spark_durable_retention.json');assert retained['status']=='passed'
        inventory=read(root/'durable_inventory.json')['files'];inventory={i['path']:i for i in inventory}
        shared=root/'shared';ready=read(shared/'ready.json');prefix=lines(shared/'prefix.jsonl')
        assert [x['state'] for x in prefix]==list(range(f+1))
        assert ready['status']=='completed' and ready['fork']==f
        read(run/'input_manifest.json');thresholds=read(shared/'thresholds.json');calibration=np.load(source(shared/'calibration.npz'));read(shared/'layout.json')
        target_units={name:calibration[str(i)].shape[1] for i,name in enumerate(thresholds['module_order'])}
        for i,module in enumerate(thresholds['module_order']):
            pools[case+'__'+module]=calibration[str(i)]
            for quantile,value in thresholds['modules'][module]['functional'].items():
                budgets.append(dict(case=case,module=module,units=target_units[module],quantile=float(quantile),
                    budget=value,first_outgoing_state=f-8,last_outgoing_state=f-1,disabled_zero_budget=value==0))
        arms={};fingerprints=set();workers=[]
        for worker in sorted(root.iterdir()):
            if not (worker/'manifest.json').exists():continue
            manifest=read(worker/'manifest.json');audit=read(worker/'verification.json');replay=read(worker/'shared_replay.json')
            assert manifest['status']=='completed' and audit['status']=='passed'
            assert audit['counts']['terminal_gradient_replays']==len(manifest['arms'])
            assert audit['counts']['terminal_heldout_split_replays']==len(manifest['arms'])*(2 if case=='transformer' else 1)
            assert manifest['config_sha256']==sha(run/'config.json')
            assert manifest['environment']==ready['environment']
            assert replay['fork_fingerprint']==ready['fork_fingerprint'] and replay['predecessor_to_fork_exact']
            fingerprints.add(audit['eight_step_fingerprint']);workers.append(worker.name)
            for name in manifest['arms']:
                assert name not in arms,name
                arms[name]=worker/name
        assert len(fingerprints)==1,'Cross-worker native replay mismatch'
        assert set(arms)=={a['id'] for a in cfg['arms']},(case,sorted(arms))
        baseline=None;case_summary=[]
        for arm in cfg['arms']:
            name=arm['id'];folder=arms[name];b=lines(folder/'boundaries.jsonl');u=lines(folder/'updates.jsonl');ev=lines(folder/'evaluations.jsonl');status=read(folder/'status.json')
            assert status['status']=='completed' and status['completed_state']==e
            assert len(b)==e-f+1 and len(u)==e-f and len(ev)==5
            states=np.array([v['state'] for v in b]);np.testing.assert_array_equal(states,np.arange(f,e+1))
            loss=np.array([v['loss'] for v in b]);assert np.isfinite(loss).all() and loss[0]==ready['fork_loss']
            history=np.r_[[v['loss'] for v in prefix[:-1]],loss];assert len(history)==e+1
            delta=np.diff(loss);pv=float(np.maximum(delta,0).sum());nv=float(np.maximum(-delta,0).sum())
            np.testing.assert_allclose(pv-nv,loss[-1]-loss[0],rtol=2e-12,atol=2e-12)
            events=event_rows(history,f,e,case,name);all_events.extend(events)
            sr=dict(case=case,run=cfg['run_id'],seed=cfg['seed'],arm=name,setting=LABELS[name],family=arm['family'],fork=f,end=e,
                updates=e-f,fork_loss=float(loss[0]),positive_variation=pv,negative_variation=nv,
                positive_variation_over_fork=pv/loss[0] if loss[0] else None,
                mean_loss=float(loss[1:].mean()),final_loss=float(loss[-1]),last32_loss=float(loss[-32:].mean()),
                maximum_loss=float(loss[1:].max()),increasing_updates=int((delta>0).sum()),
                worst_excursion_percent=max([v['window_height_percent'] for v in events if v['window_height_percent'] is not None]+[0]),
                events_overlapping_window=len(events),events_started_in_window=sum(not v['left_truncated'] for v in events),
                events_carried_in=sum(v['left_truncated'] for v in events),events_right_censored=sum(v['right_censored'] for v in events),
                recovered_event_median_duration=safe_mean([]))
            durations=[v['full_recovery_duration'] for v in events if v['full_recovery_duration'] is not None]
            sr['recovered_event_median_duration']=float(np.median(durations)) if durations else None
            for cutoff in [10,25,50,100]:sr[f'events_ge_{cutoff}pct']=sum(v['window_height_percent'] is not None and v['window_height_percent']>=cutoff for v in events)
            for v in ev:
                assert v['state'] in cfg['heldout_states']
                for split,rec in v['splits'].items():
                    heldout.append(dict(case=case,arm=name,state=v['state'],split=split,loss=rec['loss'],loss_units=rec['loss_units'],seconds=v['seconds'],shared=v['shared']))
                    if v['state']==e:sr['terminal_'+split+'_loss']=rec['loss']
            all_factors=[z for v in u for z in gate_factors(v,target_units)]
            residuals=[float(a-g['threshold']-r) for v in u for g in v['gates'].values() for a,r in zip(g.get('actual_scores',[]),g.get('rounding_bound',[]))]
            sr.update(clipped_update_fraction=float(np.mean([v['clip_factor']<1 for v in u])),
                mean_clip_factor=float(np.mean([v['clip_factor'] for v in u])),
                capped_unit_update_fraction=float(np.mean(np.array(all_factors)<1)) if all_factors else None,
                mean_cap_factor=float(np.mean(all_factors)) if all_factors else None,
                minimum_cap_factor=float(min(all_factors)) if all_factors else None,
                cap_factor_p10=float(np.quantile(all_factors,.1)) if all_factors else None,
                cap_factor_median=float(np.median(all_factors)) if all_factors else None,
                cap_factor_p90=float(np.quantile(all_factors,.9)) if all_factors else None,
                disabled_zero_budget_modules=sum(g.get('disabled_zero_reference',False) for g in u[0]['gates'].values()),
                max_cap_bound_residual=max(residuals) if residuals else None,
                mean_retained_update_norm_ratio=safe_mean([v['retained_norm_ratio'] for v in u]))
            if baseline is None:
                assert name=='control';baseline=sr.copy()
            for metric in ['positive_variation','mean_loss','final_loss','last32_loss','terminal_test_loss']+(['terminal_valid_loss'] if case=='transformer' else []):
                sr[metric+'_change_percent']=relative(sr[metric],baseline[metric])
            sr['pv_reduction_percent']=-sr['positive_variation_change_percent'] if sr['positive_variation_change_percent'] is not None else None
            summary.append(sr);case_summary.append(sr)
            key=case+'__'+name+'__';arrays[key+'state']=states;arrays[key+'loss']=loss
            for metric in ['raw_gradient_norm','rho','eta_kappa_hat','probe_loss']:
                arrays[key+metric]=np.array([v[metric] if v[metric] is not None else np.nan for v in b])
            for metric in ['clip_factor','retained_norm_ratio','proposed_norm','actual_norm','actual_update_rho','raw_gradient_dot_actual']:
                arrays[key+metric]=np.array([v[metric] if v[metric] is not None else np.nan for v in u])
            arrays[key+'cap_active']=np.array([np.mean([z<1 for z in gate_factors(v,target_units)]) if v['gates'] else np.nan for v in u])
            if case=='mlp':
                for metric in ['linear','quadratic','delta_loss','residual']:
                    arrays[key+'mse_'+metric]=np.array([v['mse_identity'][metric] for v in b])
                rises=[v['mse_identity'] for v in b[1:] if v['mse_identity']['delta_loss']>1e-8]
                sr['mse_rises_gt_1em8']=len(rises)
                sr['mse_rises_negative_linear_outweighed']=sum(v['linear']<0 and v['quadratic']>-v['linear'] for v in rises)
            names=[n for n,_ in STAGES[case]]
            for metric in ['rho','change_rms']:
                arrays[key+'stage_'+metric]=np.array([[v['representations'][n][metric] for v in b] for n in names],dtype=float)
            for stage in b[0]['representations']:
                values=[v['representations'][stage] for v in b[1:]];rms=np.array([v['change_rms'] for v in values]);rho=np.array([v['rho'] for v in values],dtype=float)
                coords=np.array([v['coordinates'] for v in values]);assert len(set(coords))==1
                energy=coords*rms**2;reversing=np.isfinite(rho)&(rho<0)
                stage_rows.append(dict(case=case,arm=name,stage=stage,coordinates=int(coords[0]),mean_rms=float(rms.mean()),
                    max_rms=float(rms.max()),median_rms=float(np.median(rms)),p95_rms=float(np.quantile(rms,.95)),
                    reversal_fraction=float(np.mean(rho[np.isfinite(rho)]<0)) if np.isfinite(rho).any() else None,
                    undefined_cosines=int((~np.isfinite(rho)).sum()),squared_motion_energy=float(energy.sum()),
                    reversing_energy=float(energy[reversing].sum()),reversing_energy_fraction=float(energy[reversing].sum()/energy.sum()) if energy.sum() else None))
            timing=dict(case=case,arm=name,worker=folder.parent.name,gpu=status['environment']['gpu'],updates=e-f,
                wall_seconds=status['elapsed_seconds'],process_peak_cuda_bytes_through_arm=status['peak_cuda_bytes'],
                gradient_seconds=sum(v['gradient_seconds'] for v in b),probe_seconds=sum(v['probe_seconds'] for v in b),
                optimizer_seconds=sum(v['optimizer_seconds'] for v in u),controller_seconds=sum(v['controller_seconds'] for v in u),
                evaluation_seconds=sum(v['seconds'] for v in ev),full_gradient_passes=len(b),probe_passes=len(b),heldout_evaluations=4)
            timing['other_seconds']=timing['wall_seconds']-sum(timing[k] for k in ['gradient_seconds','probe_seconds','optimizer_seconds','controller_seconds','evaluation_seconds'])
            assert timing['other_seconds']>=-1e-5;cost.append(timing)
        # Every source under the artifact root must match its durable Spark inventory.
        for relative_path,record in sources.items():
            path=ROOT/relative_path
            if path.is_relative_to(root) and path.name!='durable_inventory.json':
                item=inventory[path.relative_to(root).as_posix()]
                assert record['bytes']==item['bytes'] and record['sha256']==item['sha256']
        case_records.append(dict(case=case,config=cfg,environment=ready['environment'],workers=workers,
            shared_fork_fingerprint=ready['fork_fingerprint'],eight_step_fingerprint=next(iter(fingerprints)),
            calibration_states=list(range(f-8,f)),stages=STAGES[case],prefix_seconds=ready['elapsed_seconds']))
    for row in stage_rows:
        base=next(r for r in stage_rows if r['case']==row['case'] and r['stage']==row['stage'] and r['arm']=='control')
        for metric in ['squared_motion_energy','max_rms','reversing_energy']:row[metric+'_change_percent']=relative(row[metric],base[metric])
    csvwrite(output/'summary.csv',summary);csvwrite(output/'events.csv',all_events);csvwrite(output/'stage_metrics.csv',stage_rows)
    csvwrite(output/'heldout.csv',heldout);csvwrite(output/'costs.csv',cost)
    csvwrite(output/'calibration.csv',budgets);np.savez_compressed(output/'calibration.npz',**pools)
    np.savez_compressed(output/'curves.npz',**arrays)
    report=dict(status='passed',complete_matrix=len(cases)==3,cases=case_records,conditions=len(summary),
        updates=sum(r['updates'] for r in summary),boundaries=sum(r['updates']+1 for r in summary),
        heldout_evaluation_states=5*len(summary),gradient_update_audits=3*len(summary),
        terminal_gradient_replays=len(summary),terminal_heldout_split_replays=sum(2 if r['case']=='transformer' else 1 for r in summary),
        sources=sources,summary=summary,script_sha256=sha(Path(__file__)),
        definition='Primary mean and motion summaries exclude shared fork; PV covers all post-fork differences. Events use full native prefix, fixed original pre-rise references, first return and explicit censoring. No seed-level inference.',
        outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir() if p.suffix in ['.csv','.npz']})
    if len(cases)==3:assert len(summary)==36 and report['updates']==11904
    write(output/'results.json',report)
    print(json.dumps({k:report[k] for k in ['status','complete_matrix','conditions','updates','boundaries','heldout_evaluation_states','gradient_update_audits']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',choices=list(RUNS),default=list(RUNS));p.add_argument('--output',type=Path,default=HERE/'matched')
    a=p.parse_args();reduce(a.cases,a.output)
