"""Audit and reduce every prespecified H049 condition; no outcome-based selection."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('old_audit',HERE/'17_reduce_transfer.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
RUNS=[ROOT/f'runs/{n}-2026-11-08-mlp-cap-lr-seed{s}' for s,n in enumerate([286,287,288])]
HISTORY=[ROOT/'runs'/n/'artifacts/spark/scientific/worker' for n in [
    '272-2026-11-07-mlp-displacement-confirmation','282-2026-11-07-mlp-intervention-seed1','283-2026-11-07-mlp-intervention-seed2']]
# Spark originals precede the local retrieval's /spark/ directory component.
HISTORY=[p if p.exists() else p.parents[2]/'scientific/worker' for p in HISTORY]
read=old.read;rows=old.rows;check=old.check;sha=old.sha
METRICS=['mean_loss','tail32_mean','final_loss','max_loss','positive_variation',
    'test_mean','test_tail4_mean','test_terminal','mean_retained_norm','mean_modified_fraction',
    'mean_actual_norm','mean_proposed_norm','output_reversals']


def dump(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def reduce_seed(seed):
    run=RUNS[seed];w=run/'artifacts/attempt-01/worker';cfg=read(run/'config.json');manifest=read(w/'manifest.json')
    if not (w/'adam_lr1/audit_448.pt').exists():
        raise RuntimeError('Full raw checkpoints are retained on Spark; run this reducer in the retained Spark workspace (see cap_lr_compact_retrieval.json). Local figure reproduction uses 40_plot_cap_lr.py.')
    assert manifest['status']=='completed' and not manifest['mechanics_only']
    assert len(manifest['results'])==18 and all(r['status']=='completed' for r in manifest['results'])
    replay=read(w/'cap_lr_replay.json');assert replay['status']=='passed' and replay['source_sha256']==sha(HERE/'37_audit_cap_lr.py')
    assert {(v['arm'],v['state']) for v in replay['checks']}=={(a['id'],s) for a in cfg['arms'] for s in [448,480,1152]}
    qualification=read(w/'shared/qualification.json');assert qualification['status']=='passed' and qualification['direct2_vs_resume_exact']
    calibration=read(w/'shared/thresholds.json');layout=read(w/'layout.json');cal=np.load(w/'shared/calibration.npz')
    assert calibration['window']==[440,447] and layout['trunk']==['1','3'] and cfg['calibration_lr']==.001
    for i,n in enumerate(calibration['module_order']):
        assert cal[f'{i}_functional'].shape==(8,200)
        for kind in ['coordinate','functional']:
            for q,tau in calibration['modules'][n][kind].items():check(np.quantile(cal[f'{i}_{kind}'],float(q),method='linear'),tau)
    historical=HISTORY[seed]
    prior=read(historical/'shared/qualification.json')
    assert qualification['fork_fingerprint']==prior['fork_fingerprint']
    assert qualification['native_two_step_fingerprint']==prior['native_two_step_fingerprint']
    states=list(range(448,1153));eval_states=list(range(448,1153,32));curves={};metrics=[];audits=0
    first=None
    for arm in cfg['arms']:
        name=arm['id'];folder=w/name;status=read(folder/'manifest.json')
        br=rows(folder/'boundaries.jsonl');ur=rows(folder/'updates.jsonl');er=rows(folder/'evaluations.jsonl')
        assert [r['state'] for r in br]==list(range(448,1154)) and [r['state'] for r in ur]==states
        assert [r['state'] for r in er]==eval_states and len(er)==23
        assert status['completed_state']==1153 and status['updates']==705 and status['training_units']==5000
        assert status['probe_units']==5000 and status['parameter_count']==656810
        b={r['state']:r for r in br};u={r['state']:r for r in ur};ema=dict(calibration['ema'])
        for t in states:
            for n,m in b[t]['blocks'].items():check(ema[n],m['previous_ema']);ema[n]=.9*ema[n]+.1*m['norm']
            assert u[t]['all_optimizer_steps']==t+1
            assert u[t]['applied_lr']==.001*arm['lr_multiplier'] and u[t]['calibration_lr']==.001
            assert set(u[t]['gates'])==({'1','3'} if arm['kind']=='functional' else set())
            for n,gate in u[t]['gates'].items():
                tau=calibration['modules'][n]['functional'][str(arm['quantile'])];check(tau,gate['threshold']);assert tau>0
                proposed=np.asarray(gate['proposed_scores']);actual=np.asarray(gate['actual_scores'])
                assert proposed.shape==actual.shape==(200,)
                check(np.minimum(1.,tau/np.maximum(proposed,np.finfo(float).tiny)),gate['factors'])
                check(np.mean(proposed>tau),gate['affected_fraction'])
                assert np.all(actual<=tau+np.asarray(gate['rounding_bound'])+1e-10)
            if arm['kind']=='control':check(u[t]['retained_norm_ratio'],1.,rtol=0,atol=0)
        audits+=old.audit(folder,arm,layout,b,u)
        a=torch.load(folder/'audit_448.pt',map_location='cpu',weights_only=False)
        if first is None:first=a
        for k in ['before_parameters','raw_gradient','previous_gradient']:assert torch.equal(a[k],first[k])
        # Same initial moments across all LR choices; LR is the only optimizer-group change.
        old.same_state(a['optimizer_before']['state'],first['optimizer_before']['state'])
        c={k:np.asarray([b[t][k] for t in states],dtype=float) for k in ['loss','probe_loss','rho','eta_kappa_hat','raw_gradient_norm']}
        c.update({k:np.asarray([u[t][k] for t in states],dtype=float) for k in ['proposed_norm','actual_norm','retained_norm_ratio','actual_update_rho','raw_gradient_actual_cosine']})
        for n in br[0]['representations']:
            for k in ['rho','change_rms','activation_rms']:c[f'representation_{n}_{k}']=np.asarray([b[t+1]['representations'][n][k] for t in states],dtype=float)
        c['modified_fraction']=np.asarray([np.mean([g['affected_fraction'] for g in u[t]['gates'].values()]) if u[t]['gates'] else 0. for t in states])
        for e in er:
            assert set(e['splits'])=={'test'}
            assert e['splits']['test']['loss_units']==e['splits']['test']['objective_units']==10000
        c['test_loss']=np.asarray([e['splits']['test']['loss'] for e in er])
        for key,values in c.items():
            assert not np.isinf(values).any()
            if not ('rho' in key or 'cosine' in key or key=='eta_kappa_hat'):assert np.isfinite(values).all()
        assert np.isnan(c['actual_update_rho'][0])
        loss=c['loss'];test=c['test_loss']
        m=dict(seed=seed,run=cfg['run_id'],arm=name,lr_multiplier=arm['lr_multiplier'],lr=.001*arm['lr_multiplier'],
            quantile=arm.get('quantile'),mean_loss=float(loss.mean()),tail32_mean=float(loss[-32:].mean()),
            final_loss=float(loss[-1]),max_loss=float(loss.max()),peak_state=states[int(loss.argmax())],
            positive_variation=float(np.maximum(np.diff(loss),0).sum()),test_mean=float(test.mean()),
            test_tail4_mean=float(test[-4:].mean()),test_terminal=float(test[-1]),
            mean_retained_norm=float(c['retained_norm_ratio'].mean()),mean_modified_fraction=float(c['modified_fraction'].mean()),
            mean_actual_norm=float(c['actual_norm'].mean()),mean_proposed_norm=float(c['proposed_norm'].mean()),
            output_reversals=int((c['representation_output_rho']<0).sum()),seconds=status['elapsed_seconds'])
        if name in ['adam_lr1','cap_q50_lr1']:
            previous_name='control' if name=='adam_lr1' else 'functional_trunk_q50'
            prior_rows=rows(historical/previous_name/'boundaries.jsonl')
            assert len(prior_rows)==len(br)
            for new,previous in zip(br,prior_rows):
                for k in ['state','loss','probe_loss','rho','eta_kappa_hat','raw_gradient_norm','representations']:
                    assert new[k]==previous[k],(seed,name,new['state'],k)
            assert er[-1]['splits']==read(historical/previous_name/'evaluation.json')['splits']
        metrics.append(m)
        for key,value in c.items():curves[f's{seed}/{name}/{key}']=value
    lookup={m['arm']:m for m in metrics}
    for m in metrics:
        refs={'original':lookup['adam_lr1'],'same_lr':lookup[f"adam_lr{m['lr_multiplier']}"]}
        if m['quantile'] is not None:refs['same_cap_1x']=lookup[f"cap_q{round(m['quantile']*100)}_lr1"]
        for key in METRICS[:8]:
            for label,ref in refs.items():m[key+'_pct_'+label]=100*(m[key]/ref[key]-1) if ref[key] else None
    result=dict(seed=seed,status='passed',arm_count=18,updates=18*705,retained_audits=audits,
        replay_count=len(replay['checks']),historical_control_and_q50_bitwise=True,
        thresholds={n:calibration['modules'][n]['functional'] for n in layout['trunk']},
        input_manifest_sha256=sha(run/'input_manifest.json'),worker_seconds=manifest['elapsed_seconds'])
    dump(run/'reduction_verification.json',result)
    return metrics,curves,result


def write_csv(path,records):
    keys=list(dict.fromkeys(k for r in records for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(records)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[0,1,2]);args=parser.parse_args()
    if args.seed is not None:
        metrics,_,verification=reduce_seed(args.seed);print(json.dumps(dict(verification=verification,metrics=metrics)));return
    metrics=[];curves={};checks=[]
    for seed in range(3):
        m,c,v=reduce_seed(seed);metrics+=m;curves.update(c);checks.append(v)
    aggregate=[]
    for arm in read(RUNS[0]/'config.json')['arms']:
        group=[m for m in metrics if m['arm']==arm['id']];assert len(group)==3
        row=dict(arm=arm['id'],lr_multiplier=arm['lr_multiplier'],lr=.001*arm['lr_multiplier'],quantile=arm.get('quantile'),seeds=3)
        for key in [*METRICS,*[k for k in group[0] if '_pct_' in k]]:
            v=np.asarray([m[key] for m in group if m[key] is not None],dtype=float);assert np.isfinite(v).all()
            row[key+'_mean']=float(v.mean()) if len(v) else None
            row[key+'_sd']=float(v.std(ddof=1)) if len(v)>1 else None
            if len(v)!=3:row[key+'_defined_seeds']=len(v)
        aggregate.append(row)
    np.savez_compressed(HERE/'cap_lr_curves.npz',states=np.arange(448,1153),test_states=np.arange(448,1153,32),**curves)
    summary=dict(status='passed',design='54 prespecified continuations: 3 seeds x 3 LRs x (5 cap quantiles + uncapped)',
        window=[448,1152],train_samples=5000,test_samples=10000,train_states=705,test_states=23,
        reduction='Arithmetic mean and sample SD across seeds; percentages paired within seed before aggregation.',
        reducer_sha256=sha(Path(__file__)),audit_library_sha256=sha(HERE/'17_reduce_transfer.py'),
        checks=checks,metrics=metrics,aggregate=aggregate,curves_sha256=sha(HERE/'cap_lr_curves.npz'))
    dump(HERE/'cap_lr_summary.json',summary)
    write_csv(HERE/'cap_lr_per_seed.csv',metrics);write_csv(HERE/'cap_lr_aggregate.csv',aggregate)
    lines=['# MLP distributed cap x learning rate: descriptive statistics','',
        'Three initializations; mean +/- sample SD. Training states 448-1152 (705 full-objective measurements).',
        'Test: all 10,000 examples every 32 states (23 measurements). Last four: 1056,1088,1120,1152.',
        'Percentages are computed against each seed\'s control before averaging; negative is lower.',
        'No test-based tuning or early stopping. Thresholds fixed from eight untreated 1x updates.','',
        '| LR | Cap | Mean train loss | Tail-32 train loss | Maximum train loss | Positive variation | Mean train change vs original | Mean train change vs same LR | Final-four test loss | Terminal test loss |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    def fmt(r,k):return f"{r[k+'_mean']:.5f} +/- {r[k+'_sd']:.5f}"
    def pct(r,k):return f"{r[k+'_mean']:+.2f}% +/- {r[k+'_sd']:.2f}"
    for r in aggregate:
        cap='None' if r['quantile'] is None else 'q'+str(round(r['quantile']*100))
        lines.append(f"| {r['lr']:.3f} | {cap} | {fmt(r,'mean_loss')} | {fmt(r,'tail32_mean')} | {fmt(r,'max_loss')} | {fmt(r,'positive_variation')} | {pct(r,'mean_loss_pct_original')} | {pct(r,'mean_loss_pct_same_lr')} | {fmt(r,'test_tail4_mean')} | {fmt(r,'test_terminal')} |")
    lines+=['','CSV files contain all primary and secondary statistics, individual seeds, and paired changes versus the original control, same-LR control, and same cap at 1x LR. Three seeds give descriptive dispersion, not a precise population confidence interval.','']
    (HERE/'cap_lr_statistics.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    tex=[r'\begin{table*}[t]',r'\centering',r'\caption{MLP/CIFAR-10 distributed local caps with fixed 1x-calibrated budgets. Mean $\pm$ sample SD over three initializations. Training states448--1152; last-four test evaluations1056/1088/1120/1152. All conditions prespecified; no held-out tuning.}',
         r'\label{tab:mlp-cap-lr}',r'\scriptsize',r'\begin{tabular}{llccccc}',r'\toprule',
         r'LR & Cap & Mean train loss & Tail-32 train & Maximum train & Positive variation & Final-four test \\',r'\midrule']
    def texfmt(r,k):return f"${r[k+'_mean']:.4f} \\pm {r[k+'_sd']:.4f}$"
    for i,r in enumerate(aggregate):
        if i and i%6==0:tex.append(r'\midrule')
        cap='None' if r['quantile'] is None else 'q'+str(round(r['quantile']*100))
        tex.append(' & '.join([f"{r['lr']:.3f}",cap,*[texfmt(r,k) for k in ['mean_loss','tail32_mean','max_loss','positive_variation','test_tail4_mean']]])+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
    (HERE/'cap_lr_table.tex').write_text('\n'.join(tex),encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='passed',conditions=54,updates=54*705,retained_audits=162,historical_reproductions=6,curves=len(curves))))


if __name__=='__main__':main()
