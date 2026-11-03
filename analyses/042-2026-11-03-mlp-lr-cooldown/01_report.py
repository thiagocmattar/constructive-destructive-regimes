r"""Source-aligned comparison of eight-update holds and cosine recovery; no training."""
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'runs/220-2026-11-03-mlp-lr-cooldown'
ATTEMPT=RUN/'artifacts/attempts/002-audit-continuation'
CONDITIONS=['unchanged','always_half','temporary','hold8','hold8_cosine8']
NAMES={'unchanged':'Unchanged Adam','always_half':'Always-half LR','temporary':'One-step reduction',
       'hold8':'8-update hold','hold8_cosine8':'8-update hold + 8-update cosine recovery'}
COLORS={'unchanged':'.45','always_half':'#0072B2','temporary':'#D55E00','hold8':'#009E73','hold8_cosine8':'#8E44AD'}
STYLES={'unchanged':':','always_half':'--','temporary':'-','hold8':'-','hold8_cosine8':'-.'}


def read(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_out(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    plan=json.loads((ATTEMPT/'manifest.json').read_text(encoding='utf-8'));assert plan['status']=='completed'
    manifest=json.loads((RUN/'input_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:assert sha(ROOT/item['path'])==item['sha256']
    assert sha(RUN/'02_continue_after_audit.py')==plan['infrastructure_source_sha256']
    spec=importlib.util.spec_from_file_location('cooldown_audit',RUN/'02_continue_after_audit.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    parent=torch.load(audit.r.r.PARENT,map_location='cpu',weights_only=False)
    assert sha(audit.r.r.PARENT)==audit.r.r.PARENT_SHA
    oldspec=importlib.util.spec_from_file_location('old_analysis041',ROOT/'analyses/041-2026-11-03-mlp-temporary-lr/01_report.py')
    old=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(old)
    event_spec=importlib.util.spec_from_file_location('loss_events',ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py')
    detector=importlib.util.module_from_spec(event_spec);event_spec.loader.exec_module(detector)
    prefix=read(ROOT/'runs/218-2026-11-03-mlp-adam-step-actions/artifacts/attempts/001-20261103T172300Z-local-pilot/prefix/boundaries.jsonl')[:657]
    observations={};actions={};aligned=[];summaries=[];events=[];replay={}
    for condition in CONDITIONS:
        folder=ATTEMPT/'conditions'/condition;rows=read(folder/'boundaries.jsonl');decisions=read(folder/'decisions.jsonl');metrics=read(folder/'metrics.jsonl')
        assert [r['state'] for r in rows]==list(range(657,1025))
        assert [r['state'] for r in decisions]==list(range(657,1024)) and len(metrics)==367
        start=torch.load(folder/'checkpoints/start-state-000657.pt',map_location='cpu',weights_only=False)
        for key in ['model','optimizer','rng']:audit.helper.equal(parent[key],start[key])
        for key in ['previous_gradient','previous_increment']:audit.helper.equal(parent['observer'][key],start['observer'][key])
        assert start['completed_updates']==657 and start['identity']['parent_sha256']==audit.r.r.PARENT_SHA
        del start
        last_trigger=None
        for i,decision in enumerate(decisions):
            row=rows[i];value=row['eta_kappa_hat'];trigger=value is not None and value>=1.8
            if trigger:last_trigger=i
            if condition=='unchanged':expected=.001
            elif condition=='always_half':expected=.0005
            elif condition=='temporary':expected=.0005 if trigger else .001
            elif last_trigger is None:expected=.001
            else:
                age=i-last_trigger
                expected=.0005 if age<8 else .001
                if condition=='hold8_cosine8' and 8<=age<16:
                    expected=.0005+.00025*(1-math.cos(math.pi*(age-7)/8))
            assert decision['state']==row['state'] and decision['available_before_update']==row['state']+1
            assert decision['manuscript_gradient_index']==row['state']+1 and decision['eta_kappa_hat']==value
            assert decision['trigger']==trigger and abs(decision['applied_lr']-expected)<1e-18
            assert metrics[i]['decision']==decision and metrics[i]['gradient_state']==row['state'] and metrics[i]['update']==row['state']+1
            if row['previous_gradient_norm']>0:
                alternate=1-row['raw_adjacent_cosine']*row['gradient_norm']/row['previous_gradient_norm']
                assert abs(value-alternate)<=2e-12*(1+abs(value))
        if condition in ['unchanged','always_half','temporary']:
            previous=old.canonical(condition,'boundaries.jsonl')
            replay[condition]=[(r['state'],r['loss'],r['gradient_norm'],r['eta_kappa_hat']) for r in rows]==[(r['state'],r['loss'],r['gradient_norm'],r['eta_kappa_hat']) for r in previous]
            assert replay[condition]
            prior_end=(old.RETRY/'unchanged' if condition=='unchanged' else old.FIRST/'conditions'/condition)/'checkpoints/end-state-001024.pt'
            a=torch.load(prior_end,map_location='cpu',weights_only=False)
            z=torch.load(folder/'checkpoints/end-state-001024.pt',map_location='cpu',weights_only=False)
            for key in ['model','optimizer','rng']:audit.helper.equal(a[key],z[key])
            for key in ['previous_gradient','previous_increment']:audit.helper.equal(a['observer'][key],z['observer'][key])
            del a,z
        observations[condition]=rows;actions[condition]=decisions
        for i,row in enumerate(rows):
            decision=decisions[i] if i<367 else None
            aligned.append(dict(condition=condition,state=row['state'],manuscript_gradient_index=row['state']+1,
                loss=row['loss'],eta_kappa_hat=row['eta_kappa_hat'],raw_gradient_norm=row['gradient_norm'],
                applied_lr=None if decision is None else decision['applied_lr'],lr_over_base=None if decision is None else decision['lr_over_base'],
                phase=None if decision is None else decision['phase'],trigger=None if decision is None else decision['trigger'],
                actual_increment_norm=None if decision is None else metrics[i]['streams']['actual']['norm'],
                gradient_dot_actual_step=None if decision is None else metrics[i]['gradient_dot_actual_step']))
    B=observations['unchanged'][689-657]['loss'];target=.95*B
    for condition in CONDITIONS:
        rows=observations[condition];decisions=actions[condition];loss=np.array([r['loss'] for r in rows]);lrs=np.array([r['applied_lr'] for r in decisions])
        excess=np.maximum(loss[689-657:]/B-1,0);joined=[r['loss'] for r in prefix]+loss.tolist();local=[]
        for event in detector.excursions(joined):
            if event['end_state']<657:continue
            e=dict(condition=condition,**event,peak_over_prerise=joined[event['peak_state']]/joined[event['before_state']],
                peak_in_continuation=event['peak_state']>=657,later_start_after_original_recovery=event['before_state']>729)
            events.append(e);local.append(e)
        significant=[e for e in local if e['peak_in_continuation'] and e['peak_over_prerise']>=1.1]
        result=plan['results'][condition];policy=result['policy_state']
        target_state=next((r['state'] for r in rows if r['loss']<=target),None)
        summary=dict(condition=condition,status='completed',updates=367,B=B,peak_excess=float(excess.max()),excess_area=float(excess.sum()),
            mean_postupdate_loss=float(loss[1:].mean()),final_loss=float(loss[-1]),full_max_loss=float(loss.max()),full_max_state=rows[int(loss.argmax())]['state'],
            half_updates=int((lrs==.0005).sum()),below_base_updates=int((lrs<.001).sum()),below_base_fraction=float((lrs<.001).mean()),
            mean_lr_over_base=float(lrs.mean()/.001),lr_switches_within=int((lrs[1:]!=lrs[:-1]).sum()),entry_lr_change=int(lrs[0]!=.001),
            triggers=policy['triggers'],retriggers=policy['retriggers'],final_lr=float(lrs[-1]),
            target_updates=None if target_state is None else target_state-657,excursions_gt10=len(significant),censored_gt10=sum(e['right_censored'] for e in significant),
            later_excursions_gt10=sum(e['later_start_after_original_recovery'] for e in significant),
            largest_own_reference_excursion=max([e['peak_over_prerise'] for e in significant],default=None),
            runner_seconds=result['runner_seconds'],endpoint_test_loss=result['endpoint_test']['loss'],endpoint_test_images=result['endpoint_test']['loss_units'])
        assert summary['half_updates']==policy['half_updates'] and summary['below_base_updates']==policy['below_base_updates']
        assert summary['lr_switches_within']==policy['switches_within'] and summary['endpoint_test_images']==10000
        assert abs(summary['excess_area']-sum(max(0.,r['loss']/B-1) for r in rows if r['state']>=689))<1e-9
        assert abs(summary['mean_postupdate_loss']-sum(r['loss'] for r in rows[1:])/367)<1e-12
        summaries.append(summary)
    (HERE/'figures').mkdir(exist_ok=True)
    checked_scalars=0
    for bounds,name,scale,shown in [((673,745),'fixed_zoom','linear',CONDITIONS),((657,1024),'full_continuation','log',CONDITIONS),
        ((657,1024),'focal_four_arm','log',[c for c in CONDITIONS if c!='temporary']),
        ((673,745),'focal_four_arm_zoom','linear',[c for c in CONDITIONS if c!='temporary'])]:
        fig,axes=plt.subplots(4,1,figsize=(11.4,10),sharex=True)
        fig.subplots_adjust(left=.095,right=.985,bottom=.075,top=.86,hspace=.12)
        for ax in axes:
            ax.set_xlim(*bounds);ax.grid(alpha=.16)
            for state in [689,703,729]:ax.axvline(state,color='.72',ls=':',lw=.7,zorder=0)
        for condition in shown:
            rows=observations[condition];x=np.array([r['state'] for r in rows]);visible=(x>=bounds[0])&(x<=bounds[1])
            lrs=np.array([r['lr_over_base'] for r in actions[condition]]+[actions[condition][-1]['lr_over_base']])
            ys=[np.array([r[key] for r in rows]) for key in ['loss','eta_kappa_hat','gradient_norm']]+[lrs]
            for j,(ax,y) in enumerate(zip(axes,ys)):
                kwargs=dict(color=COLORS[condition],ls=STYLES[condition],lw=1.35,label=NAMES[condition])
                line,=ax.step(x[visible],y[visible],where='post',**kwargs) if j==3 else ax.plot(x[visible],y[visible],**kwargs)
                assert np.array_equal(line.get_xdata(),x[visible]) and np.array_equal(line.get_ydata(),y[visible]);checked_scalars+=int(visible.sum())
        axes[0].set_yscale(scale);axes[0].set_ylabel('Training loss'+(' (log)' if scale=='log' else ''))
        axes[1].axhline(1.8,color='.3',ls='--',lw=.9);axes[1].set_ylabel(r'$\eta\widehat\kappa_t$')
        axes[2].set_yscale('log');axes[2].set_ylabel('Raw gradient norm (log)')
        axes[3].set_ylabel('Actual LR / base LR');axes[3].set_ylim(.46,1.04);axes[3].set_yticks([.5,.75,1.])
        axes[3].set_xlabel('Parameter state s (decision applies to s -> s+1)')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.51,.955),ncol=2 if len(shown)==4 else 3,fontsize=9,frameon=False)
        fig.suptitle('Eight-update cooldown and cosine recovery: '+('fixed states 673-745' if bounds==(673,745) else 'full states 657-1024'),y=.995,fontsize=13)
        fig.text(.5,.018,r'Seed 0; same authenticated fork. Raw unsmoothed values; $\eta\widehat\kappa_t$ threshold 1.8. No LR rescaling.',ha='center',fontsize=9)
        fig.savefig(HERE/'figures'/f'{name}.pdf');fig.savefig(HERE/'scratch'/f'{name}.png',dpi=160);plt.close(fig)
    csv_out(HERE/'comparison.csv',summaries);csv_out(HERE/'aligned_values.csv',aligned);csv_out(HERE/'excursions.csv',events)
    main_updates=sum(len(read(p)) for p in (RUN/'artifacts').glob('attempts/*/conditions/*/metrics.jsonl'))
    qa_updates=sum(len(read(p)) for p in (RUN/'artifacts').glob('attempts/*/qualification/*/*/metrics.jsonl'))
    gradient_calls=sum(len(read(p)) for p in (RUN/'artifacts').rglob('boundaries.jsonl'))
    assert (main_updates,qa_updates,gradient_calls)==(1835,108,1957)
    first=json.loads((RUN/'artifacts/attempts/001-local/manifest.json').read_text(encoding='utf-8'))
    span=(datetime.fromisoformat(plan['finished_utc'])-datetime.fromisoformat(first['started_utc'])).total_seconds()
    result=dict(status='completed_with_recorded_qualification_audit_correction',single_seed_exploratory=True,B=B,target=target,comparison=summaries,
        replay_Run219_all368_observations_and_endpoint_state_exact=replay,canonical_main_updates=main_updates,qualification_updates=qa_updates,
        total_actual_updates=main_updates+qa_updates,actual_full_gradient_calls=gradient_calls,actual_image_forward_backward_evaluations=gradient_calls*5000,
        canonical_nominal_training_image_exposures=main_updates*5000,endpoint_test_images=50000,
        active_worker_seconds=plan['total_active_worker_seconds'],elapsed_attempt_span_seconds=span,cloud_spend_usd=0,
        scientific_manifest_sha256=sha(RUN/'input_manifest.json'),infrastructure_continuation_sha256=sha(RUN/'02_continue_after_audit.py'),
        predeclared_zoom=[673,745],hold_updates=8,recovery_updates=8,threshold=1.8,no_tuning=True,qualification=plan['qualification'])
    write_json(HERE/'results.json',result)
    inventory=[]
    for path in sorted((RUN/'artifacts').rglob('*')):
        if path.is_file():inventory.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    csv_out(HERE/'raw_inventory.csv',inventory)
    write_json(HERE/'verification.json',dict(status='passed_direct_checks',canonical_boundaries=1840,canonical_decisions=1835,
        rule_checked_independently_from_most_recent_trigger_age=True,all_start_model_optimizer_rng_and_previous_vectors_exact=True,
        original_three_policies_all368_observations_and_endpoint_state_exact=replay,
        notation=r'\eta\widehat\kappa_t',plotted_scalar_checks=checked_scalars,source_sha256=sha(__file__),
        raw_files=len(inventory),raw_bytes=sum(r['bytes'] for r in inventory),
        figure_sha256={p.name:sha(p) for p in sorted((HERE/'figures').glob('*.pdf'))}))
    for row in summaries:print(json.dumps(row))
    print('Worker seconds:',result['active_worker_seconds'],'attempt span:',span,'raw bytes:',sum(r['bytes'] for r in inventory),flush=True)


if __name__=='__main__':main()
