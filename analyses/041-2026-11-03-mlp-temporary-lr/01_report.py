r"""Direct reduction and fixed-window figures for \eta\widehat\kappa_t LR feedback."""
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'runs/219-2026-11-03-mlp-temporary-lr'
FIRST=RUN/'artifacts/attempts/001-local';RETRY=RUN/'artifacts/attempts/002-control-recovery'
PARENT=ROOT/'runs/218-2026-11-03-mlp-adam-step-actions/artifacts/attempts/001-20261103T172300Z-local-pilot/prefix/checkpoints/main_fork-state-000657.pt'
CONDITIONS=['unchanged','always_half','temporary']
NAMES={'unchanged':'Unchanged Adam','always_half':'Always-half LR','temporary':'Temporary reduction'}
COLORS={'unchanged':'black','always_half':'#0072B2','temporary':'#D55E00'}
STYLES={'unchanged':'-','always_half':'--','temporary':'-.'}


def read(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):Path(path).write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())

def csv_out(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def canonical(condition,name):
    if condition!='unchanged':return read(FIRST/'conditions'/condition/name)
    key='gradient_state' if name=='metrics.jsonl' else 'state'
    return [row for row in read(FIRST/'conditions/unchanged'/name) if row[key]<881]+read(RETRY/'unchanged'/name)


def main():
    original=json.loads((FIRST/'manifest.json').read_text());retry=json.loads((RETRY/'manifest.json').read_text())
    parent=torch.load(PARENT,map_location='cpu',weights_only=False)
    assert sha(PARENT)=='359097d026b45f8f0b8a33665aca4ad0382e52835745ceaeccf027ed29539ba4'
    observations={};actions={};summaries=[];aligned=[];events=[]
    spec=importlib.util.spec_from_file_location('existing_loss_events',ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py')
    detector=importlib.util.module_from_spec(spec);spec.loader.exec_module(detector)
    prefix=read(ROOT/'runs/218-2026-11-03-mlp-adam-step-actions/artifacts/attempts/001-20261103T172300Z-local-pilot/prefix/boundaries.jsonl')[:657]
    for condition in CONDITIONS:
        boundaries=canonical(condition,'boundaries.jsonl');decisions=canonical(condition,'decisions.jsonl');metrics=canonical(condition,'metrics.jsonl')
        assert [r['state'] for r in boundaries]==list(range(657,1025))
        assert [r['state'] for r in decisions]==list(range(657,1024)) and len(metrics)==367
        start=torch.load(FIRST/'conditions'/condition/'checkpoints/start-state-000657.pt',map_location='cpu',weights_only=False)
        assert start['identity']['parent_sha256']==sha(PARENT) and start['completed_updates']==657
        for key,value in parent['model'].items():assert torch.equal(value,start['model'][key])
        for index,state in parent['optimizer']['state'].items():
            for key in ['step','exp_avg','exp_avg_sq']:assert torch.equal(state[key],start['optimizer']['state'][index][key])
        for key in ['previous_gradient','previous_increment']:assert torch.equal(parent['observer'][key],start['observer'][key])
        assert torch.equal(parent['rng']['torch'],start['rng']['torch'])
        assert all(torch.equal(x,y) for x,y in zip(parent['rng']['cuda'],start['rng']['cuda']))
        assert parent['rng']['python']==start['rng']['python']
        assert np.array_equal(parent['rng']['numpy'][1],start['rng']['numpy'][1])
        for i,decision in enumerate(decisions):
            row=boundaries[i];value=row['eta_kappa_hat']
            assert decision['state']==row['state'] and decision['manuscript_gradient_index']==row['state']+1
            assert decision['available_before_update']==row['state']+1 and decision['eta_kappa_hat']==value
            expected=.001 if condition=='unchanged' else .0005 if condition=='always_half' else .0005 if value is not None and value>=1.8 else .001
            assert decision['applied_lr']==expected and decision['lr_over_base']==expected/.001
            assert metrics[i]['gradient_state']==row['state'] and metrics[i]['update']==row['state']+1
            assert metrics[i]['decision']==decision
            if row['previous_gradient_norm']>0:
                from_scalars=1-row['raw_adjacent_cosine']*row['gradient_norm']/row['previous_gradient_norm']
                assert abs(value-from_scalars)<=2e-12*(1+abs(value))
        observations[condition]=boundaries;actions[condition]=decisions
        for i,row in enumerate(boundaries):
            decision=decisions[i] if i<367 else None
            aligned.append(dict(condition=condition,state=row['state'],manuscript_gradient_index=row['state']+1,
                loss=row['loss'],eta_kappa_hat=row['eta_kappa_hat'],applied_lr=None if decision is None else decision['applied_lr'],
                lr_over_base=None if decision is None else decision['lr_over_base'],raw_gradient_norm=row['gradient_norm'],
                decision_available_before_update=None if decision is None else row['state']+1))
    B=observations['unchanged'][689-657]['loss'];target=.95*B
    for condition in CONDITIONS:
        rows=observations[condition];decisions=actions[condition];loss=np.asarray([r['loss'] for r in rows]);window=loss[689-657:]
        excess=np.maximum(window/B-1,0);lrs=np.asarray([r['applied_lr'] for r in decisions])
        source=retry if condition=='unchanged' else original['results'][condition]
        runner=(json.loads((FIRST/'conditions/unchanged/status.json').read_text())['elapsed_seconds']+retry['runner_seconds']) if condition=='unchanged' else source['runner_seconds']
        elapsed_eval=source.get('endpoint_test_seconds');endpoint=source['endpoint_test']
        joined=[r['loss'] for r in prefix]+loss.tolist();local=[]
        for event in detector.excursions(joined):
            if event['end_state']<657:continue
            e=dict(condition=condition,**event,peak_over_prerise=joined[event['peak_state']]/joined[event['before_state']],
                   peak_in_continuation=event['peak_state']>=657,later_start_after_original_recovery=event['before_state']>729)
            local.append(e);events.append(e)
        significant=[e for e in local if e['peak_in_continuation'] and e['peak_over_prerise']>=1.1]
        target_state=next((r['state'] for r in rows if r['loss']<=target),None)
        summary=dict(condition=condition,status='completed',updates=367,B=B,peak_excess=float(excess.max()),excess_area=float(excess.sum()),
             mean_postupdate_loss=float(loss[1:].mean()),final_loss=float(loss[-1]),full_max_loss=float(loss.max()),full_max_state=rows[int(loss.argmax())]['state'],
             reduced_updates=int((lrs==.0005).sum()),reduced_fraction=float((lrs==.0005).mean()),
             lr_switches_within=int((lrs[1:]!=lrs[:-1]).sum()),entry_lr_change=int(lrs[0]!=.001),
             final_lr=float(lrs[-1]),target_updates=None if target_state is None else target_state-657,
             excursions_gt10=len(significant),censored_gt10=sum(e['right_censored'] for e in significant),
             later_excursions_gt10=sum(e['later_start_after_original_recovery'] for e in significant),
             largest_own_reference_excursion=max([e['peak_over_prerise'] for e in significant],default=None),
             runner_seconds=runner,endpoint_test_loss=endpoint['loss'],endpoint_test_images=endpoint['loss_units'],
             endpoint_evaluation_or_audit_seconds=elapsed_eval,gradient_passes_canonical=368)
        assert endpoint['loss_units']==10000
        assert abs(summary['excess_area']-sum(max(0.,r['loss']/B-1) for r in rows if r['state']>=689))<1e-9
        assert abs(summary['mean_postupdate_loss']-sum(r['loss'] for r in rows[1:])/367)<1e-12
        summaries.append(summary)
    reduced_states=[r['state'] for r in actions['temporary'] if r['applied_lr']==.0005]
    assert len(reduced_states)==18 and all(b-a>1 for a,b in zip(reduced_states,reduced_states[1:]))
    assert not any(673<=s<745 for s in reduced_states)
    historical=read(ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/artifacts/attempt-01/local-mlp/lr-0.001/boundaries.jsonl')[657:1025]
    historical_exact=all((r['loss'],r['gradient_norm'])==(h['loss'],h['gradient_norm']) for r,h in zip(observations['unchanged'],historical))
    assert historical_exact
    old=ROOT/'runs/218-2026-11-03-mlp-adam-step-actions/artifacts/attempts/001-20261103T172300Z-local-pilot/branches'
    overlap={}
    for condition,arm in [('unchanged','A0'),('always_half','A1')]:
        earlier=read(old/arm/'boundaries.jsonl')
        overlap[condition]=all((r['loss'],r['gradient_norm'])==(h['loss'],h['gradient_norm']) for r,h in zip(observations[condition][:129],earlier));assert overlap[condition]
    (HERE/'figures').mkdir(exist_ok=True);(HERE/'scratch').mkdir(exist_ok=True)
    for bounds,name,yscale in [((673,745),'fixed_zoom','linear'),((657,1024),'full_continuation','log')]:
        fig,axes=plt.subplots(3,1,figsize=(10.8,8),sharex=True)
        fig.subplots_adjust(left=.09,right=.985,bottom=.09,top=.85,hspace=.12)
        for ax in axes:
            for state in reduced_states:ax.axvspan(state,state+1,color='#D55E00',alpha=.13,lw=0,zorder=0)
            for state in [689,703,729]:ax.axvline(state,color='.65',ls=':',lw=.65,zorder=0)
            ax.set_xlim(*bounds);ax.grid(alpha=.16)
        for condition in CONDITIONS:
            rows=observations[condition];x=np.asarray([r['state'] for r in rows]);loss=np.asarray([r['loss'] for r in rows]);values=np.asarray([r['eta_kappa_hat'] for r in rows])
            lrs=np.asarray([r['lr_over_base'] for r in actions[condition]]+[actions[condition][-1]['lr_over_base']])
            visible=(x>=bounds[0]) & (x<=bounds[1])
            for ax,y in [(axes[0],loss),(axes[1],values)]:
                line,=ax.plot(x[visible],y[visible],label=NAMES[condition],color=COLORS[condition],ls=STYLES[condition],lw=1.4)
                assert np.array_equal(line.get_xdata(),x[visible]) and np.array_equal(line.get_ydata(),y[visible])
            line,=axes[2].step(x[visible],lrs[visible],where='post',label=NAMES[condition],color=COLORS[condition],ls=STYLES[condition],lw=1.4)
            assert np.array_equal(line.get_ydata(),lrs[visible])
        axes[0].set_yscale(yscale);axes[0].set_ylabel('Training loss'+(' (log)' if yscale=='log' else ''))
        axes[1].axhline(1.8,color='.35',ls='--',lw=1.0);axes[1].set_ylabel(r'$\eta\widehat\kappa_t$')
        axes[2].set_ylabel('Actual LR / base LR');axes[2].set_yticks([.5,1]);axes[2].set_ylim(.44,1.06)
        axes[2].set_xlabel('Parameter state s (decision at s applies to s -> s+1)')
        handles,labels=axes[0].get_legend_handles_labels();handles.append(Patch(color='#D55E00',alpha=.13));labels.append('Temporary half-LR interval')
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.94),ncol=4,frameon=False,fontsize=9)
        fig.suptitle(('Fixed predeclared zoom: states 673-745' if name=='fixed_zoom' else 'Full continuation: states 657-1024')+'\nSeed 0 local full-objective MLP; single selected starting state',y=.995,fontsize=12)
        foot=(r'$\eta\widehat\kappa_t$ threshold 1.8; no rescaling when LR changes. '+('No temporary half-LR updates occur inside this fixed zoom.' if name=='fixed_zoom' else 'Shading shows actual one-update reductions; all recorded states shown.'))
        fig.text(.5,.025,foot,ha='center',fontsize=9)
        fig.savefig(HERE/'figures'/f'{name}.pdf');fig.savefig(HERE/'scratch'/f'{name}.png',dpi=170);plt.close(fig)
    csv_out(HERE/'comparison.csv',summaries);csv_out(HERE/'aligned_values.csv',aligned);csv_out(HERE/'excursions.csv',events)
    count_updates=sum(len(read(path)) for path in (FIRST/'conditions').glob('*/metrics.jsonl'))+len(read(RETRY/'unchanged/metrics.jsonl'))
    qualification_updates=sum(len(read(path)) for path in (FIRST/'qualification').glob('*/metrics.jsonl'))
    actual_gradient_calls=sum(len(read(path)) for path in FIRST.rglob('boundaries.jsonl'))+len(read(RETRY/'unchanged/boundaries.jsonl'))
    assert count_updates==1120 and qualification_updates==16 and actual_gradient_calls==1144
    wall=(datetime.fromisoformat(retry['finished_utc'])-datetime.fromisoformat(original['started_utc'])).total_seconds();assert wall<600
    result=dict(status='completed_with_recorded_control_recovery',single_seed_exploratory=True,B=B,target=target,comparison=summaries,
        reduced_states=reduced_states,rule_threshold=1.8,zoom_predeclared=[673,745],canonical_main_updates=1101,canonical_nominal_training_images=5505000,
        actual_main_updates=1120,recovery_replayed_updates=19,qualification_updates=16,total_actual_updates=1136,
        actual_full_gradient_calls=1144,actual_image_forward_backward_evaluations=5720000,endpoint_heldout_images=30000,
        active_worker_seconds=original['worker_elapsed_seconds']+retry['worker_elapsed_seconds'],elapsed_training_recovery_span_seconds=wall,
        failure=original['results']['unchanged'],recovery=retry,overlap_Run218_exact=overlap,unchanged_Run215_all368_exact=historical_exact,
        conditional_reductions_in_fixed_zoom=0,cloud_spend_usd=0)
    write_json(HERE/'results.json',result)
    inventory=[]
    for path in sorted((RUN/'artifacts').rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            inventory.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=digest))
    csv_out(HERE/'raw_inventory.csv',inventory)
    write_json(HERE/'verification.json',dict(status='passed_direct_checks',canonical_boundaries=1104,canonical_decisions=1101,
       parent_sha256=sha(PARENT),all_start_states_identical=True,decisions_preupdate_and_rule_exact=True,notation=r'\eta\widehat\kappa_t',
       replay_overlap19_exact=True,overlap_Run218_exact=overlap,unchanged_Run215_368_exact=True,
       point_checks='all plotted loss and available values exact against1104 source rows;1101 LR choices satisfy frozen rule and clock',
       zoom_fixed_before_results=[673,745],no_tuning=True,source_sha256=sha(__file__),raw_bytes=sum(x['bytes'] for x in inventory)))
    for row in summaries:print(json.dumps(row),flush=True)
    print('Reduced states:',reduced_states);print('Active worker seconds:',result['active_worker_seconds'],'wall span:',wall)


if __name__=='__main__':main()
