"""Compact, direct reduction of the approved Run218 local pilot; no training."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'runs/218-2026-11-03-mlp-adam-step-actions'
spec=importlib.util.spec_from_file_location('pilot_execution',RUN/'03_local_pilot.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
b=pilot.b


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    attempt=ROOT/json.loads((RUN/'local_pilot_attempt.json').read_text())['path']
    manifest=json.loads((attempt/'manifest.json').read_text());results=json.loads((attempt/'results.json').read_text())
    prefix=pilot.rows(attempt/'prefix/boundaries.jsonl');arms=list(pilot.a.SUPPORTED)
    reference={r['state']:r['loss'] for r in pilot.rows(attempt/'branches/A0/boundaries.jsonl')}
    B=reference[689];target=.95*B
    parent=attempt/'prefix/checkpoints/main_fork-state-000657.pt'
    parent_hash=b.file_hash(parent);assert parent_hash==manifest['parent_sha256']
    saved_parent=torch.load(parent,map_location='cpu',weights_only=False)
    summary=[];aligned=[];event_rows=[];fork_checks={};curves={}
    for arm in arms:
        folder=attempt/'branches'/arm;execution=results[arm]
        if execution['status']!='completed':
            summary.append(dict(arm=arm,status=execution['status']));continue
        boundaries=pilot.rows(folder/'boundaries.jsonl');metrics=pilot.rows(folder/'metrics.jsonl');actions=pilot.rows(folder/'actions.jsonl')
        assert [r['state'] for r in boundaries]==list(range(657,786))
        assert [r['gradient_state'] for r in actions]==list(range(657,785)) and len(metrics)==128
        assert boundaries[0]['loss']==prefix[-1]['loss'] and execution['heldout_evaluation']['loss_units']==10000
        fork=torch.load(folder/'checkpoints/branch_start-state-000657.pt',map_location='cpu',weights_only=False)
        assert fork['identity']['parent']['sha256']==parent_hash
        assert fork['completed_updates']==657 and len(fork['observer']['history'])==657
        assert fork['observer']['history'][-1]['pre']['state']==656
        for key in ['model','rng']:pilot.equal(fork[key],saved_parent[key])
        pilot.equal(fork['optimizer']['state'],saved_parent['optimizer']['state'])
        for key in ['previous_gradient','previous_increment']:pilot.equal(fork['observer'][key],saved_parent['observer'][key])
        fork_checks[arm]='exact weights/moments/age/RNG/previous raw gradient/actual increment; expected declared transform only'
        loss={r['state']:r['loss'] for r in boundaries};curves[arm]=boundaries
        values=np.asarray([loss[t] for t in range(689,786)]);post=np.asarray([loss[t] for t in range(658,786)])
        excess=np.maximum(values/B-1.,0.)
        hits=[r for r in boundaries if r['loss']<=target];first=hits[0] if hits else None
        run_status=json.loads((folder/'status.json').read_text())
        target_seconds=None if first is None else (datetime.fromisoformat(first['utc'])-datetime.fromisoformat(run_status['started_utc'])).total_seconds()
        joint=[r['loss'] for r in prefix[:657]]+[r['loss'] for r in boundaries]
        local_events=[]
        for e in pilot.events.excursions(joint):
            if e['end_state']<657:continue
            row=dict(arm=arm,**e,peak_over_prerise=joint[e['peak_state']]/joint[e['before_state']],
                     crosses_fork=e['before_state']<657<e['end_state'],later_than_A0_recovery=e['before_state']>729,
                     peak_in_continuation=e['peak_state']>=657)
            event_rows.append(row);local_events.append(row)
        significant=[e for e in local_events if e['peak_over_prerise']>=1.1 and e['peak_in_continuation']]
        row=dict(arm=arm,status=execution['status'],reference_loss_B=B,peak_excess=float(excess.max()),excess_area=float(excess.sum()),
                 mean_loss_128=float(post.mean()),final_loss=loss[785],time_to_target_updates=None if first is None else first['state']-657,
                 time_to_target_seconds=target_seconds,time_to_target_censored=first is None,
                 raw_clip_count=sum(r.get('input_clip_factor',1.)<1. for r in actions),
                 proposal_clip_count=sum(r.get('proposal_clip_factor',1.)<1. for r in actions),
                 fallback_count=sum(r['fallback'] for r in actions),floor_count=sum(r['floor_active'] for r in actions),
                 upper_cap_count=sum(r['upper_cap_active'] for r in actions),mean_alpha_multiple=float(np.mean([r['alpha']/.001 for r in actions])),
                 runtime_including_setup_seconds=execution['elapsed_seconds'],runner_seconds=run_status['elapsed_seconds'],
                 endpoint_test_loss=execution['heldout_evaluation']['loss'],endpoint_test_seconds=execution['heldout_evaluation']['seconds'],
                 peak_allocated_MiB=execution['peak_allocated_bytes']/2**20,gradient_objective_passes=len(boundaries),heldout_image_forwards=10000,
                 observed_max_loss=max(loss.values()),observed_max_state=max(loss,key=loss.get),
                 excursions_gt10=len(significant),right_censored_gt10=sum(e['right_censored'] for e in significant),
                 later_excursions_gt10=sum(e['later_than_A0_recovery'] for e in significant),
                 max_excursion_ratio=max([e['peak_over_prerise'] for e in significant],default=None))
        summary.append(row)
        for i,r in enumerate(boundaries):
            action=actions[i] if i<128 else {};m=metrics[i] if i<128 else {}
            aligned.append(dict(arm=arm,state=r['state'],loss=r['loss'],raw_gradient_norm=r['gradient_norm'],
                raw_adjacent_cosine=r['raw_adjacent_cosine'],raw_projected_change=r['raw_projected_change'],
                completed_step_q=r['completed_step_q'],completed_step_kappa_sec=r['completed_step_kappa_sec'],
                preceding_actual_increment_norm=r['previous_increment_norm'],outgoing_alpha=action.get('alpha'),
                outgoing_input_clip_factor=action.get('input_clip_factor',1.) if action else None,
                outgoing_proposal_clip_factor=action.get('proposal_clip_factor',1.) if action else None,
                outgoing_reference_adaptive_norm=m.get('streams',{}).get('adaptive',{}).get('norm'),
                outgoing_applied_direction_norm=(action.get('chosen_ideal_proposal_norm')/action['alpha']) if action and action['alpha'] else None,
                outgoing_quantities_available_after_moment_staging_before_parameter_update=r['state']+1 if action else None))
        del fork
    control=summary[0]
    for r in summary:
        if r['status']!='completed':continue
        r.update(peak_excess_reduction=1-r['peak_excess']/control['peak_excess'],excess_area_reduction=1-r['excess_area']/control['excess_area'],
                 mean_loss_relative_to_A0=r['mean_loss_128']/control['mean_loss_128']-1,
                 final_loss_relative_to_A0=r['final_loss']/control['final_loss']-1)
        r['pilot_promising']=(r['arm']!='A0' and r['peak_excess_reduction']>=.5 and r['excess_area_reduction']>=.5
                             and r['mean_loss_relative_to_A0']<=.01 and r['final_loss_relative_to_A0']<=.01)
        # Direct independent scalar loop checks the vectorized outcome reduction.
        assert abs(r['excess_area']-sum(max(0.,curves[r['arm']][t-657]['loss']/B-1) for t in range(689,786)))<1e-9
        assert abs(r['mean_loss_128']-sum(x['loss'] for x in curves[r['arm']][1:])/128)<1e-12
    write_csv(HERE/'comparison.csv',summary);write_csv(HERE/'aligned_diagnostics.csv',aligned);write_csv(HERE/'excursions.csv',event_rows)
    historical=pilot.rows(ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/artifacts/attempt-01/local-mlp/lr-0.001/boundaries.jsonl')
    actual=np.asarray([r['loss'] for r in prefix[:657]]+[r['loss'] for r in curves['A0']])
    original=np.asarray([r['loss'] for r in historical[:786]])
    history_check=dict(states=786,exact_loss_match=bool(np.array_equal(actual,original)),max_absolute_loss_difference=float(np.max(abs(actual-original))),
                       event=results['A0']['historical_event'],qualified=results['A0']['historical_qualified'],contemporary_replay=results['A0']['contemporary_replay'])
    # All curves are unsmoothed, complete and aligned to parameter state.
    colors={'A0':'black','A1':'#0072B2','A2':'#56B4E9','A3':'#009E73','A4':'#E69F00','A5':'#CC79A7','A6':'#8A5A20','A8':'#D55E00'}
    styles={'A0':'-','A1':'-','A2':'--','A3':'-.','A4':'-','A5':':','A6':'--','A8':'-.'}
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
    fig.subplots_adjust(left=.09,right=.985,bottom=.07,top=.875,wspace=.22,hspace=.18)
    specs=[('loss','Training objective (log)','log'),('raw_gradient_norm','Raw gradient norm (log)','log'),
           ('raw_adjacent_cosine','Adjacent raw-gradient cosine','linear'),('outgoing_alpha','Outgoing alpha / base LR','linear'),
           ('preceding_actual_increment_norm','Previous ACTUAL increment norm (log)','log'),('completed_step_q','Completed-step q (symlog)','symlog')]
    for ax,(key,label,scale) in zip(axes.flat,specs):
        for arm in curves:
            data=[r for r in aligned if r['arm']==arm]
            values=np.array([np.nan if r[key] is None else r[key] for r in data])
            if key=='outgoing_alpha':values=values/.001
            ax.plot([r['state'] for r in data],values,color=colors[arm],ls=styles[arm],lw=1.65 if arm=='A0' else 1.15,label=arm)
        ax.set_yscale(scale,**({'linthresh':1} if scale=='symlog' else {}));ax.set_ylabel(label);ax.grid(alpha=.18)
        ax.axvline(689,color='gray',lw=.8,ls=':');ax.set_xlim(657,785)
        if key=='loss':
            ax.axhline(B,color='gray',ls='--',lw=.8);ax.axhline(target,color='gray',ls=':',lw=.8)
        if key=='raw_adjacent_cosine':ax.set_ylim(-1.05,1.05)
    for ax in axes[-1]:ax.set_xlabel('Parameter state (657 common fork;128 new updates)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=8,loc='upper center',bbox_to_anchor=(.54,.935),frameon=False)
    fig.suptitle('Seed0 local MLP pilot: all eight arms, full5000-image objective\nUnsmoothed; dashed horizontal loss=B, dotted=0.95B; vertical=state689',fontsize=12,y=.99)
    (HERE/'figures').mkdir(exist_ok=True);(HERE/'scratch').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures/local_pilot.pdf');fig.savefig(HERE/'scratch/local_pilot.png',dpi=170);plt.close(fig)
    main_updates=657+sum(r.get('updates',0) for r in results.values())
    grad_calls=len(prefix)+sum(len(curves[arm]) for arm in curves)
    assert main_updates==1681 and grad_calls==1690
    inventory=[]
    for path in sorted(attempt.rglob('*')):
        if path.is_file():inventory.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=b.file_hash(path)))
    write_csv(HERE/'raw_inventory.csv',inventory)
    b.write_json(HERE/'results.json',dict(single_seed_exploratory=True,attempt=attempt.relative_to(ROOT).as_posix(),
          B=B,target=target,target_window=[689,785],postfork_mean_states=[658,785],comparison=summary,
          historical=history_check,fork_checks=fork_checks,main_updates=main_updates,nominal_training_images=8405000,
          qualification_updates=12,main_gradient_calls=1690,qualification_gradient_calls=16,total_gradient_calls=1706,
          actual_training_image_forward_backward_evaluations=8530000,full_heldout_evaluations=9,heldout_images=90000,
          worker_elapsed_seconds=manifest['elapsed_seconds'],local_retained_bytes=sum(r['bytes'] for r in inventory),cloud_spend_usd=0,
          heldout_selection=False,failed_or_stopped_arms=[r['arm'] for r in summary if r['status']!='completed']))
    # Verify the figure's source series exactly against the emitted CSV.
    with (HERE/'aligned_diagnostics.csv').open() as stream:
        reread=list(csv.DictReader(stream))
    assert len(reread)==1032
    point_checks=0
    for r in reread:
        source=curves[r['arm']][int(r['state'])-657]
        for key,source_key in [('loss','loss'),('raw_gradient_norm','gradient_norm'),('raw_adjacent_cosine','raw_adjacent_cosine'),
                               ('completed_step_q','completed_step_q'),('preceding_actual_increment_norm','previous_increment_norm')]:
            assert (None if r[key]=='' else float(r[key]))==source[source_key]
            point_checks+=1
        actions=pilot.rows(attempt/'branches'/r['arm']/'actions.jsonl')
        expected=actions[int(r['state'])-657]['alpha'] if int(r['state'])<785 else None
        assert (None if r['outgoing_alpha']=='' else float(r['outgoing_alpha']))==expected
        point_checks+=1
    b.write_json(HERE/'empirical_verification.json',dict(status='passed_direct_checks',boundaries=1032,updates=1024,
        main_updates=1681,plotted_scalar_values_checked=point_checks,parent_sha256=parent_hash,all_forks_identical=True,historical=history_check,
        checks=['all8 statuses completed;129 consecutive boundary observations and128 actions each',
                'all8 branch starts: exact parent weights/moments/age/RNG/raw gradient/actual increment',
                'independent scalar sums agree with reported area and128-state mean',
                'all6192 plotted scalars match original boundaries/actions and emitted CSV exactly; no smoothing or truncation',
                'all final held-out evaluations cover10000 images; no heldout-based selection',
                'raw artifact inventory records file sizes and SHA-256; parent unchanged'],
        source_sha256=b.file_hash(__file__),figure_sha256=b.file_hash(HERE/'figures/local_pilot.pdf')))
    for r in summary:print(r['arm'],json.dumps(r,allow_nan=False),flush=True)
    print('Worker elapsed',manifest['elapsed_seconds'],'local bytes',sum(r['bytes'] for r in inventory))


if __name__=='__main__':main()
