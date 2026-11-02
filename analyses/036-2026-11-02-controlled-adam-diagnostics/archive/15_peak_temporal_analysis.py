"""Loss-only episodes and forward-blocked Adam diagnostic forecasting."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT/'runs/215-2026-11-02-controlled-full-batch-adam'
TASKS = ['mlp','resnet','transformer']
TEST_STARTS = [384,512,640,768,896]
MODELS = dict(loss_history=[], projected_change=[0], gradient_norm=[1,2],
              momentum_norm=[3,4], all_diagnostics=[0,1,2,3,4])
ALARMS = ['eta_kappa_above_2','eta_kappa_crosses_2','gradient_grows_20_percent',
          'momentum_grows_20_percent','crossing_and_momentum_growth']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def load_arms():
    base = read(HERE/'full_diagnostics_manifest.json')
    result = read(RUN/'results.json')
    assert result['status'] == 'complete_verified' and len(result['conditions']) == 9
    sources = {p.relative_to(ROOT).as_posix():sha(p) for p in
        [HERE/'PEAK_TEMPORAL_PROTOCOL.md', HERE/'full_diagnostics_manifest.json',
         HERE/'full_unsmoothed_manifest.json', RUN/'results.json', RUN/'config.json']}
    arms = []
    for condition in sorted(result['conditions'],key=lambda c:(TASKS.index(c['task']),c['lr'])):
        folder = ROOT/condition['source']
        for name in ['manifest.json','metrics.jsonl','boundaries.jsonl']:
            path = folder/name;relative = path.relative_to(ROOT).as_posix()
            digest = sha(path);assert digest == base['sources'][relative]
            sources[relative] = digest
        identity = read(folder/'manifest.json')
        assert sha(folder/'manifest.json') == condition['manifest_sha256']
        assert identity['status'] == 'complete' and identity['completed_updates'] == 1024
        b = [json.loads(s) for s in (folder/'boundaries.jsonl').read_text().splitlines()]
        m = [json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
        assert [v['state'] for v in b] == list(range(1025))
        assert [v['update'] for v in m] == list(range(1,1025))
        assert [v['gradient_state'] for v in m] == list(range(1024))
        gradient = np.array([v['gradient_norm'] for v in b[:-1]])
        projected = np.array([np.nan if v['projected_gradient_change'] is None
                              else v['projected_gradient_change'] for v in b[:-1]])
        np.testing.assert_array_equal(gradient,[v['streams']['raw']['norm'] for v in m])
        np.testing.assert_array_equal(projected,[np.nan if v['projected_gradient_change'] is None
                                               else v['projected_gradient_change'] for v in m])
        cosine = np.array([np.nan if v['streams']['raw']['cosine'] is None
                           else v['streams']['raw']['cosine'] for v in m])
        np.testing.assert_allclose(projected[1:],1-cosine[1:]*gradient[1:]/gradient[:-1],rtol=2e-12,atol=2e-12)
        assert np.all(gradient[1:][projected[1:]>2] > gradient[:-1][projected[1:]>2])
        assert np.all(cosine[projected>2] < 0)
        arms.append(dict(id=f'{condition["task"]}:{condition["lr"]:g}',task=condition['task'],
            lr=condition['lr'],source=condition['source'],initial_tensor_sha256=identity['initial_tensor_sha256'],
            loss=np.array([v['loss'] for v in b]),projected=projected,gradient=gradient,
            momentum=np.array([v['streams']['moment']['norm'] for v in m]),
            raw_cosine=cosine,moment_cosine=np.array([np.nan if v['streams']['moment']['cosine'] is None
                else v['streams']['moment']['cosine'] for v in m]),
            adam_cosine=np.array([np.nan if v['streams']['adaptive']['cosine'] is None
                else v['streams']['adaptive']['cosine'] for v in m]),
            retention=np.r_[np.nan,[v['streams']['actual']['B_over_A'] for v in m]]))
    return arms,sources


def quiet(arm,state):
    return bool(arm['loss'][state] < 1.05*min(arm['loss'][state-16:state+1]))


def alarms_at(arm,state):
    projected = arm['projected'];gradient = arm['gradient'];momentum = arm['momentum']
    above = projected[state]>2
    crossing = above and projected[state-1]<=2
    grow_g = gradient[state]>1.2*gradient[state-1]
    grow_m = momentum[state]>1.2*momentum[state-1]
    return dict(zip(ALARMS,map(bool,[above,crossing,grow_g,grow_m,crossing and grow_m])))


def episodes(arm):
    loss = arm['loss'];out = [];state = 64
    while state<len(loss):
        baseline_state = state-16+int(np.argmin(loss[state-16:state]))
        baseline = loss[baseline_state]
        if loss[state]<1.10*baseline:
            state += 1;continue
        recovery = next((s for s in range(state+1,len(loss)) if loss[s]<baseline),None)
        stop = recovery if recovery is not None else len(loss)
        peak = state+int(np.argmax(loss[state:stop]))
        earlier = [s for s in range(max(64,state-8),state) if quiet(arm,s)]
        leads = {name: max((state-s for s in earlier if alarms_at(arm,s)[name]),default=None)
                 for name in ALARMS}
        local = np.arange(max(1,state-8),min(1024,peak+9))
        maxima = {name:int(local[np.argmax(arm[key][local])]) for name,key in
                  [('projected_peak_state','projected'),('gradient_peak_state','gradient'),
                   ('momentum_peak_state','momentum')]}
        out.append(dict(arm=arm['id'],task=arm['task'],lr=arm['lr'],onset_state=state,
            baseline_state=baseline_state,baseline_loss=float(baseline),peak_state=peak,
            peak_loss=float(loss[peak]),peak_over_baseline=float(loss[peak]/baseline),
            recovery_state=recovery,censored=recovery is None,strong=bool(loss[peak]>=1.5*baseline),
            eligible_pre_onset_states=earlier,alarm_leads=leads,**maxima))
        state = stop+1
    return out


def scores(target, probabilities):
    target = np.asarray(target,dtype=int);probabilities = np.asarray(probabilities)
    assert len(target)>0 and np.all((probabilities>=0)&(probabilities<=1))
    return dict(rows=len(target),positive=int(sum(target)),prevalence=float(np.mean(target)),
        average_precision=float(average_precision_score(target,probabilities)) if sum(target)>0 else None,
        brier=float(brier_score_loss(target,probabilities)))


def dataset(arms,horizon,rise):
    rows=[];base=[];diagnostics=[]
    for i,arm in enumerate(arms):
        logloss=np.log(arm['loss']);logg=np.log(arm['gradient']);logm=np.log(arm['momentum'])
        for state in range(64,1025-horizon):
            if not quiet(arm,state):continue
            future=float(max(arm['loss'][state+1:state+horizon+1])/arm['loss'][state]-1)
            indicator=[float(k==i) for k in range(len(arms))]
            base.append(indicator+[state/1024,logloss[state],
                *[logloss[state]-logloss[state-lag] for lag in [1,4,8]],
                logloss[state]-min(logloss[state-16:state+1])])
            diagnostics.append([arm['projected'][state],logg[state],logg[state]-logg[state-1],
                                logm[state],logm[state]-logm[state-1]])
            rows.append(dict(arm=arm['id'],task=arm['task'],lr=arm['lr'],state=state,
                target=int(future>=rise),future_max_relative_increase=future,
                loss=float(arm['loss'][state]),eta_kappa_raw=float(arm['projected'][state]),
                gradient_norm=float(arm['gradient'][state]),momentum_norm=float(arm['momentum'][state]),
                **alarms_at(arm,state)))
    baseline=np.asarray(base);diag=np.asarray(diagnostics)
    assert np.isfinite(baseline).all() and np.isfinite(diag).all()
    return rows,baseline,diag


def forecast(arms,horizon,rise,keep_predictions=False):
    rows,baseline,diag=dataset(arms,horizon,rise)
    state=np.array([r['state'] for r in rows]);target=np.array([r['target'] for r in rows])
    output=[];folds=[]
    for j,start in enumerate(TEST_STARTS):
        end=TEST_STARTS[j+1]-1 if j+1<len(TEST_STARTS) else 1024-horizon
        train=state<start-(16+horizon);test=(state>=start)&(state<=end)
        assert sum(train)>0 and sum(test)>0
        assert max(state[train])+horizon < min(state[test])-16
        fold=dict(start=start,end=end,training_rows=int(sum(train)),test_rows=int(sum(test)),
            last_training_state=int(max(state[train])),last_training_outcome=int(max(state[train])+horizon),
            first_test_history_state=int(min(state[test])-16),gap=16+horizon,models={})
        prediction={}
        for name,indices in MODELS.items():
            x=np.column_stack([baseline,diag[:,indices]]) if indices else baseline
            coefficients=[]
            if len(np.unique(target[train]))<2:
                probability=np.full(sum(test),float(np.mean(target[train])))
                fallback=True
            else:
                estimator=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='liblinear',
                    max_iter=1000,random_state=0))
                estimator.fit(x[train],target[train])
                assert estimator[-1].n_iter_[0]<1000
                probability=estimator.predict_proba(x[test])[:,1];fallback=False
                if indices:coefficients=estimator[-1].coef_[0][-len(indices):].tolist()
            prediction[name]=probability
            fold['models'][name]=dict(**scores(target[test],probability),constant_training_prior=fallback,
                                     standardized_added_coefficients=coefficients)
        for k,index in enumerate(np.flatnonzero(test)):
            output.append(dict(rows[index],fold_start=start,
                               **{'prob_'+name:float(p[k]) for name,p in prediction.items()}))
        folds.append(fold)
    def summarize(subset):
        return {name:scores([r['target'] for r in subset],[r['prob_'+name] for r in subset]) for name in MODELS}
    per_task={task:summarize([r for r in output if r['task']==task]) for task in TASKS}
    per_arm={arm['id']:summarize([r for r in output if r['arm']==arm['id']]) for arm in arms}
    fixed={}
    for name in ALARMS:
        by_task={}
        for group in ['all']+TASKS:
            subset=output if group=='all' else [r for r in output if r['task']==group]
            tp=sum(r[name] and r['target'] for r in subset)
            fp=sum(r[name] and not r['target'] for r in subset)
            positives=sum(r['target'] for r in subset);negatives=len(subset)-positives
            by_task[group]=dict(alarms=tp+fp,true_positive=tp,false_positive=fp,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/positives if positives else None,
                false_positive_rate=fp/negatives if negatives else None)
        fixed[name]=by_task
    result=dict(horizon=horizon,relative_rise=rise,eligible_rows=len(rows),forward_test_rows=len(output),
                overall=summarize(output),per_task=per_task,per_arm=per_arm,folds=folds,fixed_alarms=fixed)
    return result,output if keep_predictions else None


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    arms,sources=load_arms()
    catalog=[event for arm in arms for event in episodes(arm)]
    examples=[]
    for task in TASKS:
        arm=max((a for a in arms if a['task']==task),key=lambda a:a['lr'])
        event=max((e for e in catalog if e['arm']==arm['id'] and not e['censored']),
                  key=lambda e:(e['peak_over_baseline'],-e['peak_state']))
        start=event['peak_state']-10
        end=next(s for s in range(event['peak_state']+1,1025) if arm['loss'][s]<arm['loss'][start])
        above=next(s for s in range(start+1,event['peak_state']+1) if arm['loss'][s]>arm['loss'][start])
        examples.append(dict(task=task,arm=arm['id'],lr=arm['lr'],peak_state=event['peak_state'],
            start_state=start,end_state=end,starting_loss=float(arm['loss'][start]),
            peak_loss=event['peak_loss'],end_loss=float(arm['loss'][end]),
            marker_state=above-1,episode_onset=event['onset_state'],
            episode_baseline_state=event['baseline_state'],episode_baseline_loss=event['baseline_loss'],
            peak_over_episode_baseline=event['peak_over_baseline']))
    summary=[]
    for arm in arms:
        selected=[e for e in catalog if e['arm']==arm['id']]
        strong=[e for e in selected if e['strong']]
        summary.append(dict(arm=arm['id'],task=arm['task'],lr=arm['lr'],episodes=len(selected),
            strong_episodes=len(strong),censored=sum(e['censored'] for e in selected),
            early_alarm_hits_all={name:sum(e['alarm_leads'][name] is not None for e in selected) for name in ALARMS},
            early_alarm_hits_strong={name:sum(e['alarm_leads'][name] is not None for e in strong) for name in ALARMS}))
    primary,predictions=forecast(arms,8,.1,True)
    sensitivity=[]
    for horizon in [4,8,16]:
        for rise in [.05,.1,.25]:
            if (horizon,rise)==(8,.1):item=primary
            else:item,_=forecast(arms,horizon,rise)
            sensitivity.append(dict(horizon=horizon,relative_rise=rise,overall=item['overall'],
                                    per_task=item['per_task'],folds=item['folds']))
    predictions_path=HERE/'peak_forecasts.csv';write_csv(predictions_path,predictions)
    events_path=HERE/'peak_events.csv'
    event_rows=[]
    for event in catalog:
        row={k:v for k,v in event.items() if k not in ['alarm_leads','eligible_pre_onset_states']}
        row.update({name+'_lead':value for name,value in event['alarm_leads'].items()})
        event_rows.append(row)
    write_csv(events_path,event_rows)
    transformer=next(a for a in arms if a['task']=='transformer' and a['lr']==.003)
    local=[]
    for state in range(782,796):
        local.append(dict(state=state,loss=float(transformer['loss'][state]),
            loss_step_relative_change=float(transformer['loss'][state]/transformer['loss'][state-1]-1),
            rise_over_local_trough=float(transformer['loss'][state]/transformer['loss'][787]-1),
            eta_kappa_raw=float(transformer['projected'][state]),
            gradient_norm=float(transformer['gradient'][state]),momentum_norm=float(transformer['momentum'][state]),
            **alarms_at(transformer,state)))
    result=dict(status='numerical_checks_passed',generator=Path(__file__).relative_to(ROOT).as_posix(),
        generator_sha256=sha(__file__),sources=sources,
        environment=dict(numpy=np.__version__,sklearn=sklearn.__version__),
        protocol=read_protocol_hash(),state_alignment_verified=True,
        algebra_above_2_implies_raw_gradient_growth_verified_all_arms=True,
        primary_horizon=8,primary_relative_rise=.1,episode_summary=summary,events=catalog,
        selected_examples=examples,transformer_local_timing=local,forecast=primary,sensitivity=sensitivity,
        outputs={p.relative_to(ROOT).as_posix():sha(p) for p in [predictions_path,events_path]},
        limitations=['Exploratory design motivated by an already inspected event; no confirmatory p-value.',
                     'Temporal blocks and repeated peaks are dependent observations from one initialization per task.',
                     'Chronological prediction is not identification of an intervention effect.'])
    dump(HERE/'peak_temporal_results.json',result)
    print(json.dumps(dict(episodes=len(catalog),strong_episodes=sum(e['strong'] for e in catalog),
        examples=examples,forecast_overall=primary['overall'],forecast_by_task=primary['per_task'])))


def read_protocol_hash():
    return dict(path=(HERE/'PEAK_TEMPORAL_PROTOCOL.md').relative_to(ROOT).as_posix(),
                sha256=sha(HERE/'PEAK_TEMPORAL_PROTOCOL.md'))


if __name__=='__main__':main()
