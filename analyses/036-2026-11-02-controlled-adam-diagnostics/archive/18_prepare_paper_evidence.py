"""Expand the episode census and audit incremental forecasting; local data only."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'paper_evidence'
spec = importlib.util.spec_from_file_location('temporal', HERE/'15_peak_temporal_analysis.py')
temporal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(temporal)
LABELS = dict(mlp='MLP', resnet='ResNet-32', transformer='Transformer', all='Pooled')


def summarize(rows, names):
    return {name: temporal.scores([r['target'] for r in rows], [r['prob_'+name] for r in rows])
            for name in names}


def robustness(arms):
    rows, base, diag = temporal.dataset(arms, 8, .1)
    lookup = {a['id']: a for a in arms}
    rich, cosine, growth, delayed = [], [], [], []
    for i, row in enumerate(rows):
        a, t = lookup[row['arm']], row['state']
        logloss = np.log(a['loss'])
        rich.append([*base[i, :11], *[logloss[t]-logloss[t-lag] for lag in range(1, 17)], base[i, -1]])
        cosine.append(a['raw_cosine'][t])
        growth.append(np.log(a['gradient'][t]/a['gradient'][t-1]))
        delayed.append([a['projected'][t-lag] for lag in [16, 32]])
    rich, cosine, growth, delayed = map(np.asarray, [rich, cosine, growth, delayed])
    matrices = dict(loss_history=base, projected_change=np.c_[base, diag[:, 0]],
        raw_cosine=np.c_[base, cosine], cosine_and_gradient_growth=np.c_[base, cosine, growth],
        rich_loss=rich, rich_loss_projected=np.c_[rich, diag[:, 0]])
    results, forecasts = {}, []
    for group, gap in [('components_and_richer_loss', 24), ('past_delay_control', 48)]:
        inputs = matrices if gap == 24 else dict(rich_loss=rich,
            rich_loss_projected=np.c_[rich, diag[:, 0]],
            rich_loss_projected_delay16=np.c_[rich, delayed[:, 0]],
            rich_loss_projected_delay32=np.c_[rich, delayed[:, 1]])
        names = ['training_prior', *inputs]
        output, blocks = [], []
        times = np.array([r['state'] for r in rows])
        target = np.array([r['target'] for r in rows])
        for j, start in enumerate(temporal.TEST_STARTS):
            end = temporal.TEST_STARTS[j+1]-1 if j < 4 else 1016
            train = times < start-gap
            test = (times >= start) & (times <= end)
            first_input = min(times[test])-(32 if gap == 48 else 16)
            assert max(times[train])+8 < first_input
            predictions = {'training_prior': np.full(sum(test), np.mean(target[train]))}
            for name, x in inputs.items():
                assert np.isfinite(x).all() and len(np.unique(target[train])) == 2
                estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1.,
                    solver='liblinear', max_iter=1000, random_state=0))
                estimator.fit(x[train], target[train])
                assert estimator[-1].n_iter_[0] < 1000
                predictions[name] = estimator.predict_proba(x[test])[:, 1]
            selected = [dict(rows[index], group=group, fold_start=start,
                **{'prob_'+name: float(p[k]) for name, p in predictions.items()})
                for k, index in enumerate(np.flatnonzero(test))]
            blocks.append(dict(start=start, end=end, gap=gap, training_rows=int(sum(train)),
                last_training_outcome=int(max(times[train])+8), first_test_input=int(first_input),
                overall=summarize(selected, names), per_task={task: summarize(
                    [r for r in selected if r['task'] == task], names) for task in temporal.TASKS}))
            output.extend(selected)
        results[group] = dict(gap=gap, models=names, overall=summarize(output, names),
            per_task={task: summarize([r for r in output if r['task']==task], names)
                      for task in temporal.TASKS}, blocks=blocks)
        forecasts.extend(output)
    # Different model sets use blank CSV fields, never invented probabilities.
    fields = list(dict.fromkeys(key for r in forecasts for key in r))
    with (OUT/'robustness_forecasts.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(forecasts)
    return results


def markdown_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def tex_table(headers, rows, alignment):
    return '\n'.join([r'\begin{tabular}{'+alignment+'}', r'\hline',
        ' & '.join(headers)+r' \\', r'\hline', *[' & '.join(map(str, row))+r' \\' for row in rows],
        r'\hline', r'\end{tabular}', ''])


def main():
    OUT.mkdir(exist_ok=True)
    arms, sources = temporal.load_arms()
    old = temporal.read(HERE/'peak_temporal_results.json')
    for name, digest in old['sources'].items():
        assert temporal.sha(ROOT/name) == digest
    for name in ['peak_temporal_results.json', 'peak_forecasts.csv', 'paper_evidence/PROTOCOL.md']:
        sources[(HERE/name).relative_to(ROOT).as_posix()] = temporal.sha(HERE/name)
    events = [e for arm in arms for e in temporal.episodes(arm)]
    assert events == old['events']
    expanded = []
    for a in arms:
        for large in [False, True]:
            subset = [e for e in events if e['arm']==a['id'] and (e['strong'] or not large)]
            for alarm in temporal.ALARMS:
                leads = [e['alarm_leads'][alarm] for e in subset if e['alarm_leads'][alarm] is not None]
                expanded.append(dict(arm=a['id'], task=a['task'], lr=a['lr'],
                    severity='large' if large else 'all', alarm=alarm, episodes=len(subset),
                    hits=len(leads), misses=len(subset)-len(leads),
                    no_eligible_pre_onset_states=sum(not e['eligible_pre_onset_states'] for e in subset),
                    censored_recoveries=sum(e['censored'] for e in subset),
                    lead_median=float(np.median(leads)) if leads else None,
                    lead_q25=float(np.quantile(leads, .25)) if leads else None,
                    lead_q75=float(np.quantile(leads, .75)) if leads else None,
                    lead_min=min(leads) if leads else None, lead_max=max(leads) if leads else None))
    temporal.write_csv(OUT/'episode_summary.csv', expanded)
    predictions = list(csv.DictReader((HERE/'peak_forecasts.csv').open(encoding='utf-8', newline='')))
    alarms = []
    for a in arms:
        rows = [r for r in predictions if r['arm']==a['id']]
        for name in temporal.ALARMS:
            n = len(rows); pos = sum(int(r['target']) for r in rows)
            tp = sum(r[name]=='True' and r['target']=='1' for r in rows)
            fp = sum(r[name]=='True' and r['target']=='0' for r in rows)
            alarms.append(dict(arm=a['id'], task=a['task'], lr=a['lr'], alarm=name,
                states=n, positive=pos, prevalence=pos/n, alarms=tp+fp, tp=tp, fp=fp,
                precision=tp/(tp+fp) if tp+fp else None, recall=tp/pos if pos else None,
                false_positive_rate=fp/(n-pos)))
    temporal.write_csv(OUT/'alarm_summary.csv', alarms)
    robust = robustness(arms)
    # Reference fits must reproduce the earlier frozen forecast, not silently replace it.
    for name in ['loss_history', 'projected_change']:
        for key in ['average_precision', 'brier']:
            assert abs(robust['components_and_richer_loss']['overall'][name][key]
                       -old['forecast']['overall'][name][key]) < 1e-12
    aligned = []
    for a in arms:
        if a['lr'] != max(x['lr'] for x in arms if x['task']==a['task']): continue
        subset = [e for e in events if e['arm']==a['id'] and e['strong']]
        for e in subset:
            times = np.arange(e['onset_state']-8, e['onset_state']+9)
            assert min(times)>=0 and max(times)<1024
            aligned.append(dict(arm=a['id'], task=a['task'], lr=a['lr'], onset=e['onset_state'],
                peak=e['peak_state'], censored=e['censored'], x=list(range(-8, 9)),
                loss=(a['loss'][times]/e['baseline_loss']).tolist(),
                projected=a['projected'][times].tolist(), raw_cosine=a['raw_cosine'][times].tolist(),
                gradient=(a['gradient'][times]/a['gradient'][times[0]]).tolist(),
                momentum=(a['momentum'][times]/a['momentum'][times[0]]).tolist(),
                previous_episode_overlaps_window=any(p['arm']==e['arm'] and p['onset_state']<e['onset_state']
                    and (p['recovery_state'] is None or p['recovery_state']>=times[0]) for p in events)))
    assert len(aligned)==27
    record = dict(generator=Path(__file__).relative_to(ROOT).as_posix(),
        generator_sha256=temporal.sha(__file__), sources=sources,
        episode_summary=expanded, alarm_summary=alarms, robustness=robust, aligned_events=aligned,
        primary_forecast=old['forecast'], sensitivity=old['sensitivity'], outputs={})
    tables = ['# Evidence tables\n',
        'Generated from verified histories. Episode counts and state forecasts have different denominators. '
        'Lead summaries are conditional on a hit, in completed updates before the 10% onset. '
        'Repeated observations are dependent; these ranges are descriptive, not confidence intervals.\n',
        '## A. All nine arms: episode recurrence and early warnings\n']
    erows=[]
    for a in arms:
        selected=lambda severity, alarm: next(r for r in expanded if r['arm']==a['id']
            and r['severity']==severity and r['alarm']==alarm)
        allr=selected('all', 'eta_kappa_crosses_2'); large=selected('large', 'eta_kappa_crosses_2')
        ratio=lambda r: f"{r['hits']}/{r['episodes']}" if r['episodes'] else '--'
        lead=f"{large['lead_median']:g} [{large['lead_min']},{large['lead_max']}]" if large['hits'] else '--'
        erows.append([LABELS[a['task']], f"{a['lr']:g}", allr['episodes'], large['episodes'],
            ratio(allr), ratio(large), lead, ratio(selected('large','gradient_grows_20_percent')),
            ratio(selected('large','momentum_grows_20_percent')), allr['censored_recoveries']])
    headers=['Case','LR','Episodes','Large','Crossing: all','Crossing: large','Lead: large median [range]',
             'Gradient growth: large','Momentum growth: large','Unrecovered']
    tables += [markdown_table(headers, erows),
        '\nA fresh crossing means projected change moves from <=2 to >2. Norm-growth alarms require >20% '
        'one-state growth. Warnings must occur 1-8 states before onset while loss is <5% above its own '
        'trailing minimum. A large episode reaches >=50% above its frozen loss baseline. '
        'The CSV also retains all-event norm alarms, joint alarms, IQRs and misses.\n',
        '## B. Specificity: fixed >2 alarm on the forward-test states\n']
    arows=[]
    percent=lambda v: '--' if v is None else f'{100*v:.1f}'
    for a in arms:
        r=next(r for r in alarms if r['arm']==a['id'] and r['alarm']=='eta_kappa_above_2')
        arows.append([LABELS[a['task']],f"{a['lr']:g}",r['states'],r['positive'],r['alarms'],r['tp'],r['fp'],
                      percent(r['precision']),percent(r['recall']),percent(r['false_positive_rate'])])
    tables += [markdown_table(['Case','LR','States','Positive','Alarms','True','False','Precision %','Recall %','FPR %'], arows),
        '\nAn alarm is a state, not an independent episode. False means no >=10% rise relative to current '
        'loss within8 updates; it does not mean no later rise at any horizon. Zero-positive arms have '
        'undefined recall. The same 4,534 decisions are used for every primary comparison.\n',
        '## C. Primary incremental prediction\n']
    prows=[]
    for task in temporal.TASKS+['all']:
        group=old['forecast']['overall'] if task=='all' else old['forecast']['per_task'][task]
        b,p=group['loss_history'],group['projected_change']
        prows.append([LABELS[task],b['rows'],b['positive'],percent(b['prevalence']),
            f"{b['average_precision']:.3f}",f"{p['average_precision']:.3f}",
            f"{p['average_precision']-b['average_precision']:+.3f}",f"{b['brier']:.4f}",f"{p['brier']:.4f}"])
    tables += [markdown_table(['Case','States','Positive','Prevalence %','AP: loss','AP: + projected','Delta AP',
                              'Brier: loss','Brier: + projected'],prows),
        '\nAverage precision (AP) summarizes precision over recall; higher is better. Brier is mean squared '
        'probability error; lower is better. Neither is an accuracy percentage or a causal effect. '
        'Prevalence helps interpret AP; AP should be compared between models on the same subset. '
        'Case rows subset pooled fits, with all three rates per case.\n',
        '## D. Temporal variation of the primary comparison\n']
    tables += [markdown_table(['Test states','N','Positive','AP: loss','AP: + projected','Brier: loss','Brier: + projected'],
        [[f"{f['start']}-{f['end']}",f['test_rows'],f['models']['loss_history']['positive'],
          *[f"{f['models'][n]['average_precision']:.3f}" for n in ['loss_history','projected_change']],
          *[f"{f['models'][n]['brier']:.4f}" for n in ['loss_history','projected_change']]]
         for f in old['forecast']['folds']]), '\nThese five ranges describe temporal heterogeneity, not five independent replications.\n']
    for group, values in robust.items():
        tables += ['## Robustness: '+group.replace('_',' ')+'\n', f"All models here use a {values['gap']}-state gap.\n"]
        for task in ['all']+temporal.TASKS:
            scores=values['overall'] if task=='all' else values['per_task'][task]
            tables += ['### '+LABELS[task]+'\n', markdown_table(['Features','AP','Brier'],
                [[name,f"{r['average_precision']:.4f}",f"{r['brier']:.5f}"] for name,r in scores.items()]),'']
    tables += ['## Existing target/horizon sensitivity\n',markdown_table(
        ['Horizon','Rise %','AP: loss','AP: + projected','AP: + all'],
        [[r['horizon'],100*r['relative_rise'],*[f"{r['overall'][n]['average_precision']:.3f}"
            for n in ['loss_history','projected_change','all_diagnostics']]] for r in old['sensitivity']])]
    tables += ['## Full primary feature comparison (existing analysis)\n', markdown_table(
        ['Case','Added features','AP','Brier'],
        [[LABELS[task],name,f"{r['average_precision']:.4f}",f"{r['brier']:.5f}"]
         for task in ['all']+temporal.TASKS
         for name,r in (old['forecast']['overall'] if task=='all' else old['forecast']['per_task'][task]).items()]),
        '\nAdding all diagnostics is not consistently best; preserve these results in the supplement.\n',
        '## All fixed alarms, pooled forward-test states\n', markdown_table(
        ['Alarm','Alarm states','True','False','Precision %','Recall %'],
        [[name,r['all']['alarms'],r['all']['true_positive'],r['all']['false_positive'],
          percent(r['all']['precision']),percent(r['all']['recall'])]
         for name,r in old['forecast']['fixed_alarms'].items()])]
    (OUT/'TABLES.md').write_text('\n\n'.join(tables)+'\n',encoding='utf-8',newline='\n')
    (OUT/'table_episodes.tex').write_text(tex_table(
        ['Case',r'$\eta$','Episodes','Large','Cross: all','Cross: large','Lead','Growth $g$','Growth $m$','Cens.'],
        erows,'lrrrrrrrrr'),encoding='utf-8',newline='\n')
    (OUT/'table_prediction.tex').write_text(tex_table(
        ['Case','States','Positive',r'Prev. (\%)','AP loss','AP added',r'$\Delta$AP','Brier loss','Brier added'],
        prows,'lrrrrrrrr'),encoding='utf-8',newline='\n')
    for p in OUT.iterdir():
        if p.name in ['episode_summary.csv','alarm_summary.csv','robustness_forecasts.csv',
                      'TABLES.md','table_episodes.tex','table_prediction.tex']:
            record['outputs'][p.relative_to(ROOT).as_posix()]=temporal.sha(p)
    temporal.dump(OUT/'evidence_results.json',record)
    print({group: values['overall'] for group,values in robust.items()})


if __name__=='__main__': main()
