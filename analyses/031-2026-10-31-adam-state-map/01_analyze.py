"""Exploratory state summaries and seed-held-out paired-benefit prediction."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'analyses/030-2026-10-30-adam-lr-decisions'
POST = {
    'raw_cosine': 'Raw gradient cosine',
    'projected_change': 'Projected gradient change',
    'input_cosine': 'Incoming gradient / prior moment cosine',
    'incoming_ratio': 'Incoming / retained moment magnitude',
    'moment_cosine': 'Moment cosine',
    'adaptive_cosine': 'Adaptive direction cosine',
    'update_cosine': 'Actual update cosine',
    'descent_cosine': 'Raw gradient / descent direction cosine',
    'log_raw_norm': 'Log raw gradient norm',
    'log_moment_norm': 'Log moment norm',
    'log_adaptive_gain': 'Log adaptive / moment norm',
    'log_step_norm': 'Log actual step norm',
    'first_order_decrease': 'First-order sampled loss decrease',
    'clipping_fraction': 'Clipping fraction',
    'update_retention': 'Actual displacement / path length',
    'raw_retention': 'Raw gradient accumulation ratio',
    'moment_retention': 'Moment accumulation ratio',
    'adaptive_retention': 'Adaptive accumulation ratio',
    'raw_growth': 'Raw norm growth',
    'moment_growth': 'Moment norm growth',
}
PRE = {
    'raw_cosine': 'Initial raw gradient cosine',
    'projected_change': 'Initial projected gradient change',
    'log_raw_norm': 'Initial log raw gradient norm',
    'log_previous_moment_norm': 'Initial log prior moment norm',
    'input_cosine': 'Initial incoming / prior moment cosine',
    'incoming_ratio': 'Initial incoming / retained magnitude',
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def signals(e):
    return dict(raw_cosine=e['adjacent_cosine'],
        projected_change=e['projected_gradient_change'],
        input_cosine=e['incoming_previous_moment_cosine'],
        incoming_ratio=e['incoming_to_retained_moment_ratio'],
        moment_cosine=e['momentum']['adjacent_cosine'],
        adaptive_cosine=e['adaptive']['adjacent_cosine'],
        update_cosine=e['actual_parameter_update']['adjacent_cosine'],
        descent_cosine=-e['raw_actual_update_cosine'],
        log_raw_norm=float(np.log(e['gradient_norm'])),
        log_moment_norm=float(np.log(e['momentum']['momentum_norm'])),
        log_adaptive_gain=float(np.log(e['adaptive_to_momentum_norm_ratio'])),
        log_step_norm=float(np.log(e['actual_parameter_update']['gradient_norm'])),
        first_order_decrease=-e['raw_actual_update_cosine'] * e['gradient_norm'] * e['actual_parameter_update']['gradient_norm'],
        clipping_fraction=float(e['clip_coefficient'] < 1.0),
        update_retention=e['actual_parameter_update']['cumulative_ratio'],
        raw_retention=e['cumulative_ratio'],
        moment_retention=e['momentum']['cumulative_ratio'],
        adaptive_retention=e['adaptive']['cumulative_ratio'])


def summarize(events, cutoff):
    early = events[1:cutoff]
    values = [signals(e) for e in early]
    result = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
    for key in ['update_retention','raw_retention','moment_retention','adaptive_retention']:
        result[key] = signals(events[cutoff-1])[key]
    for key, seq in [('raw_growth', [e['gradient_norm'] for e in early]),
                     ('moment_growth', [e['momentum']['momentum_norm'] for e in early])]:
        result[key] = float(np.log(np.mean(seq[-4:]) / np.mean(seq[:4])))
    return result


def cv(rows, target, features):
    """Cell intercepts + zero/one/two slopes. All operations fit on training seeds."""
    y = np.array([r[target] for r in rows])
    x = np.array([[r['features'][f] for f in features] for r in rows], dtype=float)
    pred = np.zeros(len(rows)); fold = []; fits = []
    for seed in [1234, 1235, 1236]:
        train = np.array([i for i, r in enumerate(rows) if r['seed'] != seed])
        test = np.array([i for i, r in enumerate(rows) if r['seed'] == seed])
        groups = {(r['size'], r['multiplier']) for r in rows}
        mx = {}; my = {}
        for group in groups:
            ix = [i for i in train if (rows[i]['size'], rows[i]['multiplier']) == group]
            mx[group] = x[ix].mean(0); my[group] = y[ix].mean()
        xr = np.array([x[i] - mx[(rows[i]['size'], rows[i]['multiplier'])] for i in train])
        yr = np.array([y[i] - my[(rows[i]['size'], rows[i]['multiplier'])] for i in train])
        scale = np.sqrt(np.mean(xr*xr, axis=0)); scale[scale < 1e-14] = 1
        z = xr / scale
        beta, _, rank, singular = np.linalg.lstsq(z, yr, rcond=1e-10)
        for i in test:
            group = (rows[i]['size'], rows[i]['multiplier'])
            pred[i] = my[group] + ((x[i] - mx[group]) / scale) @ beta
        err = pred[test] - y[test]
        fold.append(dict(seed=seed, mae=float(np.mean(abs(err))), rmse=float(np.sqrt(np.mean(err*err)))))
        fits.append(dict(seed=seed, features=features, standardized_slopes=beta.tolist(),
                         scales=scale.tolist(), rank=int(rank), singular_values=singular.tolist(),
                         training_runs=[rows[i]['run'] for i in train], test_runs=[rows[i]['run'] for i in test]))
    err = pred-y
    return dict(features=features, predictions=pred.tolist(), target=y.tolist(),
        mae=float(np.mean(abs(err))), rmse=float(np.sqrt(np.mean(err*err))), folds=fold, fits=fits)


def main():
    old = json.loads((SOURCE/'results.json').read_text())
    expected = {r['run']: r for r in old['sources']}
    cohort = json.loads((SOURCE/'cohort.json').read_text())
    data = []; sources = []
    for row in cohort:
        path = ROOT/'runs'/row['run']/'artifacts/attempt-01/full-pass'
        identity = dict(run=row['run'], events_sha256=sha(path/'events.jsonl'), summary_sha256=sha(path/'summary.json'))
        for k in ['events_sha256', 'summary_sha256']:
            assert identity[k] == expected[row['run']][k], (row['run'], k)
        events = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
        updates = [e for e in events if e['event'] == 'update']
        assert [e['local_update'] for e in updates] == list(range(1,129))
        s = json.loads((path/'summary.json').read_text())
        for v in s['validations']:
            assert v['supervised_tokens'] == 693329 and v['short_tail_included']
        initial = signals(updates[0])
        initial['log_previous_moment_norm'] = float(np.log(updates[0]['previous_moment_norm']))
        data.append(dict(**row, events=updates, pre={k:initial[k] for k in PRE},
                         post={t:summarize(updates,t) for t in [8,16,32,64]},
                         val={e['local_update']:e['loss'] for e in s['validations']},
                         probe={e['local_update']:e['loss'] for e in s['probes']}))
        sources.append(identity)
    records = []; plot_rows = []; pre_replay_max_abs = 0.
    for block in sorted({r['block'] for r in data}):
        rows = [r for r in data if r['block'] == block]
        hold = next(r for r in rows if r['multiplier'] == 1)
        for row in rows:
            for k in PRE:
                difference = abs(row['pre'][k] - hold['pre'][k])
                pre_replay_max_abs = max(pre_replay_max_abs, difference)
                assert difference < 1e-6, (block, k, difference)
            assert [e['batch_indices_sha256'] for e in row['events']] == [e['batch_indices_sha256'] for e in hold['events']]
            gain = {t:hold['val'][t]-row['val'][t] for t in [0,16,128]}
            probe_gain = {t:hold['probe'][t]-row['probe'][t] for t in row['probe']}
            if row['multiplier'] != 1:
                for cutoff in [0,8,16,32,64]:
                    if cutoff == 0:
                        # Use one common-state measurement, not arm-specific roundoff.
                        fs = {k:float(np.log2(row['multiplier'])*hold['pre'][k]) for k in PRE}
                    else:
                        fs = {k:row['post'][cutoff][k]-hold['post'][cutoff][k] for k in POST}
                        if cutoff in probe_gain: fs['probe_gain'] = probe_gain[cutoff]
                    records.append(dict(run=row['run'], block=block, size=row['size'], seed=row['seed'],
                        multiplier=row['multiplier'], cutoff=cutoff, gain_final=gain[128],
                        gain_after16=gain[128]-gain[16], gain16=gain[16], features=fs))
            series = {}
            for key in ['input_cosine','incoming_ratio','update_retention']:
                a = np.array([signals(e)[key] for e in row['events']])
                b = np.array([signals(e)[key] for e in hold['events']])
                delta = a-b
                # Retention is already a prefix statistic; other traces use trailing8.
                if key != 'update_retention':
                    series[key] = dict(x=list(range(8,129)), y=np.convolve(delta,np.ones(8)/8,'valid').tolist())
                else: series[key] = dict(x=list(range(1,129)), y=delta.tolist())
            plot_rows.append(dict(run=row['run'], block=block, size=row['size'], seed=row['seed'],
                multiplier=row['multiplier'], gain_final=gain[128], gain16=gain[16],
                state16={k:row['post'][16][k] for k in ['input_cosine','incoming_ratio']},
                probe_gain=probe_gain, validation_gain=gain, series=series))
    experiments = []
    for cutoff in [0,8,16,32,64]:
        rows = [r for r in records if r['cutoff'] == cutoff]
        targets = ['gain_final','gain_after16'] if cutoff in [0,16] else ['gain_final']
        for target in targets:
            baseline = cv(rows,target,[])
            comparisons = {}
            for feature in rows[0]['features']:
                result = cv(rows,target,[feature])
                result['mae_improvement_percent'] = 100*(1-result['mae']/baseline['mae'])
                result['fold_mae_improvements_percent'] = [100*(1-a['mae']/b['mae']) for a,b in zip(result['folds'],baseline['folds'])]
                comparisons[feature] = result
            joint = {}
            if cutoff == 16:
                loss_base = comparisons['probe_gain']
                for feature in POST:
                    r = cv(rows,target,['probe_gain',feature])
                    r['mae_improvement_over_probe_percent'] = 100*(1-r['mae']/loss_base['mae'])
                    joint[feature] = r
            experiments.append(dict(cutoff=cutoff,target=target,run_order=[r['run'] for r in rows],
                baseline=baseline, candidates=comparisons, conditional_on_probe=joint))
    save('features.json',dict(sources=sources,pre_replay_max_abs=pre_replay_max_abs,pre_labels=PRE,post_labels={**POST,'probe_gain':'Fixed-probe loss benefit'},rows=records))
    save('results.json',dict(status='exploratory; outcomes known before analysis',
        generator_sha256=sha(__file__),seed_clusters=3,starting_states=6,altered_arms=12,
        experiments=experiments))
    save('figure_data.json',dict(rows=plot_rows))
    for e in experiments:
        best=sorted(e['candidates'],key=lambda k:e['candidates'][k]['mae'])[:4]
        print(json.dumps(dict(cutoff=e['cutoff'],target=e['target'],baseline_mae=e['baseline']['mae'],
            best=[dict(feature=k,mae=e['candidates'][k]['mae'],improvement=e['candidates'][k]['mae_improvement_percent'],folds=e['candidates'][k]['fold_mae_improvements_percent']) for k in best])))


if __name__ == '__main__':
    main()
