"""Retained-data reductions for the Adam manuscript revision; no training."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'analyses/036-2026-11-02-controlled-adam-diagnostics'
PRIOR = ROOT / 'analyses/055-2026-11-05-adam-directional-amplification'
MODELS = ['rho', 'A', 'rho+A', 'rho+A+rho*A', 'rho*A', 'logA',
          'rho+logA', 'rho+logA+rho*logA']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n',
                       encoding='utf-8', newline='\n')


def write_csv(name, rows):
    with (HERE / name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def design(rows, model):
    rho = np.array([r['rho'] for r in rows])
    a = np.array([r['A'] for r in rows])
    v = {'rho': rho, 'A': a, 'rho*A': rho*a, 'logA': np.log(a),
         'rho*logA': rho*np.log(a)}
    return np.column_stack([np.ones(len(rows))] + [v[k] for k in model.split('+')])


def fit(train, test, model):
    x, xx = design(train, model), design(test, model)
    y, yy = [np.array([r['log_loss_ratio'] for r in rows]) for rows in (train, test)]
    rank = int(np.linalg.matrix_rank(x)) if len(train) else 0
    if len(train) < 5 or rank != x.shape[1]:
        return dict(status='unsupported', train_n=len(train), test_n=len(test),
                    rank=rank, parameters=x.shape[1], coefficients=None, mse=None, r2=None)
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    error = yy - xx @ coef
    denom = float(np.sum((yy - yy.mean())**2))
    return dict(status='defined', train_n=len(train), test_n=len(test), rank=rank,
                parameters=x.shape[1], coefficients=coef.tolist(),
                mse=float(np.mean(error**2)),
                r2=float(1 - np.sum(error**2)/denom) if denom else None)


def main():
    spec = importlib.util.spec_from_file_location('prior', PRIOR/'01_analyze.py')
    prior = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
    arms, sources = prior.load_arms()
    for p in [HERE/'README.md', HERE/'01_analyze.py', PRIOR/'01_analyze.py']:
        sources[p.relative_to(ROOT).as_posix()] = sha(p)
    summaries, phases, features, event_rows, phase_events, cross_rows = [], [], [], [], [], []
    for arm in arms:
        task, lr = arm['task'], arm['lr']
        armid = f'{task}:{lr:g}'
        raw = [json.loads(line) for line in (ROOT/arm['source']/'metrics.jsonl').read_text().splitlines()]
        for s, m in enumerate(raw):
            # Verify the independent recorded dot product against normalized fields.
            dot = m['gradient_dot_actual_step']
            normprod = m['streams']['raw']['norm'] * m['streams']['actual']['norm']
            if normprod:
                assert np.isclose(dot/normprod, m['gradient_actual_step_cosine'], rtol=1e-10, atol=1e-12)
            cross_rows.append(dict(arm=armid, state=s,
                gradient_moment=m['gradient_current_moment_cosine'],
                gradient_direction=m['gradient_adaptive_cosine'],
                gradient_dot_actual_step=dot, loss_delta=float(arm['loss'][s+1]-arm['loss'][s])))
        for threshold in [.05, .10, .20]:
            all_selected = [e for e in arm['events'] if float(e['peak_loss_ratio']) > 1+threshold]
            selected = [e for e in all_selected if e['right_censored'] == 'False']
            occupancy = np.zeros(1024, dtype=bool)
            for e in all_selected:
                occupancy[int(e['before_state']):int(e['end_state'])] = True
            defined = [e for e in selected if e['rho_raw_before']]
            summaries.append(dict(arm=armid, task=task, lr=lr, threshold=threshold,
                recovered=len(selected), censored=len(all_selected)-len(selected),
                occupied_updates=int(occupancy.sum()), occupancy=float(occupancy.mean()),
                defined_prerise=len(defined),
                negative_prerise=sum(float(e['rho_raw_before'])<0 for e in defined),
                mean_absolute_peak_increase=float(np.mean([float(e['peak_loss_increase']) for e in selected])) if selected else None,
                mean_relative_peak_increase=float(np.mean([float(e['peak_loss_ratio'])-1 for e in selected])) if selected else None,
                mean_rise=float(np.mean([int(e['rise_steps']) for e in selected])) if selected else None,
                mean_recovery=float(np.mean([int(e['recovery_steps']) for e in selected])) if selected else None,
                observed_loss_excess_sum=sum(float(e['excess_loss_sum']) for e in all_selected)))
        for e in arm['events']:
            if float(e['peak_loss_ratio']) <= 1.1:
                continue
            event_rows.append(dict(arm=armid, spike_id=e['spike_id'],
                a=int(e['before_state']), p=int(e['peak_state']), end=int(e['end_state']),
                censored=e['right_censored']=='True', peak_ratio=float(e['peak_loss_ratio']),
                absolute_increase=float(e['peak_loss_increase']),
                observed_duration=int(e['observed_total_steps']),
                excess_loss_sum=float(e['excess_loss_sum'])))
            if e['right_censored'] == 'True':
                continue
            a, p, c = [int(e[k]) for k in ['before_state','peak_state','recovery_state']]
            if e['rho_raw_before']:
                features.append(dict(arm=armid, task=task, lr=lr, spike_id=e['spike_id'],
                    rho=float(e['rho_raw_before']),
                    A=float(e['gradient_norm_peak'])/float(e['gradient_norm_before']),
                    log_loss_ratio=float(np.log(float(e['peak_loss_ratio'])))))
            for phase, start, end in [('rise',a,p),('recovery',p,c)]:
                ms=raw[start:end]; assert len(ms)==end-start
                defined_raw=[m['streams']['raw']['cosine'] for m in ms if m['streams']['raw']['cosine'] is not None]
                phase_events.append(dict(arm=armid, task=task, lr=lr, spike_id=e['spike_id'],phase=phase,
                    updates=len(ms), locally_descending=sum(m['gradient_dot_actual_step']<0 for m in ms),
                    loss_increasing=sum(arm['loss'][s+1]>arm['loss'][s] for s in range(start,end)),
                    descending_fraction=float(np.mean([m['gradient_dot_actual_step']<0 for m in ms])),
                    defined_raw_updates=len(defined_raw),
                    negative_raw_fraction=float(np.mean(np.array(defined_raw)<0)) if defined_raw else None,
                    mean_gradient_moment=float(np.mean([m['gradient_current_moment_cosine'] for m in ms])),
                    mean_gradient_direction=float(np.mean([m['gradient_adaptive_cosine'] for m in ms]))))
        for phase in ['rise','recovery']:
            es=[e for e in phase_events if e['arm']==armid and e['phase']==phase]
            if not es: continue
            phases.append(dict(arm=armid,task=task,lr=lr,phase=phase,events=len(es),
                updates=sum(e['updates'] for e in es), locally_descending=sum(e['locally_descending'] for e in es),
                loss_increasing=sum(e['loss_increasing'] for e in es),
                raw_defined_events=sum(e['negative_raw_fraction'] is not None for e in es),
                **{k:float(np.mean([e[k] for e in es if e[k] is not None])) for k in [
                    'descending_fraction','negative_raw_fraction','mean_gradient_moment','mean_gradient_direction']}))
    fits=[]
    for armid in sorted({e['arm'] for e in features}):
        sample=[e for e in features if e['arm']==armid]
        train=[e for e in features if e['task']==sample[0]['task'] and e['arm']!=armid]
        for model in MODELS:
            for scope, training in [('in_sample',sample),('held_out_rate',train)]:
                fits.append(dict(arm=armid,task=sample[0]['task'],lr=sample[0]['lr'],
                    model=model,scope=scope,**fit(training,sample,model)))
    write_csv('event_incidence.csv',summaries)
    write_csv('event_features.csv',features)
    write_csv('pronounced_events.csv',event_rows)
    write_csv('event_phase_statistics.csv',phase_events)
    write_csv('phase_summary.csv',phases)
    write_csv('cross_alignments.csv',cross_rows)
    dump(HERE/'regressions.json',fits)
    result=dict(status='complete',trajectories=9,loss_states=9225,update_rows=len(cross_rows),
        all_candidates=sum(len(a['events']) for a in arms),
        pronounced_candidates=len(event_rows),pronounced_recovered=sum(not e['censored'] for e in event_rows),
        defined_severity_events=len(features),sources=sources,
        outputs={n:sha(HERE/n) for n in ['event_incidence.csv','event_features.csv','pronounced_events.csv',
            'event_phase_statistics.csv','phase_summary.csv','cross_alignments.csv','regressions.json']})
    dump(HERE/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','outputs']}))
    for armid in sorted({e['arm'] for e in features}):
        print(armid,[(f['model'],round(f['r2'],4)) for f in fits if f['arm']==armid and
            f['scope']=='in_sample' and f['model'] in ['rho','rho*A','logA','rho+logA','rho+logA+rho*logA']])


if __name__=='__main__': main()
