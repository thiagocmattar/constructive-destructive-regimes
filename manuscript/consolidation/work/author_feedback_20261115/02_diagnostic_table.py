"""Recompute next-update 0.1% rise table from verified native histories.

No fitting or trajectory selection: same three high-rate arms as the figure,
all decisions 1..1023. Accuracy means (TP+TN)/N; warning precision and its
event-rate lift are reported separately, because these are not accuracy.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
SOURCE = ROOT / 'analyses/055-2026-11-05-adam-directional-amplification'
spec = importlib.util.spec_from_file_location('source055', SOURCE / '01_analyze.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
arms, sources = module.load_arms()
names = ['always_no', 'always_yes', 'previous_rise', 'negative_cosine',
         'norm_growth', 'angle_norm', 'eta_kappa_ge2']
records = []
with (SOURCE / 'derived/forward_and_severity/severity.csv').open() as stream:
    old = list(csv.DictReader(stream))
for arm in arms:
    if arm['tier'] != 2:
        continue
    t = np.arange(1, 1024)
    loss, rho, growth, q = arm['loss'], arm['rho'][t], arm['r'][t], arm['Q'][t]
    y = (loss[t+1]-loss[t])/loss[t] >= .001
    flags = [np.zeros(len(t), dtype=bool), np.ones(len(t), dtype=bool),
             loss[t] > loss[t-1], rho < 0, growth > 1, rho < -1/growth, q >= 2]
    assert not np.any(q == 2)
    np.testing.assert_array_equal(flags[-1], flags[-2])
    base = float(np.mean(y))
    for name, flag in zip(names, flags):
        tp, fp = int(np.sum(flag & y)), int(np.sum(flag & ~y))
        fn, tn = int(np.sum(~flag & y)), int(np.sum(~flag & ~y))
        precision = tp/(tp+fp) if tp+fp else None
        row = dict(case=arm['task'], condition=name, n=len(t), events=int(sum(y)),
                   base_rate=base, tp=tp, fp=fp, fn=fn, tn=tn,
                   accuracy=(tp+tn)/len(t), accuracy_ratio=(tp+tn)/len(t)/(1-base),
                   precision=precision, precision_lift=precision/base if precision is not None else None,
                   recall=tp/sum(y))
        assert tp+fp+fn+tn == 1023
        if name in ['negative_cosine', 'norm_growth', 'eta_kappa_ge2']:
            prior = next(r for r in old if r['arm'] == arm['arm'] and r['target']=='future_step'
                         and float(r['minimum_relative_increase'])==.001 and r['subset']=='all')
            prefix = 'Q_ge2' if name=='eta_kappa_ge2' else name
            assert [int(prior[f'{prefix}_{k}']) for k in ['tp','fp','fn','tn']] == [tp,fp,fn,tn]
        records.append(row)
with (HERE / 'diagnostic_0p1pct.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, records[0].keys(), lineterminator='\n')
    writer.writeheader(); writer.writerows(records)
lookup = {(r['case'],r['condition']):r for r in records}
cases = ['mlp','resnet','transformer']
labels = dict(always_no='Always predict no increase', always_yes='Always predict increase',
              previous_rise='$L_t>L_{t-1}$', negative_cosine='$\\rho_t<0$',
              norm_growth='$\\|g_t\\|>\\|g_{t-1}\\|$',
              angle_norm='$\\rho_t<-\\|g_{t-1}\\|/\\|g_t\\|$',
              eta_kappa_ge2='$\\eta\\widehat\\kappa_t\\geq2$')
lines = [r'\begin{table*}[!t]', r'\centering\small',
 r'\caption{Before-update diagnostics for a next-step loss increase of at least 0.1\%. All 1,023 decisions per model are retained. Acc. is classification accuracy, $(\mathrm{TP}+\mathrm{TN})/1023$; Hit is the percentage of warnings followed by the target (precision); lift is Hit divided by the baseline event rate. The no-increase baseline exposes class imbalance. The last two conditions agree on these data and are two forms of the same diagnostic.}',
 r'\label{tab:severity}', r'\setlength{\tabcolsep}{4pt}',
 r'\begin{tabular}{@{}lrrrrrrrrr@{}}', r'\toprule',
 r'& \multicolumn{3}{c}{MLP} & \multicolumn{3}{c}{ResNet} & \multicolumn{3}{c}{Transformer} \\',
 r'Condition & Acc. (\%) & Hit (\%) & Lift ($\times$) & Acc. (\%) & Hit (\%) & Lift ($\times$) & Acc. (\%) & Hit (\%) & Lift ($\times$) \\', r'\midrule']
event_line = 'Baseline event rate'
for case in cases:
    r = lookup[case,'always_yes']
    event_line += f" & \\multicolumn{{3}}{{c}}{{{r['events']}/1023 ({100*r['base_rate']:.1f}\\%)}}"
lines += [event_line+r' \\',r'\midrule']
for name in names:
    values = []
    for case in cases:
        r = lookup[case,name]
        values += [f"{100*r['accuracy']:.1f}", f"{100*r['precision']:.1f}" if r['precision'] is not None else '--',
                   f"{r['precision_lift']:.2f}" if r['precision_lift'] is not None else '--']
    lines.append(labels[name]+' & '+' & '.join(values)+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(BASE / 'draft/tables/diagnostic-0p1pct.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
sources[(SOURCE / '01_analyze.py').relative_to(ROOT).as_posix()] = module.sha(SOURCE/'01_analyze.py')
(HERE / 'diagnostic_manifest.json').write_text(json.dumps(dict(target='(L[t+1]-L[t])/L[t] >= 0.001',
  previous_rise='L[t]>L[t-1]', decisions='t=1..1023', sources=sources, rows=records),indent=2)+'\n')
for case in cases:
    print(case, [(n,round(100*lookup[case,n]['accuracy'],2), lookup[case,n]['precision_lift']) for n in names])
