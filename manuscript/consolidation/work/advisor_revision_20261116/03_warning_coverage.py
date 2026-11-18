"""Report warning coverage from the retained diagnostic audit; no new experiment."""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parents[1] / 'draft'
SOURCE = HERE.parent / 'author_feedback_20261115/diagnostic_0p1pct.csv'
rows = list(csv.DictReader(SOURCE.open(encoding='utf-8')))
lookup = {(r['case'], r['condition']): r for r in rows}
labels = [
    ('always_no', 'Always no'), ('always_yes', 'Always yes'),
    ('previous_rise', 'Previous rise'),
    ('negative_cosine', r'$\rho_t<0$'),
    ('norm_growth', r'$\|g_t\|>\|g_{t-1}\|$'),
    ('angle_norm', 'Angle + norm'),
    ('eta_kappa_ge2', r'$1-\rho_t\|g_t\|/\|g_{t-1}\|\geq2$'),
]
lines = [r'\begin{table}[!ht]', r'\centering',
         r'\caption{Warning coverage for the fixed rules in main Table II. Each case has 1,023 decisions; there are 446/88/142 target increases for MLP/ResNet/Transformer. $W$ counts issued warnings, $H$ counts correct warnings, and recall is $100H/N_{\mathrm{target}}$. The final two rows flag the same steps in these trajectories.}',
         r'\label{tab:supp-warning-coverage}', r'\small',
         r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{lrrr|rrr|rrr}', r'\hline',
         r'& \multicolumn{3}{c|}{MLP} & \multicolumn{3}{c|}{ResNet} & \multicolumn{3}{c}{Transformer} \\',
         r'Rule & $W$ & $H$ & Recall & $W$ & $H$ & Recall & $W$ & $H$ & Recall \\ \hline']
for condition, label in labels:
    cells = []
    for case in ('mlp', 'resnet', 'transformer'):
        row = lookup[case, condition]
        tp, fp, fn, tn = (int(row[k]) for k in ('tp', 'fp', 'fn', 'tn'))
        assert tp + fp + fn + tn == int(row['n']) == 1023
        assert tp + fn == int(row['events'])
        recall = tp / (tp + fn)
        assert abs(recall - float(row['recall'])) < 1e-12
        cells += [str(tp + fp), str(tp), f'{100*recall:.1f}']
    lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
lines += [r'\hline', r'\end{tabular}', r'\end{table}']
target = DRAFT / 'tables/supp-warning-coverage.tex'
target.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
(HERE / 'warning-coverage.json').write_text(json.dumps({
    'source': SOURCE.as_posix(), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'rows_verified': len(rows), 'output_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
}, indent=2) + '\n', encoding='utf-8')
print('Verified and reported all 21 diagnostic rows.')
