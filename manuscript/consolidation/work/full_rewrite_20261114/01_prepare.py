"""Prepare source-preserving, consolidation-local assets for the complete rewrite."""
from pathlib import Path
import csv
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
DRAFT = BASE / 'draft'
EVIDENCE = BASE / 'evidence'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + '\n', encoding='utf-8', newline='\n')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


preservation_path = HERE / 'preservation.json'
if not preservation_path.exists():
    protected = [p for d in [EVIDENCE, BASE / 'human'] for p in d.rglob('*') if p.is_file()]
    tracked = subprocess.check_output(['git', 'ls-files', 'manuscript/draft', 'manuscript/artifacts'], cwd=ROOT, text=True).splitlines()
    protected += [ROOT / p for p in tracked if (ROOT / p).is_file()]
    protected += [DRAFT / name for name in ['intervention-review.tex', 'intervention-review.pdf', '05c-interventions.tex']]
    write(preservation_path, json.dumps({p.relative_to(ROOT).as_posix(): sha(p) for p in protected}, indent=2))

# Only cited entries are compiled; the two supplied bibliography snapshots remain intact.
snapshots = [EVIDENCE / '02-related-work/bibliography-current.bib', EVIDENCE / '02-related-work/references-to-add.bib']
extra = r'''
@inproceedings{li2018visualizing,
  title={Visualizing the Loss Landscape of Neural Nets},
  author={Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={Advances in Neural Information Processing Systems}, volume={31}, year={2018},
  url={https://arxiv.org/abs/1712.09913}
}
@misc{cohen2022adaptive,
  title={Adaptive Gradient Methods at the Edge of Stability},
  author={Cohen, Jeremy M. and Ghorbani, Behrooz and Krishnan, Shankar and Agarwal, Naman and Medapati, Sourabh and Badura, Michal and Suo, Daniel and Cardoze, David and Nado, Zachary and Dahl, George E. and Gilmer, Justin},
  year={2024}, note={arXiv:2207.14484, version 2}, url={https://arxiv.org/abs/2207.14484v2}
}
@misc{terjek2025feature,
  title={Feature Learning Beyond the Edge of Stability}, author={Terj{\'e}k, D{\'a}vid},
  year={2025}, note={arXiv:2502.13110, version 2}, url={https://arxiv.org/abs/2502.13110v2}
}
@misc{jiang2025ntk,
  title={Understanding the Evolution of the Neural Tangent Kernel at the Edge of Stability},
  author={Jiang, Kaiqi and Cohen, Jeremy and Li, Yuanzhi}, year={2025},
  note={arXiv:2507.12837}, url={https://arxiv.org/abs/2507.12837}
}
'''
write(DRAFT / 'references.bib', '\n'.join(p.read_text(encoding='utf-8') for p in snapshots) + extra)

# Compact GD endpoint and approximation audit, generated from the frozen numerical record.
audit_path = EVIDENCE / '04-gd-sgd/data/24-section_iv_review_checks.json'
audit = json.loads(audit_path.read_text(encoding='utf-8'))
rows = read_csv(EVIDENCE / '04-gd-sgd/data/56-endpoints.csv')
rows = [r for r in rows if r['offset'] == '0']
names = {'mlp': 'MLP', 'resnet': 'ResNet-32', 'transformer': 'Transformer'}
lines = [r'\begin{tabular}{llrrrrr}', r'\toprule',
         r'Model & $\eta$ & $L_{3000}$ & $R_{3000}$ & $I_{\rm all}/E$ & $I_1/E$ & Sign errors \\', r'\midrule']
for r, a in zip(rows, audit['runs']):
    assert r['run'] == a['run_id']
    lines.append(f"{names[r['task']]} & {float(r['lr']):g} & {float(r['final_loss']):.4f} & {float(r['R']):.4f} & "
                 f"{a['final_all_pair']/a['final_energy']:.3f} & {a['final_lag_one']/a['final_energy']:.3f} & "
                 f"{a['sign_sensitivity'][0]['sign_disagreements']}/2999 " + r'\\')
lines += [r'\bottomrule', r'\end{tabular}']
write(DRAFT / 'tables/gd-audit.tex', '\n'.join(lines))

audit_captions = [
    'Coverage and precision versus minimum next-step relative loss increase. All 1,023 eligible decisions per model are retained; counts identify rare targets. Reversal, growth, their conjunction, and the projected-response threshold are compared with unconditional frequency.',
    'Individual loss increases of at least 10 percent at each future offset. All offsets use the same 1,008 decision states. Coverage and precision describe the current flag at that offset; they are not validation of an outcome-selected lead time.',
    'Net loss increase of at least 10 percent from the decision state to each future horizon. This target differs from an individual future step increase; common decision support is retained.',
    'Any loss at least 10 percent above the decision-state loss within the future horizon. The target set expands with the horizon, so increasing precision must be compared with its rising base frequency.',
    'Within-horizon increases restricted to currently non-rising decisions. This excludes an already-rising step but not necessarily an ongoing excursion. Sparse threshold support and target frequencies remain visible.',
    'Remembered warnings for the unique next-step target. Each target is counted once; all memory widths use decisions 16--1023. Longer memory increases both coverage and the number of non-target flags.',
]
plates = []
for page, caption in enumerate(audit_captions, 1):
    plates += [r'\clearpage', r'\begin{landscape}', r'\begin{figure}[H]', r'\centering',
               rf'\includegraphics[page={page},width=9.6in]{{\evidence/05-adam/a-characterization/support/adam_forward_and_severity.pdf}}',
               r'\caption{' + caption + '}', r'\end{figure}', r'\end{landscape}']
write(DRAFT / 'tables/audit-plates.tex', '\n'.join(plates))

# Keep all main-comparison settings in the supplement; reuse the original vector PDFs.
plate_specs = [
    ('17-matched-intervention-loss.pdf', 1, 'Complete loss trajectories for all four intervention families. Each column repeats the same unchanged-Adam control and logarithmic scale. Every state in the fixed 512--800, 256--576, and 544--928 windows is retained, without smoothing. Only the family named by each row changes; one initialization per model.'),
    ('21-matched-heldout-loss.pdf', 1, 'Complete test loss at the five fixed evaluation states. Each CIFAR-10 evaluation uses all 10,000 test images; Transformer evaluations use all 245,550 valid test tokens. Lines join only measured states. These observations are not used for checkpoint or setting selection, and cannot resolve intervening held-out excursions.'),
    ('22-matched-update-and-controller-activity.pdf', 1, 'Actual stored parameter-increment norms for every treatment and unchanged Adam. Logarithmic scales are shared within each model. Equal nominal coefficients do not imply matched applied parameter or functional movement; these curves retain their resulting differences.'),
    ('22-matched-update-and-controller-activity.pdf', 2, 'Controller activity: retained raw-gradient factors for global clipping and fractions of targeted units capped for q25/q50/q75. Clipping changes the optimizer input; the cap changes native stored increments while retaining advanced moments. Values are placed at completed-update endpoints.'),
]
settings = [r'LR $\times.75$', r'LR $\times.5$', r'LR $\times.1$', r'$\beta_2=.95$', r'$\beta_2=.9$', r'$\beta_2=.5$', 'Clipping .5', 'Clipping 1', 'Cap q25', 'Cap q50', 'Cap q75']
for page, setting in enumerate(settings, 1):
    plate_specs.append(('20-matched-all-setting-stage-heatmaps.pdf', page,
                       setting + ' versus unchanged Adam over the complete matched windows. Rows retain full loss, baseline/treatment change cosine, and baseline/treatment RMS. Each stage uses the same baseline RMS denominator for both arms. Color limits include every value and may differ across plates. Dashed lines are descriptive baseline peak references.'))
plates = []
for filename, page, caption in plate_specs:
    plates += [r'\clearpage', r'\begin{figure}[H]', r'\centering',
               rf'\includegraphics[page={page},width=7.15in]{{\vcassetroot/figures/{filename}}}',
               r'\caption{' + caption + '}', r'\end{figure}']
write(DRAFT / 'tables/intervention-plates.tex', '\n'.join(plates))
for name in ['matched-outcomes', 'matched-costs']:
    text = (EVIDENCE / f'05-adam/c-intervention/tables/{name}.tex').read_text(encoding='utf-8')
    text = text.replace(r'\begin{table*}[t]', r'\begin{table}[H]').replace(r'\end{table*}', r'\end{table}')
    write(DRAFT / f'tables/supp-{name}.tex', text)

manifest = {
    'sources': {p.relative_to(BASE).as_posix(): sha(p) for p in snapshots + [audit_path, EVIDENCE / '04-gd-sgd/data/56-endpoints.csv']},
    'gd_rows': 9,
    'all_pair_positive': sum(a['final_all_pair'] > 0 for a in audit['runs']),
    'adjacent_negative': sum(a['final_lag_one'] < 0 for a in audit['runs']),
    'generated': {p.relative_to(BASE).as_posix(): sha(p) for p in [DRAFT / 'references.bib', DRAFT / 'tables/gd-audit.tex']},
}
write(HERE / 'preparation.json', json.dumps(manifest, indent=2))
print(json.dumps({'gd_rows': 9, 'protected_files': len(json.loads(preservation_path.read_text(encoding='utf-8')))}))
