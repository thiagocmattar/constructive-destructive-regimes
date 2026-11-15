"""Publish t-clock figure variants without changing any retained measurements.

Reuse the frozen generators; redirect every write into this work folder or
draft/figures. Text is translated immediately before vector-PDF export.
Numerical Matplotlib artists are hashed before and after relabeling. The
GD/overview and tracing arrays are also compared to their retained arrays.
"""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
OUT = BASE / 'draft/figures'
SCRATCH = HERE / 'scratch/figures'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
RECORD = {'clock': 'Training step t = completed optimizer updates',
          'changes': 'Labels only; no training, reductions, windows or numerical curves changed.',
          'sources': {}, 'exports': [], 'array_checks': []}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source(p):
    p = Path(p)
    RECORD['sources'][p.relative_to(ROOT).as_posix()] = sha(p)
    return p


def load(p, name):
    spec = importlib.util.spec_from_file_location(name, source(p))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arrays(fig):
    values = []
    for ax in fig.axes:
        values.extend([np.array(ax.get_xlim()), np.array(ax.get_ylim())])
        for line in ax.lines:
            values.extend([line.get_xdata(orig=False), line.get_ydata(orig=False)])
        for im in ax.images:
            values.append(np.ma.filled(im.get_array(), np.nan))
        for coll in ax.collections:
            values.append(np.ma.filled(coll.get_offsets(), np.nan))
            if coll.get_array() is not None:
                values.append(np.ma.filled(coll.get_array(), np.nan))
    h = hashlib.sha256()
    count = 0
    for value in values:
        a = np.asarray(value, dtype='<f8')
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
        count += a.size
    return {'numerical_artist_sha256': h.hexdigest(), 'values': count}


def translate(text, name):
    # s appears as a standalone mathematical index, not in English words.
    text = re.sub(r"(?<![A-Za-z'])s(?![A-Za-z])", 't', text)
    text = text.replace('Fixed probe', 'Reference examples')
    text = text.replace(r'First observed output $\rho<0$', 'First output reversal')
    text = text.replace(r'$Q\geq2$', r'$\eta\widehat\kappa\geq2$')
    text = text.replace('Q alarms:', 'Threshold alarms:')
    text = text.replace('Alarms D/G/DG/Q:', 'Alarms D/G/DG/threshold:')
    if name == '00-three-model-loss-representation-motion.pdf':
        text = text.replace('Displacement RMS\n/ stage maximum',
                            'Relative RMS\n' + r'$a_t^\ell/\max_u a_u^\ell$')
    if name == '01-three-model-temporal-structure.pdf':
        translations = {
            'Number of modes': 'Number of patterns $k$',
            'Cumulative energy': 'Squared movement\nfraction $E_k$',
            'Top-1 energy': r'First pattern $E_1$',
            'Top-10 energy': r'First ten $E_{10}$',
            'Output overlap': r'Temporal overlap $O_{10}$',
            'Energy fraction / overlap': 'Movement fraction / overlap',
            r'Output $\alpha_1/\sqrt{P}$': r'Output $\alpha_{t,1}/\sqrt{P}$',
        }
        text = translations.get(text, text)
    return text


native_save = Figure.savefig


def save(fig, filename, *args, **kwargs):
    path = Path(getattr(filename, '_filename', str(filename)))
    before = arrays(fig)
    labels = []
    for artist in fig.findobj(match=Text):
        old = artist.get_text()
        new = translate(old, path.name)
        if old != new:
            artist.set_text(new)
            labels.append({'before': old, 'after': new})
    assert arrays(fig) == before, path.name
    result = native_save(fig, filename, *args, **kwargs)
    if path.suffix.lower() == '.pdf':
        RECORD['exports'].append({'file': path.name, **before, 'labels': labels})
    return result


Figure.savefig = save


def compare_arrays(new, old, tag, subset=False):
    oldpath = source(old)
    old = np.load(oldpath)
    new = np.load(new) if isinstance(new, Path) else new
    keys = sorted(new.keys())
    if not subset:
        assert keys == sorted(old.keys()), tag
    for key in keys:
        np.testing.assert_array_equal(new[key], old[key], err_msg=tag + ':' + key)
    RECORD['array_checks'].append({'figure_group': tag, 'arrays': len(keys),
                                  'values': sum(np.size(new[k]) for k in keys),
                                  'reference': oldpath.relative_to(ROOT).as_posix()})


with contextlib.redirect_stdout(io.StringIO()):
    # Four original GD/SGD figures and the three-model Adam overview.
    old = BASE / 'work/figure_revision_20261114'
    m = load(old / '01_figures.py', 'core_views')
    m.OUT = OUT
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 9,
                        'axes.titlesize': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
                        'pdf.fonttype': 42, 'path.simplify': False})
    m.gd_figures()
    m.adam_overview()
    compare_arrays(m.PLOTTED, old / 'plotted_arrays.npz', 'GD-SGD and overview', subset=True)
    RECORD['sources'].update(m.AUDIT['sources'])

    # Both tracing figures retain the current ResNet 390--450 event.
    for name in ['resnet_peak_revision_20261114', 'resnet_temporal_revision_20261114']:
        old = BASE / 'work' / name
        m = load(old / '01_figure.py', name)
        m.HERE = SCRATCH / name
        m.HERE.mkdir(exist_ok=True)
        m.main()
        compare_arrays(m.HERE / 'plotted_arrays.npz', old / 'plotted_arrays.npz', name)
        for p in (m.HERE / 'figures').glob('*.pdf'):
            shutil.copyfile(p, OUT / p.name)
        RECORD['sources'].update(json.loads((m.HERE / 'figure_manifest.json').read_text())['sources'])

    # Main intervention plots: original point positions, scales and heatmaps.
    m = load(BASE / 'work/intervention_manuscript_20261114/01_prepare.py', 'main_interventions')
    m.HERE = SCRATCH / 'interventions'
    m.FIGURES = OUT
    m.TABLES = m.HERE / 'tables'
    m.main()
    new = json.loads((m.HERE / 'data/presentation_manifest.json').read_text())
    old = json.loads(source(BASE / 'evidence/05-adam/c-intervention/data/main-presentation_manifest.json').read_text())
    for n, o in zip(new['figures'], old['figures']):
        assert {k: v for k, v in n.items() if k != 'sha256'} == {k: v for k, v in o.items() if k != 'sha256'}
    assert new['table']['rows'] == old['table']['rows']
    RECORD['array_checks'].append({'figure_group': 'Main intervention plots/table', 'exact_manifest_match': True})

    # Supplemental intervention panels, from the frozen consolidated reductions.
    local_input = SCRATCH / 'matched_input'
    local_input.mkdir(exist_ok=True)
    for filename in ['results.json', 'curves.npz', 'heldout.csv', 'events.csv', 'costs.csv']:
        shutil.copyfile(source(BASE / 'evidence/05-adam/c-intervention/data' / ('matched-' + filename)),
                        local_input / filename)
    m = load(ROOT / 'analyses/065-2026-11-07-mlp-frozen-lr-double/61_plot_matched.py', 'matched_interventions')
    f = m.Figures(local_input, OUT, False)
    f.loss()
    f.heldout()
    f.activity()
    with PdfPages(OUT / '20-matched-all-setting-stage-heatmaps.pdf') as pdf:
        for arm in [v['id'] for v in f.report['cases'][0]['config']['arms'] if v['id'] != 'control']:
            fig, _ = f.heatmap(arm)
            fig.suptitle(m.DISPLAY[arm] + ' versus unchanged Adam', fontsize=9, y=.985)
            pdf.savefig(fig)
            plt.close(fig)

    # Six supplemental timing/severity audit pages, using retained reductions.
    m = load(ROOT / 'analyses/055-2026-11-05-adam-directional-amplification/07_plot_forward_and_severity.py', 'adam_audit')
    for filename in ['severity.csv', 'horizons.csv', 'warning_memory.csv']:
        source(m.DATA / filename)
    severity, forward = m.read_csv('severity.csv'), m.read_csv('horizons.csv')
    pages = [(severity, {'severity': True}), (forward, {'target': 'future_step'}),
             (forward, {'target': 'net_change'}), (forward, {'target': 'any_above_current'}),
             (forward, {'target': 'any_above_current', 'subset': 'not_already_rising'}),
             (m.read_csv('warning_memory.csv'), {'memory': True})]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.unicode_minus': True})
    with PdfPages(OUT / 'adam_forward_and_severity.pdf', metadata={'CreationDate': None, 'ModDate': None}) as pdf:
        for rows, options in pages:
            fig = m.figure(rows, **options)
            pdf.savefig(fig)
            plt.close(fig)

    # These three panels use batch size / local continuation index / quantile,
    # rather than s. Keep their vectors unchanged; SI defines local k=t-t0.
    for filename in ['frozen_calibration.pdf', 'sgd_proxy_validation.pdf',
                     '14-distributed-cap-quantile-ablation-mlp.pdf']:
        p = next((BASE / 'evidence').rglob(filename))
        shutil.copyfile(source(p), OUT / filename)

RECORD['outputs'] = {p.name: sha(p) for p in sorted(OUT.glob('*.pdf'))}
assert len(RECORD['outputs']) == 17
(HERE / 'figure_verification.json').write_text(json.dumps(RECORD, indent=2, ensure_ascii=False) + '\n',
                                               encoding='utf-8', newline='\n')
print(json.dumps({'figures': len(RECORD['outputs']), 'checked_exports': len(RECORD['exports']),
                  'array_checks': RECORD['array_checks']}, indent=2))
