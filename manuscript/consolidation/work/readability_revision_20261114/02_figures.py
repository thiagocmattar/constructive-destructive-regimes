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
    text = re.sub(r'\bStage ([123])\b', r'Group \1', text)
    text = re.sub(r'\bstage\b', 'site', text)
    text = text.replace(r'First observed output $\rho<0$', 'First output reversal')
    text = text.replace(r'$Q\geq2$', r'$\eta\widehat\kappa\geq2$')
    text = text.replace('Q alarms:', 'Threshold alarms:')
    text = text.replace('Alarms D/G/DG/Q:', 'Alarms D/G/DG/threshold:')
    if name == '00-three-model-loss-representation-motion.pdf':
        text = text.replace('Displacement RMS\n/ site maximum',
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
gd_loss_panel_height = None


def save(fig, filename, *args, **kwargs):
    global gd_loss_panel_height
    path = Path(getattr(filename, '_filename', str(filename)))
    if path.name == 'gd_loss_projected_change_spectral_clean.pdf':
        gd_loss_panel_height = fig.axes[0].get_position().height * fig.get_figheight()
    elif path.name == 'gd_reversal_norm_retention.pdf':
        # Keep the margins and gaps in inches while matching each axis to Figure 1.
        grid = fig.subplotpars
        height = fig.get_figheight()
        panel_height = fig.axes[0].get_position().height * height
        rows = fig.axes[0].get_gridspec().nrows
        new_height = height + rows * (gd_loss_panel_height - panel_height)
        fig.set_figheight(new_height)
        fig.subplots_adjust(top=1 - (1 - grid.top) * height / new_height,
                            bottom=grid.bottom * height / new_height,
                            hspace=grid.hspace * panel_height / gd_loss_panel_height)
    elif path.name == '00-three-model-loss-representation-motion.pdf':
        # Layout in PDF points, with the heatmaps at the GD panel height.
        row_heights = [54.17, 64.19, 64.19]
        gap, top, bottom = 17, 27, 48
        height = sum(row_heights) + 2 * gap + top + bottom
        fig.set_figheight(height / 72)
        fig.axes[0].get_gridspec().set_height_ratios(row_heights)
        fig.subplots_adjust(top=1 - top / height, bottom=bottom / height,
                            hspace=gap / np.mean(row_heights))
        for heatmap, colorbar in zip((fig.axes[5], fig.axes[8]), fig.axes[9:]):
            panel, bar = heatmap.get_position(), colorbar.get_position()
            colorbar.set_position([bar.x0, panel.y0, bar.width, panel.height])
        legend = fig.legends[0]
        handles = legend.legend_handles
        labels = [text.get_text() for text in legend.get_texts()]
        legend.remove()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.53, .005),
                   ncol=4, frameon=False, fontsize=7.5, columnspacing=1.4, handlelength=2)
    before = arrays(fig)
    labels = []
    # Tick text is regenerated by its formatter during drawing; update the
    # fixed tick labels rather than only their transient Text artists.
    for ax in fig.axes:
        old_ticks = [t.get_text() for t in ax.get_yticklabels()]
        new_ticks = [translate(t, path.name) for t in old_ticks]
        if new_ticks != old_ticks:
            ax.set_yticks(ax.get_yticks(), labels=new_ticks)
            labels.extend({'before': a, 'after': b} for a,b in zip(old_ticks,new_ticks) if a!=b)
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

    # Historical timing/severity audit remains an archived figure; it is no longer compiled.
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

# Round-two presentation: absolute outcomes and q25 motion, using the same frozen grid.
m = load(BASE / 'work/author_feedback2_20261115/04_figures.py', 'feedback2_interventions')
m.main()
m = load(BASE / 'work/appendix_20261115/01_artifacts.py', 'compact_appendix')
m.main()
RECORD['changes'] = 't/site terminology; absolute intervention axes/table; q25 illustration; compact appendix trajectories. Original data and windows preserved.'
RECORD['outputs'] = {p.name: sha(p) for p in sorted(OUT.glob('*.pdf'))}
assert len(RECORD['outputs']) == 21
(HERE / 'figure_verification.json').write_text(json.dumps(RECORD, indent=2, ensure_ascii=False) + '\n',
                                               encoding='utf-8', newline='\n')
print(json.dumps({'figures': len(RECORD['outputs']), 'checked_exports': len(RECORD['exports']),
                  'array_checks': RECORD['array_checks']}, indent=2))
