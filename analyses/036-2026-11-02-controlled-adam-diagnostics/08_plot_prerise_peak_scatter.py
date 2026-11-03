"""Plot pre-rise raw-gradient projected change against recovered spike magnitude."""
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'spike_anatomy/events_gt10.csv'
PROVENANCE = HERE / 'spike_anatomy/manifest.json'
PDF = HERE / 'figures/adam_prerise_eta_peak_scatter.pdf'
MANIFEST = HERE / 'prerise_peak_scatter_manifest.json'
TASKS = ['mlp', 'resnet', 'transformer']
TITLES = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
COLORS = ['#0072B2', '#D55E00', '#009E73']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    provenance = json.loads(PROVENANCE.read_text())
    assert provenance['status'] == 'passed'
    assert sha(SOURCE) == provenance['outputs'][SOURCE.relative_to(ROOT).as_posix()]
    source = pd.read_csv(SOURCE, float_precision='round_trip')
    assert len(source) == 129 and source.recovery_state.notna().all()
    assert source.peak_change_percent.gt(10).all() and source.spike_id.is_unique
    plotted = source.dropna(subset=['eta_kappa_hat_raw_before', 'peak_change_percent'])
    omitted = source.loc[~source.spike_id.isin(plotted.spike_id)]
    assert len(plotted) == 127 and len(omitted) == 2 and omitted.before_state.eq(0).all()
    np.testing.assert_allclose(plotted.peak_change_percent,
        100 * (plotted.loss_peak / plotted.loss_before - 1), rtol=1e-12)
    assert np.isfinite(plotted[['eta_kappa_hat_raw_before', 'peak_change_percent']]).all().all()
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), sharex=True, sharey=True)
    groups = []
    for col, (task, ax) in enumerate(zip(TASKS, axes)):
        rates = [.0001, .0003, .001] if task == 'mlp' else [.0003, .001, .003]
        for lr, color in zip(rates, COLORS):
            group = plotted.loc[plotted.task.eq(task) & plotted.learning_rate.eq(lr)]
            points = group[['eta_kappa_hat_raw_before', 'peak_change_percent']].to_numpy()
            artist = ax.scatter(points[:, 0], points[:, 1], s=17, color=color,
                alpha=.8, linewidths=.25, edgecolors='white', label=rf'$\eta={lr:g}$')
            np.testing.assert_array_equal(np.asarray(artist.get_offsets()), points)
            groups.append(dict(task=task, learning_rate=lr, count=len(group), color=color,
                spike_ids=group.spike_id.tolist(), coordinates=points.tolist()))
        ax.set_title(f'({chr(97 + col)}) ' + TITLES[col], pad=26)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=3,
            frameon=False, fontsize=7.6, handlelength=1., columnspacing=.5,
            handletextpad=.25, borderaxespad=0, scatterpoints=1)
        ax.set(xlim=(.6, 3.1), ylim=(0, 920), xticks=[1, 2, 3],
            yticks=[0, 200, 400, 600, 800], xlabel=r'Pre-rise $\eta\widehat{\kappa}$ (raw)')
        ax.grid(alpha=.16, lw=.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=.6)
        assert all(.6 < x < 3.1 and 0 < y < 920
            for group in groups if group['task'] == task for x, y in group['coordinates'])
    axes[0].set_ylabel('Peak increase (%)')
    fig.subplots_adjust(left=.08, right=.975, bottom=.2, top=.745, wspace=.31)
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert bounds.x0 >= 0 and bounds.y0 >= 0
    assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    assert sum(group['count'] for group in groups) == 127
    fig.savefig(PDF, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    with fitz.open(PDF) as doc:
        assert len(doc) == 1 and not doc[0].get_images()
        for font in doc[0].get_fonts(full=True):
            assert doc.extract_font(font[0])[3]
        for word in doc[0].get_text('words'):
            assert 0 <= word[0] < word[2] <= doc[0].rect.width
            assert 0 <= word[1] < word[3] <= doc[0].rect.height
    record = dict(generator=Path(__file__).relative_to(ROOT).as_posix(), generator_sha256=sha(__file__),
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in [SOURCE, PROVENANCE]},
        output=PDF.relative_to(ROOT).as_posix(), output_sha256=sha(PDF),
        cohort='Peak increase >10% and recovery observed by update 1024; current anatomy sample.',
        selected_events=129, plotted_events=127, omitted_undefined_prerise_ids=omitted.spike_id.tolist(),
        x='eta_kappa_hat_raw_before: one measurement at state a, before first loss rise; paper index a+1.',
        y='100*(loss_peak/loss_before-1), in percent.',
        reduction='One unsmoothed point per event; no averaging, fitting or jitter; shared linear axes.',
        palette='Established increasing-LR colors within each model; per-panel legends give numeric LR, including empty groups.',
        groups=groups, checks=dict(source_hash=True, selection=True, magnitude_identity=True,
            exact_scatter_coordinates=True, no_clipped_points=True, embedded_fonts=True, pdf_text_inside_page=True),
        visual_review='pending')
    MANIFEST.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Verified 127 event points from 129 complete >10% events; two undefined initialization diagnostics omitted.')


if __name__ == '__main__':
    main()
