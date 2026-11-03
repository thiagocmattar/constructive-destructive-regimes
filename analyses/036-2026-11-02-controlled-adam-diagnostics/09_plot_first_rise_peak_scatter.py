"""Plot raw-gradient projected change immediately after the first loss increase."""
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'spike_anatomy/events_gt10.csv'
PROVENANCE = HERE / 'spike_anatomy/manifest.json'
STEPS = HERE / 'spike_dataset/spike_steps.csv'
EXTRACTION = HERE / 'spike_dataset/manifest.json'
PDF = HERE / 'figures/adam_first_rise_eta_peak_scatter.pdf'
MANIFEST = HERE / 'first_rise_peak_scatter_manifest.json'
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
    extraction = json.loads(EXTRACTION.read_text())
    assert extraction['verification']['status'] == 'passed'
    assert sha(STEPS) == extraction['outputs'][STEPS.relative_to(HERE).as_posix()]
    states = pd.read_csv(STEPS, float_precision='round_trip')
    plotted = source.merge(states[['spike_id', 'state', 'eta_kappa_hat_raw', 'loss',
        'rho_raw', 'gradient_norm']], left_on=['spike_id', 'onset_state'],
        right_on=['spike_id', 'state'], how='left', validate='one_to_one')
    assert len(plotted) == 129 and plotted.eta_kappa_hat_raw.notna().all()
    assert (plotted.onset_state == plotted.before_state + 1).all()
    assert (plotted.loss > plotted.loss_before).all()
    np.testing.assert_array_equal(plotted.loss, plotted.loss_onset)
    np.testing.assert_allclose(plotted.eta_kappa_hat_raw,
        1 - plotted.rho_raw * plotted.gradient_norm / plotted.gradient_norm_before,
        rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(plotted.peak_change_percent,
        100 * (plotted.loss_peak / plotted.loss_before - 1), rtol=1e-12)
    assert np.isfinite(plotted[['eta_kappa_hat_raw', 'peak_change_percent']]).all().all()
    plotted['peak_loss_ratio'] = 1 + plotted.peak_change_percent / 100
    np.testing.assert_allclose(plotted.peak_loss_ratio, plotted.loss_peak / plotted.loss_before, rtol=1e-12)
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
            points = group[['eta_kappa_hat_raw', 'peak_loss_ratio']].to_numpy()
            artist = ax.scatter(points[:, 0], points[:, 1], s=17, color=color,
                alpha=.8, linewidths=.25, edgecolors='white', label=rf'$\eta={lr:g}$')
            np.testing.assert_array_equal(np.asarray(artist.get_offsets()), points)
            groups.append(dict(task=task, learning_rate=lr, count=len(group), color=color,
                spike_ids=group.spike_id.tolist(), coordinates=points.tolist()))
        ax.set_title(f'({chr(97 + col)}) ' + TITLES[col], pad=26)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=3,
            frameon=False, fontsize=7.6, handlelength=1., columnspacing=.5,
            handletextpad=.25, borderaxespad=0, scatterpoints=1)
        ax.set_yscale('log')
        ax.set(xlim=(0, 3.6), ylim=(1, 11), xticks=[0, 1, 2, 3],
            yticks=[1, 2, 5, 10], xlabel=r'First-rise $\eta\widehat{\kappa}$ (raw)')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.16, lw=.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=.6)
        assert all(0 < x < 3.6 and 1 < y < 11
            for group in groups if group['task'] == task for x, y in group['coordinates'])
    axes[0].set_ylabel('Peak / pre-rise loss')
    fig.subplots_adjust(left=.08, right=.975, bottom=.2, top=.745, wspace=.31)
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert bounds.x0 >= 0 and bounds.y0 >= 0
    assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    assert sum(group['count'] for group in groups) == 129
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
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in [SOURCE, PROVENANCE, STEPS, EXTRACTION]},
        output=PDF.relative_to(ROOT).as_posix(), output_sha256=sha(PDF),
        cohort='Peak increase >10% and recovery observed by update 1024; current anatomy sample.',
        selected_events=129, plotted_events=129, onset_equals_peak_events=int(plotted.onset_state.eq(plotted.peak_state).sum()),
        x='eta_kappa_hat_raw at state a+1: immediately after first loss increase; paper index a+2, gradients at states a and a+1.',
        y='1 + peak_change_percent/100 = loss_peak/loss_before; log axis, with 1 meaning no increase.',
        reduction='One unsmoothed point per event; no averaging, fitting or jitter; shared linear x and logarithmic y axes.',
        palette='Established increasing-LR colors within each model; per-panel legends give numeric LR, including empty groups.',
        groups=groups, checks=dict(source_hash=True, selection=True, magnitude_identity=True,
            ratio_identity=True, first_rise_state_and_loss=True, projected_change_identity=True, exact_scatter_coordinates=True, no_clipped_points=True, embedded_fonts=True, pdf_text_inside_page=True),
        visual_review='pending')
    MANIFEST.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Verified 129 first-rise event points; 13 first-rise states coincide with the loss peak.')


if __name__ == '__main__':
    main()
