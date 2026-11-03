"""Overlay per-LR densities of positive peak percentage increase in log space."""
import argparse
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
from scipy.special import ndtr
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'spike_dataset/loss_spikes.csv'
PDF = HERE / 'figures/adam_peak_increase_density.pdf'
MANIFEST = HERE / 'peak_density_manifest.json'
TASKS = ['mlp', 'resnet', 'transformer']
TITLES = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
COLORS = ['#0072B2', '#D55E00', '#009E73']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def density(grid, samples, bandwidth):
    standardized = (grid[:, None] - samples) / bandwidth
    return np.exp(-.5 * standardized ** 2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-reviewed', action='store_true')
    args = parser.parse_args()
    if args.accept_reviewed:
        record = json.loads(MANIFEST.read_text())
        assert record['generator_sha256'] == sha(__file__)
        assert record['output_sha256'] == sha(PDF)
        for relative, digest in record['sources'].items():
            assert sha(ROOT / relative) == digest
        record['visual_review'] = 'passed: Poppler render reviewed; three panels with nine density curves, readable log-percent ticks and intact legends, no clipping'
        MANIFEST.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return
    extraction_path = HERE / 'spike_dataset/manifest.json'
    extraction = json.loads(extraction_path.read_text())
    assert extraction['verification']['status'] == 'passed'
    assert sha(SOURCE) == extraction['outputs'][SOURCE.relative_to(HERE).as_posix()]
    original = pd.read_csv(SOURCE)
    df = original.loc[~original.right_censored].copy()
    assert len(original) == 812 and len(df) == 806
    assert df.peak_change_percent.gt(0).all() and not df.peak_is_provisional.any()
    np.testing.assert_allclose(df.peak_change_percent,
        100 * (df.loss_peak - df.loss_before) / df.loss_before, rtol=1e-10, atol=1e-12)
    df['log_peak_percent'] = np.log10(df.peak_change_percent)
    bandwidths = {}
    lower, upper = [], []
    for task in TASKS:
        values = df.loc[df.task.eq(task), 'log_peak_percent'].to_numpy()
        bandwidths[task] = float(values.std(ddof=1) * len(values) ** (-1 / 5))
        assert bandwidths[task] > 0
        lower.append(values.min() - 3 * bandwidths[task])
        upper.append(values.max() + 3 * bandwidths[task])
    grid = np.linspace(np.floor(min(lower)), np.ceil(max(upper)), 4096)
    x = np.power(10., grid)
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42,
        'path.simplify': False})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), sharex=True, sharey=True)
    curves, largest = [], 0.
    for col, (task, ax) in enumerate(zip(TASKS, axes)):
        rates = sorted(df.loc[df.task.eq(task), 'learning_rate'].unique())
        assert len(rates) == 3
        for lr, color in zip(rates, COLORS):
            group = df.loc[df.task.eq(task) & df.learning_rate.eq(lr)]
            values = group.log_peak_percent.to_numpy()
            bandwidth = bandwidths[task]
            y = density(grid, values, bandwidth)
            reference = np.mean(norm.pdf(grid[:, None], loc=values, scale=bandwidth), axis=1)
            np.testing.assert_allclose(y, reference, rtol=5e-13, atol=1e-15)
            # Each Gaussian mixture has unit mass on the full log-percent axis.
            exact_visible_mass = float(np.mean(ndtr((grid[-1] - values) / bandwidth)
                - ndtr((grid[0] - values) / bandwidth)))
            numeric_visible_mass = float(np.trapezoid(y, grid))
            np.testing.assert_allclose(numeric_visible_mass, exact_visible_mass, rtol=1e-8, atol=1e-10)
            assert exact_visible_mass > .9999 and np.isfinite(y).all() and (y >= 0).all()
            assert grid[0] < values.min() <= values.max() < grid[-1]
            artist, = ax.plot(x, y, color=color, lw=1.15, label=rf'$\eta={lr:g}$')
            np.testing.assert_array_equal(artist.get_xdata(), x)
            np.testing.assert_array_equal(artist.get_ydata(), y)
            largest = max(largest, float(y.max()))
            curves.append(dict(task=task, lr=float(lr), recovered_events=len(group),
                excluded_censored=int((original.task.eq(task) & original.learning_rate.eq(lr) & original.right_censored).sum()),
                bandwidth_log10_percent=bandwidth, visible_probability=exact_visible_mass,
                numeric_visible_probability=numeric_visible_mass,
                sample_spike_ids=group.spike_id.tolist(), sample_percent_increases=group.peak_change_percent.tolist()))
        ax.set_title(f'({chr(97 + col)}) ' + TITLES[col], pad=26)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=3,
            frameon=False, fontsize=7.6, handlelength=1., columnspacing=.5,
            handletextpad=.25, borderaxespad=0)
        ax.set_xscale('log')
        ax.set(xlim=(x[0], x[-1]), xticks=[.001, .1, 1, 10, 1000], xlabel='Peak increase (%)')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.16, lw=.5)
        ax.tick_params(length=2.5, width=.6)
    axes[0].set_ylabel('Density (per log decade)')
    axes[0].set_ylim(0, largest * 1.06)
    fig.subplots_adjust(left=.095, right=.975, bottom=.185, top=.745, wspace=.31)
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert bounds.x0 >= -1 and bounds.y0 >= -1
    assert bounds.x1 <= fig.bbox.width + 1 and bounds.y1 <= fig.bbox.height + 1
    assert len(curves) == 9 and sum(c['recovered_events'] for c in curves) == 806
    assert sum(c['excluded_censored'] for c in curves) == 6
    assert len(fig.axes) == 3 and sum(len(ax.lines) for ax in axes) == 9
    fig.savefig(PDF, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    with fitz.open(PDF) as document:
        assert len(document) == 1 and not document[0].get_images()
        for font in document[0].get_fonts(full=True):
            assert document.extract_font(font[0])[3]
        for word in document[0].get_text('words'):
            assert 0 <= word[0] < word[2] <= document[0].rect.width + .01
            assert 0 <= word[1] < word[3] <= document[0].rect.height + .01
    record = dict(generator=Path(__file__).relative_to(ROOT).as_posix(), generator_sha256=sha(__file__),
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in [SOURCE, extraction_path]},
        output=PDF.relative_to(ROOT).as_posix(), output_sha256=sha(PDF),
        cohort='806 recovered excursions; six censored excursions excluded; no amplitude/startup exclusions.',
        estimator='Gaussian KDE of log10(peak increase in percent); equal event weights within each LR.',
        bandwidth='Pooled sample std of log10(percent) times pooled N^(-1/5), common to all LRs within each model.',
        bandwidths=bandwidths, log10_percent_limits=[float(grid[0]), float(grid[-1])], grid_points=len(grid),
        normalization='Each LR density integrates to one in log10-percent coordinates; y is density per log decade.',
        curves=curves, checks=dict(all_events_preserved=True, no_forced_lr_ordering=True,
            independent_kernel_evaluation=True, unit_mixture_mass=True, common_axes=True,
            all_visible_masses_above=.9999, pdf_text_inside_page=True), visual_review='pending')
    MANIFEST.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Verified nine separately normalized densities from 806 recovered excursions; common bandwidth per model and common axes.')


if __name__ == '__main__':
    main()
