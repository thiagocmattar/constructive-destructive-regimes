"""Overlay normalized peak-increase histograms with common logarithmic bins."""
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
SOURCE = HERE / 'spike_dataset/loss_spikes.csv'
PROVENANCE = HERE / 'spike_dataset/manifest.json'
DENSITY = HERE / 'peak_density_manifest.json'
PDF = HERE / 'figures/adam_peak_increase_histogram.pdf'
MANIFEST = HERE / 'peak_histogram_manifest.json'
TASKS = ['mlp', 'resnet', 'transformer']
TITLES = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
COLORS = ['#0072B2', '#D55E00', '#009E73']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    extraction = json.loads(PROVENANCE.read_text())
    assert extraction['verification']['status'] == 'passed'
    assert sha(SOURCE) == extraction['outputs'][SOURCE.relative_to(HERE).as_posix()]
    density = json.loads(DENSITY.read_text())
    assert density['sources'][SOURCE.relative_to(ROOT).as_posix()] == sha(SOURCE)
    original = pd.read_csv(SOURCE, float_precision='round_trip')
    df = original.loc[~original.right_censored].copy()
    assert len(original) == 812 and len(df) == 806 and df.spike_id.is_unique
    assert df.peak_change_percent.gt(0).all() and not df.peak_is_provisional.any()
    np.testing.assert_allclose(df.peak_change_percent,
        100 * (df.loss_peak / df.loss_before - 1), rtol=1e-10, atol=1e-12)
    # Match the density view's limits, using common half-decade bins everywhere.
    lower, upper = density['log10_percent_limits']
    log_edges = np.arange(lower, upper + .25, .5)
    edges = 10. ** log_edges
    assert log_edges[0] == lower and log_edges[-1] == upper
    assert edges[0] < df.peak_change_percent.min() < df.peak_change_percent.max() < edges[-1]
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), sharex=True, sharey=True)
    groups, largest = [], 0.
    for col, (task, ax) in enumerate(zip(TASKS, axes)):
        rates = sorted(df.loc[df.task.eq(task), 'learning_rate'].unique())
        assert len(rates) == 3
        for lr, color in zip(rates, COLORS):
            group = df.loc[df.task.eq(task) & df.learning_rate.eq(lr)]
            values = group.peak_change_percent.to_numpy()
            counts, _ = np.histogram(values, bins=edges)
            reference = np.array([sum(left <= value < right for value in values)
                for left, right in zip(edges[:-1], edges[1:])])
            np.testing.assert_array_equal(counts, reference)
            assert counts.sum() == len(group)
            percentages = 100. * counts / len(group)
            np.testing.assert_allclose(percentages.sum(), 100.)
            previous = next(c for c in density['curves'] if c['task'] == task and c['lr'] == lr)
            assert set(group.spike_id) == set(previous['sample_spike_ids'])
            ax.stairs(percentages, edges, color=color, fill=True, alpha=.10, linewidth=0)
            artist = ax.stairs(percentages, edges, color=color, lw=1.15, label=rf'$\eta={lr:g}$')
            np.testing.assert_array_equal(artist.get_data().values, percentages)
            np.testing.assert_array_equal(artist.get_data().edges, edges)
            largest = max(largest, float(percentages.max()))
            groups.append(dict(task=task, learning_rate=float(lr), n=len(group),
                counts=counts.tolist(), percentages=percentages.tolist(), color=color,
                spike_ids=group.spike_id.tolist()))
        ax.set_title(f'({chr(97 + col)}) ' + TITLES[col], pad=26)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=3,
            frameon=False, fontsize=7.6, handlelength=1., columnspacing=.5,
            handletextpad=.25, borderaxespad=0)
        ax.set_xscale('log')
        ax.set(xlim=(edges[0], edges[-1]), xticks=[.001, .1, 1, 10, 1000], xlabel='Peak increase (%)')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.16, lw=.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=.6)
    axes[0].set_ylabel('Spikes per bin (%)')
    axes[0].set_ylim(0, largest * 1.08)
    fig.subplots_adjust(left=.095, right=.975, bottom=.185, top=.745, wspace=.31)
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert bounds.x0 >= 0 and bounds.y0 >= 0
    assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    assert len(groups) == 9 and sum(g['n'] for g in groups) == 806
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
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in [SOURCE, PROVENANCE, DENSITY]},
        output=PDF.relative_to(ROOT).as_posix(), output_sha256=sha(PDF),
        cohort='Same 806 recovered excursions as density figure; six unrecovered excluded; no amplitude/startup cutoff.',
        estimator='Overlaid step histograms, common half-decade bins in log10(peak percentage increase); no kernel smoothing.',
        normalization='Within each task/LR: 100*bin_count/group_n; displayed bin heights sum to 100 percent. This is probability mass per bin, not density per linear percent or per log decade.',
        log10_percent_edges=log_edges.tolist(), percent_edges=edges.tolist(), groups=groups,
        checks=dict(source_hash=True, identical_density_cohort=True, magnitude_identity=True,
            independent_bin_counts=True, all_events_binned=True, per_lr_total_100_percent=True,
            artist_heights_and_edges=True, common_axes_and_bins=True, embedded_fonts=True, pdf_text_inside_page=True),
        visual_review='pending')
    MANIFEST.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Verified nine normalized histograms, all 806 events, common half-decade bins; original density unchanged.')


if __name__ == '__main__':
    main()
