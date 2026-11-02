"""Regenerate only the retained spike figure, with gradient/momentum cosines."""
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = HERE / 'archive'
BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'
PDF = HERE / 'figures/adam_manuscript_spikes.pdf'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    prior = read(ARCHIVE / 'paper_evidence/figure_manifest.json')
    full = read(ARCHIVE / 'full_unsmoothed_manifest.json')
    examples = read(ARCHIVE / 'peak_temporal_results.json')['selected_examples']
    results = read(ROOT / 'runs/215-2026-11-02-controlled-full-batch-adam/results.json')
    keys = ['loss', 'projected', 'raw_cosine', 'moment_cosine', 'gradient', 'momentum']
    original_rows = ['training_loss_log', 'eta_kappa_hat_raw', 'rho_raw',
                     'rho_momentum', 'raw_gradient_norm_log', 'momentum_norm_log']
    titles = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42,
        'path.simplify': False})
    fig, axes = plt.subplots(4, 3, figsize=(7.16, 4.55), sharex='col')
    points = 0
    for col, example in enumerate(examples):
        task, lr = example['task'], example['lr']
        states = np.arange(example['start_state'], example['end_state'] + 1)
        condition = next(c for c in results['conditions'] if c['task'] == task and c['lr'] == lr)
        metrics_path = ROOT / condition['source'] / 'metrics.jsonl'
        assert hashlib.sha256(metrics_path.read_bytes()).hexdigest() == prior['sources'][metrics_path.relative_to(ROOT).as_posix()]
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        for key, original_row, row in zip(keys, original_rows, [0, 1, 2, 2, 3, 3]):
            source = next(c for c in full['curves'] if c['task'] == task and c['lr'] == lr and c['row'] == original_row)
            source_x = np.asarray(source['x']) - (key != 'loss')
            mask = (source_x >= states[0]) & (source_x <= states[-1])
            np.testing.assert_array_equal(source_x[mask], states)
            assert source['smoothing_window'] is None
            values = np.asarray(source['y'])[mask]
            if row == 3:
                values = values / values[0]
            if key == 'moment_cosine':
                np.testing.assert_array_equal(values, [metrics[t]['streams']['moment']['cosine'] for t in states])
                assert all(metrics[t]['gradient_state'] == t for t in states)
            else:
                old = next(c for c in prior['curves'] if c['figure'] == 'spikes' and c['task'] == task and c['key'] == key)
                np.testing.assert_array_equal(states, old['x'])
                np.testing.assert_array_equal(values, old['y'])
            momentum = key in ['momentum', 'moment_cosine']
            color = ORANGE if momentum else BLUE if key in ['gradient', 'raw_cosine'] else GREEN
            line, = axes[row, col].plot(states, values, color=color, lw=.85,
                ls='--' if momentum else '-', marker='o', ms=1.7, markeredgewidth=0)
            np.testing.assert_array_equal(line.get_ydata(), values)
            points += len(values)
        for row in range(4):
            ax = axes[row, col]
            if row:
                ax.text(0, 1.035, f'({chr(97 + row * 3 + col)})', transform=ax.transAxes,
                        va='bottom', fontsize=9)
            else:
                ax.set_title(f'({chr(97 + col)}) ' + titles[col] + '\n' + rf'$\eta={lr:g}$', pad=8)
            ax.axvline(example['episode_onset'], color='.65', lw=.7, ls=':', zorder=0)
            ax.axvline(example['peak_state'], color='.4', lw=.7, ls='--', zorder=0)
            ax.set_xlim(states[0] - .5, states[-1] + .5)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.grid(alpha=.16, lw=.5)
            ax.tick_params(length=2.5, width=.6)
            ax.margins(y=.1)
            if row == 2:
                ax.set(ylim=(-1.06, 1.06), yticks=[-1, 0, 1])
                ax.axhline(0, color='.45', ls=(0, (4, 3)), lw=.7)
            if row == 1:
                ax.axhline(2, color='.45', ls=(0, (4, 3)), lw=.7)
            if row in [0, 3]:
                ax.set_yscale('log')
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        axes[0, col].axhline(example['starting_loss'], color='.65', lw=.55, ls=':')
        axes[0, col].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=4))
        axes[0, col].yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
        axes[-1, col].set_xlabel(r'Parameter state $t$')
    for ax, label in zip(axes[:, 0], ['Training loss', r'$\eta\widehat\kappa_t$',
                                     r'$\rho_t$', 'Relative norm']):
        ax.set_ylabel(label)
    legend = fig.legend(handles=[Line2D([], [], color=BLUE, label='Gradient'),
        Line2D([], [], color=ORANGE, ls='--', label='Momentum')], frameon=False,
        fontsize=8, loc='lower center', bbox_to_anchor=(.54, .005), ncol=2,
        handlelength=1.5)
    fig.subplots_adjust(left=.105, right=.975, top=.88, bottom=.16, hspace=.48, wspace=.31)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert legend.get_window_extent(renderer).y1 < min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in axes[-1])
    assert sum(len(ax.lines) for ax in axes.flat) == 51
    fig.savefig(PDF, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    with fitz.open(PDF) as doc:
        assert len(doc) == 1 and not doc[0].get_images()
        for word in doc[0].get_text('words'):
            assert 0 <= word[0] < word[2] <= doc[0].rect.width + .01
            assert 0 <= word[1] < word[3] <= doc[0].rect.height + .01
    print(f'Verified 18 unsmoothed curves / {points} points; 15 prior curves unchanged; 3 momentum cosines match recorded moments.')


if __name__ == '__main__':
    main()
