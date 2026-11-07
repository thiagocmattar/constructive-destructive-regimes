"""Preserve manuscript spike windows; add moment and adaptive cross-geometry."""
import csv
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
SOURCE = ROOT / 'analyses/036-2026-11-02-controlled-adam-diagnostics'
NAME = '08-adam-spike-optimizer-zoom-cifar10-mlp-resnet-wikitext2-transformer.pdf'
BLUE, ORANGE, GREEN, PURPLE = '#0072B2', '#D55E00', '#009E73', '#7B3294'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    scratch = ROOT / 'tmp/pdfs/analysis058'
    scratch.mkdir(parents=True, exist_ok=True)
    sources = {}
    def read(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    prior = read(SOURCE / 'archive/paper_evidence/figure_manifest.json')
    full = read(SOURCE / 'archive/full_unsmoothed_manifest.json')
    examples = read(SOURCE / 'archive/peak_temporal_results.json')['selected_examples']
    results = read(ROOT / 'runs/215-2026-11-02-controlled-full-batch-adam/results.json')
    manuscript = ROOT / 'manuscript/draft/figures/analysis036/adam_manuscript_spikes.pdf'
    assert sha(manuscript) == sha(SOURCE / 'figures/adam_manuscript_spikes.pdf')
    for p in [manuscript, SOURCE / '01_plot_spikes.py', SOURCE / 'figures/adam_manuscript_spikes.pdf']:
        sources[p.relative_to(ROOT).as_posix()] = sha(p)
    keys = ['loss', 'projected', 'raw_cosine', 'moment_cosine', 'gradient', 'momentum',
            'gradient_current_moment_cosine', 'moment_over_gradient_norm', 'unbiased_moment_over_gradient_norm',
            'gradient_adaptive_cosine']
    old_rows = ['training_loss_log', 'eta_kappa_hat_raw', 'rho_raw', 'rho_momentum',
                'raw_gradient_norm_log', 'momentum_norm_log']
    row_indices = [0, 1, 2, 2, 3, 3, 4, 5, 5, 6]
    titles = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42,
        'path.simplify': False})
    fig, axes = plt.subplots(7, 3, figsize=(7.16, 9.1), sharex='col')
    curves, plotted, ratio_differences = [], [], {}
    for col, example in enumerate(examples):
        task, lr = example['task'], example['lr']
        states = np.arange(example['start_state'], example['end_state'] + 1)
        condition = next(c for c in results['conditions'] if c['task'] == task and c['lr'] == lr)
        path = ROOT / condition['source'] / 'metrics.jsonl'
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        assert sha(path) == prior['sources'][path.relative_to(ROOT).as_posix()]
        metrics = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r['gradient_state'] for r in metrics] == list(range(1024))
        assert [r['update'] for r in metrics] == list(range(1, 1025))
        for index, (key, row) in enumerate(zip(keys, row_indices)):
            if index < 6:
                source = next(c for c in full['curves'] if c['task'] == task and c['lr'] == lr and c['row'] == old_rows[index])
                source_x = np.asarray(source['x']) - (key != 'loss')
                mask = (source_x >= states[0]) & (source_x <= states[-1])
                np.testing.assert_array_equal(source_x[mask], states)
                assert source['smoothing_window'] is None
                values = np.asarray(source['y'])[mask]
                if row == 3:
                    values = values / values[0]
                if key != 'moment_cosine':
                    old = next(c for c in prior['curves'] if c['figure'] == 'spikes' and c['task'] == task and c['key'] == key)
                    np.testing.assert_array_equal(states, old['x'])
                    np.testing.assert_array_equal(values, old['y'])
                else:
                    np.testing.assert_array_equal(values, [metrics[s]['streams']['moment']['cosine'] for s in states])
            else:
                values = np.asarray([metrics[s][key] for s in states])
                assert np.isfinite(values).all()
                if row in [4, 6]:
                    assert np.all(np.abs(values) <= 1)
                else:
                    numerator = [metrics[s]['streams']['moment']['norm'] if index == 7 else metrics[s]['unbiased_moment_norm'] for s in states]
                    np.testing.assert_array_equal(values, np.asarray(numerator) / [metrics[s]['streams']['raw']['norm'] for s in states])
            momentum = key in ['momentum', 'moment_cosine']
            color = ORANGE if momentum else BLUE if key in ['gradient', 'raw_cosine'] else GREEN
            if row == 5:
                color = ORANGE if index == 7 else PURPLE
            linestyle = '--' if momentum or index == 8 else '-'
            line, = axes[row, col].plot(states, values, color=color, lw=.85, ls=linestyle,
                marker='o' if index != 8 else None, ms=1.7, markeredgewidth=0)
            np.testing.assert_array_equal(line.get_xdata(), states)
            np.testing.assert_array_equal(line.get_ydata(), values)
            curves.append(dict(task=task, lr=lr, key=key, row=row, source=condition['source'],
                               x=states.tolist(), y=values.tolist()))
            plotted.extend(dict(task=task, lr=lr, metric=key, state=int(s), value=float(v)) for s, v in zip(states, values))
        ratio_differences[task] = max(abs(metrics[s]['moment_over_gradient_norm'] - metrics[s]['unbiased_moment_over_gradient_norm']) for s in states)
        for row in range(7):
            ax = axes[row, col]
            if row:
                ax.text(0, 1.035, f'({chr(97 + row * 3 + col)})', transform=ax.transAxes, va='bottom', fontsize=9)
            else:
                ax.set_title(f'({chr(97 + col)}) {titles[col]}\n' + rf'$\eta={lr:g}$', pad=8)
            ax.axvspan(example['episode_onset'], example['peak_state'], color='.5', alpha=.12, linewidth=0, zorder=-1)
            ax.axvline(example['episode_onset'], color='.65', lw=.7, ls=':', zorder=0)
            ax.axvline(example['peak_state'], color='.4', lw=.7, ls='--', zorder=0)
            ax.set_xlim(states[0] - .5, states[-1] + .5)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.grid(alpha=.16, lw=.5)
            ax.tick_params(length=2.5, width=.6)
            ax.margins(y=.1)
            if row in [2, 4, 6]:
                ax.set(ylim=(-1.06, 1.06), yticks=[-1, 0, 1])
                ax.axhline(0, color='.45', ls=(0, (4, 3)), lw=.7)
            if row == 1:
                ax.axhline(2, color='.45', ls=(0, (4, 3)), lw=.7)
            if row in [0, 3]:
                ax.set_yscale('log')
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
            if row == 5:
                ax.set_ylim(bottom=0)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axes[0, col].axhline(example['starting_loss'], color='.65', lw=.55, ls=':')
        axes[0, col].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=4))
        axes[0, col].yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
        axes[-1, col].set_xlabel(r'Parameter state $t$')
    for ax, label in zip(axes[:, 0], ['Training loss', r'$\eta\widehat\kappa_t$', r'$\rho_t$',
                                     'Relative norm', r'$\cos(g_t,m_t)$', 'Momentum / gradient',
                                     r'$\cos(g_t,d_t)$']):
        ax.set_ylabel(label)
    legend = fig.legend(handles=[Line2D([], [], color=BLUE, label='Gradient'),
        Line2D([], [], color=ORANGE, ls='--', label='Momentum'),
        Line2D([], [], color=ORANGE, label=r'$\|m_t\|/\|g_t\|$'),
        Line2D([], [], color=PURPLE, ls='--', label=r'$\|\hat m_t\|/\|g_t\|$')],
        frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(.54, .008), ncol=4, handlelength=1.5)
    fig.subplots_adjust(left=.105, right=.975, top=.935, bottom=.085, hspace=.48, wspace=.31)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert legend.get_window_extent(renderer).y1 < min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in axes[-1])
    bbox = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    assert bbox.x0 >= 0 and bbox.y0 >= 0 and bbox.x1 <= fig.bbox.width and bbox.y1 <= fig.bbox.height
    assert len(curves) == 30 and len(plotted) == 1760
    assert sum(len(ax.patches) for ax in axes.flat) == 21
    pdf = HERE / 'figures' / NAME
    fig.savefig(pdf, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    with fitz.open(pdf) as doc:
        assert len(doc) == 1 and not doc[0].get_images()
        for word in doc[0].get_text('words'):
            assert doc[0].rect.contains(fitz.Rect(word[:4]))
        for font in doc[0].get_fonts(full=True):
            assert font[2] != 'Type3' and doc.extract_font(font[0])[3]
        doc[0].get_pixmap(matrix=fitz.Matrix(2, 2)).save(scratch / NAME.replace('.pdf', '.png'))
    table = HERE / 'spike_optimizer_zoom_values.csv'
    with table.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plotted[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(plotted)
    with table.open(newline='', encoding='utf-8') as stream:
        reread = list(csv.DictReader(stream))
    assert len(reread) == len(plotted)
    assert all(float(r['value']) == p['value'] and int(r['state']) == p['state'] for r, p in zip(reread, plotted))
    manifest = dict(generator=Path(__file__).relative_to(ROOT).as_posix(), generator_sha256=sha(__file__),
        sources=sources, outputs={p.relative_to(HERE).as_posix(): sha(p) for p in [pdf, table]},
        examples=examples, curves=curves, smoothing=None, checked_values=len(plotted),
        original_values_preserved=1056, added_values=704, maximum_ratio_difference=ratio_differences,
        checks='Source hashes, original curve equality, update/state alignment, ratio identities, artist and CSV read-back, PDF bounds/fonts/vector checks passed.',
        visual_review='See observations/008-spike-optimizer-zoom.md')
    (HERE / 'spike_optimizer_zoom_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'PASS: 30 curves / 1,760 values, 1,056 original values preserved. Ratio differences: {ratio_differences}')


if __name__ == '__main__':
    main()
