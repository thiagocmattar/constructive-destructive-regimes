"""Extend Figure 00 with recorded first-moment and Adam-direction diagnostics."""
import importlib.util
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'analyses/055-2026-11-05-adam-directional-amplification'
NAME = '07-high-lr-optimizer-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf'
spec = importlib.util.spec_from_file_location('source_plot', SOURCE / '02_plot.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
a = original.analysis


def main():
    arms, sources = a.load_arms()
    assert sources == a.read(SOURCE / 'manifest.json')['sources']
    source_plot = a.read(SOURCE / 'plot_manifest.json')
    original_pdf = HERE / 'figures/00-high-lr-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf'
    assert a.sha(original_pdf) == source_plot['outputs']['figures/adam_full_trajectories_high.pdf']
    for p in [SOURCE / '01_analyze.py', SOURCE / '02_plot.py', SOURCE / 'plot_manifest.json', original_pdf]:
        sources[p.relative_to(ROOT).as_posix()] = a.sha(p)
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42,
        'path.simplify': False})
    fig, axes = plt.subplots(7, 3, figsize=(7.16, 9.5), sharex='col')
    curves, values = [], []
    for col, task in enumerate(a.TASKS):
        arm = next(v for v in arms if v['task'] == task and v['tier'] == 2)
        metrics = [json.loads(line) for line in (ROOT / arm['source'] / 'metrics.jsonl').read_text().splitlines()]
        moment = np.array([r['streams']['moment']['norm'] for r in metrics])
        assert np.all(moment > 0)
        assert metrics[0]['streams']['moment']['cosine'] is None
        assert metrics[0]['streams']['adaptive']['cosine'] is None
        keys = ['loss', 'Q', 'rho', 'relative_norm', 'moment_norm', 'moment_rho', 'adaptive_rho']
        ys = [arm['loss'], arm['Q'][1:], arm['rho'][1:], arm['gradient'] / arm['gradient'][0],
              moment, np.array([r['streams']['moment']['cosine'] for r in metrics[1:]]),
              np.array([r['streams']['adaptive']['cosine'] for r in metrics[1:]])]
        xs = [np.arange(1025), np.arange(1, 1024), np.arange(1, 1024), np.arange(1024),
              np.arange(1024), np.arange(1, 1024), np.arange(1, 1024)]
        for row, (key, x, y) in enumerate(zip(keys, xs, ys)):
            ax = axes[row, col]
            assert np.isfinite(y).all()
            line, = ax.plot(x, y, color=original.GREEN, lw=.65)
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
            record = dict(arm=arm['arm'], source=arm['source'], key=key, points=len(x),
                          x_sha256=original.numeric_hash(x), y_sha256=original.numeric_hash(y))
            if row < 4:
                previous = next(c for c in source_plot['curves'] if c['figure'] == 'adam_full_trajectories_high.pdf'
                                and c['arm'] == arm['arm'] and c['key'] == key)
                for field in ['points', 'x_sha256', 'y_sha256']:
                    assert record[field] == previous[field]
            curves.append(record)
            values.extend(dict(arm=arm['arm'], metric=key, state=int(s), value=float(v)) for s, v in zip(x, y))
            for event in arm['events']:
                if float(event['peak_change_percent']) > 10:
                    start, peak = int(event['before_state']), int(event['peak_state'])
                    if event['recovery_state']:
                        ax.axvspan(start, peak, color='.5', alpha=.12, linewidth=0, zorder=-1)
                    else:
                        ax.axvspan(start, peak, facecolor='none', edgecolor='.55', linestyle=':', lw=.65, zorder=-1)
            if row == 0:
                ax.set_title(f'({chr(97 + col)}) {original.TITLES[col]}\n' + rf'$\eta={arm["lr"]:g}$', pad=8)
            else:
                ax.text(0, 1.035, f'({chr(97 + row * 3 + col)})', transform=ax.transAxes, va='bottom', fontsize=9)
            ax.set(xlim=(0, 1024), xticks=[0, 256, 512, 768, 1024])
            ax.margins(y=.08)
            original.style(ax)
            if row in [0, 3, 4]:
                original.log_axis(ax)
            elif row in [2, 5, 6]:
                ax.set(ylim=(-1.06, 1.06), yticks=[-1, 0, 1])
                ax.axhline(0, color='.45', ls=(0, (4, 3)), lw=.65)
            else:
                all_q = np.concatenate([v['Q'][1:] for v in arms if v['task'] == task])
                padding = .08 * np.ptp(all_q)
                ax.set_ylim(float(all_q.min() - padding), float(all_q.max() + padding))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.axhline(2, color='.45', ls=(0, (4, 3)), lw=.65)
        axes[-1, col].set_xlabel(r'Parameter state $s$')
    labels = ['Training loss', r'$\eta\widehat\kappa$', r'$\rho$', 'Relative norm',
              r'Momentum norm $\|m\|$', r'Momentum $\rho^{(m)}$', r'Adam $\rho^{(d)}$']
    for ax, label in zip(axes[:, 0], labels):
        ax.set_ylabel(label)
    fig.legend(handles=[Patch(facecolor='.5', alpha=.12, label='Rise of recovered >10% excursion'),
                        Patch(facecolor='none', edgecolor='.55', ls=':', label='Recovery unobserved')],
               loc='lower center', bbox_to_anchor=(.54, .009), ncol=2, frameon=False,
               fontsize=7.7, handlelength=1.5, columnspacing=1)
    fig.subplots_adjust(left=.11, right=.975, top=.94, bottom=.075, hspace=.48, wspace=.35)
    original.HERE = HERE
    original.save(fig, NAME)
    scratch = ROOT / 'tmp/pdfs/analysis058'
    scratch.mkdir(parents=True, exist_ok=True)
    with fitz.open(HERE / 'figures' / NAME) as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(2, 2)).save(scratch / NAME.replace('.pdf', '.png'))
    a.write_csv(HERE / 'optimizer_overview_values.csv', values)
    outputs = {f'figures/{NAME}': a.sha(HERE / 'figures' / NAME),
               'optimizer_overview_values.csv': a.sha(HERE / 'optimizer_overview_values.csv')}
    a.dump(HERE / 'optimizer_overview_manifest.json', dict(
        generator=Path(__file__).relative_to(ROOT).as_posix(), generator_sha256=a.sha(__file__),
        sources=sources, outputs=outputs, curves=curves, smoothing=None,
        coordinate='gradient_state s: moment/direction used for outgoing update s -> s+1',
        moment='Stored uncorrected FP32 Adam exp_avg, norm reduced in FP64; absolute norm',
        alignment='Adjacent cosine of the same stream; initial undefined pair omitted',
        adaptive='Bias-corrected Adam direction in FP64, excluding LR; not rounded parameter increment'))
    print(f'{NAME}: {len(curves)} curves, {len(values):,} values; original four rows match exactly.')


if __name__ == '__main__':
    main()
