"""Five publication-size vector figures from the fixed descriptive reductions."""
import csv
import hashlib
import importlib.util
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('analysis055', HERE / '01_analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
GREEN, ORANGE, GRAY = '#009E73', '#D55E00', '#8A8A8A'
TITLES = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
CURVES, OUTPUTS = [], {}


def numeric_hash(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


def curve(pdf, arm, key, x, y, scale='linear', **extra):
    assert len(x) == len(y)
    CURVES.append(dict(figure=pdf, arm=arm, key=key, points=len(x), x_sha256=numeric_hash(x),
                       y_sha256=numeric_hash(y), yscale=scale, **extra))


def save(fig, name):
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert bbox.x0 >= -1 and bbox.y0 >= -1, (name, bbox)
    assert bbox.x1 <= fig.bbox.width + 1 and bbox.y1 <= fig.bbox.height + 1, (name, bbox)
    path = HERE / 'figures' / name
    fig.savefig(path, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    with fitz.open(path) as document:
        assert len(document) == 1 and not document[0].get_images()
        for font in document[0].get_fonts(full=True):
            assert document.extract_font(font[0])[3]
        for word in document[0].get_text('words'):
            assert 0 <= word[0] < word[2] <= document[0].rect.width + .01
            assert 0 <= word[1] < word[3] <= document[0].rect.height + .01
    OUTPUTS[path.relative_to(HERE).as_posix()] = analysis.sha(path)


def log_axis(ax):
    ax.set_yscale('log')
    span = np.log10(ax.get_ylim()[1] / ax.get_ylim()[0])
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5) if span < 2 else (1,), numticks=4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())


def style(ax):
    ax.grid(alpha=.16, lw=.5)
    ax.tick_params(length=2.5, width=.6)


def full_trajectories(arms):
    for tier, tier_name in enumerate(['low', 'middle', 'high']):
        name = f'adam_full_trajectories_{tier_name}.pdf'
        fig, axes = plt.subplots(4, 3, figsize=(7.16, 5.65), sharex='col')
        for col, task in enumerate(analysis.TASKS):
            arm = next(a for a in arms if a['task'] == task and a['tier'] == tier)
            keys = ['loss', 'Q', 'rho', 'relative_norm']
            values = [arm['loss'], arm['Q'][1:], arm['rho'][1:], arm['gradient'] / arm['gradient'][0]]
            xs = [np.arange(1025), np.arange(1, 1024), np.arange(1, 1024), np.arange(1024)]
            for row, (key, x, y) in enumerate(zip(keys, xs, values)):
                ax = axes[row, col]
                artist, = ax.plot(x, y, color=GREEN, lw=.65)
                np.testing.assert_array_equal(artist.get_ydata(), y)
                np.testing.assert_array_equal(artist.get_xdata(), x)
                curve(name, arm['arm'], key, x, y, 'log' if row in [0, 3] else 'linear')
                for event in arm['events']:
                    if float(event['peak_change_percent']) <= 10:
                        continue
                    a, p = int(event['before_state']), int(event['peak_state'])
                    if event['recovery_state']:
                        ax.axvspan(a, p, color='.5', alpha=.12, linewidth=0, zorder=-1)
                    else:
                        ax.axvspan(a, p, facecolor='none', edgecolor='.55', linestyle=':', lw=.65, zorder=-1)
                if row == 0:
                    ax.set_title(f'({chr(97 + col)}) {TITLES[col]}\n' + rf'$\eta={arm["lr"]:g}$', pad=8)
                else:
                    ax.text(0, 1.035, f'({chr(97 + row * 3 + col)})', transform=ax.transAxes,
                            va='bottom', fontsize=9)
                ax.set(xlim=(0, 1024), xticks=[0, 256, 512, 768, 1024])
                ax.margins(y=.08)
                style(ax)
                if row in [0, 3]:
                    log_axis(ax)
                elif row == 2:
                    ax.set(ylim=(-1.06, 1.06), yticks=[-1, 0, 1])
                    ax.axhline(0, color='.45', ls=(0, (4, 3)), lw=.65)
                else:
                    # Common projected-change limits across the three rates of this model.
                    all_q = np.concatenate([a['Q'][1:] for a in arms if a['task'] == task])
                    padding = .08 * np.ptp(all_q)
                    ax.set_ylim(float(all_q.min() - padding), float(all_q.max() + padding))
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                    ax.axhline(2, color='.45', ls=(0, (4, 3)), lw=.65)
            axes[-1, col].set_xlabel(r'Parameter state $s$')
        for ax, label in zip(axes[:, 0], ['Training loss', r'$\eta\widehat\kappa$', r'$\rho$', 'Relative norm']):
            ax.set_ylabel(label)
        fig.legend(handles=[Patch(facecolor='.5', alpha=.12, label='Rise of recovered >10% excursion'),
                            Patch(facecolor='none', edgecolor='.55', ls=':', label='Recovery unobserved')],
                   loc='lower center', bbox_to_anchor=(.54, .012), ncol=2, frameon=False,
                   fontsize=7.7, handlelength=1.5, columnspacing=1)
        fig.subplots_adjust(left=.11, right=.975, top=.90, bottom=.145, hspace=.48, wspace=.35)
        save(fig, name)


def geometry(arms, states):
    name = 'adam_direction_amplification.pdf'
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 6.6), sharex=True, sharey=True)
    minimum = min(a['r'][1:].min() for a in arms) / 1.12
    maximum = max(a['r'][1:].max() for a in arms) * 1.12
    for col, task in enumerate(analysis.TASKS):
        for tier in range(3):
            ax = axes[tier, col]
            arm = next(a for a in arms if a['task'] == task and a['tier'] == tier)
            rows = [r for r in states if r['arm'] == arm['arm'] and r['rho'] != '']
            groups = [('other', [r for r in rows if r['eligible'] == '0'], '.78', 'o', 4, .45),
                      ('quiet_no_rise', [r for r in rows if r['eligible'] == '1' and r['target'] == '0'], GREEN, 'o', 5, .48),
                      ('quiet_future_rise', [r for r in rows if r['eligible'] == '1' and r['target'] == '1'], ORANGE, '^', 9, .9)]
            for key, group, color, marker, size, alpha in groups:
                x = [float(r['rho']) for r in group]
                y = [float(r['r']) for r in group]
                artist = ax.scatter(x, y, color=color, marker=marker, s=size, alpha=alpha, linewidths=0)
                np.testing.assert_array_equal(np.asarray(artist.get_offsets()), np.column_stack([x, y]))
                curve(name, arm['arm'], key, x, y, 'log')
            ax.axvline(0, color='.6', lw=.6)
            ax.axhline(1, color='.6', lw=.6)
            x = np.linspace(-1, -1 / maximum, 200)
            ax.plot(x, -1 / x, color='.35', ls='--', lw=.7)
            ax.set(xlim=(-1.05, 1.05), ylim=(minimum, maximum), xticks=[-1, 0, 1])
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(.2, .5, 1, 2, 5), numticks=6))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
            ax.yaxis.set_minor_formatter(NullFormatter())
            style(ax)
            letter = chr(97 + tier * 3 + col)
            ax.set_title(f'({letter}) {TITLES[col]}\n' + rf'$\eta={arm["lr"]:g}$', pad=6)
            positives = sum(r['eligible'] == '1' and r['target'] == '1' for r in rows)
            eligible = sum(r['eligible'] == '1' for r in rows)
            ax.text(.97, .04, f'{positives}/{eligible} future rises', transform=ax.transAxes,
                    ha='right', fontsize=7, bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=1))
            if tier == 2:
                ax.set_xlabel(r'Consecutive cosine $\rho$')
            if col == 0:
                ax.set_ylabel(r'Norm ratio $r$')
    fig.legend(handles=[Line2D([], [], color='.78', marker='o', ls='', ms=3, label='Other states'),
                        Line2D([], [], color=GREEN, marker='o', ls='', ms=3, label='Quiet: no future rise'),
                        Line2D([], [], color=ORANGE, marker='^', ls='', ms=3, label='Quiet: future rise'),
                        Line2D([], [], color='.35', ls='--', lw=.7, label='GD quadratic reference')],
               frameon=False, fontsize=7.8, ncol=2, loc='lower center', bbox_to_anchor=(.54, .008),
               columnspacing=1.3, handlelength=1.6)
    fig.subplots_adjust(left=.10, right=.98, top=.91, bottom=.145, hspace=.67, wspace=.25)
    save(fig, name)


def matched_context(summaries):
    name = 'adam_matched_context.pdf'
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.8), sharex=True)
    contrasts = ['negative_vs_nonnegative', 'strong_vs_weak_negative']
    descriptions = [r'$\rho<0$ versus $\rho\geq0$', r'$\rho<-0.8$ versus $-0.8\leq\rho<0$']
    for row, contrast in enumerate(contrasts):
        for col, task in enumerate(analysis.TASKS):
            ax = axes[row, col]
            chosen = sorted([s for s in summaries if s['task'] == task and s['contrast'] == contrast],
                            key=lambda s: float(s['lr']))
            for y, item in enumerate(chosen):
                n = int(item['pairs'])
                if n:
                    a, b = 100 * float(item['more_rate']), 100 * float(item['less_rate'])
                    ax.plot([b, a], [y, y], color='.6', lw=.9, zorder=1)
                    ax.scatter([b], [y], facecolor='white', edgecolor='.35', marker='o', s=19, lw=.8, zorder=2)
                    ax.scatter([a], [y], color=GREEN, marker='D', s=16, linewidths=0, zorder=3)
                    curve(name, item['arm'], 'more_rate', [a], [y], contrast=contrast)
                    curve(name, item['arm'], 'less_rate', [b], [y], contrast=contrast)
                else:
                    ax.text(50, y, 'No pairs', ha='center', fontsize=8, color='.45')
                ax.text(103, y + .16, f'n={n}' + ('*' if n < 20 else ''), ha='right', fontsize=7.2)
            ax.set(xlim=(-5, 107), ylim=(-.4, 2.5), xticks=[0, 25, 50, 75, 100],
                   yticks=[0, 1, 2], yticklabels=[f'{float(s["lr"]):g}' for s in chosen])
            ax.set_title(f'({chr(97 + row * 3 + col)}) {TITLES[col].split(" / ")[-1]}\n' + descriptions[row], pad=8)
            ax.invert_yaxis()
            style(ax)
            if col == 0:
                ax.set_ylabel('Learning rate')
            if row == 1:
                ax.set_xlabel('Future rise frequency (%)')
    fig.legend(handles=[Line2D([], [], marker='o', markerfacecolor='white', color='.35', ls='', ms=4,
                               label='Less reversal'),
                        Line2D([], [], marker='D', color=GREEN, ls='', ms=4, label='More reversal')],
               loc='lower center', bbox_to_anchor=(.54, .018), ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.11, right=.98, top=.85, bottom=.18, hspace=.85, wspace=.41)
    save(fig, name)


def main():
    (HERE / 'figures').mkdir(exist_ok=True)
    manifest = analysis.read(HERE / 'manifest.json')
    for relative, digest in manifest['outputs'].items():
        assert analysis.sha(HERE / relative) == digest
    arms, sources = analysis.load_arms()
    assert sources == manifest['sources']
    def table(name):
        with (HERE / f'derived/{name}.csv').open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42,
        'path.simplify': False})
    full_trajectories(arms)
    geometry(arms, table('states'))
    matched_context(table('matched_summary'))
    analysis.dump(HERE / 'plot_manifest.json', dict(generator=Path(__file__).relative_to(analysis.ROOT).as_posix(),
        generator_sha256=analysis.sha(__file__), reduction_manifest_sha256=analysis.sha(HERE / 'manifest.json'),
        outputs=OUTPUTS, curves=CURVES, dimensions='All figures 7.16 inches wide; embedded vector fonts.',
        smoothing=None, visual_review='See observations/001-complete-trajectories.md and 002-direction-context.md.'))
    print(f'Five vector PDFs; {len(CURVES)} plotted data records / {sum(c["points"] for c in CURVES):,} points verified.')


if __name__ == '__main__':
    main()
