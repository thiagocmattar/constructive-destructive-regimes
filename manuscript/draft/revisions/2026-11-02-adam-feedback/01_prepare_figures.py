"""Compact manuscript views of verified Analysis036 evidence; no training."""
import hashlib
import json
import shutil
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1]
A36 = ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics'
TASKS = ['mlp', 'resnet', 'transformer']
TITLES = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
COLORS = ['#0072B2', '#D55E00', '#009E73']
ROWS = ['training_loss_log', 'eta_kappa_hat_raw', 'rho_raw', 'rho_momentum',
        'raw_gradient_norm_log', 'momentum_norm_log']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    full = read(A36/'full_diagnostics_manifest.json')
    raw = read(A36/'full_unsmoothed_manifest.json')
    micro = read(A36/'peak_micro_three_models_manifest.json')
    temporal = read(A36/'peak_temporal_results.json')
    assert full['completed_counts'] == dict(mlp=3, resnet=3, transformer=3)
    assert full['window'] == 50 and len(full['curves']) == 72
    manifest = dict(generator=Path(__file__).relative_to(ROOT).as_posix(),
        generator_sha256=sha(__file__), sources={}, curves=[], figures=[],
        clock='Horizontal state s: loss at theta_s; vector diagnostics use g_(s+1), m_(s+1), available before update s+1.',
        overview='All nine arms; loss raw, other rows complete 50-observation trailing means.',
        peaks='Three existing loss-selected completed episodes; all values raw. Norms divided by their own first value in each window.',
        uncertainty='One fixed initialization per case; no uncertainty bands.',
        selection=micro['examples'])
    for record in [full, raw, micro, temporal]:
        for name, digest in record.get('sources', {}).items():
            assert sha(ROOT/name) == digest, name
            manifest['sources'][name] = digest
    for name in ['full_diagnostics_manifest.json', 'full_unsmoothed_manifest.json',
                 'peak_micro_three_models_manifest.json', 'peak_temporal_results.json']:
        manifest['sources'][(A36/name).relative_to(ROOT).as_posix()] = sha(A36/name)
    plt.rcParams.update({'font.family':'Times New Roman', 'mathtext.fontset':'stix',
        'font.size':8.5, 'axes.labelsize':8.5, 'axes.titlesize':8.5,
        'xtick.labelsize':7.5, 'ytick.labelsize':7.5, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.linewidth':.55, 'pdf.fonttype':42,
        'path.simplify':False})

    def curve(ax, source, name, kind, normalized=False, **style):
        x = np.asarray(source['x'], dtype=int)
        if kind == 'overview' and name != 'training_loss_log':
            x = x-1
        y = np.asarray(source['y'], dtype=float)
        divisor = float(y[0]) if normalized else 1.
        y = y/divisor
        artist, = ax.plot(x, y, **style)
        np.testing.assert_array_equal(artist.get_xdata(), x)
        np.testing.assert_array_equal(artist.get_ydata(), y)
        manifest['curves'].append(dict(figure=kind, task=source['task'], lr=source['lr'],
            row=name, x=x.tolist(), y=y.tolist(), normalization=divisor,
            smoothing_window=source['smoothing_window']))
        return x, y

    def decorate(ax):
        ax.grid(axis='y', alpha=.15, lw=.5)
        ax.tick_params(length=2.5, width=.5)

    def logarithmic(ax, loss=False):
        ax.set_yscale('log')
        span = np.log10(ax.get_ylim()[1]/ax.get_ylim()[0])
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5) if loss and span < 2 else (1,), numticks=4))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if loss: ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))

    def save(fig, name):
        folder = A36/'figures'; folder.mkdir(exist_ok=True)
        target_folder = DRAFT/'figures/analysis036'; target_folder.mkdir(parents=True, exist_ok=True)
        path = folder/name; fig.savefig(path, metadata={'CreationDate':None, 'ModDate':None})
        plt.close(fig)
        target = target_folder/name; shutil.copyfile(path, target)
        with fitz.open(path) as doc:
            assert len(doc) == 1 and not doc[0].get_images()
            for word in doc[0].get_text('words'):
                assert word[0] >= 0 and word[1] >= 0 and word[2] <= doc[0].rect.width+.01 and word[3] <= doc[0].rect.height+.01, word
        manifest['figures'].append(dict(source=path.relative_to(ROOT).as_posix(),
            target=target.relative_to(ROOT).as_posix(), sha256=sha(path)))

    fig, axes = plt.subplots(6, 3, figsize=(7.16, 6.8), sharex=True)
    q_values = []
    for col, task in enumerate(TASKS):
        rates = sorted({c['lr'] for c in full['curves'] if c['task'] == task})
        axes[0, col].set_title(f'({chr(97+col)}) {TITLES[col]}', pad=21)
        for color, rate in zip(COLORS, rates):
            for row, name in enumerate(ROWS):
                source = next(c for c in full['curves'] if (c['task'], c['lr'], c['row']) == (task, rate, name))
                _, y = curve(axes[row, col], source, name, 'overview', color=color, lw=.72,
                             label=rf'$\eta={rate:g}$' if row == 0 else None)
                if row == 1: q_values.extend(y)
        axes[0, col].legend(loc='lower center', bbox_to_anchor=(.5, 1.005), ncol=3,
            frameon=False, fontsize=6.7, handlelength=1.1, handletextpad=.22, columnspacing=.55,
            borderaxespad=0)
        for row in [0, 4, 5]: logarithmic(axes[row, col], loss=row == 0)
        for row in [2, 3]:
            axes[row, col].set(ylim=(-1.07, 1.07), yticks=[-1, 0, 1])
            axes[row, col].axhline(0, color='.55', lw=.55, zorder=0)
        axes[1, col].axhline(2, color='.4', lw=.65, ls='--', zorder=0)
        for row in range(6):
            axes[row, col].set(xlim=(0, 1024), xticks=[0, 256, 512, 768, 1024])
            decorate(axes[row, col])
        axes[-1, col].set_xlabel(r'Completed updates $s$')
    for ax in axes[1]: ax.set(ylim=(min(-.05, min(q_values)-.08), max(2.15, max(q_values)+.08)), yticks=[0, 1, 2])
    labels = ['Training loss', r'$\eta\widehat\kappa_{s+1}$',
              r'$\rho_{s+1}$'+'\n(raw gradients)', r'$\rho_{s+1}^{(m)}$'+'\n(momentum)',
              r'$\|g_{s+1}\|$', r'$\|m_{s+1}\|$']
    for ax, label in zip(axes[:, 0], labels): ax.set_ylabel(label)
    fig.subplots_adjust(left=.105, right=.985, bottom=.065, top=.91, hspace=.29, wspace=.27)
    save(fig, 'adam_manuscript_overview.pdf')

    fig, axes = plt.subplots(3, 3, figsize=(7.16, 3.45), sharex='col')
    for col, example in enumerate(micro['examples']):
        task, rate = example['task'], example['lr']
        subset = [c for c in micro['curves'] if c['task'] == task]
        axes[0, col].set_title(f'({chr(97+col)}) {TITLES[col]}'+f'\n'+rf'$\eta={rate:g}$', pad=7)
        for row, name in [(0, 'training_loss_log'), (1, 'eta_kappa_hat_raw')]:
            source = next(c for c in subset if c['row'] == name)
            curve(axes[row, col], source, name, 'peaks', color=COLORS[2], lw=.8, marker='o', ms=1.6, markeredgewidth=0)
        for name, label, color in [('raw_gradient_norm_log', 'Gradient', COLORS[0]),
                                   ('momentum_norm_log', 'Momentum', COLORS[1])]:
            source = next(c for c in subset if c['row'] == name)
            curve(axes[2, col], source, name, 'peaks', normalized=True,
                  label=label, color=color, lw=.8)
        axes[0, col].axhline(example['starting_loss'], color='.6', lw=.55, ls=':')
        axes[1, col].axhline(2, color='.4', lw=.65, ls=':')
        logarithmic(axes[0, col], loss=True)
        logarithmic(axes[2, col])
        start, end = example['start_state'], example['end_state']
        for row in range(3):
            ax = axes[row, col]
            ax.axvline(example['episode_onset'], color='.65', lw=.65, ls=':', zorder=0)
            ax.axvline(example['peak_state'], color='.4', lw=.65, ls='--', zorder=0)
            ax.set_xlim(start-.5, end+.5)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            decorate(ax)
        axes[-1, col].set_xlabel(r'Parameter state $s$')
    axes[2, 0].legend(frameon=False, fontsize=6.8, loc='upper right', handlelength=1.1, borderpad=.1)
    for ax, label in zip(axes[:, 0], ['Training loss', r'$\eta\widehat\kappa_{s+1}$', 'Relative norm']): ax.set_ylabel(label)
    fig.subplots_adjust(left=.105, right=.985, bottom=.125, top=.835, hspace=.30, wspace=.29)
    save(fig, 'adam_manuscript_spikes.pdf')
    table = [r'\begin{tabular}{lrrr}', r'\hline',
             r'Case & Positive (\%) & Loss history & + raw change \\', r'\hline']
    manifest['table_values'] = []
    for task, label in [('mlp', 'MLP'), ('resnet', 'ResNet-32'),
                        ('transformer', 'Transformer'), ('all', 'Pooled')]:
        group = temporal['forecast']['overall'] if task == 'all' else temporal['forecast']['per_task'][task]
        baseline, added = group['loss_history'], group['projected_change']
        if task == 'all': table.append(r'\hline')
        table.append(f"{label} & {100*baseline['prevalence']:.1f} & {baseline['average_precision']:.3f} & {added['average_precision']:.3f} "+r'\\')
        manifest['table_values'].append(dict(task=task, prevalence=baseline['prevalence'],
            baseline=baseline['average_precision'], projected_change=added['average_precision']))
    table += [r'\hline', r'\end{tabular}']
    (DRAFT/'tables/analysis036_forecast.tex').write_text('\n'.join(table)+'\n', encoding='utf-8', newline='\n')
    (HERE/'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(figures=manifest['figures'], curves=len(manifest['curves']),
        points=sum(len(c['x']) for c in manifest['curves']))))


if __name__ == '__main__': main()
