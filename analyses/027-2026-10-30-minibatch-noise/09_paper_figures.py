"""Focused batch-size comparisons from the unchanged, verified H023 evidence."""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]
BATCHES = [32, 128, 512, 2048, 5000]
STATES = [0, 1000, 2000]
BLUE, ORANGE = '#0072B2', '#D55E00'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


def main():
    verification = read(HERE / 'verification.json')
    assert verification['status'] == 'passed'
    assert digest(HERE / 'results.json') == verification['results_sha256']
    reduced = read(HERE / 'reduction.json')
    assert reduced['source_results_sha256'] == verification['results_sha256']
    assert digest(HERE / 'reduction.json') == read(HERE / 'scientific_summary.json')['source_reduction_sha256']
    manifest = dict(source_results_sha256=digest(HERE / 'results.json'),
                    source_reduction_sha256=digest(HERE / 'reduction.json'),
                    script_sha256=digest(Path(__file__)), numpy=np.__version__,
                    matplotlib=matplotlib.__version__, curves=[], pdfs={},
                    reduction='Repetition means within initialization; equal-weight mean and range of three initialization means. No temporal smoothing.',
                    visual_review='pending')
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
                         'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'pdf.fonttype': 42})
    folder = HERE / 'figures' / 'paper'
    folder.mkdir(parents=True, exist_ok=True)

    def values(stage, checkpoint, field):
        return np.array([[next(c[field] for c in reduced[stage]
                               if (c['seed'], c['checkpoint'], c['batch']) == (seed, checkpoint, b))
                          for b in BATCHES] for seed in range(3)])

    def record(name, checkpoint, field, blocks, line):
        center = blocks.mean(axis=0)
        np.testing.assert_array_equal(line.get_ydata(), center)
        manifest['curves'].append(dict(figure=name, checkpoint=checkpoint, field=field,
                                      batches=BATCHES, initialization_means=blocks.tolist(),
                                      mean=center.tolist(), minimum=blocks.min(axis=0).tolist(),
                                      maximum=blocks.max(axis=0).tolist()))

    def axes_style(ax, letter):
        ax.set_xscale('log', base=2)
        ax.set_xlim(25, 6400)
        ax.xaxis.set_major_locator(FixedLocator(BATCHES))
        ax.xaxis.set_major_formatter(FixedFormatter(['32', '128', '512', '2,048', '5,000']))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(length=3, width=.6)
        ax.grid(axis='y', alpha=.18, lw=.5)
        ax.text(0, 1.035, f'({letter})', transform=ax.transAxes, va='bottom', ha='left')

    titles = ['At initialization', 'After 1,000 GD updates', 'After 2,000 GD updates']

    def save(fig, name):
        path = folder / (name + '.pdf')
        fig.savefig(path, metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)
        target = ROOT / 'manuscript/draft/figures/analysis027' / path.name
        shutil.copyfile(path, target)
        manifest['pdfs'][path.name] = dict(sha256=digest(path),
                                         source=path.relative_to(ROOT).as_posix(),
                                         target=target.relative_to(ROOT).as_posix())

    name = 'batch_size_fixed_parameters'
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), sharey=True)
    for col, checkpoint in enumerate(STATES):
        ax = axes[col]
        measured = values('frozen', checkpoint, 'mean_rho')
        prediction = values('frozen', checkpoint, 'predicted_rho')
        ax.fill_between(BATCHES, measured.min(0), measured.max(0), color=BLUE, alpha=.13, lw=0)
        line, = ax.plot(BATCHES, measured.mean(0), color=BLUE, marker='o', ms=3.8, lw=1.1)
        record(name, checkpoint, 'mean_rho', measured, line)
        line, = ax.plot(BATCHES, prediction.mean(0), color=ORANGE, marker='s', mfc='none',
                        ms=6, mew=.9, linestyle='none')
        record(name, checkpoint, 'predicted_rho', prediction, line)
        axes_style(ax, chr(97 + col))
        ax.set(title=titles[col], xlabel=r'Batch size $b$', ylim=(-.035, 1.055))
        ax.set_yticks([0, .25, .5, .75, 1])
    axes[0].set_ylabel(r'Mean cosine similarity, $\rho_t$')
    fig.legend(handles=[Line2D([], [], color=BLUE, marker='o', ms=3.8, lw=1.1, label='Measured'),
                        Line2D([], [], color=ORANGE, marker='s', mfc='none', ms=6,
                               linestyle='none', label='Approximation in Eq. (19)')],
               loc='lower center', ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.095, right=.985, top=.82, bottom=.29, wspace=.2)
    save(fig, name)

    name = 'batch_size_sgd_outcomes'
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 4.75), sharex=True)
    for col, checkpoint in enumerate(STATES):
        axes[0, col].set_title(titles[col], pad=17)
        for row, field in enumerate(['mean_rho', 'R', 'loss_change']):
            ax = axes[row, col]
            blocks = values('moving', checkpoint, field)
            ax.fill_between(BATCHES, blocks.min(0), blocks.max(0), color=BLUE, alpha=.13, lw=0)
            line, = ax.plot(BATCHES, blocks.mean(0), color=BLUE, marker='o', ms=3.8, lw=1.1)
            record(name, checkpoint, field, blocks, line)
            if row == 0:
                full = values('moving', checkpoint, 'mean_full_rho')
                ax.fill_between(BATCHES, full.min(0), full.max(0), color=ORANGE, alpha=.10, lw=0)
                line, = ax.plot(BATCHES, full.mean(0), color=ORANGE, marker='s', mfc='none',
                                ms=4.8, mew=.9, lw=1.1)
                record(name, checkpoint, 'mean_full_rho', full, line)
                ax.set_ylim(-1.08, 1.08)
                ax.set_yticks([-1, -.5, 0, .5, 1])
                ax.axhline(0, color='.5', lw=.6, zorder=0)
            elif row == 1:
                ax.set_ylim(-.035, 1.055)
                ax.set_yticks([0, .25, .5, .75, 1])
            else:
                ax.axhline(0, color='.5', lw=.6, zorder=0)
                ax.set_ylim((-.28, .025) if col == 0 else (-.025, .055))
                ax.set_yticks([-.2, -.1, 0] if col == 0 else [-.02, 0, .02, .04])
                ax.set_xlabel(r'Batch size $b$')
            low, high = ax.get_ylim()
            assert blocks.min() >= low and blocks.max() <= high, (checkpoint, field)
            axes_style(ax, chr(97 + row * 3 + col))
    axes[0, 0].set_ylabel('Mean cosine\nsimilarity ' + r'$\rho_t$')
    axes[1, 0].set_ylabel('Displacement\nratio ' + r'$R_{200}$')
    axes[2, 0].set_ylabel('Training loss\nchange')
    fig.legend(handles=[Line2D([], [], color=BLUE, marker='o', ms=3.8, lw=1.1,
                               label='Cosine: mini-batch gradients'),
                        Line2D([], [], color=ORANGE, marker='s', mfc='none', ms=4.8, lw=1.1,
                               label='Cosine: full gradients at the same SGD states')],
               loc='lower center', ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.10, right=.985, top=.88, bottom=.14, hspace=.29, wspace=.28)
    save(fig, name)
    write(HERE / 'paper_figure_manifest.json', manifest)
    print(json.dumps({'figures': list(manifest['pdfs']), 'curves': len(manifest['curves']),
                      'values_per_curve': 5, 'source_evidence_changed': False}))


if __name__ == '__main__':
    main()
