"""Retain only the nine unsmoothed training-loss curves in the Adam overview."""
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PDF = HERE / 'figures/adam_manuscript_overview.pdf'
SOURCE = HERE / 'archive/paper_evidence/presentation_manifest.json'
MANIFEST = HERE / 'loss_overview_manifest.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    prior = json.loads(SOURCE.read_text())
    results_path = ROOT / 'runs/215-2026-11-02-controlled-full-batch-adam/results.json'
    assert sha(results_path) == prior['sources'][results_path.relative_to(ROOT).as_posix()]
    results = json.loads(results_path.read_text())
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in [SOURCE, results_path]}
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42,
        'path.simplify': False})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    tasks = ['mlp', 'resnet', 'transformer']
    titles = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
    colors = ['#0072B2', '#D55E00', '#009E73']
    checks = []
    for col, (task, ax) in enumerate(zip(tasks, axes)):
        curves = sorted([c for c in prior['curves'] if c['task'] == task and c['key'] == 'loss'],
                        key=lambda c: c['lr'])
        assert len(curves) == 3
        for curve, color in zip(curves, colors):
            condition = next(c for c in results['conditions'] if c['task'] == task and c['lr'] == curve['lr'])
            path = ROOT / condition['source'] / 'boundaries.jsonl'
            assert sha(path) == prior['sources'][path.relative_to(ROOT).as_posix()]
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
            boundaries = [json.loads(line) for line in path.read_text().splitlines()]
            x, y = np.asarray(curve['x']), np.asarray(curve['y'])
            np.testing.assert_array_equal(x, [b['state'] for b in boundaries])
            np.testing.assert_array_equal(y, [b['loss'] for b in boundaries])
            assert len(y) == 1025 and (y > 0).all() and curve['window'] is None
            artist, = ax.plot(x, y, color=color, lw=.9, zorder=2, label=rf'$\eta={curve["lr"]:g}$')
            np.testing.assert_array_equal(artist.get_ydata(), y)
            checks.append(dict(task=task, lr=curve['lr'], points=len(y), exact_prior_loss_values=True))
        ax.set_title(f'({chr(97 + col)}) ' + titles[col], pad=26)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=3,
            frameon=False, fontsize=7.6, handlelength=1., columnspacing=.5,
            handletextpad=.25, borderaxespad=0)
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(NullFormatter())
        span = np.log10(ax.get_ylim()[1] / ax.get_ylim()[0])
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5) if span < 2 else (1,), numticks=4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        ax.set(xlim=(0, 1024), xticks=[0, 256, 512, 768, 1024], xlabel=r'Parameter state $t$')
        ax.grid(alpha=.16, lw=.5)
        ax.tick_params(length=2.5, width=.6)
    axes[0].set_ylabel('Training loss')
    fig.subplots_adjust(left=.105, right=.975, bottom=.185, top=.745, wspace=.31)
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert bounds.x0 >= -1 and bounds.y0 >= -1
    assert bounds.x1 <= fig.bbox.width + 1 and bounds.y1 <= fig.bbox.height + 1
    assert len(fig.axes) == 3 and sum(len(ax.lines) for ax in axes) == 9
    fig.savefig(PDF, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    with fitz.open(PDF) as doc:
        assert len(doc) == 1 and not doc[0].get_images()
        for font in doc[0].get_fonts(full=True):
            assert doc.extract_font(font[0])[3]
        for word in doc[0].get_text('words'):
            assert 0 <= word[0] < word[2] <= doc[0].rect.width + .01
            assert 0 <= word[1] < word[3] <= doc[0].rect.height + .01
    record = dict(generator=Path(__file__).relative_to(ROOT).as_posix(), generator_sha256=sha(__file__),
        sources=sources, output=PDF.relative_to(ROOT).as_posix(), output_sha256=sha(PDF),
        panels=3, rows=['training_loss'], smoothing=None, dimensions_inches=[7.16, 2.55],
        checks=checks, total_points=sum(c['points'] for c in checks), visual_review='pending')
    MANIFEST.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('Verified nine unchanged unsmoothed loss curves / 9,225 points in one row of three panels.')


if __name__ == '__main__':
    main()
