"""Rebuild Figure 4 from the verified seed-0, per-repetition trajectories."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'runs/298-2026-11-19-transformer-sgd-seed0'
TASKS = [('mlp', 'CIFAR-10 / MLP', [32, 128, 512, 2048, 5000]),
         ('resnet', 'CIFAR-10 / ResNet-32', [1, 5, 20, 50]),
         ('transformer', 'WikiText-2 / Transformer', [8, 32, 128, 512, 59680])]


def load():
    accept = json.loads((HERE / 'cohort_acceptance.json').read_text())
    mlp = ROOT / 'analyses/027-2026-10-30-minibatch-noise/results.json'
    resnet = HERE / 'results.json'
    for path, key in [(mlp, 'reused_mlp_results_sha256'), (resnet, 'results_sha256')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == accept[key]
    reps = {}
    for row in json.loads(mlp.read_text())['moving']:
        if row['seed'] == 0 and row['checkpoint'] in (0, 2000):
            reps['mlp', row['checkpoint'], row['batch'], row['replicate']] = row['mean_rho']
    for row in json.loads(resnet.read_text())['forks']:
        if row['task'] == 'resnet' and row['seed'] == 0:
            reps['resnet', row['state'], row['batch'], row['rep']] = row['mean_rho']
    verified = json.loads((RUN / 'verified.json').read_text())
    assert verified['verified_arms'] == len(verified['arms']) == 66
    for row in verified['arms']:
        reps['transformer', row['state'], row['batch'], row['repetition']] = row['mean_rho']
    assert len(reps) == 182
    return reps


def main():
    reps = load()
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
                         'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'pdf.fonttype': 42})
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), sharey=True)
    rows = []
    for col, (task, title, batches) in enumerate(TASKS):
        ax = axes[col]
        x = np.array(batches) * (1000 if task == 'resnet' else 1)
        ax.set_title(f'({chr(97 + col)})  {title}', pad=6)
        for state, color, marker in [(0, '#0072B2', 'o'), (2000, '#D55E00', 's')]:
            values = []
            for batch in batches:
                n = 1 if batch == batches[-1] else 8
                arm = [reps[task, state, batch, rep] for rep in range(n)]
                assert np.isfinite(arm).all()
                values.append([np.mean(arm), min(arm), max(arm)])
                rows.append(dict(model=task, seed=0, state=state, batch=batch,
                                 plotted_batch=batch * (1000 if task == 'resnet' else 1),
                                 repetitions=n, mean=values[-1][0], minimum=min(arm), maximum=max(arm)))
            mean, low, high = np.array(values).T
            ax.fill_between(x[:-1], low[:-1], high[:-1], color=color, alpha=.16, lw=0)
            line, = ax.plot(x, mean, color=color, marker=marker, ms=3.2, lw=1,
                            label='At initialization' if state == 0 else 'After 2,000 GD updates')
            np.testing.assert_array_equal(line.get_ydata(), mean)
            ax.plot(x[-1], mean[-1], marker=marker, ms=4, color=color, mfc='white', lw=0)
        ax.set(xscale='log', xlim=(x[0] / 1.2, x[-1] * (2 if task == 'transformer' else 1.4)),
               ylim=(-1.08, 1.08), yticks=[-1, 0, 1], xlabel='Batch size $b$ (' +
               ('sequences' if task == 'transformer' else 'images') + ')')
        labels = [f'{value:,}' for value in x]
        labels[-1] += '\n(full)'
        ax.set_xticks(x, labels)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis='x', labelsize=7.5)
        ax.tick_params(length=3, width=.6)
        ax.grid(axis='y', alpha=.15, lw=.5)
        ax.axhline(0, color='.5', lw=.6, zorder=0)
    axes[0].set_ylabel(r'Mean cosine $\rho_t$')
    figure.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
                  bbox_to_anchor=(.53, 1.005), ncol=2, frameon=False, fontsize=9)
    figure.subplots_adjust(left=.085, right=.985, top=.76, bottom=.28, wspace=.18)
    pdf = HERE / 'figures/sgd_seed0_batch_size.pdf'
    figure.savefig(pdf, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(figure)
    with (HERE / 'seed0_cells.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(pdf, ROOT / 'manuscript/consolidation/draft/figures/sgd_batch_size_compact.pdf')


if __name__ == '__main__':
    main()
