"""Paper-sized GD view: identical unsmoothed arrays, compact serif typography."""
from importlib import import_module
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

GD = import_module('09_plot_gd')
HERE = Path(__file__).resolve().parent


def main():
    base = json.loads((HERE / 'gd_figure_manifest.json').read_text())
    assert base['status'] == 'passed' and base['mean_values_checked'] == 0
    original = HERE / 'figures/cifar10_gd_diagnostics.pdf'
    original_hash = GD.sha256(original)
    assert original_hash == base['figures'][original.name]['sha256']
    expected = {(c['run_id'], c['metric']): c for c in base['curves']}
    records = [r for r in json.loads((HERE / 'cohort.json').read_text())['runs'] if r['optimizer'] == 'gd']
    results = json.loads((HERE / 'results.json').read_text())
    assert results['status'] == 'passed'
    summary = {r['run_id']: r for r in results['runs']}
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
        'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
        'path.simplify': False, 'savefig.dpi': 300})
    fig, axes = plt.subplots(3, 2, figsize=(6.8, 5.65), sharex=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.085, top=.915, wspace=.29, hspace=.36)
    labels = [r'$\eta=0.004$', r'$\eta=2/300$', r'$\eta=0.01$', r'$\eta=0.02$']
    fig.legend([Line2D([], [], color=c, linewidth=1.4) for c in GD.COLORS], labels,
               ncol=4, loc='upper center', bbox_to_anchor=(.535, .995), frameon=False,
               handlelength=2.2, columnspacing=1.6, fontsize=9)
    definitions = [
        ('(a) Training loss', r'$\mathcal{L}(\theta)$', 'train_loss_with_final_state'),
        ('(b) Adjacent alignment', r'$\rho_t$', 'raw_cosine'),
        ('(c) Directional curvature', r'$\eta\widehat\kappa_t$', 'eta_kappa_proxy'),
        ('(d) Maximum curvature', r'$\eta\lambda_{\max}$', 'eta_lambda_max'),
        ('(e) Normalized displacement', r'$B_t/A_t$', 'raw_B_over_A'),
        ('(f) Gradient norm', r'$\|g_t\|_2$', 'raw_norm')]
    for ax, (title, ylabel, _) in zip(axes.flat, definitions):
        ax.set_title(title, loc='left', pad=4)
        ax.set(ylabel=ylabel, xlim=(0, 20000))
        ax.set_xticks([0, 5000, 10000, 15000, 20000], ['0', '5k', '10k', '15k', '20k'])
        ax.tick_params(width=.6, length=3)
        ax.grid(axis='y', color='.89', linewidth=.45)
        ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel('Step')
    axes[0, 0].set_ylim(0, .62)
    axes[0, 1].set_ylim(-1.08, 1.08)
    axes[0, 1].set_yticks([-1, 0, 1])
    for ax in axes[1]:
        ax.set_ylim(-.08, 2.3)
        ax.set_yticks([0, 1, 2])
    axes[1, 1].axhline(2, color='.4', linestyle='--', linewidth=.6)
    axes[2, 0].set_ylim(-.02, 1.04)
    axes[2, 1].set_yscale('log')
    curves = []
    for record, color in zip(records, GD.COLORS):
        run_id = record['run_id']
        path = HERE / 'derived' / f'Run{run_id}.npz'
        source = next(s for s in base['sources'] if s['run_id'] == run_id)
        assert GD.sha256(path) == source['derived_sha256']
        with np.load(path, allow_pickle=False) as values:
            data = dict(values)
        for ax, (_, _, metric) in zip(axes.flat, definitions):
            x = data['update']
            if metric == 'train_loss_with_final_state':
                x = np.r_[data['train_state'], 20000]
                y = np.r_[data['train_loss'], summary[run_id]['final_train']['loss']]
            elif metric == 'eta_lambda_max':
                x, y = data['curvature_state'], record['lr'] * data['hessian_eigenvalues'][:, 0]
            else:
                y = data[metric]
            line, = ax.plot(x, y, color=color, linewidth=.8, alpha=.9,
                            rasterized=metric in ['train_loss_with_final_state', 'raw_norm', 'raw_cosine', 'eta_kappa_proxy'])
            reference = expected[run_id, metric]
            assert GD.digest(line.get_xdata()) == reference['x_sha256']
            assert GD.digest(line.get_ydata()) == reference['plotted_y_sha256'] == reference['source_y_sha256']
            curves.append({'run_id': run_id, 'metric': metric, 'points': len(x),
                'finite_points': int(np.isfinite(y).sum()), 'x_sha256': GD.digest(x),
                'y_sha256': GD.digest(y), 'smoothing': None})
    assert len(curves) == 24
    pending = HERE / 'scratch/cifar10_gd_diagnostics_paper.pending.pdf'
    pending.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pending)
    plt.close(fig)
    destination = HERE / 'figures/cifar10_gd_diagnostics_paper.pdf'
    pending.replace(destination)
    assert GD.sha256(original) == original_hash
    GD.write_json(HERE / 'gd_paper_figure_manifest.json', {
        'status': 'passed', 'script': Path(__file__).name, 'script_sha256': GD.sha256(__file__),
        'figure': destination.relative_to(HERE).as_posix(), 'sha256': GD.sha256(destination),
        'bytes': destination.stat().st_size, 'dimensions_inches': [6.8, 5.65],
        'panels': 6, 'curves': curves, 'smoothing': None, 'thinning': None,
        'source_figure_sha256_unchanged': original_hash,
        'source_manifest_sha256': GD.sha256(HERE / 'gd_figure_manifest.json'),
        'sources': base['sources'],
        'caption': 'Matched CIFAR-10 full-batch GD rates: unsmoothed loss, adjacent cosine, eta-scaled directional curvature estimate, '
                   'eta*largest algebraic Hessian eigenvalue, full-prefix B/A and log raw-gradient norm. '
                   'All updates and all 201 Hessian samples per run retained; no reduction changed.'})
    print(json.dumps({'figure': str(destination), 'curves_identical_to_original': 24}))


if __name__ == '__main__':
    main()
