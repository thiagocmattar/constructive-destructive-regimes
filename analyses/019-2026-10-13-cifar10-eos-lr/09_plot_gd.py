"""Archive the detailed PDFs and plot six GD diagnostics from verified evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = ['#0072B2', '#009E73', '#D55E00', '#6A3D9A']
WINDOW = 100


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def archive_detailed():
    """Move only the four declared PDFs; retain and check their original bytes."""
    original = json.loads((HERE / 'figure_manifest.json').read_text())
    archive = HERE / 'figures/archive'
    archive.mkdir(parents=True, exist_ok=True)
    records = []
    for name, item in original['figures'].items():
        source = (HERE / 'figures' / name).resolve()
        target = (archive / name).resolve()
        assert source.is_relative_to(HERE.resolve()) and target.is_relative_to(HERE.resolve())
        if source.exists():
            assert not target.exists(), f'Refusing to overwrite {target}'
            assert source.stat().st_size == item['bytes'] and sha256(source) == item['sha256']
            source.rename(target)
        assert target.stat().st_size == item['bytes'] and sha256(target) == item['sha256']
        records.append({'original_path': source.relative_to(HERE).as_posix(),
                        'archived_path': target.relative_to(HERE).as_posix(),
                        'bytes': item['bytes'], 'sha256': item['sha256'], 'unchanged': True})
    write_json(HERE / 'archive_manifest.json', {
        'status': 'passed', 'reason': 'User requested simpler figures; detailed originals retained byte-for-byte.',
        'original_numeric_manifest': 'figure_manifest.json',
        'original_visual_verification': 'visual_verification.json', 'figures': records})


def trailing_mean(values):
    """Full right-aligned windows; retain the undefined first adjacent pair."""
    assert np.isnan(values[0]) and np.isfinite(values[1:]).all()
    output = np.full(len(values), np.nan)
    sums = np.r_[0., np.cumsum(values[1:], dtype=np.float64)]
    output[WINDOW:] = (sums[WINDOW:] - sums[:-WINDOW]) / WINDOW
    direct = np.lib.stride_tricks.sliding_window_view(values[1:], WINDOW).mean(axis=1)
    np.testing.assert_allclose(output[WINDOW:], direct, rtol=1e-9, atol=2e-10)
    assert np.flatnonzero(np.isfinite(output))[0] == 100
    assert np.isfinite(output).sum() == 19900
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-only', action='store_true')
    args = parser.parse_args()
    archive_detailed()
    if args.archive_only:
        print('Four original PDFs archived; size and SHA256 unchanged.')
        return
    assert json.loads((HERE / 'verification.json').read_text())['status'] == 'passed'
    results = json.loads((HERE / 'results.json').read_text())
    assert results['status'] == 'passed'
    summary = {r['run_id']: r for r in results['runs']}
    records = [r for r in json.loads((HERE / 'cohort.json').read_text())['runs'] if r['optimizer'] == 'gd']
    assert [r['run_id'] for r in records] == ['059', '060', '061', '062']
    manifest = {'status': 'passed', 'script': Path(__file__).name, 'script_sha256': sha256(__file__),
                'panels': 6, 'learning_rates': [r['lr'] for r in records],
                'dimensions_inches': [14, 11.5], 'thinning': None, 'curves': [], 'sources': [],
                'mean_values_checked': 0,
                'smoothing': 'Only cosine and eta-kappa: complete 100-pair trailing means, first at update 101.',
                'y_scales': ['linear', 'linear', 'linear', 'linear', 'linear', 'log'],
                'kappa': 'eta*kappa_hat=1-rho_t*norm(g_t)/norm(g_(t-1)); explicit noise term zero.',
                'coverage': 'Four GD runs; all 20k updates, loss states 0..20k, Hessian states 0..20k every 100.'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 13, 'axes.labelsize': 12, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'path.simplify': False, 'savefig.dpi': 180})
    fig, axes = plt.subplots(3, 2, figsize=(14, 11.5))
    fig.subplots_adjust(left=.075, right=.98, bottom=.105, top=.865, hspace=.44, wspace=.22)
    fig.suptitle('CIFAR-10 | Full-batch gradient descent', fontsize=20, y=.98)
    fig.text(.5, .943, '5,000 training images | tanh 200-200 MLP | 20,000 updates', ha='center', fontsize=11)
    labels = [r'$\eta=0.004$', r'$\eta=2/300\;\approx\;0.006667$', r'$\eta=0.01$', r'$\eta=0.02$']
    fig.legend([Line2D([], [], color=c, linewidth=2.5) for c in COLORS], labels,
               ncol=4, loc='upper center', bbox_to_anchor=(.5, .927), frameon=False, fontsize=12)
    definitions = [
        ('(a) Unsmoothed training loss', r'$f(\theta)$', 'train_loss_with_final_state', None),
        ('(b) Raw-gradient adjacent cosine', r'$\rho_t$', 'raw_cosine', WINDOW),
        (r'(c) $\eta\,\widehat{\kappa}_t$', r'$\eta\,\widehat{\kappa}_t$', 'eta_kappa_proxy', WINDOW),
        (r'(d) $\eta\,\lambda_{\max}$', r'$\eta\,\lambda_{\max}$', 'eta_lambda_max', None),
        ('(e) Raw-gradient cumulative displacement', r'$B_t/A_t$', 'raw_B_over_A', None),
        ('(f) Raw-gradient norm', r'$\|g_t\|_2$', 'raw_norm', None)]
    for ax, (title, ylabel, _, _) in zip(axes.flat, definitions):
        ax.set_title(title, loc='left', pad=10)
        ax.set(xlabel='Step', ylabel=ylabel, xlim=(0, 20000))
        ax.set_xticks([0, 5000, 10000, 15000, 20000], ['0', '5k', '10k', '15k', '20k'])
        ax.grid(axis='y', color='#E2E7EC', linewidth=.8)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylim(0, .62)
    axes[0, 1].set_ylim(-1.08, 1.08)
    axes[0, 1].set_yticks([-1, 0, 1])
    for ax in axes[1]:
        ax.set_ylim(-.08, 2.3)
        ax.set_yticks([0, 1, 2])
    axes[1, 1].axhline(2, color='.4', linestyle='--', linewidth=.9)
    axes[1, 1].text(.985, .08, 'GD reference: 2', ha='right', va='bottom',
                    transform=axes[1, 1].transAxes, fontsize=10,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
    axes[2, 0].set_ylim(-.02, 1.04)
    axes[2, 1].set_yscale('log')
    for record, color in zip(records, COLORS):
        run_id = record['run_id']
        path = HERE / 'derived' / f'Run{run_id}.npz'
        verified = json.loads((HERE / 'verification' / f'Run{run_id}.json').read_text())
        assert verified['status'] == 'passed' and sha256(path) == verified['derived_sha256']
        assert path.stat().st_size == verified['derived_bytes']
        with np.load(path, allow_pickle=False) as values:
            data = dict(values)
        np.testing.assert_array_equal(data['update'], np.arange(1, 20001))
        np.testing.assert_array_equal(data['train_state'], np.arange(20000))
        np.testing.assert_array_equal(data['curvature_state'], np.arange(0, 20001, 100))
        direct_kappa = 1 - data['raw_cosine'][1:] * data['raw_norm'][1:] / data['raw_norm'][:-1]
        np.testing.assert_allclose(data['eta_kappa_proxy'][1:], direct_kappa, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(data['raw_B_over_A'], data['raw_B'] / data['raw_A'], rtol=2e-14)
        assert data['hessian_eigenvalues'].shape == (201, 2)
        assert np.all(data['hessian_eigenvalues'][:, 0] >= data['hessian_eigenvalues'][:, 1])
        manifest['sources'].append({'run_id': run_id, 'learning_rate': record['lr'],
                                    'derived_path': path.relative_to(HERE).as_posix(),
                                    'derived_sha256': verified['derived_sha256'],
                                    'source_metrics_sha256': verified['source_metrics_sha256'],
                                    'source_curvature_sha256': verified['source_curvature_sha256']})
        for ax, (_, _, metric, window) in zip(axes.flat, definitions):
            x = data['update']
            if metric == 'train_loss_with_final_state':
                x = np.r_[data['train_state'], 20000]
                raw = np.r_[data['train_loss'], summary[run_id]['final_train']['loss']]
            elif metric == 'eta_lambda_max':
                x = data['curvature_state']
                raw = record['lr'] * data['hessian_eigenvalues'][:, 0]
            else:
                raw = data[metric]
            plotted = trailing_mean(raw) if window else raw
            if window:
                manifest['mean_values_checked'] += int(np.isfinite(plotted).sum())
            noisy = metric in ['train_loss_with_final_state', 'raw_norm']
            line, = ax.plot(x, plotted, color=color, linewidth=1.25, alpha=.9 if noisy else 1.,
                            rasterized=noisy)
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), plotted)
            manifest['curves'].append({'run_id': run_id, 'panel': ax.get_title(loc='left'),
                                       'metric': metric, 'points': len(x),
                                       'finite_points': int(np.isfinite(plotted).sum()),
                                       'x_sha256': digest(x), 'source_y_sha256': digest(raw),
                                       'plotted_y_sha256': digest(plotted), 'trailing_window': window,
                                       'y_scale': ax.get_yscale()})
    fig.text(.075, .047, 'Cosine and '+r'$\eta\widehat{\kappa}$'+': 100-pair trailing means. Loss and gradient norms: unsmoothed.', fontsize=10.5)
    fig.text(.075, .023, 'Gradient norms use log Y; all other panels use linear Y. Hessian measured every 100 steps.', fontsize=10.5)
    assert len(fig.axes) == 6 and len(manifest['curves']) == 24
    assert manifest['mean_values_checked'] == 159200
    destination = HERE / 'figures/cifar10_gd_diagnostics.pdf'
    # Replace only a complete PDF; some Windows viewers allow rename but lock writes.
    scratch = HERE / 'scratch'
    scratch.mkdir(exist_ok=True)
    pending = scratch / 'cifar10_gd_diagnostics.pending.pdf'
    fig.savefig(pending)
    plt.close(fig)
    pending.replace(destination)
    manifest['figures'] = {destination.name: {'bytes': destination.stat().st_size,
        'sha256': sha256(destination),
        'caption': 'Four matched full-batch GD learning rates: (a) unsmoothed linear training loss; '
                   '(b) raw adjacent cosine and (c) eta-kappa, both 100-pair trailing means; '
                   '(d) eta times largest algebraic Hessian eigenvalue; (e) full-prefix raw B/A; '
                   '(f) unsmoothed raw-gradient norm on log Y. All 20k updates are retained.'}}
    write_json(HERE / 'gd_figure_manifest.json', manifest)
    print(json.dumps({'figure': str(destination), 'curves_checked': 24,
                      'mean_values_checked': manifest['mean_values_checked'], 'archived_pdfs': 4}))


if __name__ == '__main__':
    main()
