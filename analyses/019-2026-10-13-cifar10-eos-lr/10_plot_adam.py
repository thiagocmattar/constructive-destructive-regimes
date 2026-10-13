"""Full and early-step Adam counterparts of the six-panel GD figure."""
import argparse
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
COLORS, WINDOW = GD.COLORS, GD.WINDOW
sha256, digest, write_json, trailing_mean = GD.sha256, GD.digest, GD.write_json, GD.trailing_mean
DEFINITIONS = [
    ('(a) Unsmoothed training loss', r'$f(\theta)$', 'train_loss_with_final_state', None),
    ('(b) Raw-gradient adjacent cosine', r'$\rho_t$', 'raw_cosine', WINDOW),
    ('(c) Momentum adjacent cosine', r'$\cos(\widehat m_{t-1},\widehat m_t)$', 'unbiased_momentum_cosine', WINDOW),
    (r'(d) $\eta\,\widehat{\kappa}_t$', r'$\eta\,\widehat{\kappa}_t$', 'eta_kappa_proxy', WINDOW),
    (r'(e) $\eta\,\lambda_{\max}$', r'$\eta\,\lambda_{\max}$', 'eta_lambda_max', None),
    ('(f) Raw-gradient cumulative displacement', r'$B_t/A_t$ (raw)', 'raw_B_over_A', None),
    ('(g) Momentum cumulative displacement', r'$B_t/A_t$ (momentum)', 'unbiased_momentum_B_over_A', None),
    ('(h) Raw-gradient norm', r'$\|g_t\|_2$', 'raw_norm', None),
    ('(i) Momentum norm', r'$\|\widehat m_t\|_2$', 'unbiased_momentum_norm', None),
    ('(j) Gradient / momentum norm', r'$\|g_t\|_2/\|\widehat m_t\|_2$', 'raw_to_unbiased_momentum_norm_ratio', None)]


def load_data():
    assert json.loads((HERE / 'verification.json').read_text())['status'] == 'passed'
    results = json.loads((HERE / 'results.json').read_text())
    assert results['status'] == 'passed'
    summary = {r['run_id']: r for r in results['runs']}
    records = [r for r in json.loads((HERE / 'cohort.json').read_text())['runs'] if r['optimizer'] == 'adam']
    assert [r['run_id'] for r in records] == ['063', '064', '065', '066']
    sources, data = [], {}
    for record in records:
        run_id = record['run_id']
        path = HERE / 'derived' / f'Run{run_id}.npz'
        verified = json.loads((HERE / 'verification' / f'Run{run_id}.json').read_text())
        assert verified['status'] == 'passed'
        assert path.stat().st_size == verified['derived_bytes'] and sha256(path) == verified['derived_sha256']
        with np.load(path, allow_pickle=False) as values:
            d = dict(values)
        np.testing.assert_array_equal(d['update'], np.arange(1, 20001))
        np.testing.assert_array_equal(d['train_state'], np.arange(20000))
        np.testing.assert_array_equal(d['curvature_state'], np.arange(0, 20001, 100))
        assert d['hessian_eigenvalues'].shape == (201, 2)
        assert np.all(d['hessian_eigenvalues'][:, 0] >= d['hessian_eigenvalues'][:, 1])
        assert np.all(d['raw_norm'] > 0) and np.all(d['unbiased_momentum_norm'] > 0)
        np.testing.assert_allclose(d['unbiased_momentum_norm'], d['momentum_norm'] / (1 - .9**d['update']), rtol=2e-14)
        np.testing.assert_allclose(d['eta_kappa_proxy'][1:],
            1 - d['raw_cosine'][1:] * d['raw_norm'][1:] / d['raw_norm'][:-1], rtol=1e-10, atol=1e-12)
        for stream in ['raw', 'unbiased_momentum']:
            np.testing.assert_allclose(d[stream + '_B_over_A'], d[stream + '_B'] / d[stream + '_A'], rtol=2e-14)
        d['raw_to_unbiased_momentum_norm_ratio'] = d['raw_norm'] / d['unbiased_momentum_norm']
        # The recorded ratio uses stored m/raw: invert it and apply bias correction.
        np.testing.assert_allclose(d['raw_to_unbiased_momentum_norm_ratio'],
            (1 - .9**d['update']) / d['momentum_to_raw_norm_ratio'], rtol=2e-14)
        series = {}
        for _, _, metric, window in DEFINITIONS:
            x = d['update']
            if metric == 'train_loss_with_final_state':
                x = np.r_[d['train_state'], 20000]
                raw = np.r_[d['train_loss'], summary[run_id]['final_train']['loss']]
            elif metric == 'eta_lambda_max':
                x, raw = d['curvature_state'], record['lr'] * d['hessian_eigenvalues'][:, 0]
            else:
                raw = d[metric]
            plotted = trailing_mean(raw) if window else raw
            series[metric] = (x, raw, plotted)
        data[run_id] = series
        sources.append({'run_id': run_id, 'learning_rate': record['lr'],
                        'derived_path': path.relative_to(HERE).as_posix(),
                        'derived_sha256': verified['derived_sha256'],
                        'source_metrics_sha256': verified['source_metrics_sha256'],
                        'source_curvature_sha256': verified['source_curvature_sha256']})
    return records, data, sources


def draw(records, data, manifest, end, name):
    fig, axes = plt.subplots(5, 2, figsize=(14, 18))
    fig.subplots_adjust(left=.075, right=.98, bottom=.075, top=.91, hspace=.48, wspace=.22)
    subtitle = '20,000 updates' if end == 20000 else 'First 2,000 updates'
    fig.suptitle('CIFAR-10 | Full-batch Adam', fontsize=20, y=.988)
    fig.text(.5, .962, '5,000 training images | tanh 200-200 MLP | ' + subtitle, ha='center', fontsize=11)
    labels = [r'$\eta=5\times10^{-5}$', r'$\eta=10^{-4}$', r'$\eta=2\times10^{-4}$', r'$\eta=4\times10^{-4}$']
    fig.legend([Line2D([], [], color=c, linewidth=2.5) for c in COLORS], labels,
               ncol=4, loc='upper center', bbox_to_anchor=(.5, .95), frameon=False, fontsize=12)
    for ax, (title, ylabel, metric, window) in zip(axes.flat, DEFINITIONS):
        ax.set_title(title, loc='left', pad=10)
        ax.set(xlabel='Step', ylabel=ylabel, xlim=(0, end))
        ticks = [0, 5000, 10000, 15000, 20000] if end == 20000 else [0, 500, 1000, 1500, 2000]
        labels = ['0', '5k', '10k', '15k', '20k'] if end == 20000 else ['0', '500', '1k', '1.5k', '2k']
        ax.set_xticks(ticks, labels)
        ax.grid(axis='y', color='#E2E7EC', linewidth=.8)
        ax.set_axisbelow(True)
        if 'cosine' in metric:
            ax.set_ylim(-1.08, 1.08)
            ax.set_yticks([-1, 0, 1])
        elif metric.endswith('_B_over_A'):
            ax.set_ylim(-.02, 1.04)
        elif metric.endswith('_norm'):
            ax.set_yscale('log')
            ax.set_ylim(.0015, 8.)
        elif metric == 'train_loss_with_final_state':
            ax.set_ylim(0, 1.06)
        elif metric == 'eta_kappa_proxy':
            ax.set_ylim(-.08, 2.3)
            ax.set_yticks([0, 1, 2])
        elif metric == 'eta_lambda_max':
            ax.set_ylim(0, .1)
        elif metric == 'raw_to_unbiased_momentum_norm_ratio':
            if end == 2000:
                ax.set_yscale('log')
                ax.set_ylim(.5, 78.)
            else:
                ax.set_ylim(0, 78.)
        for record, color in zip(records, COLORS):
            x, raw, plotted = data[record['run_id']][metric]
            mask = x <= end
            noisy = metric in ['train_loss_with_final_state', 'raw_norm', 'unbiased_momentum_norm',
                               'raw_to_unbiased_momentum_norm_ratio']
            line, = ax.plot(x[mask], plotted[mask], color=color, linewidth=1.25,
                            alpha=.9 if noisy else 1., rasterized=noisy)
            np.testing.assert_array_equal(line.get_xdata(), x[mask])
            np.testing.assert_array_equal(line.get_ydata(), plotted[mask])
            finite = plotted[mask][np.isfinite(plotted[mask])]
            low, high = ax.get_ylim()
            assert len(finite) and finite.min() >= low and finite.max() <= high, (metric, low, high)
            manifest['curves'].append({'figure': name, 'run_id': record['run_id'],
                'panel': title, 'metric': metric, 'points': int(mask.sum()),
                'finite_points': len(finite), 'x_sha256': digest(x[mask]),
                'source_y_sha256': digest(raw[mask]), 'plotted_y_sha256': digest(plotted[mask]),
                'trailing_window': window, 'y_scale': ax.get_yscale(), 'y_limits': [low, high]})
    fig.text(.075, .046, r'Momentum is bias-corrected: $\widehat m_t=m_t/(1-0.9^t)$. Cosines and $\eta\widehat\kappa$: 100-pair trailing means.', fontsize=10.5)
    scale_note = 'Norms and the ratio use log Y' if end == 2000 else 'Only norms use log Y'
    fig.text(.075, .029, 'Loss, norms and norm ratio are unsmoothed. ' + scale_note + '; other panels use linear Y.', fontsize=10.5)
    fig.text(.075, .012, r'Ordinary Hessian every 100 steps. $\eta\widehat\kappa$ is a gradient trajectory proxy along Adam.', fontsize=10.5)
    assert len(fig.axes) == 10
    scratch = HERE / 'scratch'
    scratch.mkdir(exist_ok=True)
    pending = scratch / (name + '.pending.pdf')
    fig.savefig(pending)
    plt.close(fig)
    destination = HERE / 'figures' / name
    pending.replace(destination)
    manifest['figures'][name] = {'bytes': destination.stat().st_size, 'sha256': sha256(destination),
        'panels': 10, 'x_limits': [0, end], 'script_sha256': sha256(__file__),
        'caption': f'Four matched Adam learning rates, states/updates 0..{end}: unsmoothed linear loss; '
                   '100-pair raw/mhat adjacent cosine and eta-kappa; eta times ordinary lambda_max; '
                   'full-prefix raw/mhat B/A; unsmoothed log raw/mhat norms; '
                   + ('log' if end == 2000 else 'linear') + ' raw/mhat norm ratio.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--early-only', action='store_true', help='Update only the early-step PDF; preserve the full-horizon figure.')
    args = parser.parse_args()
    records, data, sources = load_data()
    gd_path = HERE / 'figures/cifar10_gd_diagnostics.pdf'
    gd_hash = sha256(gd_path)
    archive_hash = sha256(HERE / 'archive_manifest.json')
    manifest = {'status': 'passed', 'script': Path(__file__).name, 'script_sha256': sha256(__file__),
        'shared_helper': '09_plot_gd.py', 'shared_helper_sha256': sha256(HERE / '09_plot_gd.py'),
        'learning_rates': [r['lr'] for r in records], 'dimensions_inches': [14, 18],
        'sources': sources, 'curves': [], 'figures': {}, 'thinning': None,
        'momentum': 'Bias-corrected first moment mhat=m/(1-beta1^t), beta1=0.9; no second-moment division.',
        'norm_ratio': 'Raw gradient norm / bias-corrected momentum norm at matching update t; unsmoothed.',
        'zero_denominator': 'Undefined if mhat norm is zero; all 80,000 measured denominators are positive.',
        'smoothing': 'Only raw/mhat adjacent cosine and eta-kappa: complete 100-pair right-aligned means, first update 101.',
        'unique_mean_values_checked': 238800, 'displayed_mean_values': 261600,
        'curvature': 'eta*largest algebraic eigenvalue of the ordinary objective Hessian; no preconditioning or division by 2.',
        'kappa': 'eta*kappa_hat=1-rho_t*norm(g_t)/norm(g_(t-1)); explicit noise term zero; trajectory proxy on Adam.',
        'gd_pdf_sha256_unchanged': gd_hash, 'archive_manifest_sha256_unchanged': archive_hash}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 13, 'axes.labelsize': 12, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'path.simplify': False, 'savefig.dpi': 180})
    full_path = HERE / 'figures/cifar10_adam_diagnostics.pdf'
    if args.early_only:
        prior = json.loads((HERE / 'adam_figure_manifest.json').read_text())
        assert sources == prior['sources']
        full_item = dict(prior['figures'][full_path.name])
        assert sha256(full_path) == full_item['sha256']
        full_item.setdefault('script_sha256', prior['script_sha256'])
        manifest['figures'][full_path.name] = full_item
        manifest['curves'] = [c for c in prior['curves'] if c['figure'] == full_path.name]
        manifest['full_pdf_sha256_unchanged'] = full_item['sha256']
    else:
        draw(records, data, manifest, 20000, full_path.name)
    draw(records, data, manifest, 2000, 'cifar10_adam_diagnostics_early.pdf')
    assert len(manifest['curves']) == 80
    assert sum(c['finite_points'] for c in manifest['curves'] if c['trailing_window']) == 261600
    assert sha256(gd_path) == gd_hash and sha256(HERE / 'archive_manifest.json') == archive_hash
    if args.early_only:
        assert sha256(full_path) == manifest['full_pdf_sha256_unchanged']
    write_json(HERE / 'adam_figure_manifest.json', manifest)
    print(json.dumps({'figures': list(manifest['figures']), 'curves_checked': 80,
                      'unique_means_checked': 238800, 'early_horizon': 2000}))


if __name__ == '__main__':
    main()
