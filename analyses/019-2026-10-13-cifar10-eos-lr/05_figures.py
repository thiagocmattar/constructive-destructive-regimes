"""Four full-horizon views; source-array hashes and complete-window checks."""
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
FIGURES = HERE / 'figures'
FIGURES.mkdir(exist_ok=True)
COHORT = json.loads((HERE / 'cohort.json').read_text())
RESULTS = json.loads((HERE / 'results.json').read_text())
PARSER = argparse.ArgumentParser()
PARSER.add_argument('--gd-only', action='store_true', help='Review the completed GD cohort while Adam continues.')
ARGS = PARSER.parse_args()
assert (ARGS.gd_only and len(RESULTS['runs']) >= 4) or (RESULTS['status'] == 'passed' and len(RESULTS['runs']) == 8)
RECORDS = COHORT['runs']
DATA = {r['run_id']: dict(np.load(HERE / 'derived' / ('Run' + r['run_id'] + '.npz'))) for r in RECORDS if (HERE / 'derived' / ('Run' + r['run_id'] + '.npz')).exists()}
SUMMARY = {r['run_id']: r for r in RESULTS['runs']}
COLORS = ['#0072B2', '#009E73', '#D55E00', '#6A3D9A']
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'legend.fontsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'path.simplify': False, 'savefig.dpi': 180})
MANIFEST = {'coverage': 'All20,000 updates; unsmoothed train states0..19,999 plus final state20,000; test states1..20,000; Hessian states0..20,000 every100.',
            'smoothing': 'Cosines and eta-kappa: complete100-pair trailing means; norm view: complete10-update raw means. No padded or shortened windows.',
            'momentum': 'mhat=m/(1-beta1^t) is the primary momentum; stored m is labeled separately.',
            'figures': {}, 'curves': [], 'mean_values_checked': 0}


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest()


def mean(x, window):
    result = np.full(len(x), np.nan)
    first = 1 if np.isnan(x[0]) else 0
    assert np.isfinite(x[first:]).all()
    cumulative = np.r_[0., np.cumsum(x[first:], dtype=np.float64)]
    result[first + window - 1:] = (cumulative[window:] - cumulative[:-window]) / window
    # Independent direct window sums cover every displayed mean.
    reference = np.lib.stride_tricks.sliding_window_view(x[first:], window).mean(axis=1)
    np.testing.assert_allclose(result[first + window - 1:], reference, rtol=1e-9, atol=2e-10)
    MANIFEST['mean_values_checked'] += len(reference)
    return result


def plot(ax, record, x, y, metric, *, style='-', alpha=1., width=1.2, label=None, window=None, raster=False):
    original = y
    if window:
        y = mean(y, window)
    color = COLORS[RECORDS.index(record) % 4]
    line, = ax.plot(x, y, style, color=color, alpha=alpha, linewidth=width, label=label, rasterized=raster)
    np.testing.assert_array_equal(line.get_xdata(), x)
    np.testing.assert_array_equal(line.get_ydata(), y)
    MANIFEST['curves'].append({'run_id': record['run_id'], 'panel': ax.get_title(), 'metric': metric,
                               'points': len(x), 'finite_points': int(np.isfinite(y).sum()),
                               'x_sha256': digest(x), 'source_y_sha256': digest(original),
                               'plotted_y_sha256': digest(y), 'trailing_window': window})
    return y


def setup(ax, title, ylabel, *, log=False, cosine=False, ratio=False):
    ax.set_title(title, loc='left', pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Parameter state / optimizer update')
    ax.set_xlim(0, 20000)
    ax.set_xticks([0, 5000, 10000, 15000, 20000], ['0', '5k', '10k', '15k', '20k'])
    ax.grid(True, alpha=.18, linewidth=.6)
    if log:
        ax.set_yscale('log')
    if cosine:
        ax.set_ylim(-1.04, 1.04)
        ax.axhline(0, color='.45', linewidth=.7, linestyle=':')
    if ratio:
        ax.set_ylim(-.02, 1.03)


def lr_label(record):
    return f"LR {record['lr']:.6g}"


def lr_legend(ax, records):
    ax.legend([Line2D([], [], color=COLORS[i], linewidth=2) for i in range(4)],
              [lr_label(r) for r in records], ncol=4, loc='upper right', framealpha=.9)


def styles(ax, values, loc='best'):
    ax.legend([Line2D([], [], color='.2', linestyle=s) for _, s in values],
              [label for label, _ in values], loc=loc, framealpha=.9)


def save(fig, name, caption):
    fig.savefig(FIGURES / name, bbox_inches='tight')
    path = FIGURES / name
    MANIFEST['figures'][name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'caption': caption}
    plt.close(fig)


def loss(ax, records, title):
    setup(ax, title, 'Squared training loss', log=True)
    for record in records:
        data, result = DATA[record['run_id']], SUMMARY[record['run_id']]
        x = np.r_[data['train_state'], 20000]
        y = np.r_[data['train_loss'], result['final_train']['loss']]
        plot(ax, record, x, y, 'train_loss_with_final_state', alpha=.85, raster=True)
    lr_legend(ax, records)


def overview():
    fig, axes = plt.subplots(8, 2, figsize=(16, 25), layout='constrained')
    fig.suptitle('CIFAR-10 full-batch learning-rate sweep | GD and Adam\n5,000 training images | tanh 200-200 MLP | one frozen seed | 20,000 updates', fontsize=18)
    for column, method in enumerate(['gd', 'adam']):
        records = [r for r in RECORDS if r['optimizer'] == method]
        loss(axes[0, column], records, '(a) GD: unsmoothed training loss' if method == 'gd' else '(b) Adam: unsmoothed training loss')
        labels = [('Training and test accuracy', 'Accuracy', False),
                  ('Raw-gradient adjacent cosine | 100 pairs', 'Cosine', False),
                  ('Update adjacent cosine | 100 pairs' if method == 'gd' else 'Momentum adjacent cosine | 100 pairs', 'Cosine', False),
                  ('Ordinary Hessian: two largest eigenvalues', 'Eigenvalue', True),
                  ('GD spectral reference' if method == 'gd' else 'Adam preconditioned spectral reference', 'Normalized largest eigenvalue', False),
                  ('Raw-gradient cumulative displacement', 'B / A', False),
                  ('Gradient identity | 100 pairs', r'$\eta\hat\kappa$', False)]
        for row, (title, ylabel, log) in enumerate(labels, 1):
            setup(axes[row, column], f'({chr(97 + row * 2 + column)}) {title}', ylabel,
                  log=log, cosine=row in [2, 3], ratio=row == 6)
        for record in records:
            d = DATA[record['run_id']]
            x = d['update']
            plot(axes[1, column], record, d['train_state'], d['train_accuracy'], 'train_accuracy', width=1)
            plot(axes[1, column], record, d['test_state'], d['test_accuracy'], 'test_accuracy', style='--', width=1)
            plot(axes[2, column], record, x, d['raw_cosine'], 'raw_cosine', window=100)
            stream = 'displacement' if method == 'gd' else 'unbiased_momentum'
            plot(axes[3, column], record, x, d[stream + '_cosine'], stream + '_cosine', window=100)
            for i, style in enumerate(['-', '--']):
                plot(axes[4, column], record, d['curvature_state'], d['hessian_eigenvalues'][:, i], f'hessian_lambda_{i+1}', style=style)
            if method == 'gd':
                spectral_x, spectral_y = d['curvature_state'], record['lr'] * d['hessian_eigenvalues'][:, 0] / 2
                metric = 'eta_lambda_max_over_2'
            else:
                spectral_x, spectral_y = d['preconditioned_state'], record['lr'] * d['preconditioned_eigenvalues'][:, 0] / 38
                metric = 'eta_preconditioned_lambda_max_over_38'
            plot(axes[5, column], record, spectral_x, spectral_y, metric)
            plot(axes[6, column], record, x, d['raw_B_over_A'], 'raw_B_over_A')
            plot(axes[7, column], record, x, d['eta_kappa_proxy'], 'eta_kappa_proxy', window=100)
        styles(axes[1, column], [('Train (5k)', '-'), ('Test (10k)', '--')])
        styles(axes[4, column], [(r'$\lambda_1$', '-'), (r'$\lambda_2$', '--')])
        axes[5, column].axhline(1, color='.2', linestyle='--', linewidth=1.2)
        axes[5, column].text(.02, .96, 'Reference = 1', transform=axes[5, column].transAxes, va='top', fontsize=10, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
        axes[7, column].axhline(2, color='.5', linestyle=':', linewidth=.8)
        axes[7, column].text(.02, .96, 'Dotted line: 2 (visual guide)', transform=axes[7, column].transAxes, va='top', fontsize=10, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
    fig.supxlabel('Train and gradient state t-1; moments and displacement at update t; test and Hessian at their labeled state.\nAdam: P=(1-beta1^t)(sqrt(vhat)+epsilon); 38/eta is a frozen-preconditioner local reference. Kappa is a proxy, not an Adam curvature identity.', fontsize=10)
    save(fig, 'cifar10_eos_comparison.pdf', 'Matched GD/Adam learning-rate comparison: unsmoothed loss, full train/test accuracy,100-pair raw/update or mhat cosine,ordinary top2 eigenvalues,optimizer-specific spectral reference,raw B/A,and100-pair eta-kappa. No shared LR values or universal stability inference.')


def geometry(method):
    records = [r for r in RECORDS if r['optimizer'] == method]
    is_adam = method == 'adam'
    definitions = [('Raw-gradient adjacent cosine | 100 pairs', 'Cosine', ['raw_cosine'], False, 100),
                   ('Momentum adjacent cosine | 100 pairs' if is_adam else 'Actual-update adjacent cosine | 100 pairs', 'Cosine', ['unbiased_momentum_cosine' if is_adam else 'displacement_cosine'], False, 100),
                   ('Raw-gradient and momentum norms' if is_adam else 'Raw-gradient norm', 'Vector norm', ['raw_norm', 'unbiased_momentum_norm', 'momentum_norm'] if is_adam else ['raw_norm'], True, None),
                   ('Adaptive direction and actual update norms' if is_adam else 'Actual-update norm', 'Vector norm', ['adaptive_norm', 'displacement_norm'] if is_adam else ['displacement_norm'], True, None),
                   ('Raw-gradient cumulative lengths', 'Cumulative length', ['raw_A', 'raw_B'], True, None),
                   ('Momentum cumulative lengths' if is_adam else 'Actual-update cumulative lengths', 'Cumulative length', ['unbiased_momentum_A', 'unbiased_momentum_B'] if is_adam else ['displacement_A', 'displacement_B'], True, None),
                   ('Raw-gradient cumulative displacement', 'B / A', ['raw_B_over_A'], False, None),
                   ('Momentum cumulative displacement' if is_adam else 'Actual-update cumulative displacement', 'B / A', ['unbiased_momentum_B_over_A', 'momentum_B_over_A'] if is_adam else ['displacement_B_over_A'], False, None)]
    if is_adam:
        definitions += [('Adaptive and actual-update displacement', 'B / A', ['adaptive_B_over_A', 'displacement_B_over_A'], False, None),
                        ('Raw-gradient / momentum alignment', 'Cosine', ['unbiased_momentum_raw_cosine'], False, 100),
                        ('Adaptive and actual-update adjacent cosine | 100 pairs', 'Cosine', ['adaptive_cosine', 'displacement_cosine'], False, 100),
                        ('Vector norm ratios', 'Ratio', ['momentum_to_raw_norm_ratio', 'adaptive_to_raw_norm_ratio'], True, None)]
    rows = len(definitions) // 2 + 1
    fig = plt.figure(figsize=(16, 3.2 * rows), layout='constrained')
    grid = fig.add_gridspec(rows, 2)
    top = fig.add_subplot(grid[0, :])
    loss(top, records, '(a) Unsmoothed full-batch training loss')
    fig.suptitle(f'CIFAR-10 | {method.upper() if method == "gd" else "Adam"} geometry | four learning rates | 20,000 updates', fontsize=18)
    display = {'raw': 'Raw gradient', 'momentum': 'Stored m', 'unbiased_momentum': 'Bias-corrected mhat',
               'adaptive': 'Adaptive direction d', 'displacement': 'Actual update'}
    for i, (title, ylabel, keys, log, window) in enumerate(definitions):
        ax = fig.add_subplot(grid[1 + i // 2, i % 2])
        setup(ax, f'({chr(98+i)}) {title}', ylabel, log=log, cosine=ylabel == 'Cosine', ratio=ylabel == 'B / A')
        style_list = ['-', '--', ':']
        for record in records:
            d = DATA[record['run_id']]
            for j, key in enumerate(keys):
                plot(ax, record, d['update'], d[key], key, style=style_list[j], window=window, raster=window is None and '_norm' in key, alpha=.85)
        if len(keys) > 1:
            labels = []
            for key, style in zip(keys, style_list):
                if (key.endswith('_A') or key.endswith('_B')) and not key.endswith('_B_over_A'):
                    label = key[-1]
                elif key.endswith('_to_raw_norm_ratio'):
                    label = 'Stored m / raw' if key.startswith('momentum') else 'Adaptive d / raw'
                else:
                    name = next(name for name in sorted(display, key=len, reverse=True) if key.startswith(name))
                    label = display[name]
                labels.append((label, style))
            styles(ax, labels)
    footer = 'Color identifies LR. A=sum of vector norms; B=norm of the vector sum; ratios use the full prefix.\n'
    if is_adam:
        footer += 'Momentum means bias-corrected mhat unless explicitly labeled stored m. '
    footer += 'Actual updates include FP32 rounding. All cosine means use 100 complete pairs.'
    fig.supxlabel(footer, fontsize=10)
    save(fig, f'cifar10_{method}_geometry.pdf', f'{method.upper()} full-horizon scalar geometry with unsmoothed loss,100-pair cosine means,log vector norms and cumulative A/B,linear B/A. Adam separates stored m,bias-corrected mhat,adaptive d and actual displacement; GD has no momentum.')


def norms():
    fig = plt.figure(figsize=(16, 17), layout='constrained')
    grid = fig.add_gridspec(5, 2)
    top = fig.add_subplot(grid[0, :])
    setup(top, '(a) Unsmoothed training loss | all eight runs', 'Squared training loss', log=True)
    handles, labels = [], []
    for record in RECORDS:
        d, result = DATA[record['run_id']], SUMMARY[record['run_id']]
        style = '-' if record['optimizer'] == 'gd' else '--'
        plot(top, record, np.r_[d['train_state'], 20000], np.r_[d['train_loss'], result['final_train']['loss']], 'train_loss_with_final_state', style=style, raster=True, alpha=.8)
        handles.append(Line2D([], [], color=COLORS[RECORDS.index(record) % 4], linestyle=style))
        labels.append(record['optimizer'].upper() + ' | ' + lr_label(record))
    top.legend(handles, labels, ncol=4, framealpha=.9, fontsize=10)
    norm_axes = []
    for i in range(4):
        for col in range(2):
            record = RECORDS[i + col * 4]
            d = DATA[record['run_id']]
            ax = fig.add_subplot(grid[i+1, col])
            norm_axes.append(ax)
            setup(ax, f'({chr(98+i*2+col)}) {record["optimizer"].upper()} | {lr_label(record)}', 'Vector norm', log=True)
            plot(ax, record, d['update'], d['raw_norm'], 'raw_norm', alpha=.15, width=.6, raster=True)
            smooth = plot(ax, record, d['update'], d['raw_norm'], 'raw_norm', window=10, width=1.5)
            if col == 1:
                other = d['unbiased_momentum_norm']
                plot(ax, record, d['update'], other, 'unbiased_momentum_norm', style='--', width=1.)
                fill_label = 'Raw / mhat gap'
                labels = [('Raw gradient | 10-update mean', '-'), ('Bias-corrected momentum mhat', '--')]
            else:
                other = d['displacement_norm'] / record['lr']
                plot(ax, record, d['update'], other, 'actual_update_norm_divided_by_lr', style='--', width=.9)
                fill_label = 'Raw / update-scale gap'
                labels = [('Raw gradient | 10-update mean', '-'), ('Actual update norm / LR', '--')]
            ax.fill_between(d['update'], smooth, other, alpha=.12, color=COLORS[i], label=fill_label)
            styles(ax, labels)
    displayed_norms = []
    for record in RECORDS:
        d = DATA[record['run_id']]
        displayed_norms.append(d['raw_norm'])
        displayed_norms.append(d['unbiased_momentum_norm'] if record['optimizer'] == 'adam' else d['displacement_norm'] / record['lr'])
    low = min(norm.min() for norm in displayed_norms)
    high = max(norm.max() for norm in displayed_norms)
    for ax in norm_axes:
        ax.set_ylim(low * .5, high * 2)
    fig.suptitle('CIFAR-10 | loss and vector norms by learning rate\nGD (left) and Adam (right) | common logarithmic norm scale', fontsize=18)
    fig.supxlabel('Faint raw samples remain visible beneath a complete 10-update trailing mean; fill spans that mean and the dashed curve.\nGD has no momentum: its dashed curve is actual update norm / LR. Adam uses bias-corrected mhat. All 20,000 raw samples are included.', fontsize=10)
    save(fig, 'loss_and_norms_by_learning_rate.pdf', 'Unsmoothed loss followed by one panel per run,common log norm Y limits,faint original raw-gradient norms,10-update raw trailing means,and fill to unsmoothed mhat on Adam or actual-update norm/LR on GD.')


if __name__ == '__main__':
    if ARGS.gd_only:
        geometry('gd')
        destination = HERE / 'scratch/gd_figure_manifest.json'
    else:
        overview()
        geometry('gd')
        geometry('adam')
        norms()
        destination = HERE / 'figure_manifest.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(MANIFEST, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'figures': list(MANIFEST['figures']), 'curves_checked': len(MANIFEST['curves']),
                      'mean_values_checked': MANIFEST['mean_values_checked']}))
