"""Two publication figures from frozen Rosenbrock histories; no optimizer rerun."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
import numpy as np

from kappa import HERE, ROOT, digest, load_sources, rolling, sha256, write_json

COLORS = ('#0072B2', '#8C4AA8', '#D55E00')
LABELS = (r'$\eta=10^{-2}$', r'$\eta=3.16\times10^{-3}$', r'$\eta=10^{-3}$')
WINDOW = 100
FIGURES = ('rosenbrock_trajectories_paper.pdf', 'rosenbrock_diagnostics_paper.pdf')


def verify_momentum_identity():
    """Independent quadratic case, with nonparallel momentum and update direction."""
    m, g = np.array([.7, -.2]), np.array([1., .4])
    h = np.array([[2., .3], [.3, 4.]])
    d, eta, beta = np.array([.2, .5]), .03, .9
    current_g = g-eta*h @ d
    current_m = beta*m+(1-beta)*current_g
    cosine = m @ current_m/(np.linalg.norm(m)*np.linalg.norm(current_m))
    tracking = m @ (g-m)/(m @ m)
    inferred = tracking+(1-cosine*np.linalg.norm(current_m)/np.linalg.norm(m))/(1-beta)
    np.testing.assert_allclose(inferred, eta*(m @ h @ d)/(m @ m), rtol=1e-12)
    np.testing.assert_allclose(inferred, m @ (g-current_g)/(m @ m), rtol=1e-12)


def reduce(history, source, prior):
    cfg = json.loads((ROOT/source['directory']/'config.json').read_text())
    assert cfg['updates'] == 2000 and cfg['initial_point'] == [0., 1.]
    assert cfg['betas'] == [.9, .95] and cfg['epsilon'] == 1e-8 and cfg['weight_decay'] == .1
    eta, beta1 = cfg['learning_rate'], cfg['betas'][0]
    theta, g, m, d = (history[k] for k in ('theta', 'raw_vectors', 'momentum_vectors', 'adaptive_vectors'))
    x, y = theta.T
    np.testing.assert_allclose(history['loss'], (1-x)**2+100*(y-x*x)**2, rtol=1e-13)
    gradient = np.column_stack((2*(x[:-1]-1)-400*x[:-1]*(y[:-1]-x[:-1]**2), 200*(y[:-1]-x[:-1]**2)))
    np.testing.assert_allclose(g, gradient, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(m, beta1*np.vstack((np.zeros(2), m[:-1]))+(1-beta1)*g, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(theta[1:], theta[:-1]*(1-eta*cfg['weight_decay'])-eta*d, rtol=1e-13, atol=1e-14)
    h = history['hessian']
    analytic_h = np.empty_like(h)
    analytic_h[:, 0, 0] = 2-400*y+1200*x*x
    analytic_h[:, 0, 1] = analytic_h[:, 1, 0] = -400*x
    analytic_h[:, 1, 1] = 200
    np.testing.assert_allclose(h, analytic_h, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(history['hessian_lambda_max'], np.linalg.eigvalsh(h)[:, -1], rtol=1e-13)
    for name, vectors in [('raw', g), ('momentum', m)]:
        norm = np.linalg.norm(vectors, axis=1)
        a = np.cumsum(norm)
        b = np.linalg.norm(np.cumsum(vectors, axis=0), axis=1)
        np.testing.assert_allclose(history[name+'_norm'], norm, rtol=1e-13)
        np.testing.assert_allclose(history[name+'_ratio'], b/a, rtol=1e-13)
        cos = np.r_[np.nan, np.sum(vectors[:-1]*vectors[1:], axis=1)/(norm[:-1]*norm[1:])]
        np.testing.assert_allclose(history[name+'_cosine'], cos, rtol=1e-13, atol=1e-14, equal_nan=True)
    derived = ROOT/prior['derived_history']
    assert sha256(derived) == prior['derived_history_sha256']
    with np.load(derived, allow_pickle=False) as saved:
        raw_eta = eta*saved['kappa_proxy']
    raw_direct = np.r_[np.nan, np.sum(g[:-1]*(g[:-1]-g[1:]), axis=1)/np.sum(g[:-1]**2, axis=1)]
    np.testing.assert_allclose(raw_eta, raw_direct, rtol=3e-10, atol=1e-12, equal_nan=True)
    m2 = np.sum(m[:-1]**2, axis=1)
    tracking = np.sum(m[:-1]*(g[:-1]-m[:-1]), axis=1)/m2
    momentum_eta = np.r_[np.nan, np.sum(m[:-1]*(g[:-1]-g[1:]), axis=1)/m2]
    inversion = np.r_[np.nan, tracking+(1-history['momentum_cosine'][1:]*history['momentum_norm'][1:]/history['momentum_norm'][:-1])/(1-beta1)]
    np.testing.assert_allclose(momentum_eta, inversion, rtol=2e-9, atol=2e-11, equal_nan=True)
    delta = np.diff(theta, axis=0)[:-1]
    remainder = g[1:]-g[:-1]-np.einsum('nij,nj->ni', h[:-2], delta)
    curvature = eta*np.einsum('ni,nij,nj->n', m[:-1], h[:-2], d[:-1])/m2
    decay = eta*cfg['weight_decay']*np.einsum('ni,nij,nj->n', m[:-1], h[:-2], theta[:-2])/m2
    nonlinear = -np.sum(m[:-1]*remainder, axis=1)/m2
    np.testing.assert_allclose(momentum_eta[1:], curvature+decay+nonlinear, rtol=2e-9, atol=2e-11)
    values = dict(state=np.arange(2001), update=np.arange(1, 2001), theta=theta, loss=history['loss'],
        raw_norm=history['raw_norm'], momentum_norm=history['momentum_norm'],
        raw_cosine=history['raw_cosine'], momentum_cosine=history['momentum_cosine'],
        eta_lambda_max=eta*history['hessian_lambda_max'], raw_eta_kappa=raw_eta,
        momentum_eta_kappa=momentum_eta, raw_ratio=history['raw_ratio'], momentum_ratio=history['momentum_ratio'],
        momentum_tracking=np.r_[np.nan, tracking], momentum_eta_curvature=np.r_[np.nan, curvature],
        momentum_eta_decay=np.r_[np.nan, decay], momentum_nonlinear_remainder=np.r_[np.nan, nonlinear])
    for metric in ('raw_cosine', 'momentum_cosine', 'raw_eta_kappa', 'momentum_eta_kappa'):
        average = rolling(values[metric], WINDOW)
        expected = np.array([values[metric][i-99:i+1].mean() for i in range(100, 2000)])
        np.testing.assert_allclose(average[100:], expected, rtol=2e-12, atol=2e-13)
        assert np.isfinite(average).sum() == 1900 and np.isnan(average[:100]).all()
        values[metric+'_mean100'] = average
    overlay = json.loads((HERE/'figure_manifest_rosenbrock_kappa_eta_overlay.json').read_text())
    original = next(v for v in overlay['rendered_series'] if v['run_id'] == source['run_id'])
    assert digest(values['raw_eta_kappa_mean100']) == original['plotted_y_sha256']
    destination = HERE/'artifacts'/f"{source['run_id']}_paper.npz"
    np.savez_compressed(destination, **values)
    row = dict(run_id=source['run_id'], learning_rate=eta, final_point=theta[-1].tolist(),
        final_loss=float(values['loss'][-1]), final_eta_lambda_max=float(values['eta_lambda_max'][-1]),
        final_raw_ratio=float(values['raw_ratio'][-1]), final_momentum_ratio=float(values['momentum_ratio'][-1]),
        late100={name: float(np.nanmean(values[name][-100:])) for name in
            ('raw_cosine', 'momentum_cosine', 'raw_eta_kappa', 'momentum_eta_kappa')},
        momentum_estimate_range=[float(np.nanmin(momentum_eta)), float(np.nanmax(momentum_eta))],
        derived=destination.relative_to(ROOT).as_posix(), derived_sha256=sha256(destination),
        arrays={name: dict(shape=list(v.shape), sha256=digest(v)) for name, v in values.items()},
        verification='Source/reconstruction, Eq.25 inversion, decay/remainder decomposition and all rolling means passed.')
    return values, row


def record_line(records, ax, x, y, color, *, run, panel, metric, style='-', width=1.35):
    line, = ax.plot(x, y, color=color, ls=style, lw=width, rasterized=True)
    np.testing.assert_array_equal(line.get_xdata(), x)
    np.testing.assert_array_equal(line.get_ydata(), y)
    records.append(dict(run_id=run, panel=panel, metric=metric, points=len(x),
        finite_points=int(np.isfinite(y).sum()), x_sha256=digest(line.get_xdata()),
        y_sha256=digest(line.get_ydata()), linestyle=line.get_linestyle()))


def trajectories(histories, sources):
    fig, ax = plt.subplots(figsize=(8.4, 5.9))
    fig.subplots_adjust(left=.08, right=.88, bottom=.105, top=.89)
    bounds = (-.12, 1.22, .22, 1.16)
    x = np.linspace(bounds[0], bounds[1], 600)
    y = np.linspace(bounds[2], bounds[3], 500)
    xx, yy = np.meshgrid(x, y)
    z = (1-xx)**2+100*(yy-xx*xx)**2
    levels = np.geomspace(1e-4, 300, 24)
    cmap = LinearSegmentedColormap.from_list('quiet_objective', ['#FFFFFF', '#F0F3F5', '#CED7DE'])
    background = ax.contourf(xx, yy, z, levels=levels, norm=LogNorm(1e-4, 300), cmap=cmap, extend='both')
    contours = ax.contour(xx, yy, z, levels=[.001, .01, .1, 1, 10, 100], colors='#7E8B97', linewidths=.55, alpha=.8)
    for level, position in [(1, (.45, .27)), (10, (.2, .5)), (100, (.26, 1.08))]:
        ax.clabel(contours, levels=[level], manual=[position], fmt='%g', fontsize=8, inline_spacing=5)
    records, arrows = [], []
    for values, source, color in zip(histories, sources, COLORS):
        path = values['theta']
        assert np.all((path[:, 0] > bounds[0]) & (path[:, 0] < bounds[1]))
        assert np.all((path[:, 1] > bounds[2]) & (path[:, 1] < bounds[3]))
        record_line(records, ax, path[:, 0], path[:, 1], color, run=source['run_id'], panel='trajectory', metric='theta', width=1.8)
        ax.scatter(*path[-1], s=32, facecolor='white', edgecolor=color, linewidth=1.5, zorder=8)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        for fraction in (.25, .55, .75):
            i = min(int(np.searchsorted(arc, fraction*arc[-1])), len(path)-6)
            ax.annotate('', xy=path[i+5], xytext=path[i],
                arrowprops=dict(arrowstyle='->', color=color, lw=1.1, mutation_scale=11), zorder=7)
            arrows.append(dict(run_id=source['run_id'], from_state=i, to_state=i+5))
    ax.scatter(0, 1, marker='s', s=50, color='#26313D', zorder=10)
    ax.scatter(1, 1, marker='*', s=160, facecolor='#E6B349', edgecolor='#26313D', linewidth=.8, zorder=10)
    ax.annotate('Start (0, 1)', (0, 1), xytext=(9, 10), textcoords='offset points', fontsize=10)
    ax.annotate('Optimum (1, 1)', (1, 1), xytext=(-8, 17), textcoords='offset points', ha='right', fontsize=10)
    ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel='$x$', ylabel='$y$')
    ax.set_aspect('equal', adjustable='box')
    handles = [Line2D([], [], marker='s', color='none', markerfacecolor='#26313D', label='Starting point', markersize=6),
        Line2D([], [], marker='*', color='none', markerfacecolor='#E6B349', markeredgecolor='#26313D', label='Global optimum', markersize=10),
        Line2D([], [], marker='o', color='none', markerfacecolor='white', markeredgecolor='#26313D', label='Final point (2,000 updates)', markersize=6)]
    ax.legend(handles=handles, loc='lower right', fontsize=9.5, frameon=True, facecolor='white', framealpha=.95, edgecolor='#DCE2E7')
    fig.legend([Line2D([], [], color=c, lw=1.8) for c in COLORS], LABELS, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.49, .98))
    colorbar = fig.colorbar(background, ax=ax, fraction=.035, pad=.04, ticks=[.001, .01, .1, 1, 10, 100])
    colorbar.set_label('Objective level $f(x,y)$')
    return fig, dict(dimensions_inches=[8.4, 5.9], domain=list(bounds), series=records,
        arrow_state_indices=arrows, all_trajectory_states=2001, contours=contours.levels.tolist())


def diagnostics(histories, sources):
    fig, axes = plt.subplots(4, 2, figsize=(8.4, 8.8), sharex=True)
    fig.subplots_adjust(left=.105, right=.985, bottom=.105, top=.925, wspace=.31, hspace=.45)
    titles = ['(a) Loss', '(b) Gradient and momentum norms', '(c) Gradient alignment', '(d) Momentum alignment',
        '(e) Maximum curvature', '(f) Gradient directional curvature', '(g) Momentum directional curvature', '(h) Normalized accumulation']
    ylabels = [r'$f(x_t,y_t)$', 'Vector norm', r'$\rho_t$', r'$c_t$', r'$\eta\lambda_{\max}$',
        r'$\eta\widehat\kappa_t$', r'$\eta\widehat\kappa_t^{(m)}$', '$B_t/A_t$']
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_title(title, loc='left', pad=5)
        ax.set(ylabel=ylabel, xlim=(0, 2000))
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.grid(axis='y', color='#E2E7EB', lw=.55)
        ax.set_axisbelow(True)
    axes[0, 0].set_yscale('log')
    axes[0, 1].set_yscale('log')
    for ax in axes[1]:
        ax.set_ylim(-1.04, 1.04)
        ax.axhline(0, color='#9CA7B2', lw=.6)
    for ax in axes[-1]:
        ax.set_xlabel('Optimizer update')
    records = []
    definitions = [('a', 0, 'loss', 'state', '-'), ('b', 1, 'raw_norm', 'update', '--'),
        ('b', 1, 'momentum_norm', 'update', '-'), ('c', 2, 'raw_cosine_mean100', 'update', '-'),
        ('d', 3, 'momentum_cosine_mean100', 'update', '-'), ('e', 4, 'eta_lambda_max', 'state', '-'),
        ('f', 5, 'raw_eta_kappa_mean100', 'update', '-'), ('g', 6, 'momentum_eta_kappa_mean100', 'update', '-'),
        ('h', 7, 'raw_ratio', 'update', '--'), ('h', 7, 'momentum_ratio', 'update', '-')]
    for values, source, color in zip(histories, sources, COLORS):
        for panel, i, metric, clock, style in definitions:
            record_line(records, axes.flat[i], values[clock], values[metric], color,
                run=source['run_id'], panel=panel, metric=metric, style=style,
                width=1.0 if metric.endswith('norm') else 1.35)
    stream_legend = [Line2D([], [], color='#26313D', ls='--', lw=1.2, label='Raw gradient'),
        Line2D([], [], color='#26313D', lw=1.2, label='Momentum')]
    fig.legend(handles=stream_legend, fontsize=10.5, frameon=False, loc='lower center',
        ncol=2, bbox_to_anchor=(.545, .002))
    ratio = np.concatenate([v[k] for v in histories for k in ['raw_ratio', 'momentum_ratio']])
    padding = .07*(ratio.max()-ratio.min())
    axes[3, 1].set_ylim(ratio.min()-padding, ratio.max()+padding)
    axes[3, 1].set_yticks([.85, .90, .95, 1.00])
    fig.legend([Line2D([], [], color=c, lw=1.7) for c in COLORS], LABELS, loc='upper center',
        ncol=3, frameon=False, bbox_to_anchor=(.545, .988))
    return fig, dict(dimensions_inches=[8.4, 8.8], panels=8, series=records,
        y_scales={letter: ax.get_yscale() for letter, ax in zip('abcdefgh', axes.flat)},
        ratio_y_limits=list(axes[3, 1].get_ylim()), smoothing_panels=['c', 'd', 'f', 'g'],
        smoothing_window=100, smoothing='Full 100-pair trailing mean after forming each per-pair metric', thinning=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accept-reviewed', action='store_true', help='Record review after inspecting the current Poppler renders; do not regenerate.')
    args = parser.parse_args()
    if args.accept_reviewed:
        record = json.loads((HERE/'paper_figures_manifest.json').read_text())
        assert record['script_sha256'] == sha256(__file__)
        for name, item in record['figures'].items():
            assert sha256(HERE/'figures'/name) == item['sha256']
            review = item['visual_verification']
            assert sha256(HERE/review['render']) == review['render_sha256']
            review.update(status='passed', review='Inspected current Poppler render for labels, legends, contours, clipping, overlap and readable data-scaled axes.')
        record['status'] = 'passed'
        write_json(HERE/'paper_figures_manifest.json', record)
        print(json.dumps({'visual_review':'passed', 'figures':list(record['figures'])}))
        return
    verify_momentum_identity()
    assert json.loads((HERE/'verification.json').read_text())['passed']
    histories, sources, _ = load_sources()
    prior = json.loads((HERE/'results.json').read_text())
    protected = {p.relative_to(HERE).as_posix(): sha256(p) for p in (HERE/'figures').glob('*.pdf') if p.name not in FIGURES}
    values, rows = [], []
    for history, source, row in zip(histories, sources, prior['sources']):
        assert row['run_id'] == source['run_id'] and row['history_sha256'] == source['history_sha256']
        result, summary = reduce(history, source, row)
        values.append(result)
        rows.append(summary)
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 11,
        'axes.titlesize': 11.5, 'axes.labelsize': 11, 'legend.fontsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'pdf.fonttype': 42, 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False, 'path.simplify': False, 'savefig.dpi': 300})
    manifest = dict(status='awaiting_visual_review', script=Path(__file__).name, script_sha256=sha256(__file__),
        metric_source_sha256=sha256(HERE/'kappa.py'), sources=sources, source_results_sha256=sha256(HERE/'results.json'),
        preserved_pdf_hashes=protected, momentum='Stored uncorrected m_t; adaptive updates use mhat_t',
        gradient_noise=0, weight_decay=.1, precision='float64', pair_updates='2..2000',
        curvature_state_clock='theta[t] for panel e; theta[t-2] for decomposition of pair t',
        momentum_estimate='a_t + (1-c_t*||m_t||/||m_(t-1)||)/(1-beta1) = m_(t-1)^T(g_(t-1)-g_t)/||m_(t-1)||^2',
        verification=dict(passed=True, pairs=5997, checked_complete_means=22800,
            raw_eta_overlay_arrays_identical=True, momentum_identity_and_decay_remainder_decomposition=True), figures={})
    for name, creator in zip(FIGURES, [trajectories, diagnostics]):
        figure, record = creator(values, sources)
        destination = HERE/'figures'/name
        pending = HERE/'scratch'/(name+'.pending.pdf')
        pending.parent.mkdir(exist_ok=True)
        figure.savefig(pending)
        plt.close(figure)
        pending.replace(destination)
        record.update(sha256=sha256(destination), bytes=destination.stat().st_size)
        render = HERE/'scratch'/(Path(name).stem+'_review.png')
        subprocess.run([shutil.which('pdftoppm'), '-r', '150', '-png', '-singlefile',
            str(destination), str(render.with_suffix(''))], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        with fitz.open(destination) as pdf:
            assert len(pdf) == 1
            page = pdf[0]
            outside = [s['text'] for block in page.get_text('dict')['blocks']
                for line in block.get('lines', []) for s in line['spans']
                if not fitz.Rect(-1, -1, page.rect.width+1, page.rect.height+1).contains(fitz.Rect(s['bbox']))]
            assert not outside, (name, outside)
        record['visual_verification'] = dict(status='awaiting_visual_review', renderer='Poppler',
            render=render.relative_to(HERE).as_posix(), render_sha256=sha256(render), text_spans_outside_page=outside)
        manifest['figures'][name] = record
    for path, expected in protected.items():
        assert sha256(HERE/path) == expected
    write_json(HERE/'paper_results.json', dict(runs=rows, no_new_optimizer_run=True,
        source_analysis='Analysis017 / Analysis016, immutable Runs055-057'))
    manifest['results_sha256'] = sha256(HERE/'paper_results.json')
    write_json(HERE/'paper_figures_manifest.json', manifest)
    print(json.dumps({'figures':list(manifest['figures']), 'verified_pairs':5997, 'verified_means':22800,
        'rows':[{k:r[k] for k in ['run_id', 'final_loss', 'final_raw_ratio', 'final_momentum_ratio', 'late100']} for r in rows]}))


if __name__ == '__main__':
    main()
