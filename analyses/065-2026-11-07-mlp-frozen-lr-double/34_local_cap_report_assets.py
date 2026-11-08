"""Manuscript-style cap-only figures and source-checked report tables; no training."""
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
REPORT = HERE / 'report'
FIG = HERE / 'figures'
BLUE, GREEN = '#2166ac', '#00876c'
EARLY = {'mlp': '2', 'resnet': 'conv1', 'transformer': 'transformer_encoder.layers.0.self_attn'}
TITLES = {'mlp': 'CIFAR-10 / MLP', 'resnet': 'CIFAR-10 / ResNet-32', 'transformer': 'WikiText-2 / Transformer'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def plot_values(worker, arm):
    c = worker['curves'][arm]
    rep = c['representations']
    return [c['loss'], c['eta_kappa_hat'], c['rho'], c['raw_gradient_norm'],
            rep[EARLY[worker['case']]]['rho'], rep[EARLY[worker['case']]]['change_rms'],
            rep['output']['rho'], rep['output']['change_rms'],
            c['retained_norm_ratio'], c['mean_target_fraction_modified']]


def make_figure(specs, target, label, short=False):
    indices = [0, 6, 7] if short else list(range(10))
    labels = [r'$\mathcal{L}(\theta_s)$', r'$\eta\widehat\kappa_{s+1}$', r'$\rho_{s+1}$', r'$\|g_{s+1}\|_2$',
              r'$\rho_{s+1}^{\rm early}$', r'$r_{s+1}^{\rm early}$', r'$\rho_{s+1}^{F}$', r'$r_{s+1}^{F}$',
              r'$\gamma_{s+1}$', 'Units modified']
    fig, axes = plt.subplots(len(indices), 3, figsize=(7.16, 3.75 if short else 9.55),
                             sharex='col', squeeze=False)
    checks = 0
    columns = []
    for col, (w, arm, title) in enumerate(specs):
        x = np.asarray(w['states'])
        assert np.all(np.diff(x) == 1)
        for name, color, style in [('control', BLUE, '-'), (arm, GREEN, '-')]:
            values = plot_values(w, name)
            for row, index in enumerate(indices):
                y = np.asarray(values[index], dtype=float)
                assert y.shape == x.shape and not np.isinf(y).any()
                if index not in [2, 4, 6]:
                    assert np.isfinite(y).all()
                if index == 0:
                    assert np.all(y > 0)
                line, = axes[row, col].plot(x, y, color=color, ls=style, lw=.75,
                                          rasterized=False)
                np.testing.assert_array_equal(line.get_xdata(), x)
                np.testing.assert_array_equal(line.get_ydata(), y)
                checks += len(x)
        for row, index in enumerate(indices):
            ax = axes[row, col]
            if index in [0, 3, 5, 7]:
                ax.set_yscale('log', nonpositive='mask')
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=3))
                ax.yaxis.set_minor_formatter(NullFormatter())
            elif index == 1:
                ax.set_yscale('linear')
                ax.yaxis.set_major_locator(MaxNLocator(3))
                ax.axhline(1, color='.55', lw=.5, ls=':')
            elif index in [2, 4, 6]:
                ax.set_ylim(-1.06, 1.06)
                ax.set_yticks([-1, 0, 1])
                ax.axhline(0, color='.65', lw=.45)
            elif index == 8:
                ax.axhline(1, color='.55', lw=.5, ls=':')
                ax.yaxis.set_major_locator(MaxNLocator(3))
            else:
                ax.set_ylim(-.03, 1.03)
                ax.set_yticks([0, .5, 1])
            ax.grid(alpha=.16, linewidth=.4)
            ax.set_xlim(x[0], x[-1])
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            ax.tick_params(length=2.5, width=.6, pad=2)
            if col == 0:
                ax.set_ylabel(f'({chr(97 + row)}) ' + labels[index], labelpad=4)
        axes[0, col].set_title(title, pad=7)
        axes[-1, col].set_xlabel('Completed updates $s$', labelpad=4)
        columns.append(dict(run=w['run_id'], worker=w['worker'], arm=arm,
                            window=[int(x[0]), int(x[-1])], rows=indices,
                            independent_y_axes=True, loss_scale='log', eta_kappa_scale='linear'))
    legend = fig.legend(handles=[Line2D([], [], color=BLUE, lw=1, label='Unchanged Adam'),
                                 Line2D([], [], color=GREEN, lw=1, label=label)],
                        loc='lower center', bbox_to_anchor=(.52, .004), ncol=2,
                        frameon=False, fontsize=8)
    fig.subplots_adjust(left=.128, right=.987, top=.947 if short else .972,
                        bottom=.16 if short else .065,
                        wspace=.39, hspace=.22 if short else .23)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert legend.get_window_extent(renderer).y1 < min(
        ax.xaxis.label.get_window_extent(renderer).y0 for ax in axes[-1])
    bounds = fig.bbox
    for ax in axes.flat:
        xticks = [label for loc, label in zip(ax.get_xticks(), ax.get_xticklabels()) if ax.get_xlim()[0] <= loc <= ax.get_xlim()[1]]
        yticks = [label for loc, label in zip(ax.get_yticks(), ax.get_yticklabels()) if ax.get_ylim()[0] <= loc <= ax.get_ylim()[1]]
        for artist in [ax.title, ax.xaxis.label, ax.yaxis.label, *xticks, *yticks]:
            if not artist.get_visible() or not artist.get_text():
                continue
            b = artist.get_window_extent(renderer)
            assert b.x0 >= bounds.x0 - 1 and b.x1 <= bounds.x1 + 1, artist.get_text()
            assert b.y0 >= bounds.y0 - 1 and b.y1 <= bounds.y1 + 1, artist.get_text()
    fig.savefig(target, metadata={'Title': label, 'Creator': 'Analysis 065, 34_local_cap_report_assets.py'})
    plt.close(fig)
    return dict(file=target.relative_to(HERE).as_posix(), sha256=sha(target),
                columns=columns, plotted_values_checked=checks)


def main():
    REPORT.mkdir(exist_ok=True)
    original = {p.name: sha(p) for p in FIG.glob('0[1-8]-*.pdf')}
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 8.6,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
        'pdf.fonttype': 42, 'path.simplify': False})
    workers, sources = [], {}
    filenames = ['transfer_comparison.json'] + [f'transfer_comparison_run{n}.json' for n in [272, 273, 274, 282, 283]]
    for name in filenames:
        data = read(HERE / name)
        assert data['status'] == 'verified' and data['generator_sha256'] == sha(HERE / '17_reduce_transfer.py')
        workers += data['workers']
        sources[name] = sha(HERE / name)
    summary = read(HERE / 'transfer_summary.json')
    assert summary['status'] == 'verified'
    assert summary['generator_sha256'] == sha(HERE / '24_summarize_transfer.py')
    assert all(summary['sources'][n] == digest for n, digest in sources.items())
    sources['transfer_summary.json'] = sha(HERE / 'transfer_summary.json')
    def find(run, arm):
        matches = [w for w in workers if w['run_id'] == run and arm in w['curves']]
        assert len(matches) == 1
        return matches[0]
    outputs = []
    outputs.append(make_figure([(find(n, 'functional_entry'), 'functional_entry', TITLES[c])
        for n, c in [(269, 'mlp'), (270, 'resnet'), (271, 'transformer')]],
        FIG / '09-local-cap-entry-manuscript.pdf', 'Entry cap (q75)'))
    outputs.append(make_figure([(find(n, 'functional_trunk_q50'), 'functional_trunk_q50', TITLES[c])
        for n, c in [(272, 'mlp'), (273, 'resnet'), (274, 'transformer')]],
        FIG / '10-local-cap-distributed-manuscript.pdf', 'Distributed cap (q50)'))
    outputs.append(make_figure([(find(n, 'functional_trunk_q50'), 'functional_trunk_q50', f'MLP / initialization {seed}')
        for n, seed in [(272, 0), (282, 1), (283, 2)]],
        FIG / '11-local-cap-mlp-replication.pdf', 'Distributed cap (q50)', short=True))

    selected = []
    main_rows = []
    def metric(run, arm, begin, end):
        rows = [r for r in summary['metrics'] if (r['run'], r['arm'], r['begin'], r['end']) == (run, arm, begin, end)]
        assert len(rows) == 1
        r = rows[0]
        w = find(run, arm)
        ix = (np.asarray(w['states']) >= begin) & (np.asarray(w['states']) <= end)
        def reduce(name):
            loss = np.asarray(w['curves'][name]['loss'])[ix]
            return dict(mean_loss=loss.mean(), tail32_mean=loss[-32:].mean(), max_loss=loss.max(),
                        positive_variation=np.maximum(np.diff(loss), 0).sum())
        native, treatment = reduce('control'), reduce(arm)
        for key in native:
            np.testing.assert_allclose(r[key], treatment[key], rtol=1e-13)
            np.testing.assert_allclose(r[key + '_pct'], 100 * (treatment[key] / native[key] - 1), rtol=1e-12, atol=1e-12)
        selected.append(r)
        return r
    def percent(value):
        return f'${value:+.2f}\\%$'
    for run, arm, begin, end, case, scope in [
        (269, 'functional_entry', 448, 800, 'MLP', 'Entry q75'),
        (270, 'functional_entry', 264, 576, 'ResNet', 'Entry q75'),
        (271, 'functional_entry', 552, 928, 'Transformer', 'Entry q75'),
        (272, 'functional_trunk_q50', 448, 1152, 'MLP', 'Distributed q50'),
        (273, 'functional_trunk_q50', 264, 576, 'ResNet', 'Distributed q50'),
        (274, 'functional_trunk_q50', 552, 928, 'Transformer', 'Distributed q50')]:
        r = metric(run, arm, begin, end)
        main_rows.append(' & '.join([case, scope, f'{begin}--{end}'] +
            [percent(r[k + '_pct']) for k in ['mean_loss', 'tail32_mean', 'max_loss', 'positive_variation']]) + r' \\')
    def table(name, columns, header, rows):
        content = '\\begin{tabular}{@{}' + columns + '@{}}\n\\toprule\n' + header + r' \\' + '\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n'
        (REPORT / name).write_text(content, encoding='utf-8', newline='\n')
    table('cap-results-table.tex', 'llrrrrr', 'Model & Cap & States & Mean & Tail & Maximum & PV', main_rows)
    seed_rows = []
    for run, seed in [(272, 0), (282, 1), (283, 2)]:
        r = metric(run, 'functional_trunk_q50', 448, 1152)
        test = next(x for x in summary['terminal_heldout_evaluations'] if x['run'] == run and x['arm'] == r['arm'] and x['split'] == 'test')
        assert test['loss_units'] == test['objective_units'] == 10000
        seed_rows.append(' & '.join([str(seed)] + [percent(r[k + '_pct']) for k in ['mean_loss', 'tail32_mean', 'positive_variation']] +
                                   [percent(test['loss_change_pct'])]) + r' \\')
    table('cap-seeds-table.tex', 'rrrrr', 'Initialization & Mean training loss & Tail loss & PV & Terminal test loss', seed_rows)
    sensitivity = []
    for run, arm, case, label, begin, end in [
        (272, 'functional_entry_q50', 'MLP', 'Entry q50', 448, 1152),
        (272, 'functional_trunk_q75', 'MLP', 'Distributed q75', 448, 1152),
        (270, 'functional_trunk', 'ResNet', 'Distributed q75', 264, 576)]:
        r = metric(run, arm, begin, end)
        sensitivity.append(' & '.join([case, label] + [percent(r[k + '_pct']) for k in ['mean_loss', 'tail32_mean', 'positive_variation']]) + r' \\')
    table('cap-sensitivity-table.tex', 'llrrr', 'Model & Additional cap variant & Mean & Tail & PV', sensitivity)
    assert original == {p.name: sha(p) for p in FIG.glob('0[1-8]-*.pdf')}
    manifest = dict(status='verified', generator_sha256=sha(Path(__file__)),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources=sources, outputs=outputs, selected_metrics=selected,
        earlier_figures_preserved=original, rows={'early': EARLY},
        table_sources={p.name: sha(p) for p in REPORT.glob('cap-*-table.tex')},
        indexing=dict(axis='completed updates s', loss='L(theta_s)', gradient='g_(s+1)',
                      representation_change='A_(s+1)-A_s', intervention_update='t=s+1',
                      step_ratio='norm(Delta theta cap)/norm(Delta theta Adam)', recorded_values_shifted=False),
        scope='Only local affine-displacement caps and their own unchanged Adam baselines. No smoothing, truncation or new training.',
        plotted_values_checked=sum(o['plotted_values_checked'] for o in outputs))
    (HERE / 'local_cap_report_assets.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'figures': len(outputs), 'plotted_values_checked': manifest['plotted_values_checked'], 'result_rows': len(selected)}))


if __name__ == '__main__':
    main()
