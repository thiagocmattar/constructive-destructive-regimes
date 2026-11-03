"""Plot pre-rise raw cosine times rise gradient norm ratio versus spike loss ratio."""
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import linregress

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'spike_anatomy/events_gt10.csv'
PROVENANCE = HERE / 'spike_anatomy/manifest.json'
PDF = HERE / 'figures/adam_rho_gradient_rise_peak_scatter.pdf'
MANIFEST = HERE / 'rho_gradient_rise_peak_scatter_manifest.json'
TASKS = ['mlp', 'resnet', 'transformer']
TITLES = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
COLORS = ['#0072B2', '#D55E00', '#009E73']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    provenance = json.loads(PROVENANCE.read_text())
    assert provenance['status'] == 'passed'
    assert sha(SOURCE) == provenance['outputs'][SOURCE.relative_to(ROOT).as_posix()]
    source = pd.read_csv(SOURCE, float_precision='round_trip')
    assert len(source) == 129 and source.recovery_state.notna().all()
    assert source.peak_change_percent.gt(10).all() and source.spike_id.is_unique
    plotted = source.dropna(subset=['rho_raw_before']).copy()
    omitted = source.loc[~source.spike_id.isin(plotted.spike_id)]
    assert len(plotted) == 127 and len(omitted) == 2 and omitted.before_state.eq(0).all()
    assert plotted.rho_raw_before.between(-1, 1).all()
    plotted['rho_times_gradient_rise_ratio'] = plotted.rho_raw_before * plotted.gradient_rise_ratio
    np.testing.assert_allclose(plotted.rho_times_gradient_rise_ratio,
        plotted.rho_raw_before * plotted.gradient_norm_peak / plotted.gradient_norm_before, rtol=1e-12)
    assert plotted.gradient_rise_ratio.notna().all() and plotted.gradient_rise_ratio.gt(0).all()
    np.testing.assert_allclose(plotted.gradient_rise_ratio,
        plotted.gradient_norm_peak / plotted.gradient_norm_before, rtol=1e-12)
    np.testing.assert_allclose(plotted.peak_change_percent,
        100 * (plotted.loss_peak / plotted.loss_before - 1), rtol=1e-12)
    assert np.isfinite(plotted[['gradient_rise_ratio', 'peak_change_percent']]).all().all()
    plotted['peak_loss_ratio'] = 1 + plotted.peak_change_percent / 100
    np.testing.assert_allclose(plotted.peak_loss_ratio, plotted.loss_peak / plotted.loss_before, rtol=1e-12)
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), sharex=True, sharey=True)
    groups = []
    for col, (task, ax) in enumerate(zip(TASKS, axes)):
        rates = [.0001, .0003, .001] if task == 'mlp' else [.0003, .001, .003]
        annotation_row = 0
        for lr, color in zip(rates, COLORS):
            group = plotted.loc[plotted.task.eq(task) & plotted.learning_rate.eq(lr)]
            points = group[['rho_times_gradient_rise_ratio', 'peak_loss_ratio']].to_numpy()
            artist = ax.scatter(points[:, 0], points[:, 1], s=17, color=color,
                alpha=.8, linewidths=.25, edgecolors='white', label=rf'$\eta={lr:g}$')
            np.testing.assert_array_equal(np.asarray(artist.get_offsets()), points)
            regression = None
            if len(group) >= 2:
                x, log_y = points[:, 0], np.log10(points[:, 1])
                design = np.column_stack([np.ones(len(x)), x])
                coefficients, _, rank, _ = np.linalg.lstsq(design, log_y, rcond=None)
                assert rank == 2
                intercept, slope = coefficients
                fitted = design @ coefficients
                residual = log_y - fitted
                sse = float(residual @ residual)
                sst = float(np.sum((log_y - log_y.mean()) ** 2))
                assert sst > 0
                r_squared = 1 - sse / sst
                reference = linregress(x, log_y)
                np.testing.assert_allclose([slope, intercept, r_squared],
                    [reference.slope, reference.intercept, reference.rvalue ** 2],
                    rtol=1e-11, atol=1e-13)
                np.testing.assert_allclose(design.T @ residual, 0, atol=1e-10)
                line_x = np.linspace(x.min(), x.max(), 200)
                line_y = 10 ** (intercept + slope * line_x)
                line, = ax.plot(line_x, line_y, color=color, lw=1.1, zorder=1)
                np.testing.assert_array_equal(line.get_ydata(), line_y)
                ax.text(.97, .96 - .11 * annotation_row,
                    rf'$R^2={r_squared:.2f}$', transform=ax.transAxes,
                    ha='right', va='top', color=color, fontsize=8)
                annotation_row += 1
                regression = dict(n=len(x), intercept=float(intercept), slope=float(slope),
                    r_squared_log10_loss_ratio=float(r_squared), sse_log10=sse, sst_log10=sst,
                    x_range=[float(x.min()), float(x.max())],
                    line_x=line_x.tolist(), line_y=line_y.tolist())
            groups.append(dict(task=task, learning_rate=lr, count=len(group), color=color,
                spike_ids=group.spike_id.tolist(), coordinates=points.tolist(), regression=regression))
        ax.set_title(f'({chr(97 + col)}) ' + TITLES[col], pad=26)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=3,
            frameon=False, fontsize=7.6, handlelength=1., columnspacing=.5,
            handletextpad=.25, borderaxespad=0, scatterpoints=1)
        ax.set_yscale('log')
        ax.set(xlim=(-16, 2.5), ylim=(1, 11), xticks=[-15, -10, -5, 0],
            yticks=[1, 2, 5, 10], xlabel=r'$\rho_{a+1}\,\|g_{p+1}\|/\|g_{a+1}\|$')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.16, lw=.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=.6)
        assert all(-16 < x < 2.5 and 1 < y < 11
            for group in groups if group['task'] == task for x, y in group['coordinates'])
    axes[0].set_ylabel('Peak / pre-rise loss')
    fig.subplots_adjust(left=.08, right=.975, bottom=.2, top=.745, wspace=.31)
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert bounds.x0 >= 0 and bounds.y0 >= 0
    assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    assert sum(group['count'] for group in groups) == 127
    assert sum(group['regression'] is not None for group in groups) == 7
    fig.savefig(PDF, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    with fitz.open(PDF) as doc:
        assert len(doc) == 1 and not doc[0].get_images()
        for font in doc[0].get_fonts(full=True):
            assert doc.extract_font(font[0])[3]
        for word in doc[0].get_text('words'):
            assert 0 <= word[0] < word[2] <= doc[0].rect.width
            assert 0 <= word[1] < word[3] <= doc[0].rect.height
    record = dict(generator=Path(__file__).relative_to(ROOT).as_posix(), generator_sha256=sha(__file__),
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in [SOURCE, PROVENANCE]},
        output=PDF.relative_to(ROOT).as_posix(), output_sha256=sha(PDF),
        cohort='Peak increase >10% and recovery observed by update 1024; current anatomy sample.',
        selected_events=129, plotted_events=127, omitted_undefined_prerise_ids=omitted.spike_id.tolist(),
        x='rho_raw_before * gradient_rise_ratio = rho_(a+1) * ||g_(p+1)|| / ||g_(a+1)||; signed pre-rise consecutive cosine times endpoint rise norm ratio; no negation or absolute value.',
        y='1 + peak_change_percent/100 = loss_peak/loss_before; log axis, with 1 meaning no increase.',
        reduction='One unsmoothed point per event; no averaging or jitter; shared linear x and logarithmic y axes.',
        regression='Separate equal-event-weight OLS with intercept per task/LR: log10(loss_peak/loss_before) = intercept + slope*x. R-squared is in-sample on log10 loss ratios. Lines are back-transformed and limited to each group observed x range; standard axes clipping applies. Empty groups have no fit.',
        palette='Established increasing-LR colors within each model; per-panel legends give numeric LR, including empty groups.',
        groups=groups, checks=dict(source_hash=True, selection=True, magnitude_identity=True,
            ratio_identity=True, gradient_endpoint_ratio_identity=True, signed_product_identity=True, exact_scatter_coordinates=True, no_clipped_points=True, independent_scipy_regressions=True, ols_residual_orthogonality=True, seven_regression_lines=True, embedded_fonts=True, pdf_text_inside_page=True),
        visual_review='pending')
    MANIFEST.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Verified 127 signed pre-rise cosine times rise norm ratio points; two undefined initialization cosines omitted.')
    for group in groups:
        if group['regression'] is not None:
            print(group['task'], group['learning_rate'], 'n=', group['count'],
                'log-response R2=', round(group['regression']['r_squared_log10_loss_ratio'], 6))


if __name__ == '__main__':
    main()
