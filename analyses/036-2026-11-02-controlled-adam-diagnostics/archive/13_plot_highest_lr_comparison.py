"""Compare highest tested Adam and full-batch SGD rates over 1,024 updates."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GD = ROOT/'analyses/024-2026-10-17-eos-gd-lr-calibration'
RUN = ROOT/'runs/215-2026-11-02-controlled-full-batch-adam'
WINDOW, UPDATES = 50, 1024
TASKS = ['mlp', 'resnet', 'transformer']
TITLES = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
ROWS = ['training_loss_log', 'eta_kappa_hat_raw', 'rho_raw', 'raw_gradient_norm_log', 'R_t']
GD_ROWS = ['loss_log', 'eta_kappa_hat_smoothed', 'rho_smoothed',
           'gradient_norms_smoothed', 'normalized_displacement']
MANIFEST = HERE/'highest_lr_comparison_manifest.json'
PDF = HERE/'figures/adam_vs_sgd_highest_lr.pdf'
RENDER = HERE/'scratch/adam_vs_sgd_highest_lr.png'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(record):
    MANIFEST.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n',
                        encoding='utf-8', newline='\n')


def main(accept_reviewed=False):
    if accept_reviewed:
        record = read(MANIFEST)
        assert record['generator_sha256'] == sha(__file__)
        for path, digest in record['sources'].items():
            assert sha(ROOT/path) == digest
        assert sha(PDF) == record['output_sha256'] and RENDER.exists()
        record.update(status='passed', visual_review=(
            'All15 panels reviewed in the Poppler render: two optimizer curves per panel, '
            'correct rates/rows, unsmoothed loss and retention, complete50-step means, '
            'legible labels and legends without overlap or clipping.'),
            render=RENDER.relative_to(ROOT).as_posix(), render_sha256=sha(RENDER))
        write(record)
        print(json.dumps(dict(status=record['status'], curves=len(record['curves']))))
        return

    record = dict(generator=Path(__file__).relative_to(ROOT).as_posix(),
        generator_sha256=sha(__file__), sources={}, rows=ROWS, window=WINDOW,
        horizon=dict(displayed_updates=UPDATES, adam_available=1024, sgd_available=3000,
                     reduction='Common first1,024 updates; later SGD history is not displayed.'),
        selection='Largest tested LR per task and optimizer; no outcome-based selection.',
        smoothing='Loss and R_t unsmoothed; other rows use complete50-observation trailing arithmetic means, aligned right.',
        clock='Loss states0..1024; raw gradient at state t-1 belongs to update t; adjacent quantities start at update2.',
        retention=dict(adam='Measured actual Adam parameter increments.',
            sgd='Original Analysis024 constant-LR raw-gradient prefix ratio; identical to GD update retention in exact arithmetic.'),
        uncertainty='One fixed initialization per case; no seed averaging or uncertainty band.',
        caveats=['Optimizer and LR vary together; highest tested rates are not equal or claimed optimal.',
            'Hardware and FP32 reduction differences limit causal optimizer attribution.',
            'The raw-gradient projected-change expression under Adam is not a measured Hessian curvature or stability threshold.'],
        conditions=[], curves=[], checks={})

    def source(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, path
        record['sources'][path.relative_to(ROOT).as_posix()] = digest

    for path in [GD/'cohort.json', GD/'results.json', GD/'single_rows_figure_manifest.json',
                 HERE/'full_diagnostics_manifest.json', RUN/'results.json', RUN/'config.json']:
        source(path)
    cohort, gd_results = read(GD/'cohort.json'), read(GD/'results.json')
    gd_figure, adam_figure = read(GD/'single_rows_figure_manifest.json'), read(HERE/'full_diagnostics_manifest.json')
    adam_results, config = read(RUN/'results.json'), read(RUN/'config.json')
    assert gd_results['status'] == 'passed' and adam_results['status'] == 'complete_verified'
    assert adam_figure['window'] == WINDOW and config['updates'] == UPDATES
    source(ROOT/gd_figure['arrays'], gd_figure['arrays_sha256'])
    gd_arrays = np.load(ROOT/gd_figure['arrays'], allow_pickle=False)

    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .6,
        'pdf.fonttype': 42, 'path.simplify': False})
    fig, axes = plt.subplots(5, 3, figsize=(7.16, 8.5), sharex=True)
    max_mean_error = 0.
    projected = []
    for col, task in enumerate(TASKS):
        gd_arm = max((r for r in cohort['runs'] if r['task'] == task), key=lambda r:r['lr'])
        adam_arm = max((r for r in adam_results['conditions'] if r['task'] == task), key=lambda r:r['lr'])
        assert adam_arm['lr'] == max(config['learning_rates_by_task'][task])
        assert gd_arm['method'] == 'gd' and gd_arm['seed'] == 0
        gd_folder = ROOT/'runs'/gd_arm['run']/'artifacts/attempt-01-spark/production'
        adam_folder = ROOT/adam_arm['source']
        gd_identity, adam_identity = read(gd_folder/'manifest.json'), read(adam_folder/'manifest.json')
        assert gd_identity['status'] == adam_identity['status'] == 'complete'
        assert gd_identity['full_batch'] and adam_identity['full_batch']
        assert gd_identity['initial_tensor_sha256'] == adam_identity['initial_tensor_sha256']
        reference = next(r for r in gd_results['runs'] if r['run_id'] == gd_arm['run_id'])
        for name in ['manifest.json', 'metrics.jsonl']:
            source(gd_folder/name, reference['source_sha256'][name])
        source(ROOT/'runs'/gd_arm['run']/'config.json', gd_arm['config_sha256'])
        assert gd_identity['config_sha256'] == gd_arm['config_sha256']
        for name in ['manifest.json', 'metrics.jsonl', 'boundaries.jsonl']:
            path = adam_folder/name
            source(path, adam_figure['sources'][path.relative_to(ROOT).as_posix()])
        gd_metrics = [json.loads(s) for s in (gd_folder/'metrics.jsonl').read_text().splitlines()]
        adam_metrics = [json.loads(s) for s in (adam_folder/'metrics.jsonl').read_text().splitlines()]
        adam_losses = [json.loads(s)['loss'] for s in (adam_folder/'boundaries.jsonl').read_text().splitlines()]
        assert len(gd_metrics) >= UPDATES+1 and len(adam_metrics) == UPDATES
        assert [r['update'] for r in gd_metrics[:UPDATES]] == list(range(1, UPDATES+1))
        assert [r['gradient_state_step'] for r in gd_metrics[:UPDATES]] == list(range(UPDATES))
        gd_raw = [r['streams']['raw'] for r in gd_metrics[:UPDATES]]
        gd_values = [[r['batch_objective_estimate'] for r in gd_metrics[:UPDATES+1]],
            [r['eta_kappa_gradient_proxy'] for r in gd_metrics[:UPDATES]],
            [r['cosine'] for r in gd_raw], [r['norm'] for r in gd_raw],
            [r['B_over_A'] for r in gd_raw]]
        adam_values = [adam_losses, [r['projected_gradient_change'] for r in adam_metrics],
            [r['streams']['raw']['cosine'] for r in adam_metrics],
            [r['streams']['raw']['norm'] for r in adam_metrics],
            [r['streams']['actual']['B_over_A'] for r in adam_metrics]]
        for raw, values in [(gd_raw, gd_values),
                            ([r['streams']['raw'] for r in adam_metrics], adam_values)]:
            np.testing.assert_allclose(values[1][1:],
                [1-raw[t]['cosine']*raw[t]['norm']/raw[t-1]['norm'] for t in range(1, UPDATES)],
                rtol=2e-10, atol=2e-10)
        record['conditions'].append(dict(task=task, sgd_run=gd_arm['run_id'],
            sgd_lr=gd_arm['lr'], adam_run='215', adam_lr=adam_arm['lr'],
            initial_tensor_sha256=gd_identity['initial_tensor_sha256']))
        axes[0,col].set_title(TITLES[col], pad=43)
        for optimizer, arm, values_by_row, color in [
                ('SGD', gd_arm, gd_values, '#0072B2'),
                ('Adam', adam_arm, adam_values, '#D55E00')]:
            for row, (name, values) in enumerate(zip(ROWS, values_by_row)):
                x = np.arange(UPDATES+1) if row == 0 else np.arange(1, UPDATES+1)
                y = np.array([np.nan if v is None else v for v in values], dtype=float)
                assert len(x) == len(y)
                smooth = row in [1,2,3]
                if smooth:
                    blocks = np.lib.stride_tricks.sliding_window_view(y, WINDOW)
                    valid = np.isfinite(blocks).all(axis=1)
                    xx, yy = x[WINDOW-1:][valid], blocks[valid].mean(axis=1)
                    independent = [math.fsum(block)/WINDOW for block in blocks[valid]]
                    np.testing.assert_allclose(yy, independent, rtol=3e-14, atol=3e-14)
                    max_mean_error = max(max_mean_error, max(abs(yy-independent)))
                else:
                    valid = np.isfinite(y)
                    xx, yy = x[valid], y[valid]
                if row in [0,3]:assert np.all(yy > 0)
                if optimizer == 'SGD':
                    tag = f'{GD_ROWS[row]}_run{arm["run_id"]}'
                    gx, gy = gd_arrays[tag+'_x'], gd_arrays[tag+'_y']
                    prefix = gx <= UPDATES
                    np.testing.assert_array_equal(xx, gx[prefix])
                    np.testing.assert_array_equal(yy, gy[prefix])
                else:
                    prior = next(c for c in adam_figure['curves']
                        if c['task'] == task and c['lr'] == arm['lr'] and c['row'] == name)
                    np.testing.assert_array_equal(xx, prior['x'])
                    np.testing.assert_array_equal(yy, prior['y'])
                label = optimizer+rf', $\eta={arm["lr"]:g}$' if row == 0 else None
                artist, = axes[row,col].plot(xx, yy, color=color, lw=1., label=label)
                np.testing.assert_array_equal(artist.get_xdata(), xx)
                np.testing.assert_array_equal(artist.get_ydata(), yy)
                record['curves'].append(dict(task=task, optimizer=optimizer, lr=arm['lr'], row=name,
                    smoothing_window=WINDOW if smooth else None,
                    source_x=x.tolist(), source_y=[None if not np.isfinite(v) else float(v) for v in y],
                    x=xx.tolist(), y=yy.tolist()))
                if row == 1:projected.extend(yy)
        axes[0,col].legend(loc='lower center', bbox_to_anchor=(.5,1.015),
            fontsize=8, frameon=False, ncol=1, handlelength=1.5, labelspacing=.25)
        for row in [0,3]:
            axes[row,col].set_yscale('log')
            axes[row,col].yaxis.set_minor_formatter(NullFormatter())
        axes[0,col].yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        axes[0,col].yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
        axes[1,col].axhline(2, color='.4', lw=.7, ls='--', zorder=0)
        axes[2,col].set(ylim=(-1.08,1.08), yticks=[-1,0,1])
        axes[2,col].axhline(0, color='.5', lw=.6, zorder=0)
        axes[4,col].set(ylim=(0,1.04), yticks=[0,.5,1], xlabel='Training step')
        for row in range(5):
            ax = axes[row,col]
            ax.set(xlim=(0,UPDATES), xticks=[0,256,512,768,1024])
            ax.text(0,1.025,f'({chr(97+row*3+col)})', transform=ax.transAxes, va='bottom', fontsize=8)
            ax.grid(axis='y', alpha=.15, lw=.5)
            ax.tick_params(length=3, width=.6)
    limits = min(-.08, min(projected)), max(2.18, max(projected)*1.04)
    for ax in axes[1]:ax.set(ylim=limits, yticks=[0,1,2])
    labels = ['Training loss', r'$\eta\widehat\kappa_t$'+'\n(raw gradients)',
              r'$\rho_t$'+'\n(raw gradients)', r'$\|g_t\|$', r'$R_t$']
    for ax, label in zip(axes[:,0], labels):ax.set_ylabel(label)
    fig.subplots_adjust(left=.105, right=.975, top=.875, bottom=.125, hspace=.40, wspace=.29)
    fig.text(.54,.048,'Common first 1,024 updates; highest tested LR per optimizer and model.',
             ha='center', fontsize=8)
    fig.text(.54,.022,'Loss and '+r'$R_t$'+' unsmoothed; other rows: 50-step trailing means.',
             ha='center', fontsize=8)
    fig.savefig(PDF, metadata={'CreationDate':None, 'ModDate':None})
    plt.close(fig)
    doc = fitz.open(PDF)
    assert len(doc) == 1
    page = doc[0]
    for word in page.get_text('words'):
        assert word[0] >= 0 and word[1] >= 0 and word[2] <= page.rect.width+.01 and word[3] <= page.rect.height+.01
    assert len(page.search_for('Training step')) == 3
    assert len(record['curves']) == 30
    record['checks'] = dict(exact_match_to_both_prior_figures=True,
        source_hashes_and_initial_tensors_verified=True, highest_rate_selection_verified=True,
        independent_mean_max_absolute_error=max_mean_error, curves=30, panels=15,
        plotted_points=sum(len(c['x']) for c in record['curves']), pdf_text_inside_page=True)
    record.update(status='numerical_and_pdf_checks_passed', visual_review='pending_poppler_render',
                  output=PDF.relative_to(ROOT).as_posix(), output_sha256=sha(PDF))
    write(record)
    print(json.dumps(dict(output=record['output'], **record['checks'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-reviewed', action='store_true')
    main(parser.parse_args().accept_reviewed)
