"""Zoom the large Transformer Adam loss peak near step768, without smoothing."""
import argparse
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT/'runs/215-2026-11-02-controlled-full-batch-adam'
PDF = HERE/'figures/adam_transformer_peak_zoom.pdf'
RENDER = HERE/'scratch/adam_transformer_peak_zoom.png'
MANIFEST = HERE/'transformer_peak_zoom_manifest.json'
ANCHOR, RADIUS, BEFORE = 768, 64, 10
ROWS = ['training_loss_log', 'eta_kappa_hat_raw', 'rho_raw', 'rho_momentum',
        'rho_adam', 'raw_gradient_norm_log', 'momentum_norm_log', 'R_t']


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
            'All nine unsmoothed panels reviewed in the Poppler render: individual points '
            'and signed loss differences retained; correct peak, baseline and endpoint; '
            'legible axes and footers without overlap or clipping.'),
            render=RENDER.relative_to(ROOT).as_posix(), render_sha256=sha(RENDER))
        write(record)
        print(json.dumps(dict(status=record['status'], window=record['window'])))
        return

    record = dict(generator=Path(__file__).relative_to(ROOT).as_posix(),
        generator_sha256=sha(__file__), sources={}, task='transformer', optimizer='Adam',
        lr=.003, run_id='215', smoothing='None in any row.', curves=[],
        clock='Loss L_t at parameter state t; vector rows retain update t, consuming the raw gradient from state t-1.',
        loss_difference='L_t-L_(t-1), for each adjacent pair entirely inside the displayed window.',
        retention='Original cumulative R_t of measured actual Adam parameter updates; no reset at the zoom start.',
        y_axes='Loss and both norm rows logarithmic; cosines use [-1.08,1.08]; other limits fit the local observations.',
        uncertainty='One recorded trajectory; no seed aggregation, uncertainty band, or new training.',
        caveats=['The window is selected retrospectively around a visible spike.',
                 'Projected raw-gradient change under Adam is not measured Hessian curvature or a stability threshold.'])

    def source(path, expected=None):
        digest = sha(path)
        if expected is not None:assert digest == expected, path
        record['sources'][path.relative_to(ROOT).as_posix()] = digest

    for path in [RUN/'results.json', HERE/'full_unsmoothed_manifest.json', HERE/'full_diagnostics_manifest.json']:
        source(path)
    results = read(RUN/'results.json')
    assert results['status'] == 'complete_verified'
    arm = next(r for r in results['conditions'] if r['task'] == 'transformer' and r['lr'] == .003)
    folder = ROOT/arm['source']
    original = read(HERE/'full_unsmoothed_manifest.json')
    parent = read(HERE/'full_diagnostics_manifest.json')
    assert sha(HERE/'full_diagnostics_manifest.json') == original['source_preview_sha256']
    source(ROOT/original['output'], original['output_sha256'])
    for name in ['manifest.json', 'boundaries.jsonl', 'metrics.jsonl']:
        path = folder/name
        source(path, parent['sources'][path.relative_to(ROOT).as_posix()])
    identity = read(folder/'manifest.json')
    assert identity['status'] == 'complete' and identity['completed_updates'] == 1024
    assert sha(folder/'manifest.json') == arm['manifest_sha256']
    boundaries = [json.loads(s) for s in (folder/'boundaries.jsonl').read_text().splitlines()]
    metrics = [json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
    assert [r['state'] for r in boundaries] == list(range(1025))
    assert [r['update'] for r in metrics] == list(range(1,1025))
    losses = np.array([r['loss'] for r in boundaries])
    candidates = np.arange(ANCHOR-RADIUS, ANCHOR+RADIUS+1)
    peak = int(candidates[np.argmax(losses[candidates])])
    assert losses[peak] > losses[peak-1] and losses[peak] > losses[peak+1]
    start = peak-BEFORE
    end = next(t for t in range(peak+1, len(losses)) if losses[t] < losses[start])
    assert np.all(losses[peak+1:end] >= losses[start])
    record['window'] = dict(requested_near_step=ANCHOR, peak_search_states=[int(candidates[0]), int(candidates[-1])],
        peak_selection='Maximum recorded loss within +/-64 states of the approximate user anchor.',
        peak_state=peak, peak_loss=float(losses[peak]), start_state=start, before_peak_steps=BEFORE,
        baseline='Loss at the first displayed state, ten steps before the peak.',
        baseline_loss=float(losses[start]), end_state=end, end_loss=float(losses[end]),
        loss_at_previous_state=float(losses[end-1]),
        endpoint='First state strictly after the peak whose loss is strictly below the starting loss.',
        states_displayed=end-start+1)
    values = [losses,
        [r['projected_gradient_change'] for r in metrics],
        [r['streams']['raw']['cosine'] for r in metrics],
        [r['streams']['moment']['cosine'] for r in metrics],
        [r['streams']['adaptive']['cosine'] for r in metrics],
        [r['streams']['raw']['norm'] for r in metrics],
        [r['streams']['moment']['norm'] for r in metrics],
        [r['streams']['actual']['B_over_A'] for r in metrics]]
    plt.rcParams.update({'font.family':'Times New Roman', 'mathtext.fontset':'stix',
        'font.size':9, 'axes.labelsize':9, 'xtick.labelsize':8, 'ytick.labelsize':8,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.6,
        'pdf.fonttype':42, 'path.simplify':False})
    fig, axes = plt.subplots(9, 1, figsize=(7.16,12.8), sharex=True)
    destinations = [0,2,3,4,5,6,7,8]
    for name, full, destination in zip(ROWS, values, destinations):
        x = np.arange(1025) if name == 'training_loss_log' else np.arange(1,1025)
        y = np.array([np.nan if v is None else v for v in full])
        mask = (x >= start) & (x <= end)
        xx, yy = x[mask], y[mask]
        assert len(xx) == end-start+1 and np.isfinite(yy).all()
        prior = next(c for c in original['curves']
                     if c['task'] == 'transformer' and c['lr'] == .003 and c['row'] == name)
        pairs = [(a,b) for a,b in zip(prior['x'], prior['y']) if start <= a <= end]
        np.testing.assert_array_equal(xx, [a for a,b in pairs])
        np.testing.assert_array_equal(yy, [b for a,b in pairs])
        assert prior['smoothing_window'] is None
        artist, = axes[destination].plot(xx, yy, color='#009E73', lw=.85,
                                        marker='o', ms=2.5, markeredgewidth=0)
        np.testing.assert_array_equal(artist.get_xdata(), xx)
        np.testing.assert_array_equal(artist.get_ydata(), yy)
        record['curves'].append(dict(row=name, x=xx.tolist(), y=yy.tolist(),
                                     smoothing_window=None, exact_original_slice=True))
    difference_x = np.arange(start+1, end+1)
    differences = np.diff(losses[start:end+1])
    independent = [boundaries[t]['loss']-boundaries[t-1]['loss'] for t in difference_x]
    np.testing.assert_array_equal(differences, independent)
    np.testing.assert_allclose(differences.sum(), losses[end]-losses[start], rtol=0, atol=2e-15)
    bars = axes[1].bar(difference_x, differences, width=.72,
                      color=np.where(differences > 0, '#D55E00', '#0072B2'), linewidth=0)
    np.testing.assert_array_equal([bar.get_height() for bar in bars], differences)
    record['curves'].insert(1, dict(row='loss_difference', x=difference_x.tolist(),
                                   y=differences.tolist(), smoothing_window=None))
    labels = ['Training loss', r'$L_t-L_{t-1}$', r'$\eta\widehat\kappa_t$'+'\n(raw gradients)',
        r'$\rho_t$'+'\n(raw gradients)', r'$\rho_t$'+'\n(momentum)', r'$\rho_t$'+'\n(Adam)',
        r'$\|g_t\|$', r'$\|m_t\|$', r'$R_t$'+'\n(Adam)']
    for row, (ax, label) in enumerate(zip(axes, labels)):
        ax.set_ylabel(label)
        ax.axvline(peak, color='.45', lw=.7, ls='--', zorder=0)
        ax.grid(axis='y', alpha=.17, lw=.5)
        ax.tick_params(length=3, width=.6)
        ax.text(-.075,1.02,f'({chr(97+row)})', transform=ax.transAxes, va='bottom', fontsize=8)
        ax.set_xlim(start-.6,end+.6)
        if row in [0,6,7]:
            ax.set_yscale('log')
            ax.yaxis.set_minor_formatter(NullFormatter())
        if row in [3,4,5]:ax.set(ylim=(-1.08,1.08), yticks=[-1,0,1])
    axes[0].axhline(losses[start], color='.45', lw=.7, ls=':',
                    label=rf'$L_{{{start}}}={losses[start]:.6f}$')
    axes[0].legend(loc='upper right', frameon=False, fontsize=8)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value,p:f'{value:g}'))
    axes[1].axhline(0, color='.45', lw=.6)
    axes[2].axhline(2, color='.45', lw=.7, ls=':')
    for ax in axes[3:6]:ax.axhline(0, color='.55', lw=.6)
    axes[8].yaxis.set_major_formatter(FuncFormatter(lambda value,p:f'{value:.3f}'))
    ticks = list(range(start,end+1,10))
    if ticks[-1] != end:ticks.append(end)
    axes[-1].set(xticks=ticks, xlabel=r'Training step $t$')
    fig.subplots_adjust(left=.15,right=.975,top=.928,bottom=.09,hspace=.38)
    fig.suptitle('WikiText-2 / Transformer - Adam, '+r'$\eta=0.003$', y=.985, fontsize=11)
    fig.text(.56,.958,f'Unsmoothed steps {start}-{end}; peak at {peak}; first return below '+rf'$L_{{{start}}}$'+f' at {end}.',
             ha='center',fontsize=9)
    fig.text(.56,.043,'Loss: state '+r'$t$'+'. Vector rows: update '+r'$t$'+', using gradient from state '+r'$t-1$'+'.',
             ha='center',fontsize=8)
    fig.text(.56,.022,'All points shown; '+r'$R_t$'+' retains its prefix from initialization. Dashed vertical line: loss peak.',
             ha='center',fontsize=8)
    fig.savefig(PDF,metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    doc = fitz.open(PDF)
    assert len(doc) == 1
    page = doc[0]
    for word in page.get_text('words'):
        assert word[0] >= 0 and word[1] >= 0 and word[2] <= page.rect.width+.01 and word[3] <= page.rect.height+.01
    record.update(status='numerical_and_pdf_checks_passed', visual_review='pending_poppler_render',
        output=PDF.relative_to(ROOT).as_posix(), output_sha256=sha(PDF),
        checks=dict(original_curves_exact=8, original_points=8*(end-start+1),
            loss_difference_points=len(differences), loss_difference_exact=True,
            difference_sum_matches_endpoint_loss_change=True, first_postpeak_crossing_verified=True,
            all_sources_hash_verified=True, pdf_text_inside_page=True))
    write(record)
    print(json.dumps(dict(output=record['output'], window=record['window'], checks=record['checks'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-reviewed', action='store_true')
    main(parser.parse_args().accept_reviewed)
