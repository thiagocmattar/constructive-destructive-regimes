"""Publication view of the original sigma comparison; preserve all 90 scalars."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.controlled.rosenbrock import sha256, write_json

SIGMAS = [0., .01, .1, 1., 10.]
COLORS = ('#0072B2', '#8C4AA8', '#D55E00')
STYLES = ('-', ':', '--')
LABELS = (r'$\eta=10^{-2}$', r'$\eta=3.16\times10^{-3}$', r'$\eta=10^{-3}$')


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accept-reviewed', action='store_true', help='Record review of the current Poppler render; do not regenerate.')
    args = parser.parse_args()
    manifest_path = HERE/'sigma_paper_manifest.json'
    destination = HERE/'figures/sigma_comparison_paper.pdf'
    if args.accept_reviewed:
        record = json.loads(manifest_path.read_text())
        assert sha256(__file__) == record['script_sha256']
        assert sha256(destination) == record['sha256']
        assert sha256(HERE/record['visual_verification']['render']) == record['visual_verification']['render_sha256']
        record['status'] = record['visual_verification']['status'] = 'passed'
        record['visual_verification']['review'] = 'Inspected current Poppler render: serif labels, shared LR legend, readable axes and no clipping or overlaps.'
        write_json(manifest_path, record)
        print(json.dumps({'visual_review':'passed', 'figure':str(destination)}))
        return
    assert json.loads((HERE/'verification.json').read_text())['passed']
    original = HERE/'figures/sigma_comparison.pdf'
    frozen = json.loads((HERE/'figures_manifest.json').read_text())
    reference = next(r for r in frozen['figures'] if r['kind'] == 'sigma_comparison')
    assert sha256(original) == reference['sha256']
    protected = {p.relative_to(HERE).as_posix():sha256(p) for p in (HERE/'figures').glob('*.pdf') if p != destination}
    cohort = json.loads((HERE/'cohort.json').read_text())
    results = {r['run_id']:r for r in json.loads((HERE/'results.json').read_text())['conditions']}
    sources, curves, values = [], [], []
    rates = sorted({r['learning_rate'] for r in cohort}, reverse=True)
    assert len(cohort) == 15 and len(rates) == 3
    for eta in rates:
        rows = []
        for sigma in SIGMAS:
            entry = next(e for e in cohort if e['learning_rate'] == eta and e['sigma'] == sigma)
            folder = ROOT/entry['directory']
            path = folder/'artifacts/attempts/001-local/history.npz'
            receipt = json.loads((folder/'verification.json').read_text())
            assert receipt['passed'] and receipt['history_sha256'] == sha256(path)
            assert receipt['config_sha256'] == sha256(folder/'config.json')
            row = results[entry['run_id']]
            assert row['history_sha256'] == sha256(path)
            with np.load(path, allow_pickle=False) as h:
                scalar = [float(h['loss'][-1]), float(np.median(h['loss'][-1000:])),
                    float(h['raw_ratio'][-1]), float(h['momentum_ratio'][-1]),
                    float(np.nanmean(h['raw_cosine'][-1000:])), float(np.nanmean(h['momentum_cosine'][-1000:]))]
                assert len(h['loss']) == 2001 and len(h['raw_cosine']) == 2000
            expected = [row['final_loss'], row['late_loss_median'],
                row['streams']['raw']['final_B_over_A'], row['streams']['momentum']['final_B_over_A'],
                row['streams']['raw']['late_mean_cosine'], row['streams']['momentum']['late_mean_cosine']]
            np.testing.assert_allclose(scalar, expected, rtol=1e-13, atol=1e-14)
            rows.append(scalar)
            sources.append(dict(entry, history_sha256=sha256(path), config_sha256=sha256(folder/'config.json'),
                verification_sha256=sha256(folder/'verification.json')))
        values.append(np.array(rows))
    plt.rcParams.update({'font.family':'STIXGeneral', 'mathtext.fontset':'stix', 'font.size':11,
        'axes.titlesize':11.5, 'axes.labelsize':11, 'xtick.labelsize':10, 'ytick.labelsize':10,
        'pdf.fonttype':42, 'axes.linewidth':.7, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 6.9), sharex=True)
    fig.subplots_adjust(left=.105, right=.985, bottom=.09, top=.91, wspace=.3, hspace=.42)
    titles = ['(a) Final objective', '(b) Late objective (median)', '(c) Raw-gradient accumulation',
        '(d) Momentum accumulation', '(e) Gradient alignment', '(f) Momentum alignment']
    labels = [r'$f(\theta_{2000})$', r'Median $f$', r'$B_T^{(g)}/A_T^{(g)}$',
        r'$B_T^{(m)}/A_T^{(m)}$', r'Mean $\rho_t$', r'Mean $c_t$']
    for i, (ax, title, label) in enumerate(zip(axes.flat, titles, labels)):
        ax.set_title(title, loc='left', pad=5)
        ax.set_ylabel(label)
        ax.set_xscale('symlog', linthresh=.01)
        ax.set_xticks(SIGMAS, ['0', r'$10^{-2}$', r'$10^{-1}$', '1', '10'])
        ax.grid(axis='y', color='#E2E7EB', lw=.55)
        ax.set_axisbelow(True)
        if i < 2:
            ax.set_yscale('log')
        elif i < 4:
            ax.set_ylim(.18, 1.02)
            ax.set_yticks([.2, .4, .6, .8, 1.])
        else:
            ax.set_ylim(-1.04, 1.04)
            ax.set_yticks([-1, -.5, 0, .5, 1])
            ax.axhline(0, color='#9CA7B2', lw=.65)
    for ax in axes[-1]:
        ax.set_xlabel(r'Noise standard deviation $\sigma$')
    for eta, matrix, color, style in zip(rates, values, COLORS, STYLES):
        for i, ax in enumerate(axes.flat):
            line, = ax.plot(SIGMAS, matrix[:, i], color=color, ls=style, marker='o',
                markersize=4.4, lw=1.4, markeredgecolor='white', markeredgewidth=.5)
            np.testing.assert_array_equal(line.get_ydata(), matrix[:, i])
            curves.append(dict(panel=chr(97+i), learning_rate=eta, points=5,
                x_sha256=digest(line.get_xdata()), y_sha256=digest(line.get_ydata()), values=matrix[:, i].tolist()))
    fig.legend([Line2D([], [], color=c, ls=s, marker='o', markersize=4.4, lw=1.4) for c,s in zip(COLORS,STYLES)],
        LABELS, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.545,.988))
    pending = HERE/'scratch/sigma_comparison_paper.pending.pdf'
    pending.parent.mkdir(exist_ok=True)
    fig.savefig(pending)
    plt.close(fig)
    pending.replace(destination)
    render = HERE/'scratch/sigma_comparison_paper_review.png'
    subprocess.run([shutil.which('pdftoppm'),'-r','150','-png','-singlefile',str(destination),str(render.with_suffix(''))],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    with fitz.open(destination) as pdf:
        assert len(pdf) == 1
        page = pdf[0]
        outside = [s['text'] for block in page.get_text('dict')['blocks'] for line in block.get('lines',[]) for s in line['spans']
            if not fitz.Rect(-1,-1,page.rect.width+1,page.rect.height+1).contains(fitz.Rect(s['bbox']))]
        assert not outside
    assert all(sha256(HERE/p) == h for p,h in protected.items())
    write_json(manifest_path, dict(status='awaiting_visual_review', script=Path(__file__).name, script_sha256=sha256(__file__),
        figure='figures/sigma_comparison_paper.pdf', sha256=sha256(destination), dimensions_inches=[8.4,6.9],
        original_figure_sha256_unchanged=sha256(original), original_manifest_sha256=sha256(HERE/'figures_manifest.json'),
        source_results_sha256=sha256(HERE/'results.json'), preserved_pdf_hashes=protected, sources=sources,
        sigma_levels=SIGMAS, seed=1234, paired_noise=True, momentum='Stored uncorrected m',
        late_window='Objective states and cosine updates 1001..2000; median objective and means of unsmoothed cosines',
        verification=dict(passed=True, scalar_values_checked=90, plotted_curves=18), series=curves,
        x_scale='symlog with linear region through sigma=.01', y_scales={p:ax.get_yscale() for p,ax in zip('abcdef',axes.flat)},
        visual_verification=dict(status='awaiting_visual_review', renderer='Poppler', render=render.relative_to(HERE).as_posix(),
            render_sha256=sha256(render), text_spans_outside_page=outside), no_new_optimizer_run=True))
    print(json.dumps({'figure':str(destination),'verified_original_scalars':90,'source_trajectories':15}))


if __name__ == '__main__':
    main()
