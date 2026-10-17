"""Reviewed four-panel MLP comparison of the three selected 3k GD rates."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, write_json
from importlib.util import spec_from_file_location, module_from_spec

MANIFEST = HERE / 'mlp_learning_rates_figure_manifest.json'


def module(path, name):
    spec = spec_from_file_location(name, path)
    result = module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accept-reviewed', action='store_true')
    args = parser.parse_args()
    if args.accept_reviewed:
        record = json.loads(MANIFEST.read_text())
        assert record['script_sha256'] == file_hash(__file__)
        for key in ['pdf', 'render', 'arrays', 'results']:
            assert file_hash(ROOT / record[key]) == record[key+'_sha256']
        record.update(status='passed', visual_review='Current Poppler render inspected: all four panels, three LR colors and shared legend, unsmoothed curves, linear first three rows, logarithmic gradient norms, reference lines and unclipped labels.')
        write_json(MANIFEST, record)
        return
    cohort = json.loads((HERE / 'cohort.json').read_text())
    records = sorted([r for r in cohort['runs'] if r['task'] == 'mlp'], key=lambda r: r['lr'])
    assert [r['lr'] for r in records] == [.004, .02, .04]
    checks = module(ROOT / 'analyses/021-2026-10-15-eos-multitask-lr/03_collect.py', 'mlp_plot_verifier')
    loader = module(ROOT / 'analyses/020-2026-10-14-eos-multitask/07_plot_training.py', 'mlp_plot_loader')
    expected = {r['run_id']: r for r in json.loads((HERE / 'results.json').read_text())['runs']}
    inputs = json.loads((ROOT / 'cache/eos-multitask/inputs.json').read_text())
    data, results = [], []
    for r in records:
        assert r['updates'] == 3000 and r['method'] == 'gd' and r['seed'] == 0
        if r['kind'] == 'reused_gd_prefix': receipt = checks.verify_legacy(r)
        else:
            assert checks.verified_local(r['run'])
            receipt = json.loads((ROOT / 'runs' / r['run'] / 'retrieval_verification.json').read_text())
        run = ROOT / 'runs' / r['run']
        manifest = json.loads((run / 'artifacts/attempt-01-spark/production/manifest.json').read_text())
        assert manifest['config_sha256'] == r['config_sha256'] == file_hash(run / 'config.json')
        assert manifest['initial_tensor_sha256'] == inputs['initial']['mlp'][0]['initial_tensor_sha256']
        d = loader.load(r)
        assert len(d['loss']) == 3000 and np.isfinite(d['loss']).all() and np.isfinite(d['final_loss'])
        assert np.isfinite(d['norm']).all() and (d['norm'] > 0).all()
        assert np.isnan(d['q'][0]) and np.isfinite(d['q'][1:]).all()
        np.testing.assert_array_equal(d['spectral_x'], np.arange(0, 3001, 100))
        np.testing.assert_allclose(d['q'][1:], 1-d['cosine'][1:]*d['norm'][1:]/d['norm'][:-1], rtol=2e-10, atol=2e-10)
        np.testing.assert_array_equal(d['cosine'][1:] < 0, d['q'][1:] > 1)
        assert expected[r['run_id']]['source_sha256'] == d['source_sha256']
        assert expected[r['run_id']]['final_loss'] == d['final_loss']
        results.append(dict(expected[r['run_id']], verified_files=receipt['files'],
            transfer_inventory_sha256=receipt['inventory_sha256']))
        data.append(d)
    assert len({d['loss'][0] for d in data}) == 1
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix', 'font.size':10,
        'axes.labelsize':11, 'axes.titlesize':10.5, 'xtick.labelsize':9, 'ytick.labelsize':9,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.7, 'pdf.fonttype':42})
    fig, axes = plt.subplots(4, 1, figsize=(6.6, 9.4), sharex=True)
    fig.suptitle('CIFAR-10 / tanh MLP', x=.13, y=.985, ha='left', fontsize=14)
    fig.text(.13, .947, 'Full-batch GD, 5,000 training images, same seed 0 initialization', fontsize=9.5, color='.3')
    colors = ['#0072B2', '#D55E00', '#009E73']
    arrays, curves, handles = {}, [], []
    for d, color in zip(data, colors):
        values = [(np.arange(3001), np.r_[d['loss'], d['final_loss']]),
                  (d['spectral_x'], d['eta_lambda']), (np.arange(1, 3000), d['q'][1:]),
                  (np.arange(3000), d['norm'])]
        for row, (x, y) in enumerate(values):
            artist, = axes[row].plot(x, y, color=color, lw=.65 if row == 2 else 1.15,
                alpha=.85 if row == 2 else 1, marker='o' if row == 1 else None,
                ms=2.6, label=rf'$\eta={d["record"]["lr"]:g}$')
            np.testing.assert_array_equal(artist.get_xdata(), x)
            np.testing.assert_array_equal(artist.get_ydata(), y)
            if row == 0: handles.append(artist)
            tag = f'run{d["record"]["run_id"]}_row{row}'
            arrays[tag+'_x'], arrays[tag+'_y'] = x, y
            curves.append(dict(tag=tag, points=len(x),
                x_sha256=hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest(),
                y_sha256=hashlib.sha256(np.asarray(y, dtype='<f8').tobytes()).hexdigest()))
    titles = ['(a) Full-objective training loss', '(b) Maximum Hessian curvature (full 5k)',
              '(c) Gradient-derived directional curvature', '(d) Raw-gradient norm']
    labels = ['Loss', r'$\eta\lambda_{\max}$', r'$\eta\widehat\kappa_t$', r'$\|g_t\|_2$']
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], loc='left', pad=6)
        ax.set_ylabel(labels[i]); ax.set_xlim(0, 3000); ax.set_xticks(np.arange(0, 3001, 500))
        ax.grid(alpha=.16, lw=.55); ax.margins(y=.08)
    axes[1].axhline(2, color='.45', ls=(0, (4, 3)), lw=.8)
    axes[2].axhline(1, color='.45', ls=(0, (4, 3)), lw=.8)
    axes[3].set_yscale('log'); axes[3].set_xlabel('GD iteration')
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .933),
        ncol=3, frameon=False, handlelength=2.5, columnspacing=2.7)
    fig.text(.13, .016, 'Unsmoothed curves. Hessian measurements every 100 updates.', fontsize=8.5, color='.35')
    fig.subplots_adjust(left=.13, right=.975, bottom=.08, top=.865, hspace=.34)
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert bounds.x0 >= -1 and bounds.y0 >= -1 and bounds.x1 <= fig.bbox.width+1 and bounds.y1 <= fig.bbox.height+1
    pdf = HERE / 'figures/eos_mlp_learning_rates_3k.pdf'; pdf.parent.mkdir(exist_ok=True)
    fig.savefig(pdf, metadata={'CreationDate':None, 'ModDate':None}); plt.close(fig)
    png = HERE / 'scratch/eos_mlp_learning_rates_3k.png'; png.parent.mkdir(exist_ok=True)
    subprocess.run([shutil.which('pdftoppm'), '-r', '170', '-singlefile', '-png', str(pdf), str(png.with_suffix(''))],
        check=True, capture_output=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    arrayfile = HERE / 'derived/mlp_learning_rates_curves.npz'; arrayfile.parent.mkdir(exist_ok=True)
    np.savez_compressed(arrayfile, **arrays)
    resultfile = HERE / 'mlp_learning_rates_results.json'
    write_json(resultfile, dict(status='passed', rates=[.004, .02, .04], updates=3000, seed=0, runs=results,
        adjacent_pairs=8997, spectral_states=93, raw_norm_points=9000, plotted_points=sum(c['points'] for c in curves),
        temporal_smoothing='none', seed_aggregation='none', gradient_norm_y_scale='log',
        checks=['Verified source inventories and exact frozen initial/config hashes',
                'Full used-row and residual-checked spectral coverage', 'Exact q identity and reversal boundary',
                'Exact plotted artist arrays', 'Values match independently verified Analysis024 reduction']))
    record = dict(status='awaiting_visual_review', script_sha256=file_hash(__file__), loader_sha256=file_hash(loader.__file__),
        cohort_sha256=file_hash(HERE / 'cohort.json'), curves=curves, layout=dict(rows=4, columns=1, panels=4), visual_review=None)
    for key, path in dict(pdf=pdf, render=png, arrays=arrayfile, results=resultfile).items():
        record[key] = path.relative_to(ROOT).as_posix(); record[key+'_sha256'] = file_hash(path)
    write_json(MANIFEST, record)
    print(json.dumps(dict(status='awaiting_visual_review', rates=[.004, .02, .04], panels=4, plotted_points=27093)))


if __name__ == '__main__': main()
