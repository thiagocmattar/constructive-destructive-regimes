"""Three-column overview of completed, verified 3k GD learning-rate conditions.

The output keeps its original MLP filename at the user's request.
"""
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
TASKS = ['mlp', 'resnet', 'transformer']


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
        record.update(status='passed', visual_review='Current Poppler render inspected: three model columns and twelve panels, per-model LR legends, completed conditions only, unsmoothed curves, linear first three rows, logarithmic gradient norms, full/subset spectral labels, reference lines and unclipped labels.')
        write_json(MANIFEST, record)
        return
    cohort = json.loads((HERE / 'cohort.json').read_text())
    checks = module(ROOT / 'analyses/021-2026-10-15-eos-multitask-lr/03_collect.py', 'mlp_plot_verifier')
    loader = module(ROOT / 'analyses/020-2026-10-14-eos-multitask/07_plot_training.py', 'mlp_plot_loader')
    expected = {r['run_id']: r for r in json.loads((HERE / 'results.json').read_text())['runs']}
    records = sorted([r for r in cohort['runs'] if r['run_id'] in expected],
        key=lambda r: (TASKS.index(r['task']), r['lr']))
    assert len(records) == len(expected), 'Every plotted condition must pass the independent reduction.'
    planned_rates = {t: sorted(r['lr'] for r in cohort['runs'] if r['task'] == t) for t in TASKS}
    rates = {t: [r['lr'] for r in records if r['task'] == t] for t in TASKS}
    assert all(rates.values()), 'At least one complete condition per model is required.'
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
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
        assert manifest['initial_tensor_sha256'] == inputs['initial'][r['task']][0]['initial_tensor_sha256']
        if r['kind'] != 'reused_gd_prefix':
            assert manifest['full_batch'] and manifest['completed_updates'] == 3000
            assert manifest['inputs_sha256'] == file_hash(ROOT / 'cache/eos-multitask/inputs.json')
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
    for task in TASKS:
        assert len({d['loss'][0] for d in data if d['record']['task'] == task}) == 1
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix', 'font.size':10,
        'axes.labelsize':11, 'axes.titlesize':10.5, 'xtick.labelsize':9, 'ytick.labelsize':9,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.7, 'pdf.fonttype':42})
    fig, axes = plt.subplots(4, 3, figsize=(11.4, 9.2), sharex='col')
    titles = ['CIFAR-10 / tanh MLP\n5,000 training images',
              'CIFAR-10 / ResNet-32\n50,000 training images',
              'WikiText-2 / Transformer\n2.09M training target tokens']
    spectral_labels = ['Hessian: full 5k objective', 'Hessian: fixed 5k-image subset',
                       'Hessian: fixed 2,500-sequence subset']
    colors = ['#0072B2', '#D55E00', '#009E73']
    arrays, curves = {}, []
    handles = {t: [] for t in TASKS}
    for d in data:
        task = d['record']['task']; col = TASKS.index(task)
        color = colors[planned_rates[task].index(d['record']['lr'])]
        values = [(np.arange(3001), np.r_[d['loss'], d['final_loss']]),
                  (d['spectral_x'], d['eta_lambda']), (np.arange(1, 3000), d['q'][1:]),
                  (np.arange(3000), d['norm'])]
        for row, (x, y) in enumerate(values):
            artist, = axes[row,col].plot(x, y, color=color, lw=.65 if row == 2 else 1.15,
                alpha=.85 if row == 2 else 1, marker='o' if row == 1 else None,
                ms=2.6, label=rf'$\eta={d["record"]["lr"]:g}$')
            np.testing.assert_array_equal(artist.get_xdata(), x)
            np.testing.assert_array_equal(artist.get_ydata(), y)
            if row == 0: handles[task].append(artist)
            tag = f'run{d["record"]["run_id"]}_row{row}'
            arrays[tag+'_x'], arrays[tag+'_y'] = x, y
            curves.append(dict(tag=tag, points=len(x),
                x_sha256=hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest(),
                y_sha256=hashlib.sha256(np.asarray(y, dtype='<f8').tobytes()).hexdigest()))
    if previous:
        current = {c['tag']: c for c in curves}
        for prior in previous['curves']:
            assert current[prior['tag']] == prior, f'Previously plotted curve changed: {prior["tag"]}'
    labels = ['Training loss', r'$\eta\lambda_{\max}$', r'$\eta\widehat\kappa_t$', r'$\|g_t\|_2$']
    for col, task in enumerate(TASKS):
        axes[0,col].set_title(titles[col], pad=33, fontsize=11)
        axes[0,col].legend(handles=handles[task], loc='lower center', bbox_to_anchor=(.5,1.01),
            ncol=len(handles[task]), frameon=False, fontsize=9, handlelength=1.25,
            columnspacing=.9, handletextpad=.45, borderaxespad=0)
        for row in range(4):
            ax = axes[row,col]
            if col == 0: ax.set_ylabel(labels[row])
            ax.set_xlim(0,3000); ax.set_xticks([0,1000,2000,3000])
            ax.grid(alpha=.16,lw=.55); ax.margins(y=.08)
            ax.text(.025,.94,f'({chr(97+row*3+col)})',transform=ax.transAxes,va='top',fontsize=9,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.8))
        axes[1,col].axhline(2, color='.45', ls=(0,(4,3)), lw=.8)
        axes[1,col].text(0,1.025,spectral_labels[col],transform=axes[1,col].transAxes,
            fontsize=8,ha='left',va='bottom',color='.3')
        axes[2,col].axhline(1, color='.45', ls=(0,(4,3)), lw=.8)
        axes[3,col].set_yscale('log'); axes[3,col].set_xlabel('GD iteration')
    fig.text(.075,.015,'Seed 0; 3,000 updates per completed run. Unsmoothed curves; Hessians every 100 updates.',
        fontsize=8.5,color='.35')
    fig.subplots_adjust(left=.075,right=.985,bottom=.08,top=.88,wspace=.27,hspace=.28)
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
    write_json(resultfile, dict(status='passed', scope='Completed conditions across three model cases; original MLP filename retained.',
        rates_by_task=rates, completed_conditions=len(results), planned_conditions=len(cohort['runs']),
        pending_run_ids=[r['run_id'] for r in cohort['runs'] if r['run_id'] not in expected],
        updates=3000, seed=0, runs=results, verified_source_inventory_files=sum(r['verified_files'] for r in results),
        adjacent_pairs=2999*len(results), spectral_states=31*len(results), raw_norm_points=3000*len(results),
        plotted_points=sum(c['points'] for c in curves),
        temporal_smoothing='none', seed_aggregation='none', gradient_norm_y_scale='log',
        checks=['Verified source inventories and exact frozen initial/config hashes',
                'Full used-row and residual-checked spectral coverage', 'Exact q identity and reversal boundary',
                'Exact plotted artist arrays', 'Values match independently verified Analysis024 reduction',
                'All previously plotted curve hashes unchanged', 'Running/queued conditions excluded']))
    record = dict(status='awaiting_visual_review', script_sha256=file_hash(__file__), loader_sha256=file_hash(loader.__file__),
        cohort_sha256=file_hash(HERE / 'cohort.json'), curves=curves, layout=dict(rows=4, columns=3, panels=12),
        series_per_model={t:len(rates[t]) for t in TASKS},
        previous_curves_verified_unchanged=[c['tag'] for c in previous['curves']] if previous else [], visual_review=None)
    for key, path in dict(pdf=pdf, render=png, arrays=arrayfile, results=resultfile).items():
        record[key] = path.relative_to(ROOT).as_posix(); record[key+'_sha256'] = file_hash(path)
    write_json(MANIFEST, record)
    print(json.dumps(dict(status='awaiting_visual_review', rates_by_task=rates, panels=12,
        plotted_points=sum(c['points'] for c in curves))))


if __name__ == '__main__': main()
