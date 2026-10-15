"""Three-by-three GD comparison with actual learning-rate trajectories."""
import argparse
import hashlib
import importlib.util
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

TASKS = ['mlp', 'resnet', 'transformer']
TITLES = ['CIFAR-10\ntanh MLP', 'CIFAR-10\nResNet-32', 'WikiText-2\nTransformer']
COLOURS = ['#0072B2', '#D55E00', '#8C4AA8']


def load(record):
    path = ROOT / 'analyses/020-2026-10-14-eos-multitask/07_plot_training.py'
    spec = importlib.util.spec_from_file_location('eos_training_plot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proxy = dict(record)
    legacy = record['run_id'] in ['059', '061', '062']
    proxy['kind'] = 'reused_gd_prefix' if legacy else 'new_training'
    d = module.load(proxy)
    run = ROOT / 'runs' / record['run']
    output = run / 'artifacts/attempt-01-spark/production'
    manifest = json.loads((output / 'manifest.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    assert file_hash(run / 'config.json') == manifest['config_sha256']
    assert record['lr'] == (config['optimizers']['gd']['lr'] if legacy else config['lr'])
    assert config['seed'] == record['seed']
    if not legacy:
        assert manifest['full_batch'] is True and config['method'] == 'gd' and config['updates'] == 500
    q = 1 - d['cosine'][1:] * d['norm'][1:] / d['norm'][:-1]
    np.testing.assert_allclose(q, d['q'][1:], rtol=2e-10, atol=2e-10)
    assert np.isnan(d['q'][0]) and np.isfinite(d['q'][1:]).all()
    assert len(d['loss']) == 500 and np.isfinite(d['loss']).all() and np.isfinite(d['final_loss'])
    np.testing.assert_array_equal(d['spectral_x'], np.arange(0, 501, 100))
    d.update(record=record, initial_tensor_sha256=manifest['initial_tensor_sha256'],
        manifest_sha256=file_hash(output / 'manifest.json'), config_sha256=file_hash(run / 'config.json'))
    return d


def draw(data, path, layout_only=False):
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 9,
        'axes.labelsize': 9, 'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(3, 3, figsize=(7.3, 6.55), sharex='col')
    arrays, curves = {}, []
    def line(ax, x, y, tag, **kw):
        artist, = ax.plot(x, y, **kw)
        np.testing.assert_array_equal(artist.get_xdata(), x)
        np.testing.assert_array_equal(artist.get_ydata(), y)
        arrays[tag + '_x'], arrays[tag + '_y'] = x, y
        curves.append(dict(tag=tag, points=len(x),
            x_sha256=hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest(),
            y_sha256=hashlib.sha256(np.asarray(y, dtype='<f8').tobytes()).hexdigest()))
    for col, task in enumerate(TASKS):
        selected = [d for d in data if d['record']['task'] == task]
        rates = sorted({d['record']['lr'] for d in selected})
        assert len(rates) == 3
        axes[0, col].set_title('Layout check\nMLP data' if layout_only else TITLES[col], pad=34)
        for rank, lr in enumerate(rates):
            seeds = sorted([d for d in selected if d['record']['lr'] == lr], key=lambda d: d['record']['seed'])
            assert len(seeds) == (1 if layout_only else 3)
            for row in range(3):
                ys = []
                for d in seeds:
                    x, y = ((np.arange(501), np.append(d['loss'], d['final_loss'])) if row == 0 else
                        ((d['spectral_x'], d['eta_lambda']) if row == 1 else (np.arange(1, 500), d['q'][1:])))
                    ys.append(y)
                    line(axes[row, col], x, y, f'{task}_{lr:g}_{row}_seed{d["record"]["seed"]}',
                        color=COLOURS[rank], alpha=.23, lw=.45, zorder=2)
                line(axes[row, col], x, np.mean(ys, axis=0), f'{task}_{lr:g}_{row}_mean',
                    color=COLOURS[rank], lw=1.15, zorder=3, label=rf'$\eta={lr:g}$',
                    marker='o' if row == 1 else None, ms=2.6)
        axes[0, col].legend(loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=3,
            frameon=False, fontsize=7.1, handlelength=1.1, columnspacing=.65, handletextpad=.3, borderaxespad=0)
        for row in range(3):
            ax = axes[row, col]
            ax.set_xlim(0, 500)
            ax.set_xticks([0, 250, 500])
            ax.grid(alpha=.16, lw=.55)
            ax.text(.035, .94, f'({chr(97 + row * 3 + col)})', transform=ax.transAxes,
                va='top', fontsize=8.5, bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.5))
            if row > 0:
                ax.axhline(2 if row == 1 else 1, color='.4', ls=(0, (4, 3)), lw=.75, zorder=1)
                ax.margins(y=.08)
            if row == 2:
                ax.set_xlabel('GD iteration')
        if col == 0:
            for row, label in enumerate(['Training loss', r'$\eta\lambda_{\max}$', r'$\eta\widehat\kappa_t$']):
                axes[row, col].set_ylabel(label)
    fig.subplots_adjust(left=.09, right=.985, bottom=.085, top=.855, wspace=.32, hspace=.24)
    if layout_only:
        fig.text(.5, .98, 'LAYOUT CHECK ONLY - repeated MLP evidence, not a multitask result', ha='center', fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, artist.get_text()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    return arrays, curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-check', action='store_true')
    parser.add_argument('--accept-reviewed', action='store_true')
    args = parser.parse_args()
    mf = HERE / 'figure_manifest.json'
    if args.accept_reviewed:
        m = json.loads(mf.read_text())
        assert file_hash(__file__) == m['script_sha256']
        for key in ['pdf', 'render', 'arrays']:
            assert file_hash(ROOT / m[key]) == m[key + '_sha256']
        m.update(status='passed', visual_review='Current 3x3 render inspected: labels, nine panels, rates, seed traces, references and margins.')
        write_json(mf, m)
        return
    cohort = json.loads((HERE / 'cohort.json').read_text())
    if args.layout_check:
        originals = [load(r) for r in cohort['runs'] if r['task'] == 'mlp' and r['seed'] == 0]
        data = []
        for task in TASKS:
            for d in originals:
                item = dict(d)
                item['record'] = dict(d['record'], task=task)
                data.append(item)
        pdf = HERE / 'scratch/layout_check.pdf'
    else:
        assert json.loads((HERE / 'retrieval_verification.json').read_text())['status'] == 'passed'
        data = [load(r) for r in cohort['runs']]
        assert len(data) == 27
        for task in TASKS:
            for seed in [0, 1, 2]:
                group = [d for d in data if d['record']['task'] == task and d['record']['seed'] == seed]
                assert len({d['initial_tensor_sha256'] for d in group}) == 1, 'Unmatched initialization.'
        pdf = HERE / 'figures/eos_multitask_learning_rates.pdf'
    arrays, curves = draw(data, pdf, args.layout_check)
    png = HERE / ('scratch/layout_check.png' if args.layout_check else 'scratch/eos_multitask_learning_rates.png')
    poppler = shutil.which('pdftoppm') or str(Path.home() / 'AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdftoppm.exe')
    subprocess.run([poppler, '-r', '160', '-singlefile', '-png', str(pdf), str(png.with_suffix(''))], check=True, capture_output=True)
    if args.layout_check:
        write_json(HERE / 'layout_verification.json', dict(status='mechanical_check_passed_awaiting_visual_review',
            scope='Repeated real MLP seed0 curves for layout only; not a multitask result.',
            pairs_verified=1497, pdf_sha256=file_hash(pdf), render_sha256=file_hash(png), script_sha256=file_hash(__file__)))
        print('Mechanical layout and real MLP scalar checks passed; inspect scratch/layout_check.png.')
        return
    arrayfile = HERE / 'derived/curves.npz'
    arrayfile.parent.mkdir(exist_ok=True)
    np.savez_compressed(arrayfile, **arrays)
    summaries = [dict(run=d['record']['run'], task=d['record']['task'], seed=d['record']['seed'], lr=d['record']['lr'],
        final_loss=d['final_loss'], q_min=float(d['q'][1:].min()), q_max=float(d['q'][1:].max()),
        spectral_max=float(d['eta_lambda'].max()), source_sha256=d['source_sha256'],
        manifest_sha256=d['manifest_sha256'], config_sha256=d['config_sha256']) for d in data]
    write_json(HERE / 'results.json', dict(status='passed', runs=summaries, conditions=27,
        gradient_pairs=27 * 499, spectral_points=27 * 6, time_smoothing='none', reduction='pointwise mean across three seeds; thin individual traces'))
    write_json(mf, dict(status='awaiting_visual_review', script_sha256=file_hash(__file__),
        cohort_sha256=file_hash(HERE / 'cohort.json'),
        pdf=pdf.relative_to(ROOT).as_posix(), pdf_sha256=file_hash(pdf), render=png.relative_to(ROOT).as_posix(), render_sha256=file_hash(png),
        arrays=arrayfile.relative_to(ROOT).as_posix(), arrays_sha256=file_hash(arrayfile),
        results_sha256=file_hash(HERE / 'results.json'), curves=curves, visual_review=None))
    observation = HERE / 'observations/001-learning-rates.md'
    observation.parent.mkdir(exist_ok=True)
    observation.write_bytes(('# Full-batch learning-rate comparison\n\n'
        'Source: the27 frozen cohort conditions and their retrieved, size/SHA-checked\n'
        'attempts; see results.json and figure_manifest.json. The generator is04_plot.py.\n\n'
        'Coverage:500 updates, seeds0/1/2 at three LRs in each of three cases.\n'
        'Loss uses states0-500; q uses499 actual adjacent-gradient pairs at states1-499;\n'
        'six Hessians per run use states0,100,...,500. No time smoothing. Thick lines\n'
        'are pointwise seed means; faint lines are individual seeds, not confidence\n'
        'intervals. The caption and measurement limitations are in PROTOCOL.md and\n'
        'figure.tex. Subset spectra and finite-step projected change differ.\n\n'
        'All27 losses,13,473 pairs and162 spectral measurements pass the scalar,\n'
        'initialization, cadence and source-to-artist checks. Final loss and ranges\n'
        'are reported per run in results.json. Visual review and scientific pattern\n'
        'interpretation remain pending; no universal transition claim is automated.\n').encode())
    print(json.dumps(dict(status='awaiting_visual_review', figure=str(pdf), conditions=27)))


if __name__ == '__main__':
    main()
