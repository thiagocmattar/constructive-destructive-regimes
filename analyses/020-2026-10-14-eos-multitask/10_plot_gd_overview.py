"""The requested 3x3 overview from the complete, matched seed0 GD trajectories."""
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

IDS = ['061', '071', '074']
TITLES = ['CIFAR-10\ntanh MLP', 'CIFAR-10\nResNet-32', 'WikiText-2\nTransformer']
COLOURS = ['#0072B2', '#D55E00', '#8C4AA8']
MANIFEST = HERE / 'gd_overview_seed0_manifest.json'


def sources():
    cohort = json.loads((HERE / 'cohort.json').read_text())
    spec = importlib.util.spec_from_file_location('eos_training_plot', HERE / '07_plot_training.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data, checks = [], []
    for number in IDS:
        record = next(r for r in cohort['runs'] if r['run_id'] == number)
        assert record['method'] == 'gd' and record['seed'] == 0 and record['updates'] == 500
        run = ROOT / 'runs' / record['run']
        attempt = run / 'artifacts/attempt-01-spark'
        inventory = json.loads((attempt / 'transfer_inventory.json').read_text())
        for item in inventory['files']:
            path = attempt / item['path']
            assert path.resolve().is_relative_to(attempt.resolve())
            assert path.stat().st_size == item['bytes'] and file_hash(path) == item['sha256']
        if number == '061':
            receipt = json.loads((ROOT / 'analyses/019-2026-10-13-cifar10-eos-lr/retrieval.json').read_text())
            assert receipt['status'] == 'complete' and number in receipt['runs']
        else:
            receipt = json.loads((run / 'retrieval_verification.json').read_text())
            assert receipt['status'] == 'passed' and receipt['inventory_sha256'] == file_hash(attempt / 'transfer_inventory.json')
        d = module.load(record)
        config = json.loads((run / 'config.json').read_text())
        manifest = json.loads((attempt / 'production/manifest.json').read_text())
        assert file_hash(run / 'config.json') == manifest['config_sha256']
        assert config['seed'] == 0
        assert record['lr'] == (config['optimizers']['gd']['lr'] if number == '061' else config['lr'])
        if number != '061':
            assert manifest['full_batch'] and config['method'] == 'gd'
        expected_q = 1 - d['cosine'][1:] * d['norm'][1:] / d['norm'][:-1]
        np.testing.assert_allclose(d['q'][1:], expected_q, rtol=2e-10, atol=2e-10)
        assert np.isnan(d['q'][0]) and np.isfinite(d['q'][1:]).all()
        np.testing.assert_array_equal(d['cosine'][1:] < 0, d['q'][1:] > 1)
        np.testing.assert_array_equal(d['spectral_x'], np.arange(0, 501, 100))
        assert len(d['loss']) == 500 and np.isfinite(d['loss']).all() and np.isfinite(d['final_loss'])
        first = np.flatnonzero(d['q'][1:] > 1)
        checks.append(dict(run_id=number, run=record['run'], task=record['task'], seed=0, lr=record['lr'],
            updates=500, initial_loss=float(d['loss'][0]), final_loss=float(d['final_loss']),
            initial_tensor_sha256=manifest['initial_tensor_sha256'],
            maximum_measured_eta_lambda=float(d['eta_lambda'].max()),
            q_min=float(d['q'][1:].min()), q_max=float(d['q'][1:].max()),
            first_negative_adjacent_gradient_update=int(first[0] + 2) if len(first) else None,
            negative_adjacent_fraction=float(np.mean(d['cosine'][1:] < 0)),
            source_sha256=d['source_sha256'], config_sha256=file_hash(run / 'config.json'),
            transfer_inventory_sha256=file_hash(attempt / 'transfer_inventory.json'), verified_files=len(inventory['files'])))
        data.append(d)
    return data, checks


def insert_manuscript():
    m = json.loads(MANIFEST.read_text())
    assert m['status'] == 'passed' and file_hash(ROOT / m['pdf']) == m['pdf_sha256']
    assert file_hash(__file__) == m['script_sha256']
    tex = ROOT / 'manuscript/2026-10-11-restatement/gd-formulation.tex'
    before = tex.read_bytes()
    text = before.decode('utf-8')
    marker = '\\input{../../analyses/' + HERE.name + '/gd-overview-seed0.tex}'
    if marker in text:
        print('Verified overview already included.'); return
    assert '021-2026-10-15-eos-multitask-lr/figure.tex' not in text, 'Review an existing multi-LR insertion first.'
    anchor = text.index('\\label{tab:sign}')
    end = text.index('\\end{table}', anchor) + len('\\end{table}')
    addition = '\n\n' + marker
    revised = text[:end] + addition + text[end:]
    assert revised.replace(addition, '', 1) == text
    backup = HERE / 'scratch/manuscript-before-gd-overview.tex'
    backup.write_bytes(before)
    assert tex.read_bytes() == before, 'Concurrent manuscript edit.'
    tex.write_bytes(revised.encode('utf-8'))
    print('Inserted the verified seed0 overview immediately after tab:sign; all surrounding source preserved.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accept-reviewed', action='store_true')
    parser.add_argument('--insert-manuscript', action='store_true')
    args = parser.parse_args()
    if args.insert_manuscript:
        insert_manuscript(); return
    if args.accept_reviewed:
        m = json.loads(MANIFEST.read_text())
        assert file_hash(__file__) == m['script_sha256']
        for key in ['pdf', 'render', 'arrays']:
            assert file_hash(ROOT / m[key]) == m[key + '_sha256']
        m.update(status='passed', visual_review='Current Poppler render inspected: nine panels, dataset/model titles, LR/seed labels, unsmoothed traces, references and margins.')
        write_json(MANIFEST, m); return
    data, checks = sources()
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 9,
        'axes.labelsize': 9, 'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(3, 3, figsize=(7.3, 6.35), sharex='col')
    arrays, artists = {}, []
    for col, d in enumerate(data):
        axes[0, col].set_title(TITLES[col], pad=30)
        values = [(np.arange(501), np.append(d['loss'], d['final_loss'])),
            (d['spectral_x'], d['eta_lambda']), (np.arange(1, 500), d['q'][1:])]
        for row, (x, y) in enumerate(values):
            ax = axes[row, col]
            line, = ax.plot(x, y, color=COLOURS[col], lw=.75 if row == 2 else 1.25,
                marker='o' if row == 1 else None, ms=3,
                label=rf'GD, $\eta={d["record"]["lr"]:g}$, seed 0')
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
            tag = f'{d["record"]["task"]}_row{row}'
            arrays[tag + '_x'], arrays[tag + '_y'] = x, y
            artists.append(dict(tag=tag, points=len(x),
                x_sha256=hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest(),
                y_sha256=hashlib.sha256(np.asarray(y, dtype='<f8').tobytes()).hexdigest()))
            ax.set_xlim(0, 500); ax.set_xticks([0, 250, 500]); ax.grid(alpha=.16, lw=.55)
            ax.text(.035, .94, f'({chr(97 + row * 3 + col)})', transform=ax.transAxes, va='top', fontsize=8.5,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.5))
            if row > 0:
                ax.axhline(2 if row == 1 else 1, color='.4', ls=(0, (4, 3)), lw=.75)
                ax.margins(y=.08)
            if row == 2: ax.set_xlabel('GD iteration')
        axes[0, col].legend(loc='lower center', bbox_to_anchor=(.5, 1.01), frameon=False,
            fontsize=8, handlelength=1.5, handletextpad=.5, borderaxespad=0)
    for row, label in enumerate(['Training loss', r'$\eta\lambda_{\max}$', r'$\eta\widehat\kappa_t$']):
        axes[row, 0].set_ylabel(label)
    fig.subplots_adjust(left=.09, right=.985, bottom=.085, top=.855, wspace=.32, hspace=.24)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text(): continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, artist.get_text()
    pdf = HERE / 'figures/eos_gd_overview_seed0.pdf'
    fig.savefig(pdf, metadata={'CreationDate': None, 'ModDate': None}); plt.close(fig)
    png = HERE / 'scratch/eos_gd_overview_seed0.png'
    poppler = shutil.which('pdftoppm') or str(Path.home() / 'AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdftoppm.exe')
    subprocess.run([poppler, '-r', '170', '-singlefile', '-png', str(pdf), str(png.with_suffix(''))], check=True, capture_output=True)
    arrayfile = HERE / 'derived/gd_overview_seed0_curves.npz'
    arrayfile.parent.mkdir(exist_ok=True); np.savez_compressed(arrayfile, **arrays)
    resultfile = HERE / 'gd_overview_seed0_results.json'
    write_json(resultfile, dict(status='passed', scope='Three complete seed0 GD trajectories; one LR per case; 500 updates.',
        runs=checks, gradient_pairs=1497, spectral_points=18, temporal_smoothing='none', seed_aggregation='none'))
    write_json(MANIFEST, dict(status='awaiting_visual_review', script_sha256=file_hash(__file__),
        loader_sha256=file_hash(HERE / '07_plot_training.py'), cohort_sha256=file_hash(HERE / 'cohort.json'),
        pdf=pdf.relative_to(ROOT).as_posix(), pdf_sha256=file_hash(pdf), render=png.relative_to(ROOT).as_posix(), render_sha256=file_hash(png),
        arrays=arrayfile.relative_to(ROOT).as_posix(), arrays_sha256=file_hash(arrayfile),
        results_sha256=file_hash(resultfile), curves=artists, visual_review=None))
    print(json.dumps(dict(status='awaiting_visual_review', results=checks), indent=2))


if __name__ == '__main__':
    main()
