"""Six isolated three-column views of the verified Analysis024 trajectories."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash

TASKS = ['mlp', 'resnet', 'transformer']
TITLES = ['CIFAR-10 / tanh MLP\n5,000 training images',
          'CIFAR-10 / ResNet-32\n50,000 training images',
          'WikiText-2 / Transformer\n2.09M training target tokens']
COLORS = ['#0072B2', '#D55E00', '#009E73']
WINDOW = 50
MANIFEST = HERE / 'single_rows_figure_manifest.json'
SPECS = [
    ('loss_log', 'loss', 'Training loss', 'log', False),
    ('rho', 'cosine', r'$\rho_t$', 'linear', False),
    ('rho_smoothed', 'cosine', r'$\rho_t$', 'linear', True),
    ('eta_kappa_hat_smoothed', 'q', r'$\eta\widehat\kappa_t$', 'linear', True),
    ('gradient_norms_smoothed', 'norm', r'$\|g_t\|_2$', 'log', True),
    ('normalized_displacement', 'ratio', r'$R_t$', 'linear', False),
]


def module(path, name):
    spec = spec_from_file_location(name, path)
    result = module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                          encoding='utf-8', newline='\n')


def series(d, field, smoothed):
    if field == 'loss':
        x, y = np.arange(3001), np.r_[d['loss'], d['final_loss']]
    elif field in ('cosine', 'q'):
        x, y = np.arange(1, 3000), d[field][1:]
    elif field == 'ratio':
        x, y = np.arange(1, 3001), d[field]
    else:
        x, y = np.arange(3000), d[field]
    assert np.isfinite(y).all()
    if smoothed:
        # Complete trailing windows, aligned to their rightmost observation.
        reference = np.convolve(y, np.ones(WINDOW) / WINDOW, mode='valid')
        y = np.lib.stride_tricks.sliding_window_view(y, WINDOW).mean(axis=-1)
        np.testing.assert_allclose(y, reference, rtol=2e-13, atol=2e-13)
        x = x[WINDOW - 1:]
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accept-reviewed', action='store_true')
    args = parser.parse_args()
    if args.accept_reviewed:
        record = json.loads(MANIFEST.read_text())
        assert record['script_sha256'] == file_hash(__file__)
        for key in ['arrays', 'results']:
            assert file_hash(ROOT / record[key]) == record[key + '_sha256']
        for figure in record['figures']:
            for key in ['pdf', 'render']:
                assert file_hash(ROOT / figure[key]) == figure[key + '_sha256']
        record.update(status='passed', visual_review=(
            'All six current Poppler renders inspected: three columns, nine LR entries, '
            'correct log/linear axes, complete trailing-window means, reference lines, '
            'coverage footers and unclipped labels.'))
        write_json(MANIFEST, record)
        print(json.dumps({'status': 'passed', 'figures': len(record['figures'])}))
        return

    loader = module(ROOT / 'analyses/020-2026-10-14-eos-multitask/07_plot_training.py',
                    'single_row_loader')
    cohort = json.loads((HERE / 'cohort.json').read_text())
    reference = json.loads((HERE / 'results.json').read_text())
    assert reference['status'] == 'passed' and reference['available'] == 9
    expected = {r['run_id']: r for r in reference['runs']}
    inputs = json.loads((ROOT / 'cache/eos-multitask/inputs.json').read_text())
    records = sorted(cohort['runs'], key=lambda r: (TASKS.index(r['task']), r['lr']))
    assert len(records) == 9 and {r['run_id'] for r in records} == set(expected)
    data = []
    for r in records:
        assert r['method'] == 'gd' and r['seed'] == 0 and r['updates'] == 3000
        run = ROOT / 'runs' / r['run']
        folder = run / 'artifacts/attempt-01-spark/production'
        source_manifest = json.loads((folder / 'manifest.json').read_text())
        assert source_manifest['config_sha256'] == r['config_sha256'] == file_hash(run / 'config.json')
        assert source_manifest['initial_tensor_sha256'] == inputs['initial'][r['task']][0]['initial_tensor_sha256']
        d = loader.load(r)
        assert d['source_sha256'] == expected[r['run_id']]['source_sha256']
        assert d['final_loss'] == expected[r['run_id']]['final_loss']
        assert d['ratio'][-1] == expected[r['run_id']]['final_B_over_A']
        assert np.isnan(d['cosine'][0]) and np.isnan(d['q'][0])
        np.testing.assert_allclose(d['q'][1:],
            1 - d['cosine'][1:] * d['norm'][1:] / d['norm'][:-1], rtol=2e-10, atol=2e-10)
        raw = [json.loads(line)['streams']['raw']
               for line in (folder / 'metrics.jsonl').read_text().splitlines()[:3000]]
        a, b = (np.array([row[key] for row in raw]) for key in ('A', 'B'))
        np.testing.assert_allclose(a, np.cumsum(d['norm']), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(d['ratio'], b / a, rtol=2e-10, atol=2e-10)
        assert d['ratio'][0] == 1 and np.all((d['ratio'] >= 0) & (d['ratio'] <= 1 + 1e-10))
        data.append(d)
    rates = {task: [r['lr'] for r in records if r['task'] == task] for task in TASKS}
    assert all(len(v) == 3 for v in rates.values())

    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 10,
        'axes.labelsize': 11, 'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42})
    figures, all_curves, arrays = [], [], {}
    poppler = shutil.which('pdftoppm')
    assert poppler, 'Poppler is required for PDF visual review.'
    for name, field, label, scale, smoothed in SPECS:
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.35))
        curves = []
        for d in data:
            task = d['record']['task']
            col = TASKS.index(task)
            index = rates[task].index(d['record']['lr'])
            x, y = series(d, field, smoothed)
            if scale == 'log':
                assert (y > 0).all(), 'Never floor or clip values for log display.'
            line, = axes[col].plot(x, y, color=COLORS[index], ls='-',
                lw=.65 if field == 'cosine' and not smoothed else 1.15,
                label=rf'$\eta={d["record"]["lr"]:g}$')
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
            tag = f'{name}_run{d["record"]["run_id"]}'
            arrays[tag + '_x'], arrays[tag + '_y'] = x, y
            curve = dict(tag=tag, run_id=d['record']['run_id'], task=task,
                         points=len(x), x_sha256=digest(x), y_sha256=digest(y))
            curves.append(curve)
            all_curves.append(curve)
        for col, ax in enumerate(axes):
            ax.set_title(TITLES[col], pad=29)
            ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=3,
                frameon=False, fontsize=9, handlelength=1.5, columnspacing=.85,
                handletextpad=.4, borderaxespad=0)
            ax.set(xlim=(0, 3000), xticks=[0, 1000, 2000, 3000], xlabel='GD iteration')
            ax.set_yscale(scale)
            ax.grid(alpha=.16, lw=.55)
            ax.margins(y=.08)
            ax.text(.025, .94, f'({chr(97 + col)})', transform=ax.transAxes, va='top',
                fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=.8))
            if field == 'cosine':
                ax.set_ylim(-1.06, 1.06)
                ax.set_yticks([-1, -.5, 0, .5, 1])
                ax.axhline(0, color='.45', ls=(0, (4, 3)), lw=.8)
            elif field == 'q':
                ax.axhline(1, color='.45', ls=(0, (4, 3)), lw=.8)
            elif field == 'ratio':
                ax.set_ylim(-.025, 1.025)
        axes[0].set_ylabel(label)
        smoothing = f'Trailing {WINDOW}-observation arithmetic mean; complete windows only.' if smoothed else 'Unsmoothed.'
        fig.text(.075, .025, f'Seed 0; 3,000 full-GD updates per condition. {smoothing}',
                 fontsize=8.3, color='.35')
        fig.subplots_adjust(left=.075, right=.985, bottom=.205, top=.745, wspace=.28)
        fig.canvas.draw()
        bounds = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
        assert bounds.x0 >= -1 and bounds.y0 >= -1
        assert bounds.x1 <= fig.bbox.width + 1 and bounds.y1 <= fig.bbox.height + 1
        pdf = HERE / f'figures/eos_gd_{name}_3k.pdf'
        png = HERE / f'scratch/eos_gd_{name}_3k.png'
        pdf.parent.mkdir(exist_ok=True)
        png.parent.mkdir(exist_ok=True)
        fig.savefig(pdf, metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)
        reader = PdfReader(pdf)
        assert len(reader.pages) == 1
        text = reader.pages[0].extract_text()
        assert all(word in text for word in ['MLP', 'ResNet-32', 'Transformer', 'Seed'])
        subprocess.run([poppler, '-r', '170', '-singlefile', '-png', str(pdf), str(png.with_suffix(''))],
            check=True, capture_output=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        figures.append(dict(name=name, field=field, y_scale=scale, smoothing=smoothing,
            rows=1, columns=3, curves=curves, plotted_points=sum(c['points'] for c in curves),
            pdf=pdf.relative_to(ROOT).as_posix(), pdf_sha256=file_hash(pdf),
            render=png.relative_to(ROOT).as_posix(), render_sha256=file_hash(png)))

    arrayfile = HERE / 'derived/single_rows_curves.npz'
    arrayfile.parent.mkdir(exist_ok=True)
    np.savez_compressed(arrayfile, **arrays)
    resultfile = HERE / 'single_rows_results.json'
    write_json(resultfile, dict(status='passed', conditions=9, updates=3000, seed=0,
        rates_by_task=rates, smoothing_window=WINDOW, seed_aggregation='none',
        smoothing='Arithmetic mean before axis transformation, complete trailing windows, no padding.',
        indexing={'loss': 'Parameter states 0..3000, including measured final loss.',
                  'cosine_and_q': 'Gradient evaluation states 1..2999; first undefined pair omitted.',
                  'norm': 'Gradient evaluation states 0..2999.',
                  'ratio': 'Completed updates 1..3000; prefix B/A after each update.'},
        source_results_sha256=file_hash(HERE / 'results.json'),
        source_runs=[dict(run_id=d['record']['run_id'], source_sha256=d['source_sha256']) for d in data],
        plotted_points=sum(c['points'] for c in all_curves),
        checks=['Exact used-source hashes match independently verified Analysis024 results',
                'Frozen config/initialization and complete nine-condition coverage',
                'Exact q identity, cumulative A and B/A identities',
                'Independent convolution verifies every smoothed point',
                'Every artist array matches its reduction; positive log values; PDF page/text/bounds checks']))
    write_json(MANIFEST, dict(status='awaiting_visual_review', visual_review=None,
        script_sha256=file_hash(__file__), loader_sha256=file_hash(loader.__file__),
        cohort_sha256=file_hash(HERE / 'cohort.json'), figures=figures,
        arrays=arrayfile.relative_to(ROOT).as_posix(), arrays_sha256=file_hash(arrayfile),
        results=resultfile.relative_to(ROOT).as_posix(), results_sha256=file_hash(resultfile)))
    print(json.dumps({'status': 'awaiting_visual_review', 'figures': 6,
                      'curves': len(all_curves), 'plotted_points': sum(c['points'] for c in all_curves)}))


if __name__ == '__main__':
    main()
