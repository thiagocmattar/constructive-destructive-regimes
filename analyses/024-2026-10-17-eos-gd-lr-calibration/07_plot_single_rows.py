"""Six isolated IEEE-width views and two manuscript composites of Analysis024."""
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
from matplotlib.ticker import NullFormatter, ScalarFormatter
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
        x, y = np.arange(2, 3001), d[field][1:]
    elif field == 'ratio':
        x, y = np.arange(1, 3001), d[field]
    else:
        x, y = np.arange(1, 3001), d[field]
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
            'All eight current Poppler renders inspected: '
            'solid colored LR curves, panel letters outside axes, no embedded footers, '
            'IEEE-width typography, correct axes, complete trailing-window means and '
            'unclipped labels. All scientific Y arrays are unchanged.'))
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

    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 9,
        'axes.labelsize': 9.5, 'axes.titlesize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42})
    figures, all_curves, arrays = [], [], {}
    poppler = shutil.which('pdftoppm')
    assert poppler, 'Poppler is required for PDF visual review.'
    layouts = [(s[0], [s]) for s in SPECS]
    by_name = {s[0]: s for s in SPECS}
    layouts += [('alignment_displacement', [by_name['rho_smoothed'], by_name['normalized_displacement']]),
                ('norms_projected_change', [by_name['gradient_norms_smoothed'], by_name['eta_kappa_hat_smoothed']])]
    for name, rows in layouts:
        nrows = len(rows)
        fig, axes = plt.subplots(nrows, 3, figsize=(7.16, 2.55 if nrows == 1 else 3.65),
                                 squeeze=False, sharex='col')
        curves = []
        reductions = []
        for row, (metric_name, field, label, scale, smoothed) in enumerate(rows):
            smoothing = f'Trailing {WINDOW}-observation arithmetic mean; complete windows only.' if smoothed else 'Unsmoothed.'
            reductions.append(dict(metric=metric_name, field=field, y_scale=scale, smoothing=smoothing))
            for d in data:
                task = d['record']['task']
                col = TASKS.index(task)
                index = rates[task].index(d['record']['lr'])
                x, y = series(d, field, smoothed)
                if scale == 'log':
                    assert (y > 0).all(), 'Never floor or clip values for log display.'
                if field == 'q':
                    assert y.min() >= -.08 and y.max() <= 2.2, 'Do not clip the q curves.'
                line, = axes[row, col].plot(x, y, color=COLORS[index], ls='-',
                    lw=.55 if field == 'cosine' and not smoothed else 1.0,
                    label=rf'$\eta={d["record"]["lr"]:g}$')
                np.testing.assert_array_equal(line.get_xdata(), x)
                np.testing.assert_array_equal(line.get_ydata(), y)
                assert line.get_linestyle() == '-'
                tag = f'{metric_name}_run{d["record"]["run_id"]}'
                arrays[tag + '_x'], arrays[tag + '_y'] = x, y
                curve = dict(tag=tag, run_id=d['record']['run_id'], task=task,
                             points=len(x), x_sha256=digest(x), y_sha256=digest(y))
                curves.append(curve)
                all_curves.append(curve)
            for col, ax in enumerate(axes[row]):
                letter = f'({chr(97 + row * 3 + col)})'
                if row == 0:
                    ax.set_title(letter + ' ' + TITLES[col], pad=25)
                    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=3,
                        frameon=False, fontsize=8, handlelength=1.15, columnspacing=.65,
                        handletextpad=.35, borderaxespad=0)
                else:
                    ax.text(0, 1.035, letter, transform=ax.transAxes, va='bottom', fontsize=9)
                ax.set(xlim=(0, 3000), xticks=[0, 1000, 2000, 3000])
                if row == nrows - 1:
                    ax.set_xlabel(r'GD update $t$')
                ax.set_yscale(scale)
                ax.grid(alpha=.16, lw=.5)
                ax.margins(y=.08)
                if field == 'cosine':
                    ax.set_ylim(-1.06, 1.06)
                    ax.set_yticks([-1, 0, 1])
                    ax.axhline(0, color='.45', ls=(0, (4, 3)), lw=.7)
                elif field == 'q':
                    ax.set_ylim(-.08, 2.2)
                    ax.set_yticks([0, 1, 2])
                    ax.axhline(1, color='.45', ls=(0, (4, 3)), lw=.7)
                elif field == 'ratio':
                    ax.set_ylim(-.025, 1.025)
                    ax.set_yticks([0, .5, 1])
                elif field == 'loss':
                    ax.set_yticks([[.15, .2, .3, .4, .6], [.5, 1, 2], [6, 7, 8, 10]][col])
                    ax.yaxis.set_major_formatter(ScalarFormatter())
                    ax.yaxis.set_minor_formatter(NullFormatter())
            axes[row, 0].set_ylabel(label)
        fig.subplots_adjust(left=.095, right=.97, bottom=.185 if nrows == 1 else .11,
                            top=.745 if nrows == 1 else .815, wspace=.30, hspace=.40)
        fig.canvas.draw()
        bounds = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
        assert bounds.x0 >= -1 and bounds.y0 >= -1, (name, bounds.bounds)
        assert bounds.x1 <= fig.bbox.width + 1 and bounds.y1 <= fig.bbox.height + 1, (name, bounds.bounds, fig.bbox.bounds)
        pdf = HERE / f'figures/eos_gd_{name}_3k.pdf'
        png = HERE / f'scratch/eos_gd_{name}_3k.png'
        pdf.parent.mkdir(exist_ok=True)
        png.parent.mkdir(exist_ok=True)
        fig.savefig(pdf, metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)
        reader = PdfReader(pdf)
        assert len(reader.pages) == 1
        text = reader.pages[0].extract_text()
        assert all(word in text for word in ['MLP', 'ResNet-32', 'Transformer'])
        assert 'Seed 0' not in text and 'Unsmoothed' not in text and 'Trailing' not in text
        subprocess.run([poppler, '-r', '170', '-singlefile', '-png', str(pdf), str(png.with_suffix(''))],
            check=True, capture_output=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        figures.append(dict(name=name, reductions=reductions,
            rows=nrows, columns=3, curves=curves, plotted_points=sum(c['points'] for c in curves),
            pdf=pdf.relative_to(ROOT).as_posix(), pdf_sha256=file_hash(pdf),
            render=png.relative_to(ROOT).as_posix(), render_sha256=file_hash(png)))

    arrayfile = HERE / 'derived/single_rows_curves.npz'
    arrayfile.parent.mkdir(exist_ok=True)
    np.savez_compressed(arrayfile, **arrays)
    resultfile = HERE / 'single_rows_results.json'
    write_json(resultfile, dict(status='passed', conditions=9, updates=3000, seed=0,
        rates_by_task=rates, smoothing_window=WINDOW, seed_aggregation='none',
        presentation='IEEE two-column width 7.16in; solid LR curves; external panel labels; no embedded footers.',
        smoothing='Arithmetic mean before axis transformation, complete trailing windows, no padding.',
        indexing={'loss': 'Parameter states 0..3000, including measured final loss.',
                  'cosine_and_q': 'One-based gradient indices t=2..3000, evaluated at states t-1; first undefined pair omitted.',
                  'norm': 'One-based gradient indices t=1..3000, evaluated at states t-1.',
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
    print(json.dumps({'status': 'awaiting_visual_review', 'figures': len(figures),
                      'curves': len(all_curves), 'plotted_points': sum(c['points'] for c in all_curves)}))


if __name__ == '__main__':
    main()
