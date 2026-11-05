"""Plot loss-selected multi-spike windows; retain the original micro-plots."""
import csv
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PDF = HERE / 'figures/adam_manuscript_spikes_multiple.pdf'
MANIFEST = HERE / 'multiple_spikes_manifest.json'
# Author-requested approximate ranges, adjusted using loss and recovery only.
WINDOWS = [('mlp', .001, 465, 744), ('resnet', .003, 215, 495),
           ('transformer', .003, 525, 865)]
BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    dataset_path = HERE / 'spike_dataset/manifest.json'
    dataset = read(dataset_path)
    results_path = ROOT / 'runs/215-2026-11-02-controlled-full-batch-adam/results.json'
    archive_path = HERE / 'archive/full_unsmoothed_manifest.json'
    events_path = HERE / 'spike_dataset/loss_spikes.csv'
    assert sha(events_path) == dataset['outputs']['spike_dataset/loss_spikes.csv']
    sources = {}
    for path in [dataset_path, results_path, archive_path, events_path]:
        relative = path.relative_to(ROOT).as_posix()
        sources[relative] = sha(path)
        if relative in dataset['sources']:
            assert sources[relative] == dataset['sources'][relative]
    with events_path.open(encoding='utf-8', newline='') as stream:
        events = list(csv.DictReader(stream))
    results, archive = read(results_path), read(archive_path)
    titles = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
    keys = ['loss', 'projected', 'raw_cosine', 'moment_cosine', 'gradient', 'momentum']
    archived_rows = ['training_loss_log', 'eta_kappa_hat_raw', 'rho_raw',
                     'rho_momentum', 'raw_gradient_norm_log', 'momentum_norm_log']
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42,
        'path.simplify': False})
    fig, axes = plt.subplots(4, 3, figsize=(7.16, 5.6), sharex='col')
    curves, windows = [], []
    for col, (task, lr, start, end) in enumerate(WINDOWS):
        condition = next(c for c in results['conditions'] if c['task'] == task and c['lr'] == lr)
        paths = [ROOT / condition['source'] / name for name in ['metrics.jsonl', 'boundaries.jsonl']]
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            sources[relative] = sha(path)
            assert sources[relative] == dataset['sources'][relative]
        metrics, boundaries = [[json.loads(line) for line in p.read_text().splitlines()] for p in paths]
        assert [m['gradient_state'] for m in metrics] == list(range(1024))
        assert [b['state'] for b in boundaries] == list(range(1025))
        states = np.arange(start, end + 1)
        raw_norm = np.array([m['streams']['raw']['norm'] for m in metrics])
        raw_cos = np.array([m['streams']['raw']['cosine'] for m in metrics[1:]])
        projected = np.array([m['projected_gradient_change'] for m in metrics[1:]])
        np.testing.assert_allclose(projected, 1 - raw_cos * raw_norm[1:] / raw_norm[:-1],
                                   rtol=2e-12, atol=2e-12)
        task_events = [e for e in events if e['task'] == task and float(e['learning_rate']) == lr]
        rises = [e for e in task_events if e['recovery_state']
                 and start <= int(e['before_state']) < int(e['recovery_state']) <= end
                 and float(e['peak_change_percent']) > 10]
        # The plotted endpoints sit outside any still-open fixed-reference event.
        for edge in [start, end]:
            assert not any(int(e['before_state']) < edge < int(e['end_state']) for e in task_events)
        assert len(rises) >= 3
        event_records = []
        for event in rises:
            a, p, c = [int(event[key]) for key in ['before_state', 'peak_state', 'recovery_state']]
            loss = np.array([b['loss'] for b in boundaries])
            assert np.all(loss[a + 1:c] > loss[a]) and loss[c] <= loss[a]
            assert p == a + 1 + int(np.argmax(loss[a + 1:c]))
            event_records.append(dict(spike_id=event['spike_id'], before_state=a,
                peak_state=p, recovery_state=c, peak_change_percent=float(event['peak_change_percent'])))
        windows.append(dict(task=task, lr=lr, start_state=start, end_state=end,
                            highlighted_events=event_records))
        values_by_key = [
            [boundaries[t]['loss'] for t in states],
            [metrics[t]['projected_gradient_change'] for t in states],
            [metrics[t]['streams']['raw']['cosine'] for t in states],
            [metrics[t]['streams']['moment']['cosine'] for t in states],
            [metrics[t]['streams']['raw']['norm'] for t in states],
            [metrics[t]['streams']['moment']['norm'] for t in states]]
        np.testing.assert_array_equal(values_by_key[0], [metrics[t]['loss_before'] for t in states])
        for key, old_row, row, values in zip(keys, archived_rows, [0, 1, 2, 2, 3, 3], values_by_key):
            values = np.asarray(values)
            old = next(c for c in archive['curves'] if c['task'] == task and c['lr'] == lr and c['row'] == old_row)
            old_states = np.asarray(old['x']) - (key != 'loss')
            mask = (old_states >= start) & (old_states <= end)
            assert old['smoothing_window'] is None
            np.testing.assert_array_equal(old_states[mask], states)
            np.testing.assert_array_equal(np.asarray(old['y'])[mask], values)
            assert np.isfinite(values).all()
            divisor = float(values[0]) if row == 3 else 1.
            assert divisor > 0
            values = values / divisor
            momentum = key in ['momentum', 'moment_cosine']
            color = ORANGE if momentum else BLUE if key in ['gradient', 'raw_cosine'] else GREEN
            line, = axes[row, col].plot(states, values, color=color, lw=.8,
                                       ls='--' if momentum else '-')
            np.testing.assert_array_equal(line.get_xdata(), states)
            np.testing.assert_array_equal(line.get_ydata(), values)
            curves.append(dict(task=task, lr=lr, key=key, x=states.tolist(), y=values.tolist(),
                               normalization_divisor=divisor, yscale='log' if row == 0 else 'linear'))
        for row in range(4):
            ax = axes[row, col]
            if row:
                ax.text(0, 1.035, f'({chr(97 + row * 3 + col)})', transform=ax.transAxes,
                        va='bottom', fontsize=9)
            else:
                ax.set_title(f'({chr(97 + col)}) ' + titles[col] + '\n' + rf'$\eta={lr:g}$', pad=8)
            for event in event_records:
                ax.axvspan(event['before_state'], event['peak_state'],
                           color='.5', alpha=.12, linewidth=0, zorder=-1)
            ax.set_xlim(start, end)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.grid(alpha=.16, lw=.5)
            ax.tick_params(length=2.5, width=.6)
            ax.margins(y=.1)
            if row == 2:
                ax.set(ylim=(-1.06, 1.06), yticks=[-1, 0, 1])
                ax.axhline(0, color='.45', ls=(0, (4, 3)), lw=.7)
            if row == 1:
                ax.axhline(2, color='.45', ls=(0, (4, 3)), lw=.7)
            if row == 3:
                ax.set_ylim(bottom=0)
                assert ax.get_yscale() == 'linear'
        axes[0, col].set_yscale('log')
        axes[0, col].yaxis.set_minor_formatter(NullFormatter())
        axes[0, col].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=4))
        axes[0, col].yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
        axes[-1, col].set_xlabel(r'Parameter state $t$')
    for ax, label in zip(axes[:, 0], ['Training loss', r'$\eta\widehat\kappa_t$',
                                     r'$\rho_t$', 'Relative norm']):
        ax.set_ylabel(label)
    legend = fig.legend(handles=[Line2D([], [], color=BLUE, label='Gradient'),
        Line2D([], [], color=ORANGE, ls='--', label='Momentum'),
        Patch(facecolor='.5', alpha=.12, label='Rise of >10% spike')], frameon=False,
        fontsize=8, loc='lower center', bbox_to_anchor=(.54, .01), ncol=3, handlelength=1.5)
    fig.subplots_adjust(left=.105, right=.975, top=.90, bottom=.145, hspace=.48, wspace=.32)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert legend.get_window_extent(renderer).y1 < min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in axes[-1])
    bounds = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    assert bounds.x0 >= -1 and bounds.y0 >= -1
    assert bounds.x1 <= fig.bbox.width + 1 and bounds.y1 <= fig.bbox.height + 1
    assert sum(len(ax.patches) for ax in axes.flat) == 4 * sum(len(w['highlighted_events']) for w in windows)
    fig.savefig(PDF, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    with fitz.open(PDF) as doc:
        assert len(doc) == 1 and not doc[0].get_images()
        for font in doc[0].get_fonts(full=True):
            assert doc.extract_font(font[0])[3]
        for word in doc[0].get_text('words'):
            assert 0 <= word[0] < word[2] <= doc[0].rect.width + .01
            assert 0 <= word[1] < word[3] <= doc[0].rect.height + .01
    record = dict(generator=Path(__file__).relative_to(ROOT).as_posix(), generator_sha256=sha(__file__),
        sources=sources, output=PDF.relative_to(ROOT).as_posix(), output_sha256=sha(PDF),
        dimensions_inches=[7.16, 5.6], smoothing=None,
        clock='Parameter state s; vector diagnostics use g_(s+1), m_(s+1).',
        norm_reduction='Each stream divided by its own value at the first displayed state; linear y-axis.',
        selection='Manual loss-only windows near author ranges, with context and completed recovery.',
        windows=windows, curves=curves, total_points=sum(len(c['y']) for c in curves),
        checks=dict(source_hashes=True, exact_archive_arrays=True, projected_change_identity=True,
                    event_boundaries=True, endpoints_outside_open_events=True, embedded_fonts=True,
                    vector_pdf=True, text_inside_page=True), visual_review='See observations/011-multiple-spikes.md.')
    MANIFEST.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'Verified {len(curves)} unsmoothed curves / {record["total_points"]:,} points; '
          f'{sum(len(w["highlighted_events"]) for w in windows)} complete >10% excursions.')


if __name__ == '__main__':
    main()
