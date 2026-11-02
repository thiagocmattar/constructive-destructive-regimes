"""Nine descriptive tables: model rows, ordered learning-rate columns."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'spike_dataset/loss_spikes.csv'
PDF = HERE / 'figures/adam_spike_descriptive_statistics.pdf'
CSV = HERE / 'spike_descriptive_statistics.csv'
MANIFEST = HERE / 'spike_descriptive_statistics_manifest.json'
TASKS = ['mlp', 'resnet', 'transformer']
TITLES = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
METRICS = ['peak_change_percent', 'rise_steps', 'recovery_steps', 'total_steps']
HEADERS = ['Peak rise\n(%)', 'Rise\n(steps)', 'Recovery\n(steps)', 'Total\n(steps)']
STATS = ['mean', 'std', 'min', 'p25', 'p75', 'max']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(value):
    MANIFEST.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def format_value(value, metric):
    if metric == 'peak_change_percent' and 0 < abs(value) < .1:
        return f'{value:.5f}'.rstrip('0')
    return f'{value:.2f}'.rstrip('0').rstrip('.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-reviewed', action='store_true')
    args = parser.parse_args()
    if args.accept_reviewed:
        record = json.loads(MANIFEST.read_text())
        assert record['generator_sha256'] == sha(__file__)
        for relative, digest in {**record['sources'], **record['outputs']}.items():
            assert sha(ROOT / relative) == digest
        record['visual_review'] = 'passed: all nine tables reviewed in the Poppler render; readable values, no overlap or clipping'
        dump(record)
        return
    source_manifest = HERE / 'spike_dataset/manifest.json'
    extraction = json.loads(source_manifest.read_text())
    assert extraction['verification']['status'] == 'passed'
    assert sha(SOURCE) == extraction['outputs'][SOURCE.relative_to(HERE).as_posix()]
    df = pd.read_csv(SOURCE)
    assert len(df) == 812 and int(df.right_censored.sum()) == 6
    complete = df.loc[~df.right_censored]
    assert len(complete) == 806 and not complete.peak_is_provisional.any()
    assert complete[METRICS].notna().all().all()
    np.testing.assert_array_equal(complete.total_steps, complete.rise_steps + complete.recovery_steps)
    np.testing.assert_allclose(complete.peak_change_percent,
        100 * (complete.loss_peak - complete.loss_before) / complete.loss_before, rtol=1e-10, atol=1e-12)
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 8.5, 'pdf.fonttype': 42, 'path.simplify': False})
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 5.65))
    fig.subplots_adjust(left=.055, right=.985, top=.905, bottom=.105, hspace=.30, wspace=.14)
    tables, rows = [], []
    for row, task in enumerate(TASKS):
        rates = sorted(df.loc[df.task.eq(task), 'learning_rate'].unique())
        assert len(rates) == 3
        for col, lr in enumerate(rates):
            ax = axes[row, col]
            ax.set_axis_off()
            all_events = df.loc[df.task.eq(task) & df.learning_rate.eq(lr)]
            sample = all_events.loc[~all_events.right_censored]
            summaries = []
            for metric in METRICS:
                values = sample[metric].to_numpy(dtype=float)
                result = dict(mean=float(values.mean()), std=float(values.std(ddof=1)),
                    min=float(values.min()), p25=float(np.quantile(values, .25, method='linear')),
                    p75=float(np.quantile(values, .75, method='linear')), max=float(values.max()))
                # Independently verify every statistic with Python's standard library.
                quartiles = statistics.quantiles(values.tolist(), n=4, method='inclusive')
                expected = [statistics.mean(values), statistics.stdev(values), min(values),
                            quartiles[0], quartiles[2], max(values)]
                np.testing.assert_allclose([result[s] for s in STATS], expected, rtol=3e-13, atol=1e-13)
                summaries.append(result)
                rows.append(dict(task=task, model=TITLES[row], learning_rate=float(lr),
                    candidates=len(all_events), recovered=len(sample), censored=int(all_events.right_censored.sum()),
                    metric=metric, unit='percent' if metric == 'peak_change_percent' else 'optimizer_updates',
                    **result))
            text = [[label] + [format_value(values[stat], metric) for values, metric in zip(summaries, METRICS)]
                    for stat, label in zip(STATS, ['Mean', 'Std', 'Min', 'P25', 'P75', 'Max'])]
            table = ax.table(cellText=text, colLabels=['Statistic'] + HEADERS,
                cellLoc='right', colLoc='center', colWidths=[.185, .23, .185, .205, .195],
                bbox=(0, 0, 1, .82))
            table.auto_set_font_size(False)
            table.set_fontsize(7.6)
            for (r, c), cell in table.get_celld().items():
                cell.PAD = .045
                cell.set_linewidth(.35)
                cell.set_edgecolor('.84')
                if r == 0:
                    cell.set_facecolor('#EAF2F7')
                    cell.set_text_props(fontweight='bold', fontsize=7.3)
                    cell.set_height(cell.get_height() * 1.8)
                elif c == 0:
                    cell.set_facecolor('.965')
                    cell.set_text_props(ha='left')
                else:
                    cell.set_facecolor('white')
            tables.append(table)
            ax.text(0, 1.035, f'({chr(97 + row * 3 + col)}) ' + TITLES[row], transform=ax.transAxes,
                fontsize=8.6, ha='left', va='bottom')
            ax.text(0, .905, rf'$\eta={lr:g}$' + f'   N={len(sample)} recovered; {int(all_events.right_censored.sum())} censored',
                transform=ax.transAxes, fontsize=7.7, ha='left', va='bottom')
    for ax, label in zip(axes[0], ['Lower tested LR', 'Middle tested LR', 'Higher tested LR']):
        bbox = ax.get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, .982, label, ha='center', va='top', fontsize=9, fontweight='bold')
    fig.text(.52, .049, r'Peak rise (%) = $100(\mathcal{L}_{p}/\mathcal{L}_{a}-1)$. Durations count optimizer updates.',
        ha='center', fontsize=8)
    fig.text(.52, .022, 'Recovered excursions only (806/812); no magnitude cutoff. Sample std; linearly interpolated quartiles.',
        ha='center', fontsize=7.8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for table in tables:
        for cell in table.get_celld().values():
            bounds = cell.get_window_extent(renderer)
            text_bounds = cell.get_text().get_window_extent(renderer)
            assert text_bounds.width <= bounds.width - 1, (cell.get_text().get_text(), bounds.width, text_bounds.width)
            assert text_bounds.height <= bounds.height + .2, (cell.get_text().get_text(), bounds.height, text_bounds.height)
    fig.savefig(PDF, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    with fitz.open(PDF) as document:
        assert len(document) == 1 and not document[0].get_images()
        for word in document[0].get_text('words'):
            assert 0 <= word[0] < word[2] <= document[0].rect.width + .01
            assert 0 <= word[1] < word[3] <= document[0].rect.height + .01
    with CSV.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert len(rows) == 36
    record = dict(generator=Path(__file__).relative_to(ROOT).as_posix(), generator_sha256=sha(__file__),
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in [SOURCE, source_manifest]},
        cohort='806 recovered excursions; six censored events excluded from all four metrics.',
        statistics=STATS, std_ddof=1, quantiles='linear interpolation, equivalent to inclusive quartiles',
        metrics=METRICS, checks=dict(nine_panels=True, independent_numeric_checks=216,
            shared_complete_cohort=True, all_cell_text_fits=True, pdf_text_inside_page=True),
        outputs={p.relative_to(ROOT).as_posix(): sha(p) for p in [PDF, CSV]}, visual_review='pending')
    dump(record)
    print('Created nine panels and 36 metric rows; all 216 summary values independently verified.')


if __name__ == '__main__':
    main()
