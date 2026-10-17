"""Reduce the complete, locally verified H021 cohort; generate scientific figures."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIZES = ['14M', '31M', '70M']
COLORS = dict(zip(SIZES, ['#0072B2', '#D55E00', '#009E73']))
SEEDS = [1234, 1235, 1236]


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def reduce():
    cohort = read(HERE / 'cohort.json')
    accepted = set(read(HERE / 'retrieved.json')['runs'])
    assert accepted == {r['run'] for r in cohort}, 'The full cohort must be retrieved first'
    probes, summaries, events, provenance = {}, {}, {}, []
    for run in cohort:
        name = run['run']
        directory = ROOT / 'runs' / name
        output = directory / 'artifacts/attempt-01/full-pass'
        receipt_path = directory / 'prelaunch/attempt-01-retrieval.json'
        receipt = read(receipt_path)
        assert receipt['verified'] and receipt['scientific_verification']['verified']
        assert digest(output / 'inventory.json') == receipt['inventory_sha256']
        inventory = {r['path']: r for r in read(output / 'inventory.json')['files']}
        for filename in ['events.jsonl', 'probe_events.jsonl', 'summary.json']:
            source = output / filename
            assert source.stat().st_size == inventory[filename]['bytes']
            assert digest(source) == inventory[filename]['sha256'], f'Changed scalar evidence: {name}/{filename}'
        probes[name] = rows(output / 'probe_events.jsonl')
        summaries[name] = read(output / 'summary.json')
        events[name] = rows(output / 'events.jsonl')
        provenance.append(dict(run=name, receipt_sha256=digest(receipt_path),
            inventory_sha256=digest(output / 'inventory.json'),
            probe_events_sha256=digest(output / 'probe_events.jsonl'),
            summary_sha256=digest(output / 'summary.json')))
    native, trials, effects = [], [], []
    for run in cohort:
        name = run['run']
        if run['condition'] == 'baseline':
            initial = next(r['loss'] for r in events[name] if r['event'] == 'initial_probe')
            native.append(dict(run=name, size=run['size'], seed=run['seed'], update=0,
                               loss_after=initial, log_probe_ratio=0.0))
            for row in probes[name]:
                entry = dict(row, run=name, size=run['size'], seed=run['seed'])
                trials.append(entry)
                if row['candidate'] == 'native_adamw':
                    native.append(dict(entry, log_probe_ratio=math.log(row['loss_after'] / initial)))
        else:
            parent, fork = run['parent'], run['junction']
            initial = next(r['loss'] for r in events[name] if r['event'] == 'initial_probe')
            parent_start = next(r['loss_after'] for r in probes[parent]
                if r['candidate'] == 'native_adamw' and r['update'] == fork)
            assert abs(initial - parent_start) < 1e-5, 'Fork probe does not match its parent state'
            parent_end = next(r['loss_after'] for r in probes[parent]
                if r['candidate'] == 'native_adamw' and r['update'] == fork + 50)
            end = summaries[name]['final_probe_loss']
            effects.append(dict(run=name, parent=parent, size=run['size'], seed=run['seed'],
                fork=fork, endpoint=fork + 50, initial_probe_loss=initial,
                native_final_probe_loss=parent_end, ablated_final_probe_loss=end,
                native_log_loss_change=math.log(parent_end / initial),
                ablated_log_loss_change=math.log(end / initial),
                tau=math.log(end / parent_end)))
    contrasts = []
    for fork in [175, 225]:
        for seed in SEEDS:
            by_size = {r['size']: r['tau'] for r in effects if r['fork'] == fork and r['seed'] == seed}
            contrasts.append(dict(fork=fork, seed=seed, tau70_minus_tau14=by_size['70M'] - by_size['14M'],
                tau31_minus_tau14=by_size['31M'] - by_size['14M']))
    return dict(provenance=provenance, native=native, trials=trials,
                branch_effects=effects, size_interactions=contrasts)


def style(ax):
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)


def plot_native(result, directory):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    metrics = [('log_probe_ratio', r'$\ln(F_t/F_0)$'),
               ('relative_loss_decrease', r'Actual decrease $D/F_{t-1}$'),
               ('first_order_decrease', r'Linear prediction $P/F_{t-1}$'),
               ('nonlinear_residual', r'Nonlinear remainder $R/F_{t-1}$')]
    for ax, (key, label) in zip(axes.flat, metrics):
        for size in SIZES:
            selected = [r for r in result['native'] if r['size'] == size and key in r]
            updates = sorted({r['update'] for r in selected})
            values = []
            for seed in SEEDS:
                block = sorted([r for r in selected if r['seed'] == seed], key=lambda r:r['update'])
                y = [r[key] / r['loss_before'] if key in {'first_order_decrease','nonlinear_residual'} else r[key] for r in block]
                ax.plot([r['update'] for r in block], y, color=COLORS[size], alpha=.25, lw=.9)
                values.append(y)
            ax.plot(updates, np.mean(values, axis=0), color=COLORS[size], marker='o', ms=3, label=size)
        ax.set(xlabel='Accepted update', ylabel=label)
        if key != 'log_probe_ratio':
            ax.axhline(0, color='0.5', lw=.6)
        style(ax)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Fixed-probe descent: three seed blocks per model size')
    fig.savefig(directory / 'native_descent_accounting.pdf')
    plt.close(fig)


def plot_branches(result, directory):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    offsets = [-.055, 0, .055]
    for ax, fork in zip(axes, [175, 225]):
        for i, size in enumerate(SIZES):
            values = [next(r['tau'] for r in result['branch_effects'] if r['size']==size and r['seed']==seed and r['fork']==fork) for seed in SEEDS]
            ax.scatter([i+o for o in offsets], values, color=COLORS[size], s=28, alpha=.75)
            ax.scatter([i], [np.mean(values)], marker='_', s=550, linewidths=2.5, color=COLORS[size])
        for seed in SEEDS:
            values = [next(r['tau'] for r in result['branch_effects'] if r['size']==size and r['seed']==seed and r['fork']==fork) for size in SIZES]
            ax.plot(range(3), values, color='0.6', lw=.7, alpha=.4, zorder=0)
        ax.axhline(0, color='0.4', lw=.8)
        ax.set(xticks=range(3), xticklabels=SIZES, title=f'Fork {fork}: updates {fork+1}–{fork+50}',
               ylabel=r'$\tau=\ln F_{\mathrm{no\ precond.}}-\ln F_{\mathrm{native}}$')
        style(ax)
    fig.suptitle('Effect of removing preconditioning at the native adaptive-step norm schedule')
    fig.savefig(directory / 'preconditioning_effects.pdf')
    plt.close(fig)


def plot_trials(result, directory):
    candidates = ['native_adamw','adam','momentum','current_preconditioned','gradient','adam_scale_0.5','adam_scale_2']
    labels = ['Native','Reconst.','No precond.','No averaging','Gradient','Half step','Double step']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, update in zip(axes.flat, [175, 200, 225, 250]):
        for i, size in enumerate(SIZES):
            means = []
            for k, candidate in enumerate(candidates):
                selected = [r['relative_loss_decrease'] for r in result['trials']
                    if r['size']==size and r['update']==update and r['candidate']==candidate
                    and 'relative_loss_decrease' in r]
                assert len(selected) == 3, 'Report undefined direction coverage before plotting'
                ax.scatter([k+(i-1)*.17]*3, selected, color=COLORS[size], alpha=.3, s=13)
                means.append(np.mean(selected))
            ax.plot(np.arange(7)+(i-1)*.17, means, 'o', color=COLORS[size], ms=4, label=size)
        ax.axhline(0, color='0.4', lw=.7)
        ax.set(title=f'Update {update}', xticks=range(7), xticklabels=labels,
               ylabel=r'One-step fixed-probe decrease $D/F_{t-1}$')
        ax.tick_params(axis='x', labelrotation=35)
        style(ax)
    axes[0,0].legend(frameon=False, ncol=3)
    fig.suptitle('Same-state direction and step-length interventions; dots are seeds, large dots means')
    fig.savefig(directory / 'direction_substitutions.pdf')
    plt.close(fig)


def main():
    result = reduce()
    save(HERE / 'results.json', result)
    figures = HERE / 'figures'
    figures.mkdir(exist_ok=True)
    marker = Path('C:/Users/thima/.codex/plugins/cache/openai-primary-runtime/pdf/26.1011.51657/skills/pdf/container_tools/mark_artifact_operation_started.mjs')
    subprocess.run(['node', str(marker), '--operation-kind','create','--expected-output-count','3','--output-format','pdf'], check=True)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    plot_native(result, figures)
    plot_branches(result, figures)
    plot_trials(result, figures)
    lines = []
    for fork in [175,225]:
        for size in SIZES:
            values = [r['tau'] for r in result['branch_effects'] if r['fork']==fork and r['size']==size]
            lines.append(f'| {fork} | {size} | {np.mean(values):.6g} | {min(values):.6g} | {max(values):.6g} |')
    observation = '''# Observation001: norm-controlled preconditioning interventions

Sources: all27 local retrieval receipts and SHA-256 inventories, listed in
`results.json` provenance; frozen cohort/configs/probe; raw `events.jsonl`,
`probe_events.jsonl`, and `summary.json` retained under each owning run.
Generating script: `python analyses/025-2026-10-17-pythia-descent-mechanism/08_analyze.py`.

Coverage:9 baselines (14M/31M/70M, three seed blocks,350 complete updates)
and18 matched50-update branches from175/225. Every baseline has9 probe points
and seven same-state candidates per point. The same128-block fixed probe is
disjoint from all training prefixes; it is a held-out portion of the training
corpus, not a separate validation population. Full original validation is retained
in each run's summary. Reduction is token weighted, with no display smoothing.

The primary effect is tau = ln F_ablated(end) - ln F_native(end), at identical
calendar update and common fork. Positive tau means removing preconditioning
increased probe loss. Adaptive norms follow the unmodified parent's schedule.
The per-seed70M-minus14M and31M-minus14M effects are in `size_interactions`.

| Fork | Size | Mean tau | Minimum seed tau | Maximum seed tau |
| --- | --- | ---: | ---: | ---: |
''' + '\n'.join(lines) + '''

Figure captions:

- `native_descent_accounting.pdf`: fixed-probe log-loss change and actual
  one-step D, linear prediction P, nonlinear remainder R, with D=P-R. Each
  decrease is normalized by that model's pre-step probe loss. Thin curves show
  seeds; thick curves show arithmetic means at measured updates. Segments join
  sparse measurements and do not represent intermediate probe evaluations.
- `preconditioning_effects.pdf`: primary50-update effects after each fork;
  points show all seed blocks, horizontal marks show means, grey segments pair
  the same seed across sizes. No uncertainty interval is inferred from updates.
- `direction_substitutions.pdf`: one-step relative loss decreases at175,200,225,
  250; small dots show three seeds, large dots show their mean. Direction
  substitutions match native adaptive norm within each model; half/double
  controls deliberately change it. Native/reconstructed Adam are numeric controls.

Uncertainty and interpretation: only three seed blocks, with initialization and
data ordering co-varying within a block. Report seed effects rather than treating
updates as replicates. The motivating200–250 interval was selected post hoc.
Within-model interventions identify an optimizer-mediated contribution under
this recipe; cross-size differences do not identify a causal capacity effect.
Adaptive norms are controlled before FP32 rounding; full displacement also has
decay, whose vector changes once trajectories diverge. R includes higher-order
terms and numeric error and is not a curvature measurement. One-step changes
hold state fixed and cannot alone establish a sustained convergence mechanism.
Figures require visual inspection before publication; raw metrics remain local.
'''
    observations = HERE / 'observations'
    observations.mkdir(exist_ok=True)
    (observations / '001-preconditioning-effects.md').write_text(observation, encoding='utf-8')
    print(json.dumps(dict(status='analysis_generated', verified_runs=27, figures=3)))


if __name__ == '__main__':
    main()
