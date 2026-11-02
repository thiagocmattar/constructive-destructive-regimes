"""Verify the nine complete Run213 arms and plot raw calibration diagnostics."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, flat, tensor_hash, write_json
from src.eos_tasks.task import model_for

HERE = Path(__file__).resolve().parent
RUN = ROOT/'runs/213-2026-11-02-eos-adam-from-init-calibration'
spec = importlib.util.spec_from_file_location('calibration_labels', RUN/'02_classify.py')
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)


def rows(file):
    return [json.loads(line) for line in file.read_text().splitlines()]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-available', action='store_true', help='Audit retrieved arms only; do not select rates or create figures.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = json.loads((RUN/'config.json').read_text())
    cache = ROOT/cfg['cache']
    shared = json.loads((cache/'inputs.json').read_text())
    gd_targets = json.loads((HERE/'gd_loss_milestones.json').read_text())
    expected = {(task, rate) for task, rates in cfg['learning_rates_by_task'].items() for rate in rates}
    data = []
    sources = []
    start_gradients = {}
    start_sources = {}
    for file in sorted((RUN/'artifacts/attempt-01').rglob('manifest.json')):
        manifest = json.loads(file.read_text())
        if manifest.get('run_id') != '213' or manifest.get('mechanics_only'):
            continue
        folder = file.parent
        assert manifest['status'] == 'complete', (str(folder), manifest['status'])
        assert manifest['completed_updates'] == cfg['updates'] == 128
        assert manifest['config_sha256'] == file_hash(RUN/'config.json')
        assert manifest['input_manifest_sha256'] == file_hash(RUN/'input_manifest.json')
        assert all(file_hash(RUN/name) == digest for name, digest in manifest['source_sha256'].items())
        assert not (folder/'evaluation.jsonl').exists()
        assert not list(folder.glob('*curvature*'))
        for name, identity in manifest['files'].items():
            artifact = folder/name
            assert artifact.stat().st_size == identity['bytes'] and file_hash(artifact) == identity['sha256']
        boundaries = rows(folder/'boundaries.jsonl')
        metrics = rows(folder/'metrics.jsonl')
        assert [r['state'] for r in boundaries] == list(range(129))
        assert [r['update'] for r in metrics] == list(range(1, 129))
        final = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
        model = model_for(manifest['task'], cache, shared['wikitext']['vocabulary_size'])
        initial = shared['initial'][manifest['task']][cfg['seed']]
        model.load_state_dict(torch.load(cache/initial['path'], weights_only=True))
        assert tensor_hash(model.parameters()) == manifest['initial_tensor_sha256'] == initial['initial_tensor_sha256']
        assert torch.equal(flat(model.parameters()).double(), final['initial_parameters'])
        model.load_state_dict(final['model'])
        assert tensor_hash(model.parameters()) == manifest['final_tensor_sha256']
        displacement = flat(model.parameters()).double()-final['initial_parameters']
        motion_error = float((displacement-final['diagnostics']['actual']['total']).norm())
        assert motion_error < 1e-9
        assert max(r['actual_displacement_identity_residual_norm'] for r in metrics) < 1e-9
        for value in final['optimizer']['state'].values():
            assert int(value['step']) == 128
        for stream in ('raw', 'moment', 'adaptive', 'actual'):
            recorded = [r['streams'][stream] for r in metrics]
            norms = np.asarray([r['norm'] for r in recorded])
            np.testing.assert_allclose([r['A'] for r in recorded], norms.cumsum(), rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose([r['energy'] for r in recorded], (norms**2).cumsum(), rtol=2e-12, atol=2e-12)
            saved = final['diagnostics'][stream]
            assert saved['count'] == 128
            # CPU/GPU reduction order differs for the large Transformer vectors.
            np.testing.assert_allclose(float(saved['total'].norm()), recorded[-1]['B'], rtol=1e-12, atol=1e-10)
            assert 0 <= recorded[-1]['B_over_A'] <= 1+1e-10
        parameters = final['optimizer']['param_groups'][0]['params']
        states = [final['optimizer']['state'][p] for p in parameters]
        moment = flat([state['exp_avg'] for state in states]).double()
        variance = flat([state['exp_avg_sq'] for state in states]).double()
        adaptive = (moment/(1-.9**128))/((variance/(1-.999**128)).sqrt()+1e-8)
        assert torch.equal(moment, final['diagnostics']['moment']['previous'])
        np.testing.assert_allclose(adaptive.numpy(), final['diagnostics']['adaptive']['previous'].numpy(), rtol=2e-14, atol=2e-14)
        # Audits use full CPU vectors, not the online scalar identities alone.
        first = torch.load(folder/'audit-gradient-0.pt', map_location='cpu', weights_only=True)['raw_gradient'].double()
        second = torch.load(folder/'audit-gradient-1.pt', map_location='cpu', weights_only=True)['raw_gradient'].double()
        terminal = torch.load(folder/'audit-gradient-128.pt', map_location='cpu', weights_only=True)['raw_gradient'].double()
        previous = final['diagnostics']['raw']['previous']
        for i, before, after in [(1, first, second), (128, previous, terminal)]:
            q = float(before.dot(before-after))/float(before.dot(before))
            assert abs(q-boundaries[i]['projected_gradient_change']) < 2e-10
        terminal_cosine = float(previous.dot(terminal)/(previous.norm()*terminal.norm()))
        reference = start_gradients.setdefault(manifest['task'], first.clone())
        reference_source = start_sources.setdefault(manifest['task'], folder.relative_to(ROOT).as_posix())
        start_error = float((first-reference).norm()/reference.norm())
        assert start_error < 1e-5, (manifest['task'], start_error)
        losses = [r['loss'] for r in boundaries]
        matches = []
        for condition in gd_targets['conditions']:
            if condition['task'] != manifest['task']:
                continue
            for gd_state, target in condition['targets_by_parameter_state'].items():
                matches.append(dict(gd_run=condition['run_id'], gd_state=int(gd_state), target_loss=target,
                    first_adam_state=next((i for i, loss in enumerate(losses) if loss <= target), None),
                    first_adam_trailing64_state=next((i for i in range(63, len(losses))
                        if float(np.mean(losses[i-63:i+1])) <= target), None),
                    censoring_horizon=128))
        result = dict(task=manifest['task'], lr=manifest['lr'], updates=128, status='complete',
            initial_loss=losses[0], **labels.classify(losses),
            terminal_adjacent_raw_cosine=terminal_cosine,
            tail_raw_cosine_median=float(np.median([r['streams']['raw']['cosine'] for r in metrics[-64:]])),
            tail_stream_cosine_medians={stream: float(np.median([r['streams'][stream]['cosine'] for r in metrics[-64:]]))
                for stream in ('raw', 'moment', 'adaptive', 'actual')},
            tail_moment_over_gradient_norm_median=float(np.median([r['moment_over_gradient_norm'] for r in metrics[-64:]])),
            tail_unbiased_moment_over_gradient_norm_median=float(np.median([r['unbiased_moment_over_gradient_norm'] for r in metrics[-64:]])),
            final_retention={name: metrics[-1]['streams'][name]['B_over_A'] for name in final['diagnostics']},
            gpu=manifest['environment']['gpu'], source=folder.relative_to(ROOT).as_posix(),
            manifest_sha256=file_hash(file), final_checkpoint_bytes=(folder/'final.pt').stat().st_size,
            elapsed_seconds=manifest['elapsed_seconds'],
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows(folder/'events.jsonl')),
            initial_gradient_relative_error_to_case_reference=start_error,
            initial_gradient_reference_source=reference_source,
            final_parameter_telescoping_error=motion_error, gd_loss_matches=matches)
        data.append(dict(result=result, boundaries=boundaries, metrics=metrics, terminal_cosine=terminal_cosine))
        sources.append(result)
        del final, first, second, terminal, previous, moment, variance, adaptive, model, displacement
    keys = [(d['result']['task'], d['result']['lr']) for d in data]
    if args.verify_available:
        assert len(keys) == len(set(keys)) and set(keys).issubset(expected)
        print(json.dumps(dict(verified_arms=len(keys), missing=sorted(expected-set(keys)), conditions=sources)))
        return
    assert len(keys) == len(set(keys)) and set(keys) == expected, (len(keys), expected-set(keys))
    result = dict(status='nine_arms_verified', updates=1152, conditions=sources,
        regime_candidates={task: labels.select_regimes([r for r in sources if r['task'] == task]) for task in cfg['tasks']},
        limitations=['128-step training-only pilot; loss labels are provisional.', 'One initialization per case; no seed uncertainty.',
            'No Hessian/HVP measurements or curvature/stability claims.', 'MLP on one owned laptop; each other case uses matched 5090 class across its LRs.'])
    write_json(HERE/'fast_calibration_results.json', result)
    (HERE/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'serif', 'font.size':9, 'pdf.fonttype':42,
        'axes.spines.top':False, 'axes.spines.right':False})
    colors = ['#0072B2', '#D55E00', '#009E73']
    titles = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
    figure, axes = plt.subplots(4, 3, figsize=(10, 8), squeeze=False)
    plotted = []
    def line(ax, x, y, descriptor, **kwargs):
        y = np.asarray([np.nan if v is None else v for v in y], dtype=float)
        artist, = ax.plot(x, y, **kwargs)
        np.testing.assert_array_equal(artist.get_ydata(), y)
        plotted.append(dict(descriptor, x=list(map(int, x)), y=[None if not np.isfinite(v) else float(v) for v in y]))
    for col, task in enumerate(cfg['tasks']):
        subset = sorted([d for d in data if d['result']['task'] == task], key=lambda d:d['result']['lr'])
        for color, d in zip(colors, subset):
            rate = d['result']['lr']; b = d['boundaries']; m = d['metrics']
            desc = dict(task=task, lr=rate)
            x = np.arange(129)
            line(axes[0,col], x, [r['loss'] for r in b], dict(desc, panel='loss'), color=color, label=f'{rate:g}')
            raw_cosines = [r['streams']['raw']['cosine'] for r in m]+[d['terminal_cosine']]
            line(axes[1,col], x, raw_cosines, dict(desc, panel='raw_cosine'), color=color, lw=.9)
            line(axes[2,col], x, [r['projected_gradient_change'] for r in b], dict(desc, panel='projected_change'), color=color, lw=.9)
            line(axes[3,col], np.arange(128), [r['streams']['raw']['norm'] for r in m], dict(desc, panel='raw_norm'), color=color)
            line(axes[3,col], np.arange(128), [r['streams']['moment']['norm'] for r in m], dict(desc, panel='moment_norm'), color=color, ls='--', lw=.9)
        axes[0,col].set_title(titles[col]); axes[0,col].legend(title='Constant LR', fontsize=8)
        axes[1,col].set_ylim(-1.05, 1.05); axes[1,col].axhline(0, color='.6', lw=.5)
        axes[2,col].axhline(1, color='.6', lw=.5)
        axes[3,col].set_yscale('log'); axes[3,col].set_xlabel('Parameter state / gradient index')
        for row in range(4): axes[row,col].grid(alpha=.2)
    for row, title in enumerate(['Full training loss', 'Adjacent raw cosine', 'Projected change q', 'Norm: raw solid / m dashed']):
        axes[row,0].set_ylabel(title)
    figure.tight_layout()
    figure.savefig(HERE/'figures/fast_calibration.pdf', metadata={'CreationDate':None})
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(10, 5.5), squeeze=False)
    streams = ['raw','moment','adaptive','actual']; stream_colors = ['#0072B2','#D55E00','#009E73','#777777']
    for col, task in enumerate(cfg['tasks']):
        subset = sorted([d for d in data if d['result']['task'] == task], key=lambda d:d['result']['lr'])
        for style, d in zip(['-', '--', ':'], subset):
            for stream, color in zip(streams, stream_colors):
                m = d['metrics']; desc=dict(task=task, lr=d['result']['lr'], stream=stream)
                for row, field in enumerate(['cosine','B_over_A']):
                    line(axes[row,col], np.arange(1,129), [r['streams'][stream][field] for r in m],
                        dict(desc, panel='streams_'+field), color=color, ls=style, lw=.9,
                        label=stream if style == '-' and row == 0 else None)
        axes[0,col].set_title(titles[col]); axes[0,col].set_ylim(-1.05,1.05)
        axes[1,col].set_ylim(0,1.03); axes[1,col].set_xlabel('Accepted optimizer update')
        axes[0,col].legend(fontsize=7, ncol=2)
        for row in range(2):axes[row,col].grid(alpha=.2)
    axes[0,0].set_ylabel('Adjacent stream cosine'); axes[1,0].set_ylabel('Complete-prefix B/A')
    figure.suptitle('LR within each case: lowest solid, middle dashed, highest dotted', fontsize=10)
    figure.tight_layout(rect=(0,0,1,.96))
    figure.savefig(HERE/'figures/fast_calibration_streams.pdf', metadata={'CreationDate':None})
    plt.close(figure)
    write_json(HERE/'fast_calibration_plot_values.json', dict(curves=plotted,
        source_results_sha256=file_hash(HERE/'fast_calibration_results.json'),
        source_gd_targets_sha256=file_hash(HERE/'gd_loss_milestones.json'),
        figures={name:file_hash(HERE/'figures'/name) for name in ['fast_calibration.pdf','fast_calibration_streams.pdf']}))
    print(json.dumps(dict(verified_arms=9, accepted_updates=1152, curves=len(plotted))))


if __name__ == '__main__':
    main()
