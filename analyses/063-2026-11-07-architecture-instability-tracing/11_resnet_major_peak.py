"""Re-reduce a larger ResNet loss event from authenticated untreated captures.

No training, new probe, intervention, or SVD fit. The selected completed-step
window is 390--450, around the last recovered >100% loss excursion before 512.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'runs/273-2026-11-07-resnet-distributed-displacement'
WORKER = RUN / 'artifacts/resnet-a/resnet-distributed'
HISTORY = ROOT / 'runs/223-2026-11-04-multispike-interventions/artifacts/duqoihms8w1gle/payload/worker/control/boundaries.jsonl'
EVENTS = ROOT / 'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py'
STAGES = [
    ('conv1', 'conv1.call0.output', 'Conv 1'),
    ('layer1', 'layer1.4.relu.call1.output', 'Stage 1'),
    ('layer2', 'layer2.4.relu.call1.output', 'Stage 2'),
    ('layer3', 'layer3.4.relu.call1.output', 'Stage 3'),
    ('avgpool', 'avgpool.call0.output', 'Pooling'),
    ('output', 'output.log_probabilities', 'Output'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return {r['state']: r for r in map(json.loads, Path(path).read_text().splitlines())}


def main():
    source = WORKER / 'control/boundaries.jsonl'
    inventory = read(WORKER / 'control/transfer_inventory.json')['files']
    pinned = next(r for r in inventory if r['path'] == 'boundaries.jsonl')
    assert sha(source) == pinned['sha256'] and source.stat().st_size == pinned['bytes']
    manifest = read(WORKER / 'control/manifest.json')
    assert manifest['status'] == 'completed' and not manifest['mechanics_only']
    assert manifest['arm']['kind'] == 'control'
    assert (manifest['training_units'], manifest['probe_units']) == (50000, 1000)
    audit = read(WORKER / 'representation_audit.json')
    assert audit['status'] == 'passed'
    assert audit['source_sha256'] == sha(ROOT / 'analyses/065-2026-11-07-mlp-frozen-lr-double/18_verify_representations.py')
    assert read(WORKER / 'shared/qualification.json')['direct2_vs_resume_exact']

    # Pin both the original-history identity and the identical data/initialization.
    inputs = {r['path']: r for r in read(RUN / 'input_manifest.json')['files']}
    old_inputs = {r['path']: r for r in read(ROOT / 'runs/243-2026-11-07-architecture-instability-tracing/input_manifest.json')['files']}
    identity_paths = ['cache/eos-multitask/resnet_data.pt', 'cache/eos-multitask/resnet_initial_seed0.pt']
    for path in identity_paths:
        assert inputs[path]['sha256'] == old_inputs[path]['sha256']
    history_identity = next(r for r in inputs.values() if r.get('local_path') == HISTORY.relative_to(ROOT).as_posix())
    assert sha(HISTORY) == history_identity['sha256']
    assert history_identity['sha256'] == old_inputs['runs/243-2026-11-07-architecture-instability-tracing/inputs/resnet/boundaries.jsonl']['sha256']
    b, original = rows(source), rows(HISTORY)
    assert list(b) == list(range(264, 578))
    shared = sorted(set(b) & set(original))
    np.testing.assert_array_equal([b[s]['loss'] for s in shared], [original[s]['loss'] for s in shared])

    # Apply the unchanged event definition to this continuation, not Run215's
    # numerically different historical trajectory (also true of the old trace).
    spec = importlib.util.spec_from_file_location('loss_events', EVENTS)
    detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detector)
    events = detector.excursions([r['loss'] for r in original.values()])
    for event in events:
        for key in ['before_state', 'onset_state', 'peak_state', 'recovery_state', 'end_state']:
            if event[key] is not None:
                event[key] += min(original)
        event['loss_before'] = original[event['before_state']]['loss']
        event['loss_peak'] = original[event['peak_state']]['loss']
        event['peak_change_percent'] = 100 * (event['loss_peak'] / event['loss_before'] - 1)
    candidates = [r for r in events if r['recovery_state'] and int(r['peak_state']) < 512
                  and float(r['peak_change_percent']) > 100]
    event = max(candidates, key=lambda r: int(r['peak_state']))
    base, peak, recovery = [int(event[k]) for k in ['before_state', 'peak_state', 'recovery_state']]
    assert (base, peak, recovery) == (404, 415, 439)
    assert float(event['loss_before']) == b[base]['loss'] and float(event['loss_peak']) == b[peak]['loss']
    begin, end = 390, 450
    x = np.arange(begin, end + 1)
    loss = np.array([b[int(s)]['loss'] for s in x])
    probe = np.array([b[int(s)]['probe_loss'] for s in x])
    assert x[loss.argmax()] == peak and x[:loss.argmax()][loss[:loss.argmax()].argmin()] == base
    crossing = next(s for s in range(base + 1, peak + 1) if b[s]['loss'] >= 1.1 * b[base]['loss'])

    old = read(HERE / 'resnet_results.json')
    overlap = [r for r in old['boundaries'] if r['state'] in b]
    np.testing.assert_array_equal([r['loss'] for r in overlap], [b[r['state']]['loss'] for r in overlap])
    stages, errors = [], []
    for name, old_name, label in STAGES:
        for metric in ['rho', 'change_rms']:
            left = np.array(old['curves'][old_name][metric])
            right = np.array([b[t + 1]['representations'][name][metric] for t in old['states']])
            np.testing.assert_allclose(left, right, rtol=2e-12, atol=2e-14)
            errors.append(float(np.max(np.abs(left - right))))
        values = {key: [b[int(s)]['representations'][name][key] for s in x]
                  for key in ['rho', 'change_rms', 'activation_rms']}
        assert np.isfinite(np.array(list(values.values()))).all()
        rho, rms = np.array(values['rho']), np.array(values['change_rms'])
        assert np.max(np.abs(rho)) <= 1 + 1e-12 and np.all(rms >= 0)
        stages.append(dict(name=name, original_name=old_name, label=label,
                           coordinates=b[begin]['representations'][name]['coordinates'], **values,
                           reversals=int(np.sum(rho < 0)), maximum_rms=float(rms.max()),
                           maximum_rms_step=int(x[rms.argmax()])))
    output_rho = np.array(stages[-1]['rho'])
    first_negative = int(np.flatnonzero(output_rho < 0)[0])
    for stage in stages:
        stage['output_alignment_correlation'] = float(np.corrcoef(stage['rho'], output_rho)[0, 1])
    previous = next(r for r in events if int(r['peak_state']) == 293)
    sources = [source, WORKER / 'control/transfer_inventory.json', WORKER / 'control/manifest.json',
               WORKER / 'representation_audit.json', WORKER / 'shared/qualification.json',
               RUN / 'config.json', RUN / 'input_manifest.json', HISTORY, EVENTS,
               HERE / 'resnet_results.json', ROOT / 'runs/243-2026-11-07-architecture-instability-tracing/input_manifest.json']
    result = dict(status='verified', generator_sha256=sha(__file__), case='resnet', arm='control',
                  completed_steps=x.tolist(), source_outgoing_window=[begin - 1, end - 1],
                  selection='Last recovered >100% full-loss excursion with peak before step 512; window fixed at 390--450 using loss only.',
                  loss=loss.tolist(), probe_loss=probe.tolist(), stages=stages,
                  event=dict(baseline_step=base, baseline_loss=b[base]['loss'], first_rise_step=base + 1,
                             peak_step=peak, peak_loss=b[peak]['loss'], recovery_step=recovery,
                             rise_percent=float(event['peak_change_percent']), loss_threshold_step=crossing,
                             first_observed_output_reversal_step=int(x[first_negative]),
                             reversal_left_censored=first_negative == 0),
                  previous_event=dict(peak_step=293, rise_percent=float(previous['peak_change_percent'])),
                  qualification=dict(native_run223_loss_states_exact=len(shared), prior_trace_loss_states_exact=len(overlap),
                                     trajectory_scope='Exact match to Run223 native control and prior Run243 trace on their overlap; not bitwise replay of the original Run215 overview.',
                                     prior_trace_stage_values_checked=2 * len(STAGES) * len(old['states']),
                                     maximum_prior_trace_metric_difference=max(errors),
                                     shared_input_sha256={p: inputs[p]['sha256'] for p in identity_paths},
                                     original_hardware_representation_audit_max_error=audit['max_absolute_error']),
                  measurement=dict(training_images=50000, probe_images=1000, stages=6, seed=0, lr=.003,
                                   output='All-class log probabilities', clock='s completed updates; incoming A_s-A_(s-1)',
                                   probe='First intact original 1,000-image BatchNorm group; unchanged training mode/context',
                                   smoothing='none', rms_scale='Each stage divided by its own maximum within 390--450',
                                   new_training=False, new_svd=False),
                  sources={p.relative_to(ROOT).as_posix(): sha(p) for p in sources})
    (HERE / 'resnet_390_450_motion.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: result[k] for k in ['status', 'event', 'previous_event', 'qualification']}, indent=2))


if __name__ == '__main__':
    main()
