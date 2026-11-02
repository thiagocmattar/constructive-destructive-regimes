"""Closeout records for the requested backward-anchor comparison."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
RUN=ROOT/'runs/212-2026-11-02-eos-backward-anchors'


def main():
    result=json.loads((HERE/'results.json').read_text())['profiles']
    v=json.loads((HERE/'verification.json').read_text())
    review=json.loads((HERE/'figure_review.json').read_text())
    close=json.loads((HERE/'closeout.json').read_text())
    historical=json.loads((HERE/'historical_comparison.json').read_text())['profiles']
    assert v['status']=='passed' and review['status']=='passed' and close['status']=='complete'
    elapsed=(close['closed_epoch']-min(p['created_epoch'] for p in close['pods']))/60
    cost=close['compute_estimate_usd']
    lines=[]
    for task in ['mlp','resnet','transformer']:
        for start in [500,2000,3000]:
            pair={opt:next(x for x in result if x['job']['task']==task and x['job']['start']==start
                          and x['job']['optimizer']==opt) for opt in ['gd','adam']}
            gd,adam=pair['gd'],pair['adam']
            lines.append(f'| {task} | {start} | {start+256} | {gd["job"]["lr"]:g} | {adam["job"]["lr"]:g} | '
                f'{gd["rows"][0]["cosine"]:.6f} | {adam["rows"][0]["cosine"]:.6f} | '
                f'{gd["scalar_rows"][-1]["loss"]:.6f} | {adam["scalar_rows"][-1]["loss"]:.6f} |')
    report=f'''# Analysis035: fixed endpoints looking backward through 256 updates

Completed the requested separate [GD](figures/gd_backward_lag256.pdf) and
[Adam](figures/adam_backward_lag256.pdf) figures. Every curve fixes one endpoint
gradient g_t and plots cos(g_t,g_(t-k)) for k=1..256, in order. Source starts
500/2000/3000 become endpoints 756/2256/3256 after exactly 256 optimizer updates.
Rows are these later endpoints; columns are MLP, ResNet-32 and Transformer.
Each panel has one raw curve, common linear axes and no parity split or smoothing.

This implements the user's clarified backward definition. Historical
Analysis033/034 largest-rate figures were forward anchored cos(g_s,g_(s+k));
the adjacent figures were moving lag-one pairs. Neither former scalar series
can be reversed to obtain the new profile. The new endpoint's k=1 agrees with
its final adjacent pair, and k=256 agrees with its start-to-end cosine by
symmetry. Both identities pass in all 18 profiles.

Run212 computes 18 full-objective continuations from the same nine directly saved
Run205 states (all primary): largest source GD rates 0.04/0.1/0.2 from
Run122/120/126; Adam rates 0.0003/0.001/0.001 remain those calibrated in
Analysis034. Adam moments reset at the retained source, with beta1=0.9,
beta2=0.999, epsilon 1e-8, no clipping/decay/schedule/AMSGrad, foreach/fusedFalse.
GD is constant-rate plain SGD without momentum/decay. No new rate selection.

All source tensors and saved-gradient/chunk qualifications pass. Full data,
ordering, losses and parameter scope match the earlier contracts: MLP first 5000
CIFAR-10 images with half squared error averaged over images; ResNet all 50000
images with 50 fixed 1000-image train-mode BN groups; Transformer all 59680
sequences and 2088600 valid targets with token-weighted NLL. FP32 parameters and
accumulated raw gradients, TF32 disabled, exact FP64 dot products/norms.
Transformer chunk 512 qualifies against 64. Hardware and FP32 reduction differences
can change later trajectories; this is a recomputation from identical starts,
without a claim of bitwise equality to historical continuations.
The [historical scalar comparison](historical_comparison.json) records all 257
boundaries against the earlier forward profiles, including maximum loss,
relative-norm and forward-cosine differences; it never selects a result or horizon.
The observed maxima are {max(x['maximum_loss_difference'] for x in historical):.6f}
in absolute training loss, {max(x['maximum_norm_relative_difference'] for x in historical):.6f}
in relative gradient norm, and {max(x['maximum_forward_cosine_difference'] for x in historical):.6f}
in forward cosine. Later paths differ substantially in some cases. These figures
describe newly recomputed continuations from matched starts, rather than an
exact reproduction of the historical trajectories.

| Model | Start s | Fixed endpoint t | GD LR | Adam LR | GD rho_t(1) | Adam rho_t(1) | GD endpoint train loss | Adam endpoint train loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(lines)}

Validation covers 4626 gradients and 4608 backward pairs. Every pair has independent
GPU/CPU FP64 checks (maximum cosine difference {v['maximum_cpu_gpu_cosine_error']:.3g});
90 additional local pair audits use retained full vectors (maximum error
{max(x['error'] for x in v['local_full_vector_pair_audits']):.3g}), and all 108 retained
vectors match their training-time hashes. Every plotted value matches the JSON,
CSV and artist arrays exactly. Both one-page vector PDFs pass structural and
actual-render visual review. [Checks](verification.json), [figure review](figure_review.json),
[all values](lag_profiles.csv), [results](results.json),
[observation/captions](observations/001-backward-anchors.md).

The temporary 257-gradient histories are reduced only after the endpoint is
available. Retain all raw pair products/norms, scalar histories and per-gradient
hashes, six full vectors per arm (offsets 0/128/192/254/255/256), terminal
model/optimizer/RNG recovery and source checkpoints. Dense histories are then
discarded on the worker to honor the request to simplify/accelerate execution.
No random projection or lossy gradient compression is used.

One seed-0 initialization per architecture and dependent checkpoint forks;
one pair per lag, without error bars or averaging endpoints. Optimizers change
both transformation and calibrated numerical step scale; actual update norms
are not matched. The figures do not establish permanent damping, stability,
generalization or optimizer superiority. No new validation/test or manuscript edit.

Five initial ResNet placements were retired before training: four failed the
saved-gradient threshold on RTX 5090, and one failed dependency setup. All five
were retried with unchanged scientific inputs on RTX 4090; failed logs and
verification receipts remain available. A transient FUSE-volume stall in one
Transformer arm cleared without an optimizer restart or scientific-input change.
The initial slow environment upload was replaced by direct official downloads
with source package versions pinned. [Execution notes](execution_notes.json).

All {len(close['pods'])} task Pods and Pod storage were deleted after per-worker
byte/SHA-256 and archive verification: {v['transferred_files']} files,
{v['transferred_bytes']/1e9:.3f} GB locally retained. No task network volume or
remaining cloud storage. Allocation through final deletion took {elapsed:.1f} min;
compute estimate USD {cost:.2f}, provisional pending provider billing; USD 15 task
ceiling. Unrelated account volumes and Spark were preserved. [Closeout](closeout.json).

Generators: `03_analyze.py` (audits/figures), `04_records.py` (records).
Executed scientific inputs: Run212 `config.json`, `input_manifest.json`, `01_worker.py`.
'''
    (HERE/'REPORT.md').write_text(report,encoding='utf-8',newline='\n')
    obs=HERE/'observations';obs.mkdir(exist_ok=True)
    (obs/'001-backward-anchors.md').write_text('''# Observation001: fixed endpoint raw gradients compared with their past

Question: display backward anchored cosines through lag 256, separately for GD
and Adam, using the largest GD-rate source histories and existing calibrated
Adam rates. The user chose endpoints reached along the Adam continuations,
so every earlier gradient in the Adam window belongs to that same Adam fork.

Sources: Run212 frozen input manifest/config and complete per-arm results,
scalar histories, CPU/GPU product records, source qualifications, gradient
hashes and six full audit vectors. Inputs are the identical Run205 starts used
by Analysis033/034; the old forward/adjacent scalar curves are not substituted.
Numerical audits are in `verification.json`; figure/source identities and
actual-PDF visual checks are in `figure_review.json`.

Reduction: for each retained start s=500/2000/3000, run exactly 256 updates and
fix t=s+256. For k=1..256 compare g_t with g_(t-k). One pair per lag, all trainable
parameters once in model order, complete original objective, FP32 raw gradients
and FP64 dot/norm reductions. No epsilon; a zero denominator is undefined.
No averaging across endpoints, centering, smoothing, parity split, projection,
envelope or downsampling. Exact scalar arrays are checked independently against
the CSV and each plotted artist. Each grid has 2304 values and shared x=1..256,
y=-1.05..1.05. The plotted endpoint gradient is fixed throughout every panel.

Caption, `figures/gd_backward_lag256.pdf`: Backward anchored whole-model
raw-gradient cosine cos(g_t,g_(t-k)) along GD. Columns: CIFAR-10 MLP (GD LR 0.04,
source Run122), CIFAR-10 ResNet-32 (0.1, Run120), WikiText-2 Transformer (0.2,
Run126). Rows: endpoint t=756/2256/3256 reached from saved starts 500/2000/3000.
Every lag1–256 is connected in order as one curve, with a horizontal zero line.
Increasing lag walks backward toward the retained start.

Caption, `figures/adam_backward_lag256.pdf`: The same backward anchored statistic
along cold Adam from the identical nine GD-trained source states. Adam LRs:
MLP 3e-4, ResNet-32/Transformer 1e-3; both Adam and source GD rates appear in the
column labels. Same endpoints, 256-update budget, axes, styling and coverage as
the GD figure. Equal absolute endpoint indices do not mean equal GD/Adam weights.

Coverage and uncertainty: 18 profiles from nine directly saved primary states,
one seed-0 initialization per architecture, 256 anchored pairs per profile,
no seed uncertainty bands. Full source data/order/objectives are unchanged;
Transformer chunk 512 passes the original64-chunk qualification. Hardware and
FP32 reductions change later trajectories despite matched initial tensors;
`historical_comparison.json` reports substantial differences in some cases.
These are newly recomputed continuations, not identical historical trajectories.
The complete temporary gradient histories are discarded after exact products
and audit-vector extraction; all scalar measurements and gradient hashes,
selected full vectors and optimizer/RNG recovery remain locally verified.
Losses are training only. A cosine curve alone does not establish stability,
generalization, optimizer benefit or asymptotic damping. Backward anchored
cosines, historical forward anchored cosines and moving adjacent cosines answer
different questions and must keep their labels.

Generator: `03_analyze.py`. Publication outputs are the two vector PDFs;
actual-PDF previews are ignored `scratch/gd_backward_lag256.png` and
`scratch/adam_backward_lag256.png`. The report tabulates endpoint lag-one
cosines and losses without ranking optimizers.
''',encoding='utf-8',newline='\n')
    (obs/'INDEX.md').write_text('# Analysis035 observations\n\n- [001: Fixed endpoint raw gradients compared with their past](001-backward-anchors.md).\n',encoding='utf-8',newline='\n')
    readme=HERE/'README.md';text=readme.read_text(encoding='utf-8')
    text=text.replace('Status: preparation.','Status: running.').replace('Status: running.',f'Completed: [report](REPORT.md), [GD figure](figures/gd_backward_lag256.pdf), '
        f'[Adam figure](figures/adam_backward_lag256.pdf). All task cloud deleted; compute USD{cost:.2f}, provisional.')
    readme.write_text(text,encoding='utf-8',newline='\n')
    run_readme=RUN/'README.md'
    run_readme.write_text(run_readme.read_text(encoding='utf-8').split('\nCompleted:')[0].rstrip()+'\n',encoding='utf-8',newline='\n')
    with run_readme.open('a',encoding='utf-8',newline='\n') as f:
        f.write(f'\nCompleted: 18 profiles / 4608 retrospective pairs; qualified starts, 90 local vector audits, '
                f'all cosine/recovery/figure checks pass. All task compute/storage deleted; '
                f'compute USD{cost:.2f}, provisional. [Report](../../analyses/035-2026-11-02-eos-backward-anchors/REPORT.md).\n')
    print(json.dumps(dict(report='REPORT.md',observations=1,figures=2,cost_provisional_usd=cost)))


if __name__=='__main__':main()
