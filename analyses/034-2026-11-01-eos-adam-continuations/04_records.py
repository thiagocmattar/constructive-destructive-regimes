"""Render analysis records from audited results and provider closeout."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent


def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    v=json.loads((HERE/'verification.json').read_text());c=v['counts']
    rows=json.loads((HERE/'profile_summary.json').read_text())['profiles']
    decisions=json.loads((HERE/'calibration_decision.json').read_text())['decisions']
    close=json.loads((HERE/'closeout.json').read_text())
    primary=[r for r in rows if r['primary_eligible']]
    by_model={}
    for name in ['mlp','resnet','transformer']:
        group=[r for r in primary if r['task']==name]
        by_model[name]=dict(profiles=len(group),terminal_median_adjacent_positive=sum(r['terminal64_median_adjacent']>0 for r in group),
            terminal_has_negative_pairs=sum(r['terminal64_negative_fraction']>0 for r in group))
    summary=dict(**c,selected_rates={k:d['selected_lr'] for k,d in decisions.items()},
        primary_terminal_median_adjacent_positive=sum(r['terminal64_median_adjacent']>0 for r in primary),
        primary_terminal_negative_pairs=sum(r['terminal64_negative_fraction']>0 for r in primary),
        primary_terminal_anchor_half_gap_at_most_point1=sum(r['anchor_terminal_half_gap']<=.1 for r in primary),
        pods_deleted=len(close['pods']),compute_estimate_usd=close['compute_estimate_usd'],
        elapsed_minutes=(close['closed_epoch']-min(p['created_epoch'] for p in close['pods']))/60,
        maximum_lag=max(r['horizon'] for r in rows),by_model=by_model,cloud_storage_remaining_gb=0,billing='provisional')
    save(HERE/'summary.json',summary)
    selection='\n'.join(f'| {name} | {d["selected_lr"]:g} | {d["primary_checkpoints"]} |' for name,d in decisions.items())
    table='\n'.join(f'| {r["task"]} / {r["parent_run_id"]} | {r["anchor"]} | {r["horizon"]} | '
        f'{r["adam_lr"]:g} | {r["adjacent_first"]:.6f} | {r["terminal64_median_adjacent"]:.6f} | '
        f'{100*r["terminal64_negative_fraction"]:.2f}% | {r["anchor_terminal_half_gap"]:.6f} | '
        f'{r["final_loss"]:.6f} | {"primary" if r["primary_eligible"] else "unqualified"} |' for r in rows)
    report=f'''# Analysis034: Adam from the same EOS checkpoints

Completed the authorized calibration and matched-horizon continuation at all 27
retained states from Analysis032/033. Runs209/211 test five cold-Adam rates;
Run210 restarts at the identical original model states with the selected rate.
There are 25 primary states and two previously unqualified Run062 reconstructions.
Production records {c['production_cosines']:,} anchored/adjacent cosine values at positive lags
through the same individual GD horizons, up to lag {summary['maximum_lag']}.

| Model | Selected constant Adam LR | Primary calibration states |
| --- | ---: | ---: |
{selection}

The terminal 64-update median adjacent raw-gradient cosine is positive in
{summary['primary_terminal_median_adjacent_positive']}/{len(primary)} primary profiles;
{summary['primary_terminal_negative_pairs']}/{len(primary)} contain at least one negative adjacent pair in that window.
The terminal anchored odd/even half-gap is at most 0.1 in
{summary['primary_terminal_anchor_half_gap_at_most_point1']}/{len(primary)} primary profiles.
These statistics answer different questions: anchored similarity measures
orientation relative to the initial gradient, while adjacent similarity measures
how the raw-gradient direction turns between successive states. Positive adjacent alignment can coexist
with a small or negative anchored cosine. Neither sign establishes loss benefit,
stability, or permanent/asymptotic damping.

Positive terminal median adjacent cosines occur in
{by_model['mlp']['terminal_median_adjacent_positive']}/{by_model['mlp']['profiles']} primary MLP profiles,
{by_model['resnet']['terminal_median_adjacent_positive']}/{by_model['resnet']['profiles']} ResNet-32 profiles and
{by_model['transformer']['terminal_median_adjacent_positive']}/{by_model['transformer']['profiles']} Transformer profiles.
All {by_model['transformer']['terminal_has_negative_pairs']} primary Transformer profiles contain negative adjacent pairs
in the final window. These counts describe dependent forks at the individual
256/512/2048 horizons listed below; they do not rank optimizers or models.

The same seed 0 model, full data/order and original 500/2000/3000 states are matched.
There are 21 directly saved historical starts, four already-qualified legacy
reconstructions and two previously unqualified reconstructions. Every new fork
uses the identical retained Run205 model tensors used by the previous analysis;
the latter two starts remain outside primary summaries.
The three parent GD learning rates identify different histories, not three Adam
rates. Adam starts with zero moments and step 0 at each fork, beta1=0.9,
beta2=0.999, epsilon 1e-8, constant LR, no decay/clipping/schedule/AMSGrad.
This is an optimizer-reset intervention from GD-trained parameters; it is not
Adam training from initialization or continuation of pre-existing Adam moments.
The intervention changes the optimizer and its calibrated numerical LR together;
GD/Adam actual step norms are not matched. Curve differences therefore do not
separate the causal roles of momentum, preconditioning and step scale.

Calibration uses 64 full-objective updates for each rate
1e-5/3e-5/1e-4/3e-4/1e-3 and each of 27 starts. Exclude the two unqualified starts
from selection. Require every primary state to remain finite, peak loss no more
than 4 times initial and final 16 mean loss no more than 1.05 times initial (plus 1e-6).
Minimize equal-checkpoint mean log(final 16 mean/initial), separately per model.
This is conditional short-horizon training-loss tuning, not a globally optimal
rate or independent evidence of optimizer superiority. Cosines and test data
never select the rate; no validation/test evaluation is added.
The ResNet-32 and Transformer selections are at the upper grid boundary, so this
calibration does not establish that larger rates would be worse.
The 64-update calibration does not constrain later training-loss excursions;
the complete production loss histories are retained alongside the cosines.

The MLP objective is half squared error summed over outputs and averaged over
the same first 5000 CIFAR-10 images. ResNet-32 uses all 50000 images with the same
50 fixed 1000-image train-mode BN groups and no running-stat mutation. Transformer
uses all 59680 fixed sequences and 2,088,600 valid targets, token-weighted NLL.
All trainable parameters contribute once in model order; FP32 parameters/gradients,
TF32 disabled, FP64 diagnostic reductions. Transformer chunk 512 qualifies against
the original 64 at every start within 1e-5 relative gradient and 1e-5 absolute loss.
It changes the FP32 accumulation partition, not the objective or token weighting.

Each lag k is one optimizer boundary; g[k] is the complete raw gradient at
theta[k] before the Adam update. Record cos(g0,g[k]) and cos(g[k-1],g[k]) every
boundary, with first adjacent/zero-denominator pairs undefined. No gradient
projection or quantization is used; archive compression is lossless.
The last sampled gradient is unapplied, so optimizer clocks equal the final lag.
Every GPU FP64 cosine and gradient norm is independently checked against CPU NumPy FP64;
maximum cosine disagreement {v['maximum_cpu_gpu_cosine_error']:.3g}.
The first three and terminal two full gradients, model/Adam/Torch CPU/CUDA RNG states, source
checkpoints and complete scalar histories are retained locally. All
{c['vector_audits']} arm-level vector audits pass. Selected calibration and
production prefixes are independently repeated; {c['calibration_prefixes_within_1e_minus5']}/27
have loss, norm and cosine differences below 1e-5 at every initial 64-update
boundary. All matched initial tensors and boundary checks pass; the full
64-update numerical drift is retained without a claim of bitwise trajectory replay.

Run211 distributes the unchanged 10 calibration arms at Run121 states 2000/3000
over already calibrated warm workers. This placement is canonical by a frozen
infrastructure plan before its outcomes are observed; partial/redundant original
Run209 arms remain visible and never replace a worse outcome. Production places
those two short-horizon continuations on warm ResNet workers. No data, LR grid,
objective, initialization, precision or qualification threshold changes.
Two initial 5090 workers failed the declared chunk qualification before Adam
updates. Their failure logs are retained; qualified 4090 placements were used.

| Task / parent GD run | Anchor | Last lag | Adam LR | First adjacent | Last 64 median adjacent | Last 64 negative pairs | Anchored half-gap | Final train loss | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{table}

The GD anchored reference in the figures is reused without new training.
Historical GD does not supply a complete adjacent-gradient series after lag 10;
the newly measured adjacent curves are Adam only. Horizons are matched to the
prior GD observation, not selected by the new Adam outcomes. There is one
initialization and dependent checkpoint forks; no seed uncertainty, stationary
autocorrelation, curvature interpretation of Adam, or generalization claim.

Four vector PDFs show [calibration](figures/calibration.pdf),
[anchored Adam/GD](figures/anchored_profiles.pdf),
[adjacent Adam](figures/adjacent_profiles.pdf) and its
[first 128-update view](figures/adjacent_first128.pdf). All raw points are retained;
no smoothing or path simplification. [Observation/captions](observations/001-adam-continuations.md),
[every production value](lag_profiles.csv), [results](results.json),
[summary](summary.json), [verification](verification.json).

All {summary['pods_deleted']} physical task Pods were deleted after per-file
byte/SHA-256 and archive verification. Retrieved {c['transferred_files']} files /
{c['transferred_bytes']/1e9:.3f} GB. No task cloud storage remains. Allocation through
last deletion took {summary['elapsed_minutes']:.1f} minutes; aggregate compute
estimate USD {summary['compute_estimate_usd']:.2f}, provisional pending complete
provider billing. The task stayed under its USD 50/4h30 bounds. Spark and unrelated
resources were preserved. [Closeout](closeout.json), [billing](billing.json).
'''
    (HERE/'REPORT.md').write_text(report,encoding='utf-8',newline='\n')
    observation='''# Observation001: adjacent and anchored raw gradients under cold Adam

Sources: Run205 retained start checkpoints and original gradients; Runs209/211's
five-rate calibration; Run210's full-objective production records; Analysis033
GD anchored profiles. Source/input identities, complete scalar reductions and
transfer receipts are linked from the report. Generator: `02_analyze.py`.

Reduction: each cosine is a full-vector FP64 dot divided by Euclidean norms,
checked on GPU and independently on CPU; no projection, smoothing or epsilon.
Calibration equal-weights primary checkpoints. Last 64 summaries use raw adjacent
pairs; anchored half-gap is the median of 32 complete odd/even half differences.

Coverage/uncertainty: 27 dependent forks from one seed 0 initialization per model,
25 primary, two flagged historical reconstructions. All three original anchors;
individual 256/512/2048 horizons match GD. No seed error bars or independent
LR-tuning evaluation. Training objective only, no validation/generalization claim.

Figure captions:

- `calibration.pdf`: final 16 mean training loss / initial loss after 64 Adam
  updates at five rates. Thin lines are primary retained states; the black curve
  is their geometric mean, equivalent to the mean-log selection score. Red
  crosses fail at least one declared safeguard. The dashed vertical line is the
  common model rate selected by the declared loss/safeguard rule.
- `anchored_profiles.pdf`: exact cos(g0,g[k]) along Adam (solid), with the
  previously measured GD reference (faint dashed) at identical initial states
  and horizons. Colors identify parent GD histories. Horizontal coordinate is
  symlog with linear region through 16; raw values, no smoothing.
- `adjacent_profiles.pdf`: exact cos(g[k-1],g[k]) along the new Adam trajectories,
  at every full-objective optimizer boundary. Same source colors, anchors and
  horizons; the first adjacent pair is k=1. Symlog horizontal coordinate.
- `adjacent_first128.pdf`: the same adjacent pairs restricted to 1..128 on a
  linear horizontal coordinate to expose the optimizer-reset transient.

Trajectory panels mark the two unqualified historical starts [U]; these never
select LR or enter primary summaries. Declining anchored odd/even gaps and
positive local alignment are descriptive geometry, not a convergence/stability
test or proof that Adam outperforms GD. Raw full gradients differ from actual
Adam moments, adaptive directions and parameter increments.
'''
    obs=HERE/'observations';obs.mkdir(exist_ok=True)
    (obs/'001-adam-continuations.md').write_text(observation,encoding='utf-8',newline='\n')
    original=(HERE/'README.md').read_text(encoding='utf-8')
    if not original.startswith('Completed:'):
        (HERE/'README.md').write_text(f'Completed: [report](REPORT.md), [figures](figures/anchored_profiles.pdf), [exact adjacent curves](figures/adjacent_profiles.pdf).\n\nAll task cloud deleted; USD {summary["compute_estimate_usd"]:.2f} compute estimate, provisional.\n\n'+original,encoding='utf-8',newline='\n')
    for rid,slug in [('209','eos-adam-calibration'),('210','eos-adam-long-lags'),('211','eos-parallel-adam-calibration')]:
        path=ROOT/f'runs/{rid}-2026-11-01-{slug}/README.md'
        text=path.read_text(encoding='utf-8')
        if not text.startswith('Completed:'):
            detail={'209':'125 canonical arms at 25 starts here; two delayed starts are completed by unchanged Run211 arms. Failed/partial originals retained.',
                '210':'All 27 prescribed matched-horizon profiles completed.',
                '211':'All 10 parallel arms at two retained states completed.'}[rid]
            path.write_text('Completed: '+detail+' All raw evidence retrieved and hash verified; task cloud deleted.\n\n'+text,encoding='utf-8',newline='\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
