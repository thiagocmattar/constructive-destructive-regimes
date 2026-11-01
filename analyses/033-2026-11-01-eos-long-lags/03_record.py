"""Write the long-lag observation and execution closeout from verified results."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import re

HERE=Path(__file__).parent
ROOT=HERE.parents[1]
RUN=ROOT/'runs/206-2026-11-01-eos-long-lags'


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    results=json.loads((HERE/'results.json').read_text())['profiles']
    check=json.loads((HERE/'verification.json').read_text())
    closeout=json.loads((HERE/'closeout.json').read_text())
    billing=json.loads((HERE/'billing.json').read_text()) if (HERE/'billing.json').exists() else None
    margin=json.loads((HERE/'budget_margin_adjustment.json').read_text())
    calibration=json.loads((HERE/'calibration_decision.json').read_text())
    cfg=json.loads((RUN/'config.json').read_text())
    primary=[p for p in results if p['primary_eligible']]
    reversing=[p for p in primary if p['initial_reversal']]
    damped=[p for p in reversing if p['final_envelope']<=cfg['damped_gap_threshold']]
    criterion_stops=[p for p in reversing if p['stop_reason']=='alternating_envelope_below_threshold']
    persistent=[p for p in reversing if p['final_envelope']>cfg['damped_gap_threshold']]
    counts=Counter(p['last_lag'] for p in results)
    gpu_allocations=Counter(p['gpu'] for p in closeout['pods'])
    gpu_description=', '.join(f'{n} {name}' for name,n in sorted(gpu_allocations.items()))
    retirements=[json.loads(p.read_text()) for p in (HERE/'infrastructure_attempts').glob('*/retirement.json')]
    timing={}
    for name in ['mlp','resnet','transformer']:
        times=[r['seconds'] for p in results if p['task']==name for r in p['rows'][11:]]
        timing[name]=dict(samples=len(times),median_seconds=statistics.median(times),
            minimum_seconds=min(times),maximum_seconds=max(times))
    budget_ended=[p for p in results if p['stop_reason']=='budget_or_deadline_request']
    elapsed=(max(p['deleted_epoch'] for p in closeout['pods'])-min(p['created_epoch'] for p in closeout['pods']))/60
    source_errors=[p['boundary_checks']['source_boundary_relative_error'] for p in results]
    chunk_errors=[p['boundary_checks']['compute_chunk_relative_error'] for p in results if 'compute_chunk_relative_error' in p['boundary_checks']]
    resume_errors=[p['resume_boundary_checks']['source_boundary_relative_error'] for p in results if 'resume_boundary_checks' in p]
    vector_audits=len(check['retained_full_vector_audits'])
    ownership=Counter(p['source_result'].split('/')[1][:3] for p in results)
    start_counts=Counter((j['parent_mode'],j['primary_eligible']) for j in cfg['jobs'])
    parity=[]
    for p in reversing:
        failures=[r['lag'] for r in p['rows'][1:] if r['cosine'] is not None and
                  ((r['lag']%2==1 and r['cosine']>=0) or (r['lag']%2==0 and r['cosine']<=0))]
        parity.append(dict(parent_run_id=p['parent_run_id'],anchor=p['anchor'],first_parity_sign_break=min(failures) if failures else None))
    summary=dict(status='complete',profiles=27,primary_profiles=25,flagged_profiles=2,
        direct_saved_starts=start_counts[('saved_checkpoint',True)],
        qualifying_reconstructions=start_counts[('reconstructed',True)],
        unqualified_reconstructions=start_counts[('reconstructed',False)],
        cosines=check['cosines'],new_cosines=check['new_cosines'],common_maximum_lag=256,
        largest_measured_lag=max(p['last_lag'] for p in results),horizon_counts=dict(sorted(counts.items())),
        primary_initially_reversing=len(reversing),primary_damped_by_declared_rule=len(criterion_stops),
        primary_terminal_half_gap_at_most_threshold=len(damped),
        primary_still_above_damping_threshold=len(persistent),budget_limited_profiles=len(budget_ended),
        resolved_transformer_physical=calibration['selected_physical'],timing=timing,
        maximum_source_boundary_relative_gradient_error=max(source_errors),
        maximum_compute_chunk_relative_gradient_error=max(chunk_errors) if chunk_errors else 0,
        maximum_selected_chunk_absolute_loss_error=check['maximum_selected_chunk_absolute_loss_error'],
        allocation_to_last_deletion_minutes=elapsed,aggregate_gpu_hours=closeout['aggregate_gpu_hours'],
        compute_estimate_usd=closeout['compute_estimate_usd'],billing=closeout['billing'],
        posted_billing_total_usd=billing['posted_total_usd'] if billing else None,
        raw_files=check['raw_files'],raw_bytes=check['raw_bytes'],cloud_storage_remaining_gb=closeout['cloud_storage_remaining_gb'],
        peak_recorded_cuda_allocated_bytes=check['peak_recorded_cuda_allocated_bytes'],
        peak_recorded_cuda_reserved_bytes=check['peak_recorded_cuda_reserved_bytes'],
        pods_deleted=len(closeout['pods']),gpu_allocations=dict(gpu_allocations),
        failed_gradient_qualification_attempts=len(check.get('failed_attempts',[]))+len(check.get('workers_with_later_independent_start_failure',[])),
        unreachable_setup_only_workers=len(retirements),parity_sign_checks=parity,
        scientific_runs=['206','207','208'],parallel_later_resnet_profiles=6,
        canonical_profiles_by_run=dict(ownership),qualified_handoff_profiles=len(check['handoff_sources']),
        full_vector_audits=vector_audits,maximum_staged_resume_relative_gradient_error=max(resume_errors) if resume_errors else 0,
        generated_utc=datetime.now(timezone.utc).isoformat())
    save(HERE/'summary.json',summary)
    rows=[]
    for p in results:
        eligible='primary' if p['primary_eligible'] else 'unqualified replay'
        gpu='4090' if '4090' in p['gpu'] else '5090'
        rows.append(f'| {p["task"]} | {p["parent_run_id"]} | {p["lr"]:g} | {p["anchor"]} | {eligible} | {gpu} | {p["last_lag"]} | {p["cosine_256"]:.6f} | {p["final_cosine"]:.6f} | {p["common_256_envelope"]:.6f} | {p["final_envelope"]:.6f} | {p["stop_reason"]} |')
    horizons=', '.join(f'{n} at lag{k}' for k,n in sorted(counts.items()))
    case_notes=[]
    rebound=next(p for p in results if p['parent_run_id']=='126' and p['anchor']==500)
    rebound_gaps={s['lag']:s['median_half_odd_even_gap'] for s in rebound['stages']}
    rebound_note=(f'For example, the early Transformer at LR0.2 has local half-gap '
        f'{rebound_gaps[512]:.3f} at lag512, rising to {rebound_gaps[1024]:.3f} at1024 '
        f'and ending at {rebound_gaps[2048]:.3f} at2048. More lags reveal a rebound, not monotonic damping.')
    for task in ['mlp','resnet','transformer']:
        selected=[p for p in reversing if p['task']==task]
        reached=sum(p['stop_reason']=='alternating_envelope_below_threshold' for p in selected)
        above=sum(p['final_envelope']>cfg['damped_gap_threshold'] for p in selected)
        label=dict(mlp='MLP',resnet='ResNet',transformer='Transformer')[task]
        case_notes.append(f'- {label}: {len(selected)} primary initially reversing profiles; {reached} stage-criterion stops, {above} terminal gaps above0.1; terminal gap range {min(p["final_envelope"] for p in selected):.3f}–{max(p["final_envelope"] for p in selected):.3f}.')
    report=f'''# Exact long-lag gradient profiles

Completed27 profiles / {check['cosines']:,} anchored cosines across the nine
[Analysis024](../024-2026-10-17-eos-gd-lr-calibration/README.md) conditions,
at anchors500/2000/3000. All profiles reach lag256;
the largest measured lag is{summary['largest_measured_lag']}. There are25 primary
profiles and two separately flagged historical Run062 reconstructions.
The original starts comprise21 direct saved starts, four reconstructions that
passed the prior historical checks, and two that failed. This follow-up reuses
their retained Run205 states; it does not replay initialization again.

Among the{len(reversing)} primary profiles that originally reverse at lag1,
{len(damped)} have a terminal local half-gap at most0.1, and{len(persistent)} remain above
it at their final measured horizon. Of these, {len(criterion_stops)} stopped at a
declared stage because the criterion was met. The score is a median half absolute odd/even
cosine gap of at most0.1 over the final32 adjacent pairs. It describes the
alternating component of this anchored curve; it does not require the cosine's
common offset to vanish. Persistent curves are retained as observed.
Cosine normalizes gradient magnitudes: shrinking gradient norms can coexist
with persistent angular alternation. This score does not measure decay of the
gradient norm or the amplitude of parameter oscillations.
A small anchored half-gap also does not show that consecutive gradients have
stopped reversing, even within the measured range.

{rebound_note}

{chr(10).join(case_notes)}

These endpoint ranges have unequal horizons; they are descriptive summaries,
not matched-coverage model or LR rankings. The common-range figure supports
comparisons at the same measured lags.

- [Common lag256 figure](figures/common_lag256.pdf)
- [Longer primary profiles](figures/long_lag_profiles.pdf)
- [All profiles with reconstruction flags](figures/all_profiles_with_reconstruction_flags.pdf)
- [Local odd/even half-gap](figures/alternating_gap.pdf)
- [Every value](lag_profiles.csv), [verified results](results.json), [verification](verification.json)
- [Observation and captions](observations/001-long-lag-profiles.md)
- [Execution closeout](closeout.json), [summary and timing](summary.json)

## Coverage and interpretation

Final horizons: {horizons}. Positive-lag1 controls stop at256. Initially
negative-lag1 profiles extend through the frozen stages256/512/1024/2048,
stopping at the first stage meeting the envelope threshold. There were
{len(budget_ended)} budget/deadline-limited profiles. Coverage beyond256 is
adaptive and outcome dependent; compare all conditions on the shared256-lag
figure, and use the longer figure to inspect persistent trajectories.

The half-gap companion plots the declared local stopping statistic over every
complete32-pair window. A threshold crossing is local: some profiles first
attenuate and then rebound. It does not establish asymptotic decay or guarantee
that unmeasured future lags remain small. Complete odd/even pairs are used;
an unmatched final odd lag is retained in the raw curve and omitted from this score.

The plots join measured even lags and odd lags separately, without smoothing,
centering or projection. Each value is one whole-model anchored pair,
cos(gradL(theta_anchor),gradL(theta_(anchor+k))), rather than an average over
many training positions. Late windows continue beyond the original3000-state
history. All new forks preserve original LRs, plain SGD, seed0, model and full
training objective. They do not establish seed uncertainty, stationarity,
stability, generalization or a loss benefit from alignment.

| Task | Parent | LR | Anchor | Eligibility | GPU | Last lag | Cosine 256 | Final cosine | Gap 256 | Final gap | Stop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

The GPU column identifies final execution hardware. Some handed-off prefixes
were measured on another GPU; source paths and activation records retain that
provenance. Canonical final profiles by source run: {dict(ownership)}.

## Exact streaming and numerical qualification

No compression was required: the fixed anchor and current/recent full gradients
bound memory independently of horizon. Recorded peak CUDA allocation is
{check['peak_recorded_cuda_allocated_bytes']/1e9:.3f} GB and peak reservation is
{check['peak_recorded_cuda_reserved_bytes']/1e9:.3f} GB. Parameters and accumulated gradients
are FP32, TF32 is disabled, and diagnostic products/norms are FP64. Every new
lag was checked using independent NumPy CPU and CUDA reductions. Maximum cosine
disagreement is{check['maximum_cpu_gpu_cosine_error']:.3g}; all{vector_audits} retained
full-gradient vector audits pass. The297 source rows, including all27 anchors,
are unchanged. Zero denominators are explicitly undefined.

The original full training coverage is unchanged: the first 5,000 CIFAR MLP
examples with half squared error summed over 10 outputs and averaged over examples;
50,000 ResNet examples with 50 fixed 1,000-image train-mode BN groups; and 59,680
Transformer sequences of length 35 with 2,088,600 supervised targets and token-weighted NLL.
Existing -100 padding is excluded. No held-out evaluation was run. Original source metadata
and per-file hashes are retained under Run206.

Calibration selected a common Transformer compute chunk of
{calibration['selected_physical']} from64/128/256/512. Every selected-start
comparison is within the prespecified1e-5 relative-gradient tolerance; the
maximum observed chunk discrepancy is{summary['maximum_compute_chunk_relative_gradient_error']:.3g}.
All nine selected-start losses also satisfy the declared1e-5 absolute tolerance;
the maximum disagreement is{summary['maximum_selected_chunk_absolute_loss_error']:.3g}.
This preserves the mathematical full objective while changing FP32 accumulation
partitions. Hardware and rounding can change long continuations; bitwise
agreement with a chunk64 continuation or the historical trajectory is not claimed.
The maximum source-boundary gradient discrepancy is
{summary['maximum_source_boundary_relative_gradient_error']:.3g}.
ResNet's BN groups never change. The prior failed replay qualification at
Run062 anchors2000/3000 is not repaired by this extension.

Run207 freezes the parallel placement and shared runner for the six later
ResNet windows. It reuses completed4090 control Pods after verified retrieval,
with two independent windows per worker. Both reach256 before extensions;
persistent windows then resume from their own model/optimizer/RNG state and
unapplied gradient at256. Raw baseline/extension evidence and derived phase
configs are retained separately. The largest resume-gradient discrepancy is
{summary['maximum_staged_resume_relative_gradient_error']:.3g}, within1e-5.
The merged outputs preserve all verified prefix rows. Small recomputation
rounding differences do not imply bitwise-identical future trajectories.

Run208 freezes four checkpoint handoffs to shorten the parallel tail: three
later ResNet profiles resume their measured lag256 states on independent4090
workers; parent125/anchor500 Transformer resumes its measured lag512 state on
a5090. Exact scalar prefixes and sparse raw vectors are preserved. Each new
worker must recompute the captured unapplied gradient within the original1e-5
tolerance before stepping. Old sources stop only after replacement progress
proves qualification. Hardware/rounding can alter subsequent trajectories;
these are qualified continuations, not bitwise historical replays. Source
selection follows the declared handoff activation, not curve outcome.

The new run retains exact per-lag scalar products/norms, full gradients at11
and the final adjacent pair, and final model/optimizer/RNG recovery states.
The last gradient is measured but not applied. Source references and all source
lag0..10 vectors remain in Run205. This run does not retain every new full vector
and cannot supply arbitrary future all-pairs or per-parameter lag analyses.

## Execution and cleanup

The cohort used{len(closeout['pods'])} task allocations ({gpu_description}),
with at most twelve workers concurrently. Live prices were USD1.19/GPU-hour
for Secure RTX5090 and USD0.89 for Secure RTX4090. Some starts failed strict
gradient qualification on5090 and were retried on the source4090 hardware;
the tolerance was not relaxed. Thus GPU hardware is not matched across every
LR and anchor; initial numerical agreement does not prove identical future
trajectories. All failed attempts and costs are retained in closeout.
An intentional handoff stop can leave a signal exit code after the scientific
manifest and selected window have completed. Those windows are retained only
with completed window records and verified vectors, checkpoints and file hashes;
the process exit code alone is not the scientific acceptance criterion.
The approved cap was USD30 and four hours; runtime storage used30GB container
disk and10GB Pod volume per worker after the API rejected the prepared8GB
volume request. No network volume was created. Detached jobs emitted manifests,
progress and events; the supervisor monitored, retrieved, verified and deleted
workers individually. An independent budget/deadline guard bounded compute.
After measured small retention/transfer costs, the operational graceful trigger
was refined from USD25 to USD{margin['graceful_trigger_usd']:g} within the same approved USD30 envelope, retaining
a USD{margin['hard_compute_stop_usd']:g} hard compute stop and four-hour deadline. The ten-minute deadline
retrieval margin remained. This changed infrastructure controls only; scientific
inputs and the lag-horizon rule remained frozen.

Median complete gradient plus diagnostic times were
{timing['mlp']['median_seconds']:.3f}s MLP,
{timing['resnet']['median_seconds']:.3f}s ResNet and
{timing['transformer']['median_seconds']:.3f}s Transformer.
Allocation to last deletion took{elapsed:.1f}minutes; aggregate allocation was
{closeout['aggregate_gpu_hours']:.3f}GPU-hours. Compute is approximately
USD{closeout['compute_estimate_usd']:.2f}, provisional pending posted billing/storage.
Task-ID billing reads in billing.json contain only currently posted charges;
hourly buckets may be partially posted. They are not a final settled invoice.
All{check['raw_files']} retrieved files / {check['raw_bytes']/1e9:.3f}GB passed
size/SHA-256 checks before their Pod was deleted. There were{len(retirements)}
unreachable setup-only workers: available platform logs and monitoring snapshots
were preserved, but full remote setup files could not be retrieved. No scientific
job had started on those workers. All{len(closeout['pods'])} task Pods were terminated;
no task cloud storage remains. Spark and unrelated account resources were
untouched. No manuscript edit or push was made.
'''
    (HERE/'REPORT.md').write_text(report,encoding='utf-8',newline='\n')
    observation=f'''# Observation001: exact long-lag anchored gradient profiles

Question: how far does the previously observed odd/even gradient oscillation
persist beyond lag10, for the same nine EOS conditions and three saved starts?

Sources: Runs206/207/208 frozen configs/input manifests, per-worker inventories, exact
per-window results and sparse full vectors; Run205's verified source histories
and end states. `checkpoint_provenance.json` hashes all27 source windows.
`handoff_captures.json` and activation records trace the accelerated resumes.
`02_analyze.py` audits/reduces the evidence and generates the four PDFs;
`03_record.py` writes the report/summary. Exact values are in `lag_profiles.csv`.

Coverage: 27 seed0 profiles, 25 primary, common lag256, maximum measured lag
{summary['largest_measured_lag']}; {check['cosines']:,} anchored cosines.
The{len(reversing)} primary initially negative-lag1 profiles include{len(damped)} satisfying
the declared final-pair envelope threshold and{len(persistent)} still above it.
See REPORT.md's table for each LR/anchor and stop. Failed Run062 replay starts
at2000/3000 remain flagged and excluded from primary plots.
The source cohort contains21 direct saved starts and six reconstructed starts;
four reconstructions met the earlier historical checks and enter the primary
set. Qualification does not imply bitwise equality of later continuations.

{rebound_note}

Raw cosine reduction: no random projection, averaging of positions, smoothing or centering.
FP64 dot products/norms of FP32 full-objective gradients. The envelope used
for stopping is median(|rho_(2m)-rho_(2m-1)|/2) over the last32 pairs; it is
a descriptive reduction, not a validation or optimization criterion. CPU/GPU
reductions agree within{check['maximum_cpu_gpu_cosine_error']:.3g}; {vector_audits} retained
full-vector audits pass. Each plotted artist is compared to its exact source
arrays. Numerical validation is in verification.json; visual review is separate.

Caption, `common_lag256.pdf`: Whole-model raw full-objective gradient cosine
from states500/2000/3000 for the25 primary Analysis024/032 conditions.
Columns show MLP, ResNet-32 and Transformer; rows show anchors. Color identifies
learning rate. Solid/dashed curves join measured even/odd lags separately,
without smoothing. Each lag uses one anchored pair; all panels share lag256
coverage. This figure displays the common measured range before adaptive
extension, without a stability or generalization interpretation.

Caption, `long_lag_profiles.pdf`: Same25 primary anchored profiles with
base2 logarithmic lag axis and declared adaptive terminal horizons. Initially
positive-lag1 controls stop at256. Initially reversing profiles extend through
512/1024/2048 until the final32-pair median half-gap is<=0.1 or a bound is
reached. Dotted vertical lines mark256. Curves ending at different lags have
different measured coverage; persistence at a bound is not evidence of eventual
decay. Exact endpoint gaps and stopping reasons are in the report.

Caption, `all_profiles_with_reconstruction_flags.pdf`: Same longer profiles
including all27 windows. The unqualified Run062 historical reconstructions at
anchors2000/3000 are explicitly labeled [U] and faded. They are exploratory
continuations and do not become historical matches because this run completes.

Caption, `alternating_gap.pdf`: Declared local attenuation score for the25
primary profiles: the rolling median of half the absolute difference between
an even-lag cosine and its preceding odd-lag cosine, over32 complete pairs
(64 lags). The horizontal dashed line marks0.1 and the vertical dotted line
marks common lag256 coverage. This companion is a derived summary of exact
raw cosines, not an additional sample or independent observation. Rebounds
remain visible. A small gap need not mean zero cosine offset or permanent
decorrelation. The raw figures retain every cosine without smoothing.

Uncertainty/caveats: one initialization per case and one dependent pair per lag,
no confidence interval. Adaptively selected horizons are not fixed-coverage
comparisons beyond256. Full-objective deterministic GD differs from minibatch
Pythia histories. Selected Transformer compute chunks preserve the objective
but change FP32 accumulation; hardware/rounding can change future trajectories.
Damped alternation may leave a nonzero cosine offset. Anchored geometry alone
does not measure gradient-norm or parameter-oscillation amplitude decay. It
does not establish that consecutive gradients stop reversing, even within the
measured range. It also
does not measure stability, generalization, optimizer utility or full temporal
displacement accounting. No new held-out evaluation or manuscript claim.
'''
    (HERE/'observations').mkdir(exist_ok=True)
    (HERE/'observations/001-long-lag-profiles.md').write_text(observation,encoding='utf-8',newline='\n')
    (RUN/'README.md').write_text(f'''# Run206: completed streaming exact long-lag continuations

Initial full-objective continuations from all27 Run205 starts for
[Analysis033](../../analyses/033-2026-11-01-eos-long-lags/REPORT.md).
The joint cohort reaches256; the largest measured lag is{summary['largest_measured_lag']}.
There are{check['cosines']:,} exact anchored cosine values, 25 primary profiles
and two previously unqualified Run062 reconstructions retained separately.
Run206 supplies{ownership.get('206',0)} canonical final profiles. Run207 owns
the six later ResNet starts; Run208 owns four qualified checkpoint handoffs.
The source model/optimizer/RNG states and independently verified lag0..10
histories come from Run205; no replay from initialization was needed.

`config.json`, `01_worker.py`, `input_manifest.json` and copied source protocol
metadata remain frozen. The common Transformer compute chunk selected by the
declared calibration was{calibration['selected_physical']}; all nine starts
passed numerical qualification. ResNet's fixed BN groups were unchanged.
Exact GPU/CPU FP64 streaming products avoid compression and growing history
memory. Raw evidence under ignored `artifacts/attempt-*/workerN/` retains
per-lag products, gradients at11/final pair, and model/optimizer/RNG end states.
The final sampled gradient is not applied. Full reference histories remain in
Run205. `test_streaming.py` is a synthetic mechanics check, not paper evidence.

All{check['raw_files']} files / {check['raw_bytes']/1e9:.3f}GB were locally
size/SHA-256 verified before termination. All{len(closeout['pods'])} task Pods and their storage
were deleted; no network volume was created. Allocation-to-last-deletion time
was{elapsed:.1f}minutes and compute estimate USD{closeout['compute_estimate_usd']:.2f}
is provisional pending posted billing/storage, within the approved USD30 cap.
The analysis owns figures, numerical/visual reviews, observations and closeout.
No unrelated resource, original scientific input or manuscript asset was edited.
''',encoding='utf-8',newline='\n')
    readme=f'''# Analysis033: completed exact long-lag gradient profiles

[Report](REPORT.md), [primary curves](figures/long_lag_profiles.pdf),
[common range](figures/common_lag256.pdf), [odd/even gap](figures/alternating_gap.pdf).

Follow-up to [Analysis024](../024-2026-10-17-eos-gd-lr-calibration/README.md)
and [Analysis032](../032-2026-11-01-eos-exact-lag10/REPORT.md): 27 anchored profiles at states500/2000/3000,
all through lag256 and up to{summary['largest_measured_lag']}. Exact streaming products
required no compression. There are25 primary profiles and two previously
unqualified Run062 starts. Of{len(reversing)} primary profiles with negative original
lag1, {len(damped)} have terminal local half-gap <=0.1 and{len(persistent)} remain above it.
Damping can rebound; a crossing does not establish permanent decorrelation.

Runs206/207/208 own the frozen continuations, parallel placement and qualified
checkpoint handoffs. Canonical profiles by source run: {dict(ownership)}.
All{len(closeout['pods'])} task Pods were deleted after verified retrieval; no task cloud
storage remains. Compute estimate USD{closeout['compute_estimate_usd']:.2f} is provisional,
within the approved USD30 envelope. Allocation to last deletion took{elapsed:.1f}minutes.
Spark and unrelated account resources were preserved.

[Every value](lag_profiles.csv), [results](results.json), [summary](summary.json),
[observation/captions](observations/001-long-lag-profiles.md),
[verification](verification.json), [final review](final_review.json),
[closeout](closeout.json). Numbered scripts preserve preparation, orchestration,
reduction, documentation, reviews and checkpoint captures. Do not rerun completed
allocation; changed scientific inputs need a new run ID.
'''
    (HERE/'README.md').write_text(readme,encoding='utf-8',newline='\n')
    for run_id,slug,description in [
        ('207','eos-parallel-resnet-lags','Six later ResNet profiles on three reused4090 workers; both assigned baselines reach256 before extensions.'),
        ('208','eos-checkpoint-handoffs','Four qualified handoffs: three later ResNet profiles at256 and parent125/anchor500 Transformer at512. Replacements pass the original gradient tolerance before source shutdown.')]:
        path=ROOT/f'runs/{run_id}-2026-11-01-{slug}/README.md'
        path.write_text(f'''# Run{run_id}: completed {slug.replace('-',' ')}

{description}

[Analysis033](../../analyses/033-2026-11-01-eos-long-lags/REPORT.md) owns
the complete metric/method contract, figures, observations and closeout.
This run supplies{ownership.get(run_id,0)} canonical final profiles; earlier prefixes
and unused valid partial continuations remain raw evidence. The joint cohort
contains27 profiles,25 primary, all through256 and up to{summary['largest_measured_lag']}.

Executed config, worker, input manifest and shared scientific runner remain
frozen. Original data/order, seed, full objective, LRs and plain SGD match the
parent runs. ResNet BN groups remain fixed. Exact CUDA/NumPy FP64 products
of FP32 full gradients require no compression. Hardware/rounding can perturb
future trajectories; no bitwise, stability or generalization claim is made.

Ignored artifacts retain scalar rows, sparse full vectors and terminal
model/optimizer/RNG recovery states; the final gradient is unapplied.
All retrieved sizes/SHA-256 pass. All{len(closeout['pods'])} task Pods and attached
storage were deleted, with no network volume created. Joint compute estimate
USD{closeout['compute_estimate_usd']:.2f} is provisional within the approved USD30/four-hour
envelope. See the analysis for failures, activation/source records and final QA.
''',encoding='utf-8',newline='\n')
    # Format numeric quantities in prose while preserving code, links and IDs.
    for path in [HERE/'REPORT.md',HERE/'README.md',HERE/'observations/001-long-lag-profiles.md']+[
        ROOT/f'runs/{rid}-2026-11-01-{slug}/README.md' for rid,slug in
        [('206','eos-long-lags'),('207','eos-parallel-resnet-lags'),('208','eos-checkpoint-handoffs')]]:
        text=path.read_text(encoding='utf-8')
        text=text.replace('positive-lag1','positive lag-1').replace('negative-lag1','negative lag-1')
        pieces=re.split(r'(^\|[^\r\n]*|`[^`]*`|\]\([^)]+\)|\b(?:Run|Analysis)\d+\b)',text,flags=re.MULTILINE)
        for i in range(0,len(pieces),2):
            pieces[i]=re.sub(r'(?<=[a-z])(?=\d)',' ',pieces[i])
            pieces[i]=re.sub(r'\b(USD|LR)(?=\d)',r'\1 ',pieces[i])
            pieces[i]=re.sub(r'(?<=\d)(minutes|seconds|GPU-hours|GB|s MLP|s ResNet|s Transformer)\b',r' \1',pieces[i])
        text=''.join(pieces)
        path.write_text(text,encoding='utf-8',newline='\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
