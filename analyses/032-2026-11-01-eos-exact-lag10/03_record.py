"""Record verified observations, timing and closeout for the exact lag profiles."""
from pathlib import Path
import json
import statistics

HERE=Path(__file__).parent
ROOT=HERE.parents[1]
RUN=ROOT/'runs/205-2026-11-01-eos-exact-lag10'


def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    results=json.loads((HERE/'results.json').read_text())
    verification=json.loads((HERE/'verification.json').read_text())
    closeout=json.loads((HERE/'closeout.json').read_text())
    profiles=results['profiles'];primary=[p for p in profiles if p['primary_eligible']]
    negative=[p for p in primary if p['cosines'][0] is not None and p['cosines'][0]<0]
    rejected=[p for p in profiles if not p['primary_eligible']]
    timings=[];total_bytes=total_files=0;replay_failures=[]
    for worker in range(3):
        output=RUN/f'artifacts/attempt-01/worker{worker}'
        manifest=json.loads((output/'manifest.json').read_text())
        events=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
        for task in ['mlp','resnet','transformer']:
            rows=[r for r in events if r['phase']=='gradient' and r.get('task')==task and r['offset']>0]
            timings.append(dict(worker=worker,task=task,n=len(rows),median_gradient_seconds=statistics.median(r['seconds'] for r in rows),
                minimum_gradient_seconds=min(r['seconds'] for r in rows),maximum_gradient_seconds=max(r['seconds'] for r in rows)))
        receipt=json.loads((HERE/f'retrieval-worker{worker}.json').read_text());total_bytes+=receipt['bytes'];total_files+=receipt['files']
        for path in output.glob('*/replay_checks.json'):
            checks=json.loads(path.read_text());failed=[q for q in checks if not q['passed']]
            if failed:replay_failures.append(dict(parent=path.parent.name,first_failed_state=failed[0]['state'],failed_checks=len(failed),checked_records=len(checks)))
    summary=dict(total_profiles=len(profiles),primary_profiles=len(primary),
        saved_checkpoint_profiles=sum(p['parent_mode']=='saved_checkpoint' for p in profiles),
        reconstructed_profiles=sum(p['parent_mode']=='reconstructed' for p in profiles),
        unqualified_reconstruction_profiles=len(rejected),
        total_cosines=10*len(profiles),primary_cosines=10*len(primary),
        recorded_window_updates=270,replay_updates_outside_windows=5960,total_applied_updates=6230,
        primary_negative_lag1=len(negative),negative_lag1_positive_lag2=sum(p['cosines'][1]>0 for p in negative),
        negative_lag1_positive_lag10=sum(p['cosines'][9]>0 for p in negative),
        negative_lag1_positive_all_even=sum(all(v>0 for v in p['cosines'][1::2]) for p in negative),
        negative_lag1_negative_all_odd=sum(all(v<0 for v in p['cosines'][::2]) for p in negative),
        maximum_fixed_repeat_relative_error=max(p['fixed_repeat']['relative_gradient_error'] for p in profiles),
        retained_bytes=total_bytes,retained_files=total_files,replay_failures=replay_failures,timings=timings,
        aggregate_compute_seconds=sum(json.loads((RUN/f'artifacts/attempt-01/worker{w}/manifest.json').read_text())['elapsed_seconds'] for w in range(3)))
    save(HERE/'summary.json',summary)
    table=['| Model | LR | Anchor | Lag 1 | Lag 2 | Lag 10 | Primary eligible |',
           '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for p in profiles:
        table.append(f"| {p['task']} | {p['lr']:g} | {p['anchor']} | {p['cosines'][0]:.6f} | {p['cosines'][1]:.6f} | {p['cosines'][9]:.6f} | {'Yes' if p['primary_eligible'] else 'No: replay failed'} |")
    elapsed=(max(p['deleted_epoch'] for p in closeout['pods'])-min(p['created_epoch'] for p in closeout['pods']))/60
    rn=[t['median_gradient_seconds'] for t in timings if t['task']=='resnet']
    tr=[t['median_gradient_seconds'] for t in timings if t['task']=='transformer']
    high_late={task:next(p for p in profiles if p['task']==task and p['anchor']==3000 and p['lr']==rate)['cosines'][9]
               for task,rate in [('mlp',.04),('resnet',.1),('transformer',.2)]}
    report=f'''# Analysis032: exact anchored gradient cosine through lag 10

All 27 requested windows completed: nine Analysis024 conditions at states
500–510, 2000–2010 and 3000–3010. Each profile has 11 complete-objective gradients
and ten original-rate GD updates. All 270 requested cosines use full FP32
gradient vectors and FP64 reductions, without projection or smoothing.

{len(primary)}/27 profiles qualify for the primary comparison.
Two Run062 MLP LR 0.02 reconstructions, at 2000 and 3000, fail original-history
qualification and are retained in the separately flagged companion. Its early
window and all three Run059 LR 0.004 reconstructions qualify. The other 21
profiles start from exact saved model tensors. Saved-state results are short
cloud continuations, rather than an assertion of bitwise historical replay.
At 3000, qualification for the seven saved parents checks the original endpoint
loss only; gradients 3001–3010 are new and have no original scalar reference.
Scalar qualification does not prove tensor equality for reconstructed starts.
The first failed Run062 replay check is at model state
{replay_failures[0]['first_failed_state']} (gradient update {replay_failures[0]['first_failed_state']+1}).

Among the {len(primary)} primary profiles, {len(negative)} have negative lag 1.
Of those, {summary['negative_lag1_positive_lag2']} have positive lag 2 and
{summary['negative_lag1_positive_lag10']} have positive lag 10;
{summary['negative_lag1_positive_all_even']} have positive cosine at every even
lag through 10, and {summary['negative_lag1_negative_all_odd']} are negative at every
odd lag. These conditional measurements distinguish consecutive reversal
from alignment at longer update separations. They do not establish stationarity,
independence of distant gradients, optimization stability or generalization.
The three anchors are chronological locations, not matched dynamical regimes.
Rates were selected by the earlier calibration and only seed 0 is represented.
Within each model case, learning rate varies with the original initialization,
data/order and objective matched. Cross-case columns also change architecture,
objective and data volume; they do not isolate a causal model-size effect.
Coverage ends at lag 10. One anchored Gram-matrix row does not identify the
complete norm-weighted all-pair interaction or whole-path displacement.

Even-lag positivity does not imply an unchanged gradient direction. At anchor
3000, the highest-rate MLP, ResNet and Transformer curves have lag-10 cosines
{high_late['mlp']:.4f}, {high_late['resnet']:.4f} and {high_late['transformer']:.4f},
respectively. The visible sign pattern therefore coexists with substantial
direction changes over even steps in the MLP and ResNet cases. These are
case-specific examples, with the cross-case confounds stated above.

[Primary figure](figures/exact_gradient_lag10.pdf) and
[all profiles with reconstruction flags](figures/all_profiles_with_reconstruction_flags.pdf).
The complete 270 values are in [lag_profiles.csv](lag_profiles.csv), including
eligibility and provenance labels. Each point is one anchored pair, unlike the
window-averaged CountSketch profiles in Analysis005. No uncertainty bands or
training-seed replication are inferred from these points.

The window at 3000 extends the Analysis024 horizon by ten updates, in separate
Run205 outputs. Runs059/062 originally continued to 20000, so their scalar
references also cover this extension. Original runs/configs/artifacts and the
manuscript remain unchanged. Inputs, exact objective coverage, token weighting,
fixed BN groups and numerical choices are in [README.md](README.md) and Run205's
frozen config. No Hessians, sampling-noise probes or held-out evaluation were
added. Every gradient includes the complete training objective.
Run205 also retains byte-identical source input manifests as `source_inputs.json`
and `source_cifar_inputs.json`, including pinned revisions, ordering, normalization
and token exclusions. Preparation's `source_commit` identifies its base HEAD;
the per-file SHA256 inventory freezes the new code and config before execution.

Independent verification reads all 297 stored vectors and recomputes norms,
anchored cosines and adjacent cosines using chunked NumPy FP64 reductions.
All 270 values pass, with maximum absolute cosine error
{verification['maximum_cosine_error']:.3g}. It also checks all 21 saved starts
against every tensor of their extracted source checkpoint, and checks actual
parameter displacement against the ten recorded applied gradients with an
explicit FP32 rounding bound. Fixed-state repeated gradients have maximum
relative discrepancy {summary['maximum_fixed_repeat_relative_error']:.3g}.
Original/extracted checkpoint and input/source hashes remain recorded locally.
See [verification.json](verification.json), [checkpoint provenance](checkpoint_provenance.json)
and [figure manifest](figure_manifest.json).

Three Secure RTX 4090 workers were read at USD 0.89/GPU-hour. Aggregate production
time was {summary['aggregate_compute_seconds']/60:.2f} GPU-minutes; allocation to
last deletion took {elapsed:.2f} minutes, including setup, staging and retrieval.
Per-worker median recorded gradient times, excluding fixed-state repeats, were
{min(rn):.2f}–{max(rn):.2f} s for ResNet and {min(tr):.2f}–{max(tr):.2f} s for Transformer.
These include this recorder's diagnostics/I/O and are workload measurements,
rather than an isolated hardware benchmark.
Estimated compute cost is USD {closeout['compute_estimate_usd']:.3f}, provisional
before final posted billing/storage reconciliation, below the USD 6 cap.
All {total_files} inventory-listed files ({total_bytes/1e9:.3f} GB) are locally
size/SHA-256 verified. All three task pods were deleted after acceptance; task
cloud storage is zero. Unrelated resources and Spark were untouched. The base
image reports an unused torchvision dependency mismatch after pinning PyTorch;
the experiment never imports torchvision, and its pinned Torch/NumPy/SciPy
runtime and real gradients pass checks. Raw logs preserve the warning.

Numerical endpoints below are rounded for reading; CSV/JSON retain full values.

'''+ '\n'.join(table)+'\n'
    (HERE/'REPORT.md').write_text(report,encoding='utf-8',newline='\n')
    observation=f'''# Observation001: exact short-lag profiles at retained GD states

Source: Run205's frozen config/input manifest, all three retrieved worker
inventories, 21 exact extracted checkpoint starts, and two qualified-or-flagged
MLP replay histories. The nine parent IDs are 059, 062, 120–126. Reduction:
`02_analyze.py`; independent complete-vector NumPy FP64 norms and dot products;
`03_record.py` generates the report and its displayed endpoints. Every curve
uses one anchored pair at each lag 1–10, with no projection, centering, averaging
or smoothing. All 297 vectors / 270 cosines pass verification. Primary coverage
is {len(primary)} windows / {10*len(primary)} values; the remaining two windows
are preserved as unqualified historical reconstructions.

Primary caption: Exact whole-model gradient cosine with the gradient at the
selected starting state held as reference, along ten constant-rate full-GD
updates. Rows identify anchors 500, 2000, 3000; columns identify MLP, ResNet-32 and
Transformer; colors identify the three original learning rates per model.
MLP LR 0.02 at 2000/3000 is omitted after historical replay fails qualification.
The final row is an extension of the Analysis024 horizon, with complete
training-objective coverage, including token-weighted Transformer reduction.
There is one dependent vector pair per point and one initialization per model;
no error bars or statistical population claims are supported.

Companion caption: The same full-vector measurements, adding the two failed
historical reconstructions with dashed lines and explicit legend labels.
Their numerical cosines are exact for the reconstructed vectors; those vectors
are not qualified as the original Run062 states. Small qualifying scalar
errors do not establish exact recovery of the other reconstructed starts.

Among primary profiles with negative lag 1, {summary['negative_lag1_positive_lag2']}/{len(negative)}
have positive lag 2, {summary['negative_lag1_positive_lag10']}/{len(negative)} have positive
lag 10, and {summary['negative_lag1_positive_all_even']}/{len(negative)} are positive at
all even lags. These are descriptive counts over selected states, not
independent trials or a stability criterion. The source values and displayed
arrays are in `results.json`, `lag_profiles.csv` and `figure_manifest.json`.
Publication PDFs are under `figures/`; ignored Poppler renders are under
`scratch/`. Visual-review evidence is recorded separately after inspection.
'''
    (HERE/'observations').mkdir(exist_ok=True)
    (HERE/'observations/001-exact-lag10.md').write_text(observation,encoding='utf-8',newline='\n')
    readme=HERE/'README.md';text=readme.read_text(encoding='utf-8')
    text=text.replace('Status: prepared.','Status: complete. See [REPORT.md](REPORT.md), [verified results](results.json),\n[primary figure](figures/exact_gradient_lag10.pdf) and [observation](observations/001-exact-lag10.md).')
    text=text.replace('Every available replay scalar is compared with original loss,\nnorm and adjacent cosine.','Replay losses/norms and available adjacent cosines are compared with their\noriginal scalar records.')
    text=text.replace('Analysis scripts, figures, observations and\nfinal verification will be added here at closeout.','`02_analyze.py` verifies/reduces vectors and plots; `03_record.py` records\nobservations and closeout. See `verification.json`, `summary.json` and `closeout.json`.')
    readme.write_text(text,encoding='utf-8',newline='\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
