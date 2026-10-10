"""Record facts, coverage and figure captions from the accepted final reduction."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                  ['| '+' | '.join(map(str,row))+' |' for row in rows])

def main():
 payload=json.loads((HERE/'results.json').read_text()); results=payload['results']
 observations=HERE/'observations'; observations.mkdir(exist_ok=True)
 sources=[]; phase=[]; calibration=[]; totals=[]; replay=[]; tails=[]; full_pass=[]
 for r in results:
  run=ROOT/'runs'/r['run']; lease=json.loads((run/'prelaunch/lease.json').read_text()); attempt=lease['attempt']
  receipt=json.loads((run/'prelaunch'/f'{attempt}-retrieval.json').read_text())
  assert receipt['verified'] and receipt['scientific_verification']['temporal_verified']
  sources.append(f"- [{r['size']} Run{r['run'][:3]}](../../../runs/{r['run']}/README.md), `{attempt}/full-pass`: raw sketches, update events, exact pairs/window aggregates, metadata and frozen config. Archive SHA-256: `{receipt['archive_sha256']}`.")
  replay.append([r['size'],f"{r['validation_loss']:.6f}",f"{r['reference_validation_loss']:.6f}",f"{r['reference_cosine_rmse']:.6f}"])
  for name,w in r['windows'].items():
   c=w['cosine_by_lag']; ex=w['exact_window']
   phase.append([r['size'],name,f"{w['exact_lag_one_mean_cosine']:.4f}",f"{100*w['exact_lag_one_negative_frequency']:.1f}%",f"{c[1]:.4f}",f"{c[3]:.4f}",f"{w['long_lag_mean_32_64']:.4f}",f"{w['shuffled_order_expectation']:.4f}"])
   calibration.append([r['size'],name,w['exact_calibration_pairs'],f"{w['cosine_calibration_rmse']:.5f}",f"{w['cosine_calibration_max_abs']:.5f}"])
   totals.append([r['size'],name,f"{ex['energy']:.6g}",f"{ex['lag_one_interaction']:.6g}",f"{ex['all_pair_interaction']:.6g}",f"{100*w['exact_lag_one_share_of_all_interaction']:.2f}%",f"{100*w['lag_one_truncation_relative_squared_sum_error']:.2f}%",f"{100*w['interaction_relative_error']:.2f}%"])
  tails.append([r['size'],f"{r['windows']['late']['late_tail_exclusion_max_cosine_change_1_64']:.6f}"])
  g=r['whole_pass']
  full_pass.append([r['size'],f"{g['energy']:.6g}",f"{g['lag_one_interaction']:.6g}",f"{g['all_pair_interaction']:.6g}",f"{100*g['signed_lag_one_share']:.2f}%",f"{100*g['lag_one_truncation_relative_squared_sum_error']:.2f}%"])
  (run/'README.md').write_text(f"""# Run{r['run'][:3]}: {r['size']} temporal raw-gradient sketches

Completed: one full MiniPile pass, 712 updates and 1,491,711,416 input tokens;
seed 1235. The original initialization SHA, pinned token cache and paired order
were retained. The executed recipe/config and sources are frozen.

Accepted evidence: `artifacts/{attempt}/full-pass/`; local archive and every
inventoried artifact passed SHA-256/size verification and the scientific checks.
Raw sketch history covers all 712 updates, with four fixed 16,384-bucket maps,
exact raw norms, adjacent pairs, sampled longer-lag pairs and exact window sums.
Initial, update356 and final recovery checkpoints are retained locally.

This is a measurement replay, not a bitwise copy of its earlier reference or an
independent training seed. Actual worker: Secure RTX4090, USD0.74/GPU-hour.
Infrastructure retry details and the frozen-input inventory are in `prelaunch/`.
Resource removal and provisional cost are documented by Analysis005 closeout.

[Analysis005](../../analyses/005-2026-10-04-temporal-lag-profiles/REPORT.md) owns
the cross-run conclusions, figures, calibration and limitations. Final full
token-weighted validation loss: {r['validation_loss']:.6f}; all 500 validation
documents and 693,668 input tokens were covered, including short tails.
""",encoding='utf-8')
 source_text='\n'.join(sources)
 (observations/'001-lag-profiles.md').write_text(f"""# Observation001: temporal raw-gradient cosine profiles

Question: how does signed cross-step alignment change with lag, training phase
and model size? Do the proposed early/middle/late profile shapes appear?

## Sources and coverage

{source_text}

All models use seed1235 and the original 712-update recipe. Revisions, loss
reduction, accumulation boundary, unique trainable parameter scope and source
hashes are retained in the run configs/metadata and `../results.json`.
Both pair endpoints are inside one of three preselected 128-update windows:
1–128, 293–420 and 585–712. Lags1–64 have 127 down to64 pairs per model/window;
the secondary output through127 has only one pair at its endpoint. There are
zero silently truncated training or validation tokens. Update712 is retained.

Reduction: `../03_analyze.py`, with definitions in `../METHOD.md`. Projected dots
are averaged across four maps in FP64 and divided by original exact norms.
Pair cosines are equally weighted. This is an uncentered signed cosine profile,
without a stationarity assumption. Exact lag-one values come from raw-gradient
FP64 dot products. Zero-denominator pairs are undefined and coverage is reported.

## Results

{table(['Size','Window','Exact C(1)','Negative lag1 pairs','C(2)','C(4)','Mean C(32–64)','Unordered reference'],phase)}

The individual 127-lag curves, four-map values and norm-weighted interactions
are available in `../curve_data.csv` and `../results.json`. Lag1 alone does not
determine the sign at another lag. Descriptive interpretation is in `../REPORT.md`.

## Calibration and uncertainty

{table(['Size','Window','Exact calibration pairs','Cosine RMSE','Max absolute error'],calibration)}

Exact pairs include every within-window adjacent pair and randomly anchored
lags2/4/8/16/32/64. Sample counts and errors separately by lag are in
`../results.json`. The planned full-history qualification uses RMSE<0.02 and
maximum absolute error<0.06. Four-map bands are projection sensitivity only,
not training-seed confidence intervals. Errors from a fixed map can be correlated
across pairs; small sign changes below calibration error remain unresolved.
There is one training seed, not four independent training runs.

Excluding the final partial update changes the primary late cosine profile by
at most the following absolute amount over lags1–64:

{table(['Size','Maximum absolute change'],tails)}

This sensitivity does not replace the retained primary evidence. The new replay
shares immutable initialization/order/recipe but may differ numerically from its
earlier reference; full-history differences are explicitly retained:

{table(['Size','Replay validation loss','Reference validation loss','Adjacent-cosine RMSE versus reference'],replay)}

## Figures and captions

- `../figures/lag_profiles.pdf`: Mean raw-gradient cosine at every integer lag
  1–64, for 14M/31M/70M in early/middle/late training. Both endpoints lie within
  the stated 128-update window. Lines show the four-map mean; shading is the
  range across fixed independent maps; markers at lag1 show its exact mean.
  One training seed1235; 128−k valid pairs at lagk.
- `../figures/lag_profiles_above_shuffled.pdf`: The same profiles after each map's
  unordered-window mean is subtracted. The reference is the expected cosine
  under uniform random ordering within the window. This is descriptive, without
  an exchangeability assumption, permutation p-value or stationary-process claim.
- `../figures/lag_profiles_fixed_origins.pdf`: An additional coverage control,
  keeping the same first64 source updates at every lag1–64. Each lag has64 pairs,
  with targets inside the128-update window. Markers show exact lag1 means for
  those source updates; bands are the four-map range. This control was added
  after the early probes, before middle/late spectra were available. It does
  not replace the predeclared all-pair primary profile or assume stationarity.
- `../figures/temporal_gram.pdf`: Four-map mean raw cosine Gram matrices for each
  model/window, normalized by original exact gradient norms. Local row/column
  indices cover all128 updates. The diagonal includes sketch norm error; the
  color scale saturates outside[−1,1], while numerical data are not clipped.

Generating script: `../03_analyze.py`; PDF SHA-256 records and visual QA are in
`../artifact_validation.json`. Scratch PNGs are ignored preview artifacts.

## Limits

These are raw gradients along AdamW training, not AdamW directions or parameter
updates. Phase averages mix changing local behavior. Shared update windows do
not ensure equivalent learning states across architectures. No stability,
generalization, causal dimensionality mechanism or learning-benefit claim is
established. Multi-seed validation remains deferred, as requested.
""",encoding='utf-8')
 (observations/'002-all-lag-interaction.md').write_text(f"""# Observation002: exact all-pair interaction and lag-one adequacy

Question: does the manuscript's lag-one truncation represent the temporal
cross-step term within these same training windows?

## Sources, reduction and coverage

{source_text}

The three 128-update windows, seed and raw-gradient scope match Observation001.
The recorder independently sums raw gradients in FP64 inside each window and
records E=sum||g||², B²=||sum g||², J=(B²−E)/2, exact I₁ and J−I₁. Thus the total
and the beyond-lag-one residual do not depend on the projection. Individual
I_k are estimated from projected dots; they are norm-weighted sums, rather than
the equally weighted cosine averages. The last partial update remains included.
The reduction verifies E and I₁ against the update stream, the exact aggregate
identity, and every stored projected calibration dot against the sketch history.

## Results

{table(['Size','Window','E','Exact I1','Exact J','Signed I1/J','Relative B² error using lag1','Projected J relative error'],totals)}

The signed share can be negative when lag1 cancels but the all-pair total is
constructive; it is not a fraction constrained to[0,1]. The squared-sum error
is |B²−(E+2I₁)|/B². Projected-total relative error is (estimated J−J)/|J|.
Exact numeric values and all127 lag contributions are in `../results.json`.

The same exact aggregate check is also available for each full712-update
trajectory from the retained baseline streaming sum, raw norms and adjacent dots:

{table(['Size','Full-pass E','Exact I1','Exact J','Signed I1/J','Relative B² error using lag1'],full_pass)}

This full-pass table retains update712 and does not use projected dot products.
It covers the same three trajectories, not three additional training replications.

## Figures and captions

- `../figures/cumulative_lag_interaction.pdf`: Twice the cumulative estimated
  lag interaction through k, divided by exact raw-gradient energy E, for each
  model and phase. This primary plot covers k1–64 and uses the four-map mean.
  Different lag contributions have128−k pairs and different gradient norms;
  the curve is not a cosine decay curve or an AdamW displacement.
- `../figures/cumulative_lag_interaction_all_lags.pdf`: The same cumulative
  quantity through all127 lags. Shading gives the four-map range; diamonds at127
  give the independently measured exact2J/E endpoint. Coverage shrinks to one
  pair at the final lag. Endpoint agreement calibrates the aggregate but alone
  does not validate each individual lag profile.

Generating script: `../03_analyze.py`; sources/hashes: `../results.json`;
PDF integrity and visual QA: `../artifact_validation.json`.

## Interpretation and limits

These exact raw-gradient aggregates test the lag-one approximation directly.
The SGD displacement identity requires the corresponding plain-SGD update rule
and step-size assumptions; these AdamW trajectories do not establish it for
actual parameter updates. A positive all-pair sum can coexist with negative
adjacent alignment. Neither sign alone establishes stability or learning quality.
This is a first, single-seed measurement test in three windows, not a general law
over model scale or training schedules. See `../REPORT.md` for the narrow claim.
""",encoding='utf-8')
 (observations/'INDEX.md').write_text("# Analysis005 observations\n\n- [001: Temporal lag profiles](001-lag-profiles.md): four cosine/Gram figures, coverage, calibration and sensitivity.\n- [002: All-lag interaction](002-all-lag-interaction.md): two cumulative figures, exact aggregate and lag-one approximation checks.\n",encoding='utf-8')
 protected=json.loads((HERE/'protected_inputs.json').read_text())
 assert all((ROOT/r['path']).stat().st_size==r['bytes'] and sha(ROOT/r['path'])==r['sha256'] for r in protected)
 record={'checked_at':datetime.now(timezone.utc).isoformat(),'protected_files_unchanged':len(protected),
   'results_sha256':sha(HERE/'results.json'),'analysis_script_sha256':sha(HERE/'03_analyze.py'),
   'figures':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (HERE/'figures').glob('*.pdf')},
   'raw_artifacts_verified_before_worker_deletion':True,'mathematics_and_capture_tests':'5 passed',
   'visual_qa':'pending PDF rendering and inspection'}
 (HERE/'artifact_validation.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record,indent=2))

if __name__=='__main__':main()
