# Optimizer state and paired loss benefit

Source: all18 accepted events.jsonl and summary.json files from Runs187-204,
with hashes checked against Analysis030's accepted results. Sources and reduced
features are in ../features.json; all plotting coordinates are retained in
../figure_data.json and ../results.json. Scripts: ../01_analyze.py (reductions),
../02_plot.py (figures/report), ../03_verify.py (independent verification).

Coverage: Pythia14M/70M, three seeds, one common fork stage, 128 updates per arm.
Twelve altered arms use their own matched1x control. The18 trajectories do not
supply18 independent initial states. Full validation includes693,329 target
tokens across all500 documents; the fixed training probe has262,016 targets.
No undefined values are imputed into predictions; first undefined step cosine
is excluded by the update2 start. All21 response and six initial features are
reported. The three additional stream-ratio controls were added after observing
the first actual-retention result; all comparisons are exploratory.

## Optimizer state map

File: ../figures/optimizer_state_map.pdf.
Caption: Mean incoming-gradient/prior-moment alignment and the relative norm
of incoming versus retained contributions to momentum over updates2-16. Color
is held-minus-candidate final validation loss; positive is beneficial. Shapes
identify LR multipliers, panels identify model sizes, each mark is one seed/arm.
All18 points are shown, including six zero-benefit held controls. The coordinate
means summarize a window; they are not plugged into a nonlinear EMA identity.
The color separation largely follows the intervention. This is a descriptive
state map, not evidence of a classifier or a stability boundary.

## Paired optimizer response

File: ../figures/paired_optimizer_response.pdf.
Caption: Paired responses to halving or doubling LR relative to holding it.
First row: fixed-probe loss benefit (held minus candidate, positive is better).
Other rows: candidate-minus-held incoming/prior-moment cosine, relative incoming
moment magnitude, and actual displacement/path length. Thin traces show three
seeds; thick traces show arithmetic seed means. The first16 updates are shaded.
The two instantaneous geometry traces use an8-update trailing mean starting
at8; retention is an unsmoothed prefix statistic, reset at the fork. Probe loss
is measured only at0/16/32/64/128. Connecting segments do not locate the exact
crossing time. No confidence interval or independence of adjacent updates is implied.

## All primary predictive scores

File: ../figures/predictive_value_all_features.pdf.
Caption: Reduction in held-seed prediction MAE for final validation benefit,
relative to training-seed means for the same model and LR multiplier. Positive
values improve prediction. Each response feature uses at most16 updates; initial
features use the common pre-update state with a signed LR action interaction.
One scalar slope is fitted per feature, conditional on model/LR cell intercepts.
All27 feature comparisons and each held-seed fold are shown, with pooled MAE in
the last column. Color saturates at +/-60%; printed numbers retain actual values.
Folds share training data, and no multiplicity-adjusted inference is claimed.

## Retention prediction audit

File: ../figures/retention_prediction_audit.pdf.
Caption: Exploratory inspection of the actual-step retention candidate. Left:
observed and predicted final-benefit residuals after subtracting the held-out
model/LR baseline, for all12 altered arms. Blue/orange identify0.5x/2x; circles
and triangles identify14M/70M. Each prediction excludes that seed from fitting.
Middle: mean absolute errors for each held seed. Right: sensitivity of actual,
adaptive, and moment retention to observation cutoffs8/16/32/64. Curves connect
separate analyses, not a continuously evaluated prediction process. This candidate
was highlighted after the feature scan, so its apparent performance needs new data.

## Observation and limits

At16, actual/adaptive retention reduce primary MAE by about37%, improving all
three held-seed folds. Momentum retention helps pooled MAE but not every fold;
raw cosine and projected change do not improve primary MAE. No initial summary
improves primary prediction. The initial-state measurements agree across arms
within5.42e-7 absolute; the common held-arm measurement avoids fitting roundoff.

The main retention result is sensitive to window and outcome. Alone it worsens
prediction of the additional validation benefit from16 to128. Combined with
early probe loss it improves pooled secondary error, but not every fold relative
to the probe-only comparator. There are only three seed clusters, and all states
share the same eventual best action. This suggests a candidate magnitude predictor;
it supplies neither a validated controller nor a causal optimizer mechanism.

Visual review and numerical checks are recorded in ../figure_checks.json and
../verification.json. No manuscript integration or new training was performed.
