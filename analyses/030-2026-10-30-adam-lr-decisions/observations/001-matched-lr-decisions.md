# H024 matched LR figures

Sources: all18 original events and summary files, individually hash verified and
listed in results.json. Generator: ../06_analyze.py; execution and input contracts
are in ../README.md. GPU/scientific acceptance precedes these reductions.

## Matched LR validation

File: ../figures/matched_lr_validation.pdf. Caption: Full-validation loss at the
common fork and16/128 subsequent updates for three constant LR multipliers,
holding per-update AdamW shrinkage constant. Columns show seeds, rows show sizes.
Reduction: original token-weighted evaluations on all500 validation documents,
693,329 targets including the packed tail. No smoothing or omitted evaluations.
Uncertainty: show each of the three seed clusters; curves are not confidence bands.
Caveat: one training stage and short continuations; no billion-parameter claim.

## Early geometry and later loss

File: ../figures/early_geometry_later_loss.pdf. Caption: Mean adjacent cosine over
local updates2--16 versus final loss difference from the held-LR branch, separately
for raw gradients, moments and adaptive directions. Negative Y means the rate
change improves final validation loss. Each panel has12 altered-rate arms.
Reduction: arithmetic mean of original pointwise cosines; matched validation
difference at128. Update1 is excluded because its gradient precedes intervention.
Uncertainty: six matched states with three seed clusters; points are not independent
replications across sizes/arms. Caveat: associations do not establish mediation or
a useful decision threshold; momentum/adaptive cosines are not actual displacement.

## Exploratory decision regret

File: ../figures/held_seed_decision_regret.pdf. Caption: Achieved loss above the
best observed arm per starting state when holding LR or selecting from fixed ridge
models using early loss, additional raw cosine, or joint optimizer diagnostics.
Reduction: fixed penalty10, training-fold standardization, leave-one-seed-out;
both sizes and all arms of each held seed are excluded during fitting. No tuning.
Coverage:18 arms, six states, three folds. Uncertainty: per-state outcomes shown,
no significance intervals. Caveat: the observed best arm is a finite-comparison
reference, not an oracle beyond these rates; this pilot does not validate a scheduler.
The fixed-LR comparator chooses its multiplier only on each fold's training seeds;
it tests whether diagnostics add value beyond ordinary state-independent LR tuning.
If every state favors the same rate, conditional decision value is unestablished.
Predictions use each candidate's post-intervention updates2--16; all three early
branches are observed. Deploying this selection procedure would cost160 updates
to produce a128-update selected branch, before evaluation overhead. These loss
regrets do not establish an equal-compute benefit or a checkpoint-only policy.

Visual review: passed on October30. All three final PNG previews have legible
labels, clear legends, no title/curve overlaps and no clipped text. Numerical
source reductions pass ../11_verify_results.py; receipts are in
../result_verification.json and ../figure_checks.json.

The fixed-LR and loss-baseline bars are zero in every state and therefore have
no visible height. Validation segments connect only the three measured states,
not a densely sampled validation curve. Scatter separation is also separation
by the imposed LR intervention; it does not identify an independent predictor.

## Completed observation

The2x rate yields the lowest final validation loss in all six states, improving
on the held rate by0.023816--0.050583. Half-rate arms have positive early raw
cosine and worse final loss; doubled-rate arms have negative early raw cosine
and better final loss. This separates raw-vector reinforcement from optimizer
benefit in these matched continuations; it is not a stability or curvature test.

The fixed-LR and loss-baseline selectors achieve zero observed regret. Adding
raw cosine gives mean regret0.011137; joint diagnostics give0.016247. This pilot
does not establish added decision value from the diagnostic features. A common
winning rate also means these states do not demonstrate a need for conditional
LR selection. This finding is restricted to this stage, these rates, three seed
clusters, two small Pythia models and128-update continuations.
