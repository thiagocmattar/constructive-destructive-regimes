# Analysis005: temporal raw-gradient interference

The first test is complete: Pythia 14M, 31M and 70M, one training seed (1235),
712 updates per model. **Adjacent cancellation and cumulative interaction have
different signs in the middle window at all three sizes.** The lag profiles have
a short negative trough and positive rebound, followed by a positive background.
The proposed smooth early/late decay toward zero is not a consistent description
of these trajectories over the measured horizons.

The random-projection approach worked for this measurement. Exact pair checks
qualified all three histories, and exact gradient sums independently establish
the all-pair totals. This addresses item 3 of
`manuscript/ideas/reflections.md` directly: lag one is an observable to evaluate,
rather than an assumed summary of the entire temporal interaction.

## Start with these figures

- [Lag profiles by phase](figures/lag_profiles.pdf): every integer lag 1-64,
  with early, middle and late panels and all three sizes.
- [All-lag cumulative interaction](figures/cumulative_lag_interaction_all_lags.pdf):
  all 127 lags within each window; diamonds mark independently measured exact
  totals. The middle curves start negative and finish positive.
- [Same-origin coverage control](figures/lag_profiles_fixed_origins.pdf): the
  same first 64 source updates contribute at every lag.
- [Above the unordered-window reference](figures/lag_profiles_above_shuffled.pdf):
  a descriptive view of the short-lag deviations from the window background.
- [Temporal Gram matrices](figures/temporal_gram.pdf): time-resolved geometry
  underlying the window averages.
- [Cumulative interaction through lag 64](figures/cumulative_lag_interaction.pdf):
  the primary-horizon version of the all-lag decomposition.

All six PDFs were rendered with Poppler and visually inspected. Full sources,
reductions, captions and uncertainties are in
[Observation001](observations/001-lag-profiles.md) and
[Observation002](observations/002-all-lag-interaction.md).

## What happened to the proposed three-regime picture?

The windows were fixed before measurement: updates 1-128, 293-420 and 585-712.
They describe training position; their alignment signs were not imposed.
The table uses exact adjacent cosine and projected longer-lag cosine:

| Size | Window | Mean cosine, lag 1 | Mean cosine, lags 32-64 |
| --- | --- | --- | --- |
| 14M | Early | +0.3630 | +0.3157 |
| 31M | Early | +0.1366 | +0.1218 |
| 70M | Early | +0.0601 | +0.0711 |
| 14M | Middle | -0.2537 | +0.0918 |
| 31M | Middle | -0.2722 | +0.0934 |
| 70M | Middle | -0.2695 | +0.0853 |
| 14M | Late | +0.3291 | +0.2540 |
| 31M | Late | +0.2410 | +0.2714 |
| 70M | Late | -0.0231 | +0.2059 |

**Early:** 14M has a declining envelope, particularly with fixed source coverage,
but all three curves have rebounds. At 70M, lag 2 is substantially negative
(-0.1434 in the primary profile, about -0.0935 in the fixed-origin control).
Therefore, short-lag sign changes are not exclusive to middle training. None
of the three primary early profiles approaches zero by lag 64.

The early window also mixes changing dynamics. The first-64-origin exact lag-one
means are +0.7996, +0.4435 and +0.2697, much higher than the full-window averages.
The Gram matrices show that the coherent early block has different duration
across sizes. Thus a broad early window is not a homogeneous constructive regime.
The coverage control was added after seeing the early pilot, before the middle
and late profiles were available; it is a sensitivity analysis, not a replacement
for the predeclared primary profile.

**Middle:** all three models have negative lag-one and lag-two means, followed by
a positive lag-three/lag-four rebound. Mean cosine at lag 3 is +0.2565, +0.2271
and +0.1240. The longer-lag background remains about +0.085 to +0.093.
This supports a short-horizon cancellation/rebound structure. It does not by
itself establish a periodic oscillator or identify an optimizer mechanism.

**Late:** the curves retain a positive background around +0.21 to +0.27, with
short-lag dips and rebounds. 14M has positive lag-one but negative lag-two
alignment; 70M has slightly negative mean lag-one but strongly positive lag-three
alignment (+0.3592). A uniform smooth return to zero is not supported through
lag 64. Nothing here estimates an infinite-lag asymptote.

Subtracting the unordered-window mean makes middle/late longer-lag profiles
fluctuate closer to zero. This is compatible with local temporal structure
superimposed on a persistent uncentered background. The subtraction is not a
stationarity test, permutation significance test or causal counterfactual.

## The direct answer to reflection 3

For raw gradients in a window, define

\[
I_k=\sum_{t=k+1}^{N}g_{t-k}^{\top}g_t,\qquad
E=\sum_t\|g_t\|^2,\qquad
J=\sum_{k=1}^{N-1}I_k.
\]

The exact identity is

\[
\left\|\sum_tg_t\right\|^2=E+2J.
\]

An FP64 streaming sum measures J independently of the sketches. In the middle
windows, the result is:

| Size | Exact adjacent interaction I1 | Exact all-pair interaction J | Signed I1/J | Squared-sum error using only lag 1 |
| --- | --- | --- | --- | --- |
| 14M | -10.6946 | +169.0948 | -6.32% | 96.35% |
| 31M | -10.0077 | +159.6306 | -6.27% | 96.71% |
| 70M | -9.9227 | +127.2076 | -7.80% | 96.42% |

The omitted beyond-lag-one term changes the interaction's sign. All nine
model/window totals are positive. In each full 712-update trajectory, retaining
only lag one misses 97.94-98.57% of the squared cumulative raw-gradient norm;
its signed share of J is only -0.069% to +0.084%. These totals are exact streaming
reductions, not projection estimates.

Equal-pair mean cosine and norm-weighted interaction answer different questions.
A modest persistent positive cosine, combined with many longer-lag pairs, can
dominate the aggregate despite negative adjacent interactions. Hence neither a
negative adjacent mean nor a small individual longer-lag cosine justifies the
lag-one truncation.

The squared-sum identity holds for the recorded raw gradients. Connecting it to
net parameter displacement requires the matching update rule: the reflection's
SGD formula assumes plain constant-step SGD. These models trained with AdamW;
their raw-gradient sums are not actual parameter displacements.

## Accuracy, coverage and reproducibility

Each accepted update records four fixed signed CountSketch maps, 16,384 buckets
each, with exact original norms. The complete history occupies about 187 MB per
model, rather than retaining every full gradient. Projection RNG is isolated
from training RNG. Individual longer-lag dots use FP64 offline reductions.
See [METHOD.md](METHOD.md) for the estimator and operational definitions.

Each model has 970 exact calibration pairs: every adjacent pair plus sampled
lags 2/4/8/16/32/64. Whole-history cosine RMSE is 0.00380-0.00444; the largest
observed absolute error is 0.01849. Within the nine plotted windows, RMSE is
0.00357-0.00566. Projected all-pair totals differ from exact totals by at most
0.744%. These empirical checks support the prominent dips and rebounds; tiny
features near the observed error scale remain unresolved. They are not uniform
error bounds for every pair or unmeasured lag. Shading is the range across four
maps, not a training-seed confidence interval.

Primary lags have 127 pairs at lag 1 and 64 at lag 64. The supplementary all-lag
profile falls to one pair at lag 127. The final partial update is retained.
Excluding it changes the late mean-cosine profile by at most 0.01027, 0.00979
and 0.00797 at 14M/31M/70M, without changing the reported larger-scale pattern.
All runs include 1,491,711,416 training input tokens; validation uses all 500
documents and 693,668 input tokens with the recorded token weighting and tails.

These are new measurement replays of the original recipe, with the same initial
parameter SHA, cache, order, seed, batch, optimizer and schedule. They are not
independent seed replications or bitwise copies of Runs005/006/007. The new
workers used RTX4090; the references used RTX5090. Full adjacent-cosine RMSE
against those references is 0.2702/0.2906/0.3604, demonstrating numerical
trajectory sensitivity despite matching initial inputs. Final validation losses
are 5.18223/4.62493/4.00994 versus reference 5.18202/4.61740/4.05843.
The temporal conclusions belong to the newly recorded histories; these replays
do not add independent seed support to the earlier replicated conclusions.

## Recommended central claim

> In the measured Pythia/MiniPile trajectories, raw-gradient alignment depends
> on temporal lag and training phase. Negative adjacent alignment can coexist
> with positive longer-lag alignment and a positive cumulative cross-step
> interaction. Lag-one alignment therefore measures local cancellation and is
> insufficient to characterize cumulative temporal interference in these cases.

This is narrower and better supported than a stability or dimension-mediated
performance claim. The three middle-window lag-one means are very similar,
and the early/late size patterns differ. The experiment does not isolate
dimensionality, establish a scaling law, or link these profiles causally to
learning quality or generalization. Earlier Analysis002/003 replicated
conclusions remain intact. No supplied manuscript or working TeX was edited.

Before promoting phase-specific lag shapes as a general empirical result,
validate the same predeclared windows and coverage controls with additional
training seeds on matched hardware. No additional seeds were launched.

## Execution closeout

All three complete archives passed size/SHA-256 checks, strict file inventories,
frozen-source checks, initial-tensor/order checks and independent CPU scientific
verification before worker deletion. Five targeted mathematics/capture/coverage
tests passed. All 127 protected original inputs are unchanged.

All six task-owned pods, including three infrastructure failures before training,
were deleted; all current deadline guards were stopped. No action was sent to
unrelated resources. Raw sketches, exact diagnostics, logs and initial/midpoint/
final recovery checkpoints remain local. Retained task cloud storage: zero.

Cloud elapsed time, including retries and retrieval, was about 3 hours 12 minutes.
Estimated cost is **USD6.54**, including current-worker GPU/storage allocation
and posted charges for failed attempts, below the USD15 cap. Billing is
provisional: the live API snapshot has only USD5.08 posted so far, with recent
usage still absent. This is not a final invoiced total. See
[closeout.json](closeout.json), [final billing snapshot](billing_final_snapshot.json) and
[artifact validation](artifact_validation.json).
