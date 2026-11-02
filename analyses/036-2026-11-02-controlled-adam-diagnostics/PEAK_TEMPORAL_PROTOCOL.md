# Exploratory peak timing and temporal validation

The user requested repetition checks, a three-model micro view, and stronger
temporal or causal analysis of the raw-gradient projected change and gradient /
momentum norms. Reuse all nine completed Run215 histories. No training, new
initialization, Hessian/HVP, LR/schedule intervention or cloud execution is added.
This protocol records choices before fitting the forecasting comparisons. The
question and thresholds are informed by the already inspected Transformer
peak794, so this is exploratory rather than a preregistered confirmation.

## Availability and algebra

Align everything to parameter state s. Use L_s and g_s from boundaries[s], and
the momentum/adaptive directions after consuming g_s from metrics[s], whose
accepted update number is s+1. They are functions of the state and current
gradient, before the next parameter displacement. Forecast only future losses,
L_(s+1) onward. The earlier micro PDF places these vector rows one step later;
the new three-column PDF explicitly uses the corrected state alignment.

The original expression is 1-rho_s*norm(g_s)/norm(g_(s-1)). With nonzero norms,
exceeding2 implies rho_s<0 and norm(g_s)>norm(g_(s-1)). Thus the threshold and
gradient growth are algebraically dependent, not independent corroboration.
Under Adam the expression is a projected gradient change along adaptive steps,
not an identified curvature or stability boundary.

## Loss-only episodes and examples

Skip the first64 states. At each unassigned state, take the minimum loss over
the preceding16 states. Start an episode when current loss is at least10%
above that reference. Freeze its baseline, merge subsequent rebounds until the
first loss strictly below it, and choose the maximum within the episode as its
peak. If no recovery occurs by1024, retain and flag the censored episode.
Report all episodes and the subset with peak at least50% above its baseline.
No gradient/momentum diagnostic selects an episode.

At each task's highest tested LR, display the completed episode with the largest
peak/baseline ratio. This selects conspicuous examples, not representative
average behavior. The figure starts10 states before the selected peak and ends
at the first post-peak loss below that starting loss, preserving the previous
micro-view convention. The episode baseline and figure baseline can differ;
both are recorded explicitly.

In the figure, cumulative actual-update R_t stays attached to state t after
updates1..t; it is not shifted to the input-gradient state. Only the raw-gradient,
momentum and adaptive diagnostics shift relative to the historical PDF. Thus no
actual increment from the next update is displayed as already available at t.

For each episode, inspect the preceding8 states for strictly earlier alarms,
restricted to states still below5% above their own trailing16-state loss minimum.
Alarms are: projected change>2; a fresh crossing of2; gradient norm growing>20%
in one state; momentum norm growing>20%; and a fresh crossing together with
momentum growth>20%. Record both hits and misses, including censored episodes.
These lead counts are descriptive; the forecasting analysis also counts alarms
without a future rise.

## Forecast target, controls and blocked evaluation

At each eligible state s>=64, predict whether any of the next8 losses exceeds
L_s by at least10%. Eligibility requires the current loss to remain strictly
below5% above the minimum over states s-16..s. This restriction uses only the
past/current loss and excludes already pronounced rises. Keep the same rows for
every model. Secondary sensitivity crosses horizons4/8/16 with rises5%/10%/25%.
The final horizon states are omitted, never labeled as negatives.

The loss-history baseline uses task/LR-arm indicators, state/1024, log current
loss, log loss changes over1/4/8 states and current log draw-up from the past16
minimum. Compare this with additions of (a) projected change alone, (b) log raw
gradient norm and its one-state log change, (c) log stored-momentum norm and its
one-state log change, (d) all five added features. Fit fixed L2 logistic models,
C=1, no class reweighting or hyperparameter search. Scale using training rows
only. Analysis solver randomness, if any, is fixed independently of training.

Use expanding training prefixes and five forward test blocks beginning at384,
512,640,768,896, each ending just before the next block or valid horizon end.
Remove16+horizon states between training and test decisions. Thus every
training outcome precedes the earliest loss-history input of its test block.
All tasks/LRs use the same time cuts; never randomly split individual states.
Report pooled and per-task average precision, event prevalence and Brier score,
plus each block's result and the number of blocks improved. These dependent
observations do not supply independent replicates or an initialization-level CI.

Also report the fixed-alarm precision, recall and false-positive rate on the
forward-test rows. Low-LR arms with no episodes remain important negative
controls. No threshold is tuned on a test block. Sensitivity results are retained
even when they disagree with the primary setting; no best configuration is chosen.

## What the data can establish and the next causal comparison

Earlier measured signals, recurrent patterns and incremental blocked prediction
can support a useful temporal diagnostic. They do not identify a causal effect
of the diagnostic itself or distinguish all feedback mechanisms in deterministic
training. The inspected trajectory has no randomized treatment or matched
counterfactual branch.

For a subsequent, separately authorized experiment, select alarm and quiet
checkpoints by a fixed rule. Copy theta, both Adam moments, bias-correction
clock, RNG and data state. Keep LR, schedule, data, precision and initialization
fixed. Compare unchanged Adam with separate first-moment and second-moment
perturbations, one component at a time, over a fixed short horizon. This would
test the effect of those optimizer-state changes at matched states; it would
not directly manipulate or causally identify projected change. No such forks
are launched here, and no extra initialization seeds or Hessians are proposed.

Method references: [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
describes chronological splits and an explicit gap. [Hernan and Robins, Causal
Inference: What If](https://miguelhernan.org/whatifbook) distinguishes causal
questions from temporal associations and makes identification assumptions explicit.
