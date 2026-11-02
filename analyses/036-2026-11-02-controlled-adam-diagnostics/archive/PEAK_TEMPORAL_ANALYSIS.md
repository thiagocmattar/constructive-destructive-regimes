# Do the Adam diagnostics precede pronounced loss spikes?

The inspected Transformer spike supports a short temporal lead before the loss
rise becomes large. Repetition and blocked forecasts support investigating the
diagnostic further. A crossing above 2 is not a reliable standalone alarm, and
these trajectories do not identify a causal mechanism.

The [three-column micro figure](figures/adam_peak_micro_three_models.pdf) shows
one conspicuous completed episode at each task's highest Adam LR: MLP 0.001,
ResNet 0.003 and Transformer 0.003. All nine panels per column are unsmoothed.
The [protocol](PEAK_TEMPORAL_PROTOCOL.md) records operational definitions before
forecast fitting; the study is nevertheless retrospective and exploratory.

## Timing at the Transformer spike

Align all measurements to parameter state t: loss and raw gradient are measured
there; momentum is the first moment after consuming that gradient, available
before the next parameter update. In the earlier micro PDF, these vector rows
were labeled by update t+1. This one-step shift matters for a lead-lag claim.
Actual-update R_t retains its natural state t after the first t updates.

| State | Training loss | Rise above L_787 | Raw-gradient eta*kappa_hat | Raw gradient norm | Stored momentum norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 786 | 0.648481 | 0.16% | 1.7086 | 0.3952 | 0.02428 |
| 787 | 0.647419 | 0.00% | 1.7083 | 0.3997 | 0.02903 |
| 788 | 0.650024 | 0.40% | 2.3830 | 0.6456 | 0.04506 |
| 789 | 0.661349 | 2.15% | 2.8070 | 1.2008 | 0.08340 |
| 790 | 0.725120 | 12.00% | 3.3106 | 2.8170 | 0.20951 |
| 791 | 1.200537 | 85.43% | 3.0588 | 6.0128 | 0.42422 |

The fresh crossing above 2 is at 788, two states before the 10% loss-rise onset at 790.
Both norms grow strongly at 788. Momentum already grows from 0.02428 to 0.02903
at 787 while loss still declines; the gradient's increase at 787 is small.
The projected change peaks at 790, raw gradient norm at 792, momentum norm at 793,
and loss at 794. Thus the signals precede the pronounced rise and loss maximum,
but the crossing at 788 does not precede the very first positive loss change.
Earlier quiet-time excursions above 2 are also present; this is a fresh crossing,
not the first-ever threshold exceedance.

There is an exact dependence between two of these observations:

    eta*kappa_hat = 1 - rho_t * norm(g_t) / norm(g_(t-1)).

For nonzero norms, eta*kappa_hat>2 requires negative raw cosine and a norm ratio
greater than 1. Gradient growth and this threshold are therefore not independent
pieces of evidence. Momentum growth can add a different optimizer-state view.
Under Adam, the projected change is not a measured Hessian curvature or a
calibrated stability boundary.

## Repetition, including misses

Use loss alone to define an episode: at least 10% above the preceding 16-state
loss minimum, followed through its first return below that frozen baseline.
Merge rebounds within an unfinished episode and retain right-censored episodes.
Skip the first 64 states. A large episode reaches at least 50% above its baseline.

All nine runs contain 129 episodes, 45 large, with two unfinished recoveries.
At the highest LRs, there are 27 large episodes in total. An early alarm must
occur 1-8 states before the 10% onset while current loss is still less than 5%
above its own past 16-state minimum.

| Highest-LR task | All episodes | Large episodes | Large episodes with a prior fresh crossing of 2 | With prior >20% gradient growth | With prior >20% momentum growth |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 39 | 17 | 13/17 | 13/17 | 13/17 |
| ResNet-32 | 7 | 3 | 3/3 | 3/3 | 3/3 |
| Transformer | 13 | 7 | 7/7 | 7/7 | 7/7 |

The three MLP 13/17 counts need not refer to the same 13 episodes. One MLP and
one Transformer episode have an unfinished recovery, although their observed
onsets and warnings are available. There are misses; this is not a universal
precursor. All 129 episodes and individual leads are retained in
[peak_events.csv](peak_events.csv) and peak_temporal_results.json.

The Transformer large-episode peaks are 429, 482, 605, 702, 794, 895, 992. Their fresh
crossing leads before 10% onset are 4, 5, 6, 4, 2, 8, 5 states respectively. These are
dependent recurrences along one evolving trajectory. The largest ResNet example
has projected change peak at 418, gradient norm peak at 421, momentum norm and
loss peaks at 422. In the selected MLP example, projected change peaks at 699,
momentum norm at 702, and gradient norm and loss at 703. The maxima do not all
precede loss in every task.

## Forward-blocked prediction beyond the loss history

Predict a future loss at least 10% above the current loss within the next 8 steps.
Include only current states still less than 5% above their recent loss minimum,
so the classifier does not simply identify an already pronounced rise.
Train on expanding prefixes; test on five later blocks beginning 384,512,640,
768,896. A24-state gap separates training decisions from test decisions, so
all training outcomes precede the earliest test loss-history input. There is
no random step split or test-set threshold/hyperparameter selection.

The pooled models include task/LR-arm indicators, training age and current/past
loss changes. Each added diagnostic is compared with that same baseline. Table
entries are average precision on the same 4,534 forward-test states, 495 positive
(10.92%). Per-task rows summarize those pooled models' predictions within each
task, including all three LRs; they are not separately fitted task models.
Higher average precision is better.

| Test subset | Loss history | + projected change | + gradient level/growth | + momentum level/growth | + all diagnostics |
| --- | ---: | ---: | ---: | ---: | ---: |
| All nine runs | 0.260 | 0.454 | 0.303 | 0.298 | 0.395 |
| MLP | 0.500 | 0.546 | 0.532 | 0.564 | 0.625 |
| ResNet-32 | 0.045 | 0.136 | 0.062 | 0.053 | 0.063 |
| Transformer | 0.248 | 0.433 | 0.388 | 0.432 | 0.563 |

Projected change improves average precision and Brier score in all five time
blocks. Pooled Brier score falls from 0.11972 to 0.08685 (lower is better).
Its added standardized coefficient is positive in each block. More features
are not consistently better: the all-diagnostic model underperforms projected
change alone overall, and its ResNet Brier score is slightly worse than the
loss baseline. This discourages declaring a universal combined alarm.

The fixed threshold has much poorer specificity than the learned score:

| Fixed alarm, on the same forward-test states | Alarm states | Later rise within 8 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Projected change>2 | 891 | 187 | 21.0% | 37.8% |
| Fresh crossing of 2 | 402 | 64 | 15.9% | 12.9% |
| Gradient grows>20% | 1,008 | 175 | 17.4% | 35.4% |
| Momentum grows>20% | 612 | 158 | 25.8% | 31.9% |
| Fresh crossing and momentum grows>20% | 116 | 30 | 25.9% | 6.1% |

For projected change>2,704 of 891 alarm states have no qualifying rise within 8
steps. These are state-level decisions, not independent alarms or episodes.
The earlier episode table asks whether an episode has any early alarm; it has
a different denominator and must not be read as alarm precision.

The lowest-LR ResNet and Transformer arms have no qualifying loss episode.
Nevertheless, projected change exceeds 2 at 175/633 and 190/633 forward-test
states respectively, with zero positive forecast targets in either arm.
These retained negative controls directly rule out treating the threshold as
a universal loss-spike or instability warning.

Sensitivity uses every combination of horizons4/8/16 and future rises5%/10%/25%.
The projected-change extension improves pooled average precision in all nine
settings. Baseline / projected-change / all-diagnostic average precisions are:

| Horizon | Required rise | Loss history | + projected change | + all diagnostics |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 5% | 0.478 | 0.565 | 0.707 |
| 4 | 10% | 0.316 | 0.470 | 0.598 |
| 4 | 25% | 0.096 | 0.326 | 0.470 |
| 8 | 5% | 0.390 | 0.561 | 0.569 |
| 8 | 10% | 0.260 | 0.454 | 0.395 |
| 8 | 25% | 0.193 | 0.365 | 0.248 |
| 16 | 5% | 0.438 | 0.546 | 0.528 |
| 16 | 10% | 0.384 | 0.461 | 0.441 |
| 16 | 25% | 0.345 | 0.396 | 0.383 |

No best setting is selected. Raw forecasts, source features, labels and block
identities are in [peak_forecasts.csv](peak_forecasts.csv); probability columns
have the prefix prob_. Full ablations, per-arm and per-block metrics, alarms
and sensitivity outputs are retained in peak_temporal_results.json.

## Strengthening the diagnostic next

1. Freeze one target, availability clock, alarm rule and operating point using
   early blocks; validate on untouched later training or a held-out whole LR
   arm. Existing later blocks are now inspected and cannot become a fresh
   confirmatory test. Keep non-spike states and report precision, recall,
   lead time, recovery censoring and incremental value over loss history.
2. Use whole-episode / contiguous time blocks for robustness checks, rather than
   treating adjacent steps as independent samples. Repeat threshold and horizon
   sensitivity without choosing the most favorable result. Test transfer between
   tasks separately from within-trajectory forecasting. No new initialization
   seed sweep is required by this proposal.
   Add within-arm time-shift controls that preserve contiguous diagnostic blocks
   to test whether the exact timing, rather than only training phase, adds value.
3. To test optimizer-state mechanisms causally, plan matched short forks from
   alarm and quiet checkpoints. Preserve weights, data, LR, schedule, Adam clock
   and RNG; compare unchanged Adam with separate, explicitly defined first-moment
   and second-moment perturbations. Record future loss, raw gradient changes,
   moments, adaptive directions and actual updates. These interventions would
   identify their own conditional effects, not the causal effect of a diagnostic
   number. No Hessians, LR interventions or such forks are executed here.
   Starting raw-gradient diagnostics would agree across branches because the
   weights and gradient history agree; the optimizer state would differ.
   Record actual step lengths and include a matched-step-length control before
   attributing an outcome difference specifically to direction rather than size.

Chronological splits and gaps follow the rationale in the
[TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
The causal proposal requires an explicit intervention and identification
assumptions, as distinguished from temporal prediction by
[Hernan and Robins](https://miguelhernan.org/whatifbook).

This remains one initialization per task, with different qualified hardware and
an inspected event motivating the design. Temporal predictability is useful
evidence for a candidate diagnostic; it is not a causal explanation, a validated
learning-rate controller, an Adam stability threshold or a generalization result.
There are no new training runs, cloud resources or compute charges.
