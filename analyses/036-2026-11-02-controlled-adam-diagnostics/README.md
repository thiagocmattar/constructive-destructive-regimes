# Analysis036: fixed-level loss-spike dataset

Current work isolates loss-only excursions from all nine completed Run215 Adam histories.
The fixed pre-rise reference was selected by the author. No smoothing, tolerance,
magnitude cutoff, or startup exclusion is used. Earlier exploratory work stays in archive/.

- [Mathematical definition](loss_spike_definition.md)
- [Event dataset: one row per excursion](spike_dataset/loss_spikes.csv)
- [All recorded event-state statistics](spike_dataset/spike_steps.csv)
- [Per-run counts](spike_dataset/summary.csv) and [source hashes](spike_dataset/manifest.json)
- Reproduce: `python analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py --baseline pre-rise`
- Verify: `python analyses/036-2026-11-02-controlled-adam-diagnostics/03_verify_loss_spikes.py`

## Coverage

812 candidate excursions: 806 recovered and 6 right-censored. All 9,225 loss observations (0..1,024 per run) are searched. Counts include tiny fluctuations.

| Model | LR | Candidates | Recovered | Censored | Largest observed peak / before |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 0.0001 | 95 | 94 | 1 | 1.3091 |
| MLP | 0.0003 | 140 | 139 | 1 | 3.0311 |
| MLP | 0.001 | 124 | 123 | 1 | 4.1763 |
| ResNet-32 | 0.0003 | 81 | 80 | 1 | 1.0370 |
| ResNet-32 | 0.001 | 121 | 121 | 0 | 7.0211 |
| ResNet-32 | 0.003 | 44 | 44 | 0 | 9.5877 |
| Transformer | 0.0003 | 80 | 79 | 1 | 1.0263 |
| Transformer | 0.001 | 89 | 89 | 0 | 1.8061 |
| Transformer | 0.003 | 38 | 37 | 1 | 3.3140 |

## Largest observed excursion per run

This table is a review view, ranked by peak / pre-rise loss within each run.
The CSV retains all events in chronological order. Ratios and displayed losses
are rounded here only; the CSV stores the full available precision.

| Model | LR | Before / peak / recovery state | Loss before | Peak loss | Recovery loss | Ratios before / peak / recovery | Rise | Recovery |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| MLP | 0.0001 | 901 / 910 / 914 | 0.0291109 | 0.038108 | 0.0290764 | 1 / 1.3091 / 0.9988 | 9 | 4 |
| MLP | 0.0003 | 879 / 890 / 904 | 0.015624 | 0.0473582 | 0.0152358 | 1 / 3.0311 / 0.9752 | 11 | 14 |
| MLP | 0.001 | 689 / 703 / 729 | 0.0130397 | 0.0544576 | 0.0128955 | 1 / 4.1763 / 0.9889 | 14 | 26 |
| ResNet-32 | 0.0003 | 887 / 892 / 898 | 0.599812 | 0.622001 | 0.596953 | 1 / 1.0370 / 0.9952 | 5 | 6 |
| ResNet-32 | 0.001 | 914 / 925 / 965 | 0.107322 | 0.753524 | 0.10694 | 1 / 7.0211 / 0.9964 | 11 | 40 |
| ResNet-32 | 0.003 | 415 / 422 / 482 | 0.107334 | 1.02908 | 0.106345 | 1 / 9.5877 / 0.9908 | 7 | 60 |
| Transformer | 0.0003 | 1010 / 1015 / 1020 | 2.3449 | 2.40647 | 2.33731 | 1 / 1.0263 / 0.9968 | 5 | 5 |
| Transformer | 0.001 | 913 / 921 / 960 | 1.06442 | 1.9225 | 1.06336 | 1 / 1.8061 / 0.9990 | 8 | 39 |
| Transformer | 0.003 | 787 / 794 / 855 | 0.647419 | 2.14557 | 0.645477 | 1 / 3.3140 / 0.9970 | 7 | 61 |

## Column guide

**Event identifiers and provenance.** `spike_id` identifies one event within a task/LR trajectory; `task`, `dataset`, `model`, `learning_rate`, `seed`, `gpu`, `objective_units`, `loss_units`, `source`, and `initial_tensor_sha256` retain the run identity. Within each case the initial weights and objective are matched across LRs. Different cloud devices were used for some arms.

**Boundaries.** `before_state=a`, `onset_state=a+1`, `peak_state=p`, `recovery_state=c`. `end_state` is c when recovered, otherwise the observed horizon. `right_censored` and `peak_is_provisional` flag the incomplete final event. `after_recovery_state=c+1` is the next observation after the first return, when available; it makes no assertion about sustained recovery.

**Losses and changes.** `loss_before`, `loss_onset`, `loss_peak`, `loss_recovery`, `loss_after_recovery`, and `loss_last_observed` are raw objective losses. `*_loss_ratio` divides the corresponding loss by `loss_before`; `*_change_percent` is 100 times that ratio minus 100. `peak_to_recovery_change_percent` instead divides recovery loss by peak loss. `peak_loss_increase` is the absolute peak minus pre-rise loss.

**Durations and excess loss.** `rise_steps=p-a`, `recovery_steps=c-p`, `total_steps=c-a` count optimizer updates. `observed_post_peak_steps` and `observed_total_steps` describe follow-up to `end_state`, including censored cases. `excess_loss_sum` sums positive excess above the fixed pre-rise loss at the observed interior states, in loss units times updates. Recovery quantities remain blank when unobserved.

**Optimizer anchors.** Each metric below has `_before`, `_peak`, and `_recovery` columns:

| Metric prefix | Meaning |
| --- | --- |
| `eta_kappa_hat_raw` | Recorded normalized projected change of consecutive raw gradients |
| `rho_raw`, `rho_momentum`, `rho_adam` | Consecutive cosine for raw gradients, stored first moments, and adaptive directions |
| `gradient_norm`, `momentum_norm`, `adam_direction_norm` | Whole-vector Euclidean norms |
| `unbiased_momentum_norm` | Norm of the bias-corrected first moment |
| `momentum_over_gradient_norm` | Stored first-moment norm divided by raw-gradient norm |
| `gradient_current_moment_cosine` | Alignment of the raw gradient with its current first moment |
| `incoming_gradient_prior_moment_cosine` | Alignment of the raw gradient with the preceding first moment |
| `gradient_adaptive_cosine` | Alignment of the raw gradient with the LR-free adaptive direction |
| `R_adam` | Cumulative retention of applied Adam updates through the loss state |

At parameter state s, gradient/moment/direction statistics use paper indices s+1; `R_adam` uses updates through s. No statistic is smoothed or reset at the event boundary. Terminal raw-gradient norm/projected change exist, but prospective moments/directions and their cosines do not. Undefined quantities are blank, never replaced by zero.

**Rise-window extrema.** `*_rise_max` reports the maximum projected change or gradient/momentum norm on [a,p]; `*_rise_min` reports the minimum raw/momentum/Adam cosine on [a,p]. Each has `_state` and `_lead_to_peak_steps` columns. Earliest ties win. These are retrospective descriptors using the whole rise, including the loss peak.

**State table.** `spike_steps.csv` retains the raw scalar record at every state from a through c (or the censored end), plus event identity, flags, loss difference, and loss ratio. Nested source keys are flattened with underscores. `streams_raw_*`, `streams_moment_*`, `streams_adaptive_*` and `streams_actual_*` preserve recorded norms, cosines, cumulative geometry, and undefined reasons; the remaining columns preserve cross-stream alignments, norm ratios, applied-step products and numerical rounding checks. These source stream prefixes include the update at s+1; `R_adam` and `actual_update_norm` separately refer to applied updates through s. Outgoing applied-step products and rounding checks require the next update and must not be used as pre-update covariates. Shared boundary states occur once per event.

## Scope and verification

Only existing Run215 production histories are used: three models, three constant LRs each, one frozen seed0 initialization per model, 1,024 full-objective updates per arm. MLP uses mean half squared error on its prepared 5,000 CIFAR-10 images; ResNet uses mean cross entropy on all 50,000 training images; Transformer uses token-weighted NLL on all 2,088,600 prepared targets. Loss units differ across cases.

No new training, Hessians, LR interventions, significance tests, or causal claims. Event selection uses only loss. Event rows from the same trajectory can be dependent; these are not independent random seeds. The extraction is exploratory.

Verification checks original file sizes/SHA-256, complete clocks, loss/gradient equality across logs, the projected-change identity, detector edge cases/ties/censoring, an independent streaming segmentation, every anchor, and every exported state statistic. The manifest pins the generator, mathematical definition and source histories.
