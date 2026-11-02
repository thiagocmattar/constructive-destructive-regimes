# Analysis036 paper evidence preparation

This is a local reanalysis of completed Run215, requested after inspection of the
existing results. It is exploratory. It adds no optimizer training, seeds,
Hessians, learning-rate changes or cloud work. Interventions are deferred.
The active manuscript and its copied figures are not changed.

## Questions and evidence roles

1. What do raw-gradient reversal and magnitude changes look like near a loss
   excursion under Adam? Use the same three loss-selected micro examples.
2. Does the sequence recur, and with what misses, lead times and false alarms?
   Expand the existing loss-only episode census across all nine arms. Keep
   episode coverage separate from state-level alarm precision. Display all
   27 large highest-rate episodes aligned to their 10% loss-rise onset; show
   individual traces and medians, with no independent-sample uncertainty claim.
3. Does the diagnostic add information beyond the observed loss history? Keep
   the previously verified forward-blocked comparison as the primary analysis;
   add the prespecified robustness checks below, reporting all outcomes.
4. What mechanism produces the sequence? TODO for a separately designed future
   intervention. The present evidence does not answer this causal question.

## Frozen inherited definitions

Use PEAK_TEMPORAL_PROTOCOL.md unchanged: 16 prior states for episode baseline,
10% onset, recovery below the frozen baseline, large episodes at least 50%,
start searching at state64. Early warnings occur 1-8 states before onset while
loss remains less than 5% above its trailing minimum. Retain missed episodes
and unfinished recovery. Report lead time conditional on a warning; it is the
earliest qualifying warning in the eight-state window, not the first ever.

Forecast eligibility uses the past/current 17 states (t-16 through t); target
is a 10% rise relative to current loss during the next8 updates. Use the same
4,534 forward-test decisions, all tasks/LRs, five expanding-prefix fits, and
24-state gap. A past/current input never uses a future loss or gradient.

## Additional checks specified before computing them in this revision

Keep C=1 L2 logistic regression, liblinear, training-only StandardScaler, no
class weighting or model selection. Use the same chronological masks and rows.

- Add raw cosine alone, and raw cosine plus one-state log gradient growth to
  the existing loss baseline. These are component descriptions of projected
  change, not independent mechanistic evidence.
- Strengthen the loss baseline: retain arm indicators, age, current log loss
  and current log draw-up, and replace the three selected lag changes by all
  16 log-loss changes relative to the current loss. Compare this richer baseline
  with the same baseline plus current projected change.
- Add projected change delayed by16 and32 states to that richer baseline.
  Delays are strictly past-only. For these comparisons enlarge the exclusion
  gap to48 states for every model, including the contemporaneous model, so
  training outcomes precede every test diagnostic input. This is a descriptive
  timing-specificity check, not an exchangeable permutation test or p-value.
- Report pooled, per-task and per-block AP and Brier, plus a training-prevalence
  constant-probability reference. Case rows subset pooled-model predictions;
  there are no new independent training runs and no claim of task transfer.

No favorable horizon, threshold, delay, block, model or case will be selected
after these checks. Existing 3x3 horizon/rise sensitivity stays visible.
No naive step bootstrap, independent-event standard error, or confirmatory
p-value will be reported. Repeated episodes and time blocks are dependent.

## Figure contract

Use Analysis024's Times New Roman/STIX typography, 7.16-inch width, three case
columns, blue/orange/green rate colors, embedded vector fonts, panel letters,
light grids, and unclipped data. Four rows in overview and micro view: loss
(log), raw-gradient eta*kappa_hat, raw-gradient rho, and gradient/momentum
norms divided by their own value at the displayed trajectory/window start.
Line style distinguishes gradient (solid) from stored first moment (dashed).
Normalized norms compare proportional change, not absolute magnitudes.
Absolute separate norms remain in the original full unsmoothed figure.

Overview: loss raw; other curves complete right-aligned trailing8 arithmetic
means (linear space before log axes). Eight is a display choice fixed here,
not optimized for a result. Plot every raw projected-change observation as a
faint same-color line behind its mean, so the mean cannot conceal crossings.
No threshold count or temporal conclusion is computed from smoothed curves.
Micro and onset-aligned event views: no temporal smoothing. Events align at
10% onset, not the loss peak. The event view uses the same four variable rows,
individual traces plus pointwise medians; these are descriptive aggregates.
The fourth PDF shows primary forward-test precision-recall curves by case.

All rows use state t, meaning t completed updates: L_t and g_t are measured at
theta_t; m_t here denotes the stored moment after consuming g_t, before the
next displacement. This analysis convention maps to manuscript g_(s+1),
m_(s+1) at completed-update state s; the integration draft explains the map.
Actual increment retention is excluded from these compact figures and remains
available in the full diagnostic figures.
