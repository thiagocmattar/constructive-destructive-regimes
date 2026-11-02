# Observation008: repeated Adam spikes and forward-blocked diagnostics

## Sources, question and comparison

The user requested a three-column extension of the raw peak micro view and an
assessment of whether projected raw-gradient change, gradient growth and momentum
growth precede pronounced loss spikes across tasks and repeated events. Reuse
all nine verified Run215 arms, with the same single frozen initialization per
task. No training, Hessian/HVP, new seeds, LR/schedule intervention, cloud work
or additional spending is performed.

[PEAK_TEMPORAL_PROTOCOL.md](../PEAK_TEMPORAL_PROTOCOL.md) records the loss-only
episode rule, availability clock, eligibility, target, fixed alarms, baseline,
forward blocks, gap and sensitivity choices before forecast fitting. The already
inspected Transformer event motivated the design; this is exploratory, not a
preregistered or causal confirmation. [The analysis](../PEAK_TEMPORAL_ANALYSIS.md)
reports timing, repeats, misses, false alarms, all predictive ablations and a
proposal for stronger validation and separately authorized optimizer-state forks.

## Figure caption, reduction and coverage

`figures/adam_peak_micro_three_models.pdf`: unsmoothed Adam loss-peak micro views
for CIFAR-10/MLP at LR 0.001, CIFAR-10/ResNet-32 at LR 0.003 and
WikiText-2/Transformer at LR 0.003. Select each highest-LR arm's completed loss-only
episode with the largest peak/baseline ratio. These are conspicuous examples,
not an average or diagnostic-selected sample. Peaks are 703,422,794 and windows
are 693-729,412-481,784-852 respectively. Each begins ten states before its peak
and ends at the first subsequent loss below the starting value. The frozen
episode baseline and the figure's starting-loss reference are both recorded;
they differ when the episode's loss trough is at another state.

The nine rows show training loss (log), L_t-L_(t-1), raw-gradient eta*kappa_hat,
raw-gradient rho_t, stored-momentum rho_t, adaptive Adam rho_t, raw-gradient norm
(log), stored-momentum norm (log), and cumulative actual-update R_t. All finite
points are shown without averaging, binning or thinning. Positive loss differences
are orange bars; negative differences blue; original diagnostics green. Light-grey
dashed vertical lines mark the point before loss first exceeds the starting-loss
reference, and dark-grey dashed lines mark the selected peaks. A dotted horizontal
line marks each starting loss and the projected-change reference2. Neither is an
identified Adam stability boundary. Cosines use [-1.08,1.08]; other Y limits are
local and norm rows scale separately. R_t axes do not start at zero.

Everything uses parameter-state availability: loss and raw gradient at theta_t,
with the moment/adaptive diagnostic after consuming g_t but before the next
parameter displacement. Relative to the previous full/micro PDFs, the six
gradient/momentum/adaptive rows shift one position left, from update t+1 to
state t. This is an explicit clock change, not new measurements. Loss retains
its original state index. Actual-update R_t also retains its original index,
since it accumulates only the updates completed through state t; no future
increment is attributed to the current state. Its prefix still begins at
initialization. The 24 original curves contain 1,408 exact values, and the three
loss-difference series contain 173 exact adjacent differences:1,581 values total.

## Temporal results and interpretation

The local Transformer fresh threshold crossing is at state 788, when loss is
only0.40% above its trough. It precedes the 10% rise at 790 and loss maximum794.
Norms grow strongly at 788; projected change, gradient norm and momentum norm
reach local maxima at 790,792,793 respectively. The first positive loss change
is already at 788, so the signal precedes a pronounced rise, not every increase.
The threshold>2 already implies gradient-norm growth algebraically; the two
cannot count as independent evidence.

The loss-only extraction finds129 episodes, 45 large (>=50% peak rise), with two
unfinished recoveries. At highest LRs, fresh crossings precede13/17 large MLP,
3/3 ResNet and 7/7 Transformer episodes by 1-8 states while loss remains below
a 5% draw-up. Dependent repetitions and misses are retained, not filtered out.

The primary forecasting target is >=10% loss increase in the next 8 states from
a currently quiet (<5% draw-up) state. Five expanding-prefix / forward-test
blocks with a 24-state gap produce4,534 eligible predictions,495 positive.
Adding projected change to the task/LR/age/loss-history model raises pooled
average precision 0.2602 to 0.4543 and lowers Brier 0.11972 to 0.08685, improving
both in all five blocks. All nine horizon/threshold sensitivities retain a
pooled AP improvement. The combined model is not uniformly better, particularly
for ResNet. An untrained threshold>2 has187 positives among 891 alarm states:
21.0% precision,37.8% recall,704 false-positive states. Episode coverage and
alarm precision have different denominators. See the full tables and limitations.

## Generators, checks, uncertainty and reproduction

`15_peak_temporal_analysis.py` pins the protocol and all original scalar histories,
verifies source hashes and clocks, extracts all episodes, records every selected
example, fits the fixed forecast comparisons, and writes peak_events.csv,
peak_forecasts.csv and peak_temporal_results.json. Norm source columns are distinct
from probability columns, which carry the prefix prob_. No test row scales or
trains a classifier, and all training outcomes precede the earliest test input.

`16_plot_peak_micro_three_models.py` checks every original curve against its
hash-pinned full-unsmoothed counterpart with the declared clock mapping, verifies
loss differences and Matplotlib artists, and writes the PDF and figure manifest.
All 27 panels pass page/text-boundary checks and Poppler visual review: source
points, bars, panel letters(a-aa), labels and timing footers are legible without
overlap or clipping. Scratch PNGs remain ignored.

`17_verify_peak_temporal.py` independently reconstructs all 4,534 forecast labels,
raw source features, fixed alarms, episode baselines/maxima/recovery and exact
figure values. Synthetic future perturbation leaves current/past features
unchanged while changing the forecast label; a synthetic rebound is merged and
unfinished recovery retained. A separate ranking calculation reproduces average
precision and direct squared errors reproduce Brier scores to 2.23e-16 maximum
absolute error. peak_temporal_verification.json records the passed checks.

No independent-initialization uncertainty or causal effect is measured. The
selected examples, event thresholds and forecast models are exploratory. Later
blocks are held out from fitting, but all results are now inspected; they are
not untouched evidence for a subsequent selected rule. Do not treat individual
states or repeated peaks as independent replicates. Do not infer convergence,
generalization, stability, a loss-landscape classification, or a deployable LR
controller from this analysis.

Reproduce in order:

    python analyses/036-2026-11-02-controlled-adam-diagnostics/15_peak_temporal_analysis.py
    python analyses/036-2026-11-02-controlled-adam-diagnostics/16_plot_peak_micro_three_models.py
    pdftoppm -singlefile -scale-to 2400 -png analyses/036-2026-11-02-controlled-adam-diagnostics/figures/adam_peak_micro_three_models.pdf analyses/036-2026-11-02-controlled-adam-diagnostics/scratch/adam_peak_micro_three_models
    python analyses/036-2026-11-02-controlled-adam-diagnostics/16_plot_peak_micro_three_models.py --accept-reviewed
    python analyses/036-2026-11-02-controlled-adam-diagnostics/17_verify_peak_temporal.py
