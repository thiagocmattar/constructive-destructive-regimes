# Anatomy of excursions with peak increases above 10%

This is the author's requested review of event morphology, raw-gradient
diagnostics, momentum and adaptive directions. It uses only existing Run215
histories and preserves the complete spike dataset and all figure files.
No new experiment, forecasting model, intervention or manuscript edit is made.

## Review outputs

- [Main table, printed as text](table_gt10.md): three dataset/model blocks,
  three learning rates per block, mean ± sample SD and maximum peak increase.
- [LaTeX table](table_gt10.tex): the same nine rows with `\multirow{3}` dataset/model
  labels; requires the `multirow` package. It is prepared in this analysis only.
- [Selected event rows and anatomy features](events_gt10.csv).
- [Expanded summaries with exact per-metric denominators](summary_gt10.csv).
- [Source and output hashes](manifest.json).
- Generator: [07_spike_anatomy.py](../07_spike_anatomy.py).

Reproduce from the repository root:
`python analyses/036-2026-11-02-controlled-adam-diagnostics/07_spike_anatomy.py`.
Tables are text/CSV/LaTeX, not figures.

## Selection, onset and the unit of analysis

One excursion is one row/sample, with possible temporal dependence between
samples from a trajectory. For the existing definition, a is the last state
before the first upward departure from its observed loss, a+1 is the first
elevated state, p is the earliest largest loss before recovery, and c is the
first return at or below the fixed loss at a. Partial declines and several
local maxima can belong to the same excursion. No diagnostic determines these
boundaries. See the [mathematical definition](../loss_spike_definition.md).

The new operation is a filter, not a new segmentation:

$$100(\mathcal L_p/\mathcal L_a-1)>10.$$

It selects 131 observed events: 129 recovered and 2 censored. Every summary of
amplitude, timing and anatomy uses the common recovered cohort, so there are
no provisional peaks or imputed recovery times in those statistics. The main
N column counts all selected observed events, with censored counts in
parentheses. Rows with no qualifying events are retained and show missing
statistics, not zeros.

There is no startup exclusion. Two selected MLP events start at initialization,
where previous-gradient cosine and projected change are undefined. Pre-rise
cosine/projected-change summaries therefore use 46 rather than 47 observations
at MLP LR 0.0003, and 38 rather than 39 recovered observations at LR 0.001.
All other metric-specific missingness is recorded through n in the summary CSV.
Sample standard deviations use n-1; none is presented as a standard error.

## Operational meanings of the table columns

All scalar inputs are unsmoothed. Event summaries give equal weight to events.
For phase summaries, average within each event first, then summarize those
event means; long recoveries do not automatically receive more weight.

- Peak increase: percentage relative to the event's fixed pre-rise loss.
  Its main-table cell reports mean ± SD [maximum]. Maxima of all other metrics
  remain available in the expanded summary CSV.
- Time [rise; recovery]: p-a and c-p, in accepted optimizer updates.
- Before: the diagnostic available at parameter state a, before applying the
  update that produces the first loss increase. In the paper's clock this is
  raw gradient g_(a+1), moment m_(a+1), and consecutive cosine rho_(a+1).
- Recovery cosine: the event's mean raw-gradient cosine at arrival states
  p+1,...,c, followed by a summary across events. It is not just the cosine at c.
  Projected-change cells use the same [before; mean recovery] pairing.
- Gradient norm ratio [rise; recovery]: norm(g_(p+1))/norm(g_(a+1)) and
  norm(g_(c+1))/norm(g_(p+1)). These are multiplicative endpoint changes;
  5 means fivefold growth and 0.3 means reduction to 30% of the peak-state norm.
  They are not averages of one-update ratios and do not assert that the norm
  maximum occurs at the loss peak.

The expanded event/summary files also retain:

- Gradient norm ratio across the final update before the rise; analogous
  momentum and adaptive-direction rise/recovery ratios.
- Raw, momentum and adaptive cosine means separately on rise and recovery.
- Projected-change means on rise and recovery; fractions of positive cosine
  and decreasing loss observations during recovery.
- Maxima of gradient norm, momentum norm and projected change over the **full**
  interval [a,c], their states and signed leads to p. A negative lead means
  the diagnostic maximum is after the loss peak. This avoids forcing positive
  leads by searching only within the rise.
- The first state above +10% and the first projected change above two within
  the event (including a), with separate leads to first loss increase, +10%
  crossing and loss peak. This search does not examine an arbitrary pre-event
  lookback. A missing first crossing remains missing.

Rise-phase response diagnostics use arrival states a+1,...,p; recovery uses
p+1,...,c. These windows are disjoint and correspond to the completed loss
transitions in each phase. The pre-rise diagnostic is at a. This timing is
distinct from the prospective outgoing parameter update stored at a state;
no outgoing applied-step quantity is used as a pre-update predictor.

## What the current evidence says

Pre-rise raw cosine is negative in 117/127 defined observations. Projected
change is already above two at the last pre-rise state in 21/127. Thus the
destructive-gradient starting condition is common, but a pre-onset crossing
above two is not a universal signature of these events.

The full-event projected-change maximum is before the loss peak in 82/129
events, coincident in 29, and after it in 18. The full-event gradient-norm
maximum is before the loss peak in 14, coincident in 110, and after it in 5.
Growth before a peak and a maximum strictly before that peak are different
claims. These counts include all recovered >10% events, not only illustrations.

The recovery does not uniformly return to positive raw-gradient alignment.
Mean event-averaged recovery cosine ranges from -0.89 to -0.55 across the MLP
rates, is -0.36 and -0.13 for the two eligible ResNet rates, and -0.02 and +0.09
for the two eligible Transformer rates. Nevertheless, the mean fraction of
decreasing loss steps during recovery ranges from 0.73 to 0.95 across these
groups. Loss recovery therefore can coexist with continued raw-gradient reversal.
Momentum/adaptive cosine measures in the expanded summary describe whether
optimizer averaging preserves a different directional pattern.

The largest observed recovered increase under this definition is +858.77%
(ResNet LR 0.003), corresponding to 9.59 times its pre-rise loss.

## Reviewer checks on the interpretation

1. **Do onset and a pronounced spike mean the same time?** No. For the retained
   highest-LR illustrations, first loss increases are at states 690/416/788,
   first +10% crossings at 697/418/790, and loss peaks at 703/422/794. The gray
   bands currently begin at those +10% crossings, not at the first positive
   loss increment. Respective first projected-change values above two within
   the event are at 691/415/788. A lead to the peak is not automatically a lead
   to the first loss increase.
2. **Are projected change, negative cosine and norm growth independent?** No.
   The identity eta*kappa_hat = 1-rho*(current norm / previous norm) couples
   them algebraically. Report them as complementary components of one response,
   not three independent confirmations of a mechanism.
3. **Does this prove Adam amplifies a feedback loop?** No. That remains the
   author's mechanism hypothesis. Momentum and second-moment division can
   alter or attenuate the raw-gradient changes; their growth cannot be inferred
   from raw norm alone. The expanded summaries explicitly compare momentum
   and adaptive-direction norm ratios rather than assuming amplification.
4. **Can these selected rows establish an a-priori detector?** No. They establish
   anatomy conditional on a future >10% event and permit temporal comparisons.
   Predictive evidence also needs the diagnostic behavior at non-event times
   and an evaluation using information available before the chosen outcome.
   No new forecasting setup is introduced here.
5. **Are the recovery statements universal?** No. Retain the model-dependent
   cosine results above, rather than describing every loss decline as a return
   to a constructive raw-gradient regime. Alignment and loss decrease remain
   separate observed quantities.

## Suggested paper sequence

Training-loss overview → micro-plots defining the event and its diagnostic
timing → per-LR density plots showing the distribution and tails → this table
checking how broadly the anatomy recurs among all >10% recovered events.
This moves from the visible phenomenon to an interpretable example and then
to the event population. Present the Adam feedback explanation as a hypothesis
consistent with selected trajectories, with the model-dependent recovery and
algebraic coupling stated explicitly. Intervention remains deferred.

Verification checks source SHA-256/clock identities, preservation of all
original event columns in a CSV round trip, the pre-rise projected-change
identity, phase membership, gradient ratio telescoping, full-event maximum
placement, and independent mean/SD calculations. No new training or cloud use.
