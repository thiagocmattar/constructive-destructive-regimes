# Adam state map and paired loss benefit

Analysis031 / H025. Local exploratory reanalysis of Analysis030, Runs187-204.
No training, cloud work, or manuscript changes. The outcomes were already known
when this analysis was designed; none of the comparisons below is confirmatory.

Completed: [report](REPORT.md) and four reviewed PDFs. At the 16-update cutoff,
actual/adaptive accumulation ratios reduce final-benefit prediction MAE by
37.0%/37.2% beyond the model/LR baseline, with improvement in all three seed folds.
No tested initial summary improves the primary prediction. The retention signal
is sensitive to observation window, target, and intervention; it is a candidate
for new validation, not a reliable controller. Independent checks cover 1,068
feature values, 1,956 predictions, 6,840 plotted source values, and 9,216 ratio
identities. Local analysis only; no cloud cost. H024's broader scheduling question
remains unresolved. Next analysis032, hypothesisH026, run205.

Question: which available optimizer-state summaries, if any, predict the
magnitude of a learning-rate intervention's loss benefit beyond its model size
and LR multiplier? The held-rate (1x) continuation is the paired reference.

## Comparison and target

There are 12 altered-rate arms (0.5x/2x), six common starting states, and three
seed clusters. Keep both model sizes and both altered arms of a seed together.
The primary target is full-validation loss at 128 for held LR minus altered LR;
positive means benefit. A secondary target subtracts the same benefit at 16,
measuring the additional relative loss reduction from 16 to 128. This distinction
prevents an already-realized early advantage from being described as future gain.
All validation covers the original 500 documents/693,329 target tokens, including
the tail. The separately named loss probe is the fixed 128-sequence training subset.

Baseline: predict the average benefit of the same model size and LR multiplier
on the other two seeds. Assess every candidate with leave-one-seed-out prediction,
using MAE/RMSE and all three fold errors. Do not treat updates as independent data.

## Features and time boundary

Before the first changed parameter update, the first incoming gradient provides
six common-state summaries: raw adjacent cosine, projected gradient change,
log raw norm, log previous-moment norm, incoming-gradient/prior-moment cosine,
and incoming/retained moment magnitude ratio. These describe the same common
state; the held arm supplies the shared value to avoid using tiny replay roundoff
differences (maximum absolute discrepancy below 1e-6 in these six summaries).
Multiply each by log2(LR multiplier) to permit an action-dependent prediction.
This boundary includes computing the incoming gradient, not only loading a checkpoint.

Post-intervention summaries use only local updates 2 through the cutoff. Primary
cutoff 16; cutoffs 8/32/64 are explicitly exploratory timing sensitivities. Candidate
features are raw/projected/input-moment/moment/adaptive/actual-step geometry,
descent cosine with the current raw gradient, norms and norm growth, adaptive
norm amplification, clipping fraction, first-order batch loss decrease, and
actual displacement/path-length ratio. Use the candidate-minus-held difference
of each summary. Log transforms precede averaging; growth is log(mean last4 /
mean first 4). Actual retention includes steps 1 through cutoff. The first-order
term is -g dot delta_theta on the update's sampled batch; it is not measured
validation improvement or an independent curvature estimate. The SGD projected
change expression remains a descriptive quantity under Adam.

All 21 candidate response features (20 geometry/magnitude summaries plus fixed-
probe benefit) are reported. Fit one feature at a time, after centering both
feature and target within model-by-multiplier cells on the training seeds.
Estimate the single common slope by least squares. Predict the held seed with
training cell means and slope. No hyperparameter or threshold is tuned.
At cutoff 16 also add each geometry feature to the probe-benefit comparator,
using two slopes, to assess information conditional on early loss response.
These small fits remain fragile: each training fold has only 8 arms/four states.

After the first exploratory scan identified actual-step retention, raw, moment,
and adaptive accumulation ratios were added as specificity controls. This is a
documented outcome-informed extension of the exploratory scan, not a preregistered
discovery. All four ratios use the same prefix 1 through cutoff; no full-trajectory
ratio enters an early prediction.

## Interpretation and figures

This is a map of recorded scalar state, not a reconstruction of the full Adam
second-moment tensor or Hessian. Temporal views are descriptive, not additional
replications. All arms have the same eventual LR winner, so only benefit magnitude
can vary independently of the action. A favorable cross-validation score is a
candidate for fresh validation, not a discovered scheduling law. Searching all
these features/windows on a known small cohort entails selection optimism.

Figures: a two-coordinate momentum state map colored by final paired benefit;
paired temporal responses; and all primary predictive scores with individual
held-seed errors. Solid LR traces use color, never dashed styles. Source hashes,
reduced arrays, uncertainty, captions, and limitations accompany the figures.
Generate with01_analyze.py and02_plot.py; verify with03_verify.py.
