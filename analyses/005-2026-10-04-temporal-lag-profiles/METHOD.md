# Temporal raw-gradient measurements

This experiment addresses item 3 of `manuscript/ideas/reflections.md`. Here,
“lag” means separation in accepted optimizer updates. The measured vector is the
unique whole-model accumulated gradient, unscaled and before gradient clipping
or AdamW. An update contains 1,024 sequences of 2,048 tokens, except the retained
final partial update. Loss reduction, initialization and token ordering follow
the frozen source-run recipe.

For a window of N updates, the primary descriptive profile is

\[
C(k)=\frac{1}{N-k}\sum_{t=k+1}^{N}
\frac{g_{t-k}^{\top}g_t}{\|g_{t-k}\|\|g_t\|},\qquad 1\leq k<N.
\]

This is an uncentered signed cosine profile. It does not assume stationarity or
that distant pairs become orthogonal. Both members of a pair must lie inside the
same window. Zero-denominator pairs are undefined and excluded with coverage
reported. Primary plots retain all integer lags 1–64; the full 128-update window
also permits a secondary analysis through lag 127. Coverage decreases from 127
pairs at lag 1 to 64 at lag 64 and one at lag 127.

Cosine gives each valid pair equal weight. The exact all-pair decomposition uses
the different, norm-weighted quantity

\[
I_k=\sum_{t=k+1}^{N}g_{t-k}^{\top}g_t,
\quad E=\sum_t\|g_t\|^2,
\quad J=\sum_{k=1}^{N-1}I_k=
\frac{\|\sum_tg_t\|^2-E}{2}.
\]

The recorder independently accumulates the raw gradient sum in FP64 in each
window. Thus J and the residual J−I₁ are measured exactly up to floating-point
reduction, even though individual longer-lag dot products are estimated. The
plotted cumulative interaction is 2 times the cumulative sum of I_k divided by E.
A lag-one estimate of the squared sum is E+2I₁; its relative error is assessed
against the exact squared sum. No identity between this raw-gradient sum and an
AdamW parameter displacement is asserted. The SGD displacement identity in the
reflection requires the corresponding constant-step plain-SGD updates.

Four independent fixed signed CountSketch maps with 16,384 buckets are drawn
using isolated PCG64 generators. Every coordinate contributes to one bucket with
a random sign. We average their projected dot products and normalize by the
original exact gradient norms; we do not normalize by projected norms or clip
estimates into [−1,1]. The sign-hashing inner-product estimator is unbiased over
the map distribution under the standard random-map assumptions
([Weinberger et al., 2009, revised 2010](https://arxiv.org/abs/0902.2206v5)).
Unbiasedness does not establish accuracy for this particular gradient history.
The experiment therefore records exact adjacent pairs at every update and
random anchors in each 16-update block for lags 2, 4, 8, 16, 32 and 64. Calibration
reports sample coverage, RMSE and maximum absolute cosine error by window and lag.
Passing mechanics calibration is not evidence for a scientific conclusion.

Each FP32 sketch history occupies 186,646,528 data bytes, plus its NumPy header.
Offline dot products and reductions use FP64. Four-map minimum/maximum bands
show projection sensitivity; they are not confidence intervals or independent
training-seed replications. Fixed-map errors are correlated across pairs. Small
sign changes below demonstrated projection error remain unresolved.

The unordered-window reference is the mean cosine across all distinct pairs in
the same window. A uniformly random reordering has this expected mean at every
lag. Subtracting it provides a descriptive comparison to the window's overall
alignment; it is not a permutation significance test or proof of a stationary
mean component. The temporal Gram heatmap provides a further descriptive check
of how a window average conceals time variation.

Windows were fixed before inspecting their spectra: updates 1–128, 293–420 and
585–712. They are called early, middle and late, without imposing constructive
or destructive labels. The primary late window retains update 712; a sensitivity
analysis recomputes its profile after excluding that partial batch. The first
test uses one training seed, 1235, at 14M, 31M and 70M. Architecture and initial
parameter differences prevent a causal interpretation in terms of dimension alone.

An additional offline coverage control was added after inspecting the completed
early windows, before middle/late profiles were available. It keeps the first
64 source updates in each window fixed across all lags1–64; their target updates
remain inside the128-update window. This holds source coverage constant, without
asserting that target dynamics are stationary. It is a sensitivity analysis;
the predeclared all-within-window-pair profile remains primary.
