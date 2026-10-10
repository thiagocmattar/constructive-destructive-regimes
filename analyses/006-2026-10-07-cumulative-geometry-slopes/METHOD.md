# Derivatives of cumulative geometry

## Source and coverage

Inputs are Run 002 attempt02 full-pass events, summary, verification, resolved
config and input identity; Run 002 config and generating script; and its original
figure data and comparison PDF. Exact paths, sizes and SHA-256 hashes appear in
[results.json](results.json). The source figure's 712 embedded rows match the
original events exactly, and its recorded event SHA-256 matches the current file.
All source hashes are checked again after analysis and during verification.

Pythia-14M has 14,067,712 unique trainable parameters, initialized randomly with
the recorded SmallInit/WangInit recipe, seed 1234. MiniPile/tokenizer revisions,
token cache and ordering are inherited unchanged from the
[source contract](../../runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/README.md).
No vector histories are reconstructed or new batches evaluated.

One update is an accepted optimizer boundary after accumulated,
supervised-token-mean next-token loss. Raw gradients are unscaled, before
clipping. Adaptive directions are bias-corrected first moments divided
coordinatewise by the square root of bias-corrected second moments plus epsilon,
evaluated in FP64 from actual FP32 AdamW state. Moments receive clipped gradients.
The adaptive stream excludes LR, decay, descent's minus sign and subtraction
rounding. All unique trainable parameters enter the common scope.

All 712 updates are retained: 728,375 sequences, 1,491,711,416 input tokens and
1,490,983,041 supervised targets. Primary display uses updates 1-711, each with
1,024 sequences; update 712 has 311. The initial LR is zero, although gradients
and moments are measured. Validation uses all 500 documents and 693,329 supervised
targets, with initial/final evaluations only.

## Exact inclinations

"Inclination" means slope per optimizer update. A rendered line's angle changes
with axis scales, so no angle in degrees is used. Apply the definitions separately
to the raw-gradient and adaptive-direction streams. In the equations below,
\(g_t\) denotes the raw-gradient case; the same norm algebra applies to the
recorded adaptive directions.

\[
A_t=\sum_{u=1}^{t}\|g_u\|,\qquad
B_t=\left\|\sum_{u=1}^{t}g_u\right\|.
\]

With \(A_0=B_0=0\), exact backward differences are

\[
\Delta A_t=A_t-A_{t-1}=\|g_t\|,\qquad
\Delta B_t=B_t-B_{t-1}.
\]

Here \(\Delta\) denotes a finite difference, not the manuscript's parameter
displacement \(\Delta_T\). \(\Delta A_t\) is nonnegative; \(\Delta B_t\) can be
negative if an incoming vector reduces the resultant.

The quotient identity is

\[
\boxed{
\frac{B_t}{A_t}-\frac{B_{t-1}}{A_{t-1}}
=\frac{\Delta B_t-(B_{t-1}/A_{t-1})\Delta A_t}{A_t}.
}
\]

Consequently, a flat ratio requires

\[
\Delta B_t=\frac{B_{t-1}}{A_{t-1}}\Delta A_t.
\]

Equal absolute slopes, \(\Delta B_t=\Delta A_t>0\), instead increase a ratio
below 1. Equal fractional growth, \(B_t/B_{t-1}=A_t/A_{t-1}\), leaves the ratio
unchanged. We therefore also retain exact log increments
\(\log(A_t/A_{t-1})\) and \(\log(B_t/B_{t-1})\).

For a continuous interpolation, the familiar quotient derivative is

\[
\frac{d(B/A)}{dt}=\frac{AB'-BA'}{A^2};
\qquad
\frac{d(B/A)}{dt}=0
\ \Longleftrightarrow\
\frac{B'}{B}=\frac{A'}{A}
\quad(A,B>0).
\]

No new symbol for the cumulative ratio is needed.

## Recent growth versus accumulated history

When \(\Delta A_t>0\), the exact identity can also be written

\[
\frac{B_t}{A_t}-\frac{B_{t-1}}{A_{t-1}}
=\frac{\Delta A_t}{A_t}
\left(\frac{\Delta B_t}{\Delta A_t}
-\frac{B_{t-1}}{A_{t-1}}\right).
\]

Thus, recent \(\Delta B/\Delta A\) above the historical ratio raises cumulative
B/A; equality leaves it flat; a lower value reduces it. For a window, use
\(\sum\Delta B/\sum\Delta A\), rather than averaging individual slope ratios.
This is recent resultant growth per unit vector norm, not retained information.

The prefactor \(\Delta A_t/A_t\) also matters: cumulative ratios become less
responsive as their denominators grow. By the triangle inequality,
\(|\Delta B_t|\leq\Delta A_t\), so

\[
\left|\frac{B_t}{A_t}-\frac{B_{t-1}}{A_{t-1}}\right|
\leq2\frac{\Delta A_t}{A_t}.
\]

If norms stay bounded while A continues accumulating, this bound tends to zero
even without evidence that optimization has converged.

Incoming alignment with the accumulated past is recoverable from the scalars:

\[
\cos\left(g_t,\sum_{u<t}g_u\right)
=\frac{B_t^2-B_{t-1}^2-\|g_t\|^2}
{2B_{t-1}\|g_t\|}.
\]

It differs from adjacent cosine. We retain it to check whether incoming vectors
oppose the previous cumulative sum. Its interpretation concerns these diagnostic
vectors along AdamW training, not actual optimizer displacement.

## Reduction and uncertainty

Primary figure 1 shows A/B, exact increments with trailing 31-update means, the
mean B increment required for a flat ratio, and recent window slope ratio versus
cumulative B/A. Figure 2 shows training loss, trailing 31-point ordinary least
squares loss slopes, and corresponding ratio slopes. Symmetric logarithmic
axes retain both slope signs and zero. Complete windows only are displayed.

The CSV preserves exact unsmoothed quantities for all 712 updates. The first
ratio derivative, first log increments and first alignment to accumulated past
are undefined and stored empty. These source A/B values are positive; a future
zero denominator would require a null, rather than division or a substituted zero.

Summary intervals are 1-178,179-356,357-534,535-711. Terminal windows are 32,64,100
and 128 updates, ending 711; sensitivity shifts them to end 712 and includes the
partial batch. Each mean backward difference includes the incoming increment
at the interval's first update. Ratio endpoint changes compare its first and
last recorded values; OLS slopes use the recorded values themselves.

Loss means are supervised-token weighted. Primary full-sized batches have equal
target counts, so weighted and unweighted means coincide there. Tail sensitivity
and the final CSV rolling mean retain the shorter batch's correct weight.
OLS uses equal update weights on each recorded token-mean loss. Changing-batch
loss slopes are observational trends, not fixed-objective derivatives.

Smoothing sensitivity uses 15/31/61 updates. Centered 61-update checks around the
observed ratio minima are retrospective, not online detectors. There is no
predefined plateau tolerance, significance test or convergence threshold.
Correlated updates are not independent replicates; there is one training seed.

[Verification](verification.json) checks the exact algebra, reconstruction of
cumulative curves, token-weighted tail reduction, source/output hashes and four
synthetic cases, including constant B/A with increasing B and constant B/A with
decreasing quadratic loss. The PDFs were rendered and visually inspected.

## Per-step loss-difference grid extension

The requested 3x2 grid pairs \(L_t-L_{t-1}\) on y with exact backward
differences of A, B and cumulative B/A on x. It uses no smoothing.
Rows follow that order; columns are raw gradients and Adam adaptive directions.
Updates 2-711 provide 710 points in each panel; update 1 has undefined loss
difference, and the partial update 712 is retained in the new CSV only.

Because loss_t is evaluated before update_t and batches change, this same-index
pairing is contemporaneous. It does not isolate the effect of the current
optimizer step. All panels share y limits and a common color scale for optimizer updates.
A/B axes use their own norm units. Ratio-slope x-axes use identical symmetric
logarithmic scaling so small late differences remain visible alongside the
larger early differences. The values themselves are untransformed in the data.

[The extension design](panel_design.json), [script](03_loss_difference_grid.py)
and [observation](observations/002-loss-difference-grid.md) specify full coverage,
hashes, caption and interpretation limits separately from the original reduction.

## Matched-lag grids at 2,5 and 20 updates

For each requested lag k, all axes use the same endpoints:
\(L_t-L_{t-k}\) versus \(A_t-A_{t-k}\), \(B_t-B_{t-k}\), or
\((B/A)_t-(B/A)_{t-k}\). These are lagged differences, as clarified by the user,
without logarithmic transformation, smoothing or division by k.

Display endpoints start at k+1 and end at 711, giving 709/706/691 pairs for
k=2/5/20. All 712 source rows remain in the combined CSV for each lag.
The first k differences are undefined. Colors share the original update scale;
within each PDF y limits are shared and ratio-change x limits are shared.
Numeric axis limits can differ across lag figures.

The [design](lagged_panel_design.json),
[generating script](04_lagged_loss_difference_grids.py) and
[observation 003](observations/003-matched-lag-grids.md) retain the clarification,
exact formulas, hashes, checks and captions. The limitations of changing-batch
loss and overlapping pairs continue to apply.
