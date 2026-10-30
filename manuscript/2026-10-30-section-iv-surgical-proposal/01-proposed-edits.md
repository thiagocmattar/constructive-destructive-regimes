# Proposed edits to Section IV

Use the quoted anchors to locate each edit; line numbers will change during
revision. Replacement blocks are proposed manuscript prose. Square-bracket
cross-references in this proposal are explanatory; LaTeX labels inside code
blocks are the intended final labels. New labels are defined by the S7 plan.

## S1 State the empirical contribution at the opening

**Target:** `draft/sections/04-gd-experiments.tex`, the paragraph beginning
"We examine whether the diagnostics in Section" and ending
"or preventing divergence."

**Replace with:**

```tex
Full-batch gradient descent provides a controlled setting for examining
how consecutive gradient directions relate to cumulative parameter motion
and objective progress. The sampling-noise term vanishes, and each gradient
directly determines an update. Building on the observation that oscillatory
full-GD training can reduce loss over longer timescales~\cite{cohen2021edge},
we compare learning rates at matched initialization, objective and update
budget. We examine local reversal through $\rho_t$, cumulative motion through
$R_t$ and the complete cross-step interaction, and the accuracy of the local
loss approximation in Section~\ref{subsec:loss-connection}. The comparisons
characterize the nine recorded trajectories.
```

This makes the difference from the cited EOS observation explicit without
claiming that the norm expansion or oscillatory loss reduction is new.

## S2 Shorten the main measurement protocol

**Target:** IV-A, from "We use three settings motivated" through
"we compare only the largest value."

**Replace with:**

```tex
We use three settings motivated by the Edge-of-Stability
experiments~\cite{cohen2021edge}: a tanh MLP with two 200-unit hidden layers
on the first 5,000 CIFAR-10 training images; ResNet-32 on all 50,000 CIFAR-10
training images; and a two-layer, two-head word-level Transformer on
WikiText-2, with embedding and feed-forward dimensions of 200. The MLP
minimizes half-squared error summed over the ten output coordinates and
averaged over examples. ResNet minimizes mean cross-entropy over 50 fixed
groups of 1,000 images, with batch normalization computed within each group.
The Transformer minimizes token-weighted negative log-likelihood over
2,088,600 prepared training targets. Every update accumulates the complete
fixed objective.

Each condition runs for 3,000 updates at a constant learning rate:
$\{0.004,0.02,0.04\}$ for the MLP, $\{0.025,0.05,0.1\}$ for ResNet,
and $\{0.05,0.1,0.2\}$ for the Transformer. Rates were selected from earlier
training calibrations to expose contrasting dynamics within this horizon,
without test-loss selection. Within each case, all rates share one frozen
initialization (seed 0) and identical data preparation. There is no momentum,
weight decay, gradient clipping, schedule, augmentation or dropout. These
are paired comparisons conditional on one initialization per case.

We record full-objective loss and raw-gradient geometry at every update.
Gradient $g_t$ is evaluated at $\theta_{t-1}$; loss and displacement at
index $t$ refer to $\theta_t$. The first adjacent pair is undefined and
omitted; all measured gradient norms are nonzero. Both $R_t$ and the
displacement norm in Table~\ref{tab:gd-endpoints} use the gradient-sum
identity, whose agreement with stored-parameter displacement is subject
to FP32 update roundoff. Smoothed curves use complete 50-observation
trailing arithmetic means, computed before logarithmic display.
Late means average the last 1,000 defined observations. Preparation,
precision, accumulation and spectral measurement details appear in
Appendix~\ref{appendix:gd-raw-alignment}.
```

**Move rather than delete:** the existing coordinate/channel standardization,
token streams and exclusions, vocabulary construction, physical accumulation,
FP32/FP64/TF32 settings, spectral cadence, subset coverage and solver details.
Their destination is S7. Add the exact reference tutorial/revision from the
retained input provenance when implementing; do not leave "reference tutorial"
as the sole identifier. Do not change the frozen vocabulary or training data.

## S3 Make the interpretation of loss coexistence precise

**Target:** IV-B, the final two sentences beginning
"These observations support the separation" through
"unstable or unproductive update."

**Replace with:**

```tex
These observations illustrate the distinction in
Section~\ref{subsec:loss-connection} between adjacent reversal and
longer-timescale objective progress. The accuracy of its one-step loss
approximation is examined separately below.
```

Keep the LR comparison, seven late-negative conditions, ResNet first-event
indices and MLP early-transient caveat. They supply useful empirical detail
and preserve the distinction between the first reversal and sustained behavior.

## S4 Make the cumulative geometry a substantive result

**Target:** IV-C. Change its heading from "Cumulative Cancellation and Gradient
Magnitude" to "Cumulative Geometry and Gradient Magnitude". Replace its first
two paragraphs, from "The lower row" through "a lag-one approximation."

**Replace with:**

```tex
The lower row of Fig.~\ref{fig:gd-alignment-displacement} shows that
$R_{3000}$ decreases with learning rate within each case. Absolute
displacement nevertheless increases, as reported in
Table~\ref{tab:gd-endpoints}. For the MLP, the gradient-sum displacement
norm rises from $0.951$ to $3.139$ and $4.763$, while the accumulated path
length rises from $1.198$ to $64.361$ and $185.542$. The reduction in
$R_{3000}$ therefore reflects faster growth of traveled distance than
of net displacement.

The temporal decomposition further separates adjacent reversal from
whole-trajectory interaction. At $T=3000$, the cumulative adjacent term
$2\sum_{t=2}^{T}g_{t-1}^{\top}g_t$ is negative in seven conditions,
whereas the complete term
$2\sum_{1\leq u<t\leq T}g_u^{\top}g_t$ is positive in all nine.
Positive contributions beyond lag one outweigh the adjacent negative
contribution in those seven trajectories
(Table~\ref{tab:gd-temporal-accounting}). Thus, negative adjacent
alignment and low path-length retention coexist with positive total
cross-step interaction over the complete recorded horizon. Neither
$\rho_t$ nor $R_t$ alone specifies the sign of that total interaction.
```

Keep the existing final paragraph about gradient magnitudes and Fig. 3(a)-(c).
The all-pair result is an endpoint statement; do not extend it to every prefix
or every individual pair. The displacement values are Euclidean quantities
within each fixed parameterization, not amounts of learned information.

## S5 Test the local loss approximation in IV-D

**Target:** IV-D heading "Projected Gradient Change and the Spectral Reference".

**New heading:** "Projected Gradient Change and the Local Loss Approximation".

Keep its first paragraph through the exact threshold equivalence, replacing
the final sentence beginning "Consequently, the two displays" with:

```tex
The diagnostics are algebraically linked: $\eta\widehat\kappa_t$ also
incorporates the consecutive norm ratio, but its threshold crossing
does not independently validate a curvature estimate.
```

Replace the spectral paragraph beginning "Table~\ref{tab:gd-endpoints} places"
and the existing concluding paragraph with:

```tex
We directly assess the local loss approximation by comparing
\begin{equation}
  \widehat{\delta\mathcal L}_{t-1}
  :=-\frac{\eta}{2}\|g_{t-1}\|^2
  \left(2-\eta\widehat\kappa_t\right)
  \label{eq:gd-loss-surrogate}
\end{equation}
with the recorded change
$\mathcal L(\theta_{t-1})-\mathcal L(\theta_{t-2})$ for
$t=2,\ldots,3000$. This gives 2,999 unsmoothed comparisons per condition;
the last update has no subsequent recorded gradient. The comparison is
retrospective because $\widehat\kappa_t$ uses that subsequent gradient.

The approximation's sign disagrees with the recorded change in zero
transitions for the lowest MLP and Transformer rates, but in 1,325 of
2,999 transitions ($44.2\%$) for the Transformer at $\eta=0.2$.
After excluding transitions for which either change is at most $10^{-5}$
times the larger neighboring loss in absolute value, disagreement remains
in 1,239 of 2,837 transitions ($43.7\%$).
Table~\ref{tab:gd-loss-audit} reports every condition, residual magnitudes
and threshold sensitivity. These descriptive thresholds do not establish
floating-point error bounds. Near a small predicted decrease, a residual
can reverse the sign even when it is small relative to the first-order
descent term. The audit measures the approximation error without assigning
it uniquely to finite-step, nonsmooth or numerical effects.

The nine trajectories therefore demonstrate local reversal alongside
positive final all-pair interaction and continued longer-timescale
loss reduction. Their gradient-derived diagnostics characterize observed
motion; the local loss approximation varies in accuracy across these
conditions. The spectral measurements in
Appendix~\ref{appendix:gd-raw-alignment} provide additional context with
explicit objective coverage. Extension to sampling noise, transformed
optimizer directions or prospective intervention requires separate tests.
```

Retain the unbenchmarked runtime/memory statement in the appendix's measurement
paragraph. Avoid introducing a stability threshold from the sign-disagreement
rate. The audit is evidence about the approximate relation, not a test of the
exact displacement identity or a claim of prediction before an update.

## S6 Replace one endpoint column

**Targets:** Table I caption in IV and
`draft/tables/analysis024_endpoints.tex`, generated by
`analyses/024-2026-10-17-eos-gd-lr-calibration/08_build_manuscript_assets.py`.

Keep the first six columns. Replace `eta*lambda_max,3000` with
`||Delta_3000||`, using `final_gradient_sum_displacement` from the scalar audit.
The proposed new column is:

| Model | LR | Displacement norm |
| --- | ---: | ---: |
| MLP | 0.004 | 0.9511 |
| MLP | 0.02 | 3.1391 |
| MLP | 0.04 | 4.7628 |
| ResNet-32 | 0.025 | 5.7045 |
| ResNet-32 | 0.05 | 9.0183 |
| ResNet-32 | 0.1 | 13.9329 |
| Transformer | 0.05 | 11.3183 |
| Transformer | 0.1 | 15.6026 |
| Transformer | 0.2 | 21.9693 |

**Proposed caption:**

```tex
Descriptive endpoints of the nine matched learning-rate conditions.
Late means average the last 1,000 defined gradient observations;
other entries refer to update 3,000. Displacement is computed as
$\eta\|\sum_{t=1}^{3000}g_t\|$, subject to FP32 parameter-update roundoff.
Each model case shares one initialization across rates. Values are
compared within cases; no across-initialization uncertainty is estimated.
```

Update the generator's provenance inputs and verification, rather than pasting
these rounded values into its generated output. Preserve the spectral values
unchanged in the appendix table specified next.

## S7 Extend Appendix D with the moved evidence

**Target:** `draft/sections/07-gd-diagnostics.tex`.

Rename the appendix "Measurement Details and Additional Full-Batch Diagnostics".
Preserve `appendix:gd-raw-alignment` so existing references remain valid. Use
four subsections in this order:

1. **Objectives and Measurement.** Move the detailed protocol from S2 here,
   preserving vocabulary construction, token weighting/exclusions, fixed BN
   groups, precision and physical accumulation. Include the unbenchmarked
   diagnostic runtime/memory qualification once.
2. **Unsmoothed Alignment and Temporal Accounting.** Keep the existing raw
   cosine paragraph and figure byte-for-byte. Add the all-nine-condition table
   `tab:gd-temporal-accounting` with Model, LR, path length, adjacent interaction,
   beyond-lag-one interaction and total interaction. Use the existing fields
   `final_gradient_sum_path_length`, `final_lag_one`,
   `final_beyond_lag_one`, and `final_all_pair`. State their exact definitions
   and the factor of two; use scientific notation where needed. State that
   the last three columns are raw-gradient interactions, not multiplied by
   eta squared. Round for display only; check their sum at source precision.
3. **Retrospective Loss Approximation.** Add `tab:gd-loss-audit` covering all
   nine conditions: Model, LR, raw opposite-sign count/2999, median absolute
   residual divided by `eta*||g_previous||^2`, and eligible/disagreement counts
   at relative thresholds 1e-7, 1e-6 and 1e-5. Use two subtables if needed for
   IEEE width. Define the threshold as
   `tau*max(abs(L(theta[t-2])),abs(L(theta[t-1])))`, retain a transition only
   when both compared magnitudes exceed it, and print every denominator.
   State that serial observations are not independent seed replicates.
4. **Spectral Context.** Move the existing spectral protocol and endpoints
   here, preserving all nine values. Table columns are Model, LR, Hessian
   objective coverage and final `eta*lambda_max`. Retain a short explanation
   that maximum pointwise eigenvalue, directional finite-step change and
   different objective coverage are distinct comparisons. Do not interpret
   subset mismatch as evidence for or against estimator accuracy.

The all-pair and loss tables derive from `section_iv_review_checks.json`;
the spectra derive from `results.json`. No new figure or training trajectory
is needed. Any later time-aligned comparison must pair the Hessian at state s
with the expansion for q at gradient index s+2, not with q at index s.

## S8 Align two interpretations in Section III

**Target 1:** `draft/sections/03-methods.tex`, the block beginning
"The displacement $\Delta_T$ captures" and ending
"measure of the magnitude of learned information in parameter space."

**Replace with:**

```tex
The displacement $\Delta_T$ captures the net parameter change produced
by training. Its squared norm measures the magnitude of this change in
the chosen parameterization; its relation to learned function or predictive
performance requires additional evidence.
```

**Target 2:** in the normalized displacement subsection, the two sentences
beginning "Values near one indicate" and ending "greater cumulative cancellation."

**Replace with:**

```tex
Values near one indicate that the updates retain most of their accumulated
path length as net displacement. Smaller values indicate greater loss of
path-length retention through directional variation, including cancellation;
they do not determine the sign of the total cross-step interaction in
Eq.~\eqref{eq:cumulative-interference}.
```

Keep the exact equations and the motivation for measuring displacement.
This preserves the geometric interpretation while avoiding an unsupported
information measure and the identification of low R with negative total
interaction. A broader rewrite of the Section V stability discussion remains
separate from this surgical package.
