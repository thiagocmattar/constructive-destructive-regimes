> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [METHOD.md](../../../../../../analyses/063-2026-11-07-architecture-instability-tracing/METHOD.md).

# Consecutive-representation oscillation tracing

Read the [consolidated scientific section](../../../../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/004-consolidated-instability-tracing.md)
for the three-case evidence and the [reusable workflow guide](../../../../../../research/INSTABILITY_TRACING.md)
for probe selection, stochastic-training adaptations, computational costs and retention.

## Purpose and conclusion contract

Locate where a loss-associated oscillation is first observable along a network's
forward computation and characterize its transmission through subsequent stages.
This is a descriptive localization method, not proof of a unique causal source.
Gradients originate from the complete downstream loss and optimizer state couples
updates across time. A targeted, matched intervention is a separate causal test.

The deliverable is a common architecture trace, numerical table, source-qualified
observation and candidate follow-up for each model. Negative cosine alone is not
an instability criterion: require a measured loss excursion and report motion
magnitude and recovery as well. A quiet or different trajectory is a valid result.

## 1. Freeze the trajectory and measurement contract

Specify dataset bytes/order, model and optimizer states, full training objective,
physical batches, seed, precision, model mode, buffers, RNG and update indexing.
Select the time window using recorded loss, before inspecting layer diagnostics;
label that event selection as retrospective. Check replay/resume on the execution
hardware. Cross-device continuation is not exact historical replay unless verified.

Choose a fixed ordered diagnostic probe. Training loss/gradients and representation
probe coverage are separate fields. Never silently shrink an objective or claim
probe-wide conclusions hold for all examples. For BatchNorm, retain the original
complete statistic group; for language retain token positions, masks and context.
Disable stochastic measurement drift by preserving the original model contract,
not by silently switching training-mode normalization to evaluation mode.

For future minibatch setups, preserve the native objective/update batches and
label native losses/gradients by their actual sampling coverage. A fixed/full
reference loss or fixed-probe gradient is an additional diagnostic with its own
cost. Freeze diagnostic stochastic choices and verify repeated probes at fixed
weights; restore training RNG and buffers. The current three cases use full-
objective gradients, which must not be silently assumed in a different setup.

## 2. Inventory actual computational stages

Build a forward-ordered list, including raw input, learned input projection or
embedding, pre/post nonlinearities and normalization, residual branch endpoints,
residual sums, pooled representations and final outputs. Record the full tensor
shape at each stage. Clone hook values immediately when later operations are
in-place. Repeated module calls require separate identifiers.

For a convolution, $A_t^{(l)}$ has sample, channel and spatial axes. For language,
it has sequence, token-position and feature axes. Keep every declared coordinate;
flatten only for inner products. Token IDs are fixed categorical inputs, not a
Euclidean activation whose changes should be confused with learned embeddings.
Raw logits, log-probabilities and probabilities are different output spaces and
must be labeled separately. For classification, a per-example common logit shift
does not change softmax; compare log-probabilities as a loss-relevant check.
All-vocabulary Euclidean geometry still weights every vocabulary coordinate
equally, rather than by its probability or loss sensitivity. Therefore also
extract the correct-label log probability for every image/token as an auxiliary
loss readout. Its negative mean must reproduce the probe cross-entropy, but its
cosine/SVD retain individual images/tokens without averaging them.

## 3. Consecutive motion, norm and alignment

State $t$ is the number of completed optimizer updates. For displayed
$t=a,\ldots,b$, define the outgoing displacement

$$
\Delta A_t^{(l)}=A_{t+1}^{(l)}-A_t^{(l)},\qquad
\rho_t^{(l)}=\frac{\langle\Delta A_t^{(l)},\Delta A_{t-1}^{(l)}\rangle_F}
{\|\Delta A_t^{(l)}\|_F\|\Delta A_{t-1}^{(l)}\|_F}.
$$

Capture states $a-1$ through $b+1$. The increment entering a peak at $p$ is
labeled $p-1$. Record activation coordinate RMS, displacement coordinate RMS and
per-sample/token vector RMS where meaningful; do not compare differently sized
layers by unnormalized Frobenius norm alone. Any zero denominator is undefined,
stored as null, without adding an epsilon. Verify input invariance explicitly.

Raw gradients use $\rho_t^g=\cos(g_t,g_{t-1})$ and SVD rows $g_t$ themselves.
Do **not** difference gradients. Applied parameter increments are a separate
stream, including optimizer transformations and LR, if captured.

## 4. Fixed temporal-window SVD

Let $T=b-a+1$ and $P_l$ be all retained scalar coordinates of one activation tensor.
Form

$$
D_l=\begin{bmatrix}\operatorname{vec}(\Delta A_a^{(l)})^\mathsf T\\
\vdots\\\operatorname{vec}(\Delta A_b^{(l)})^\mathsf T\end{bmatrix}
\in\mathbb R^{T\times P_l},\qquad D_l=U_l\Sigma_lV_l^\mathsf T.
$$

Rows are optimizer updates; columns identify fixed sample/token/feature positions.
No temporal centering, sample averaging or row normalization is applied. Report
$\sigma_k^2/\sum_j\sigma_j^2$, cumulative top1/top10 energy and ranks for 95/99%.
This measures representation-change energy, not loss variance.

For the actual Run 243 probes, the distinction between sample rows and time rows
is concrete:

| Representation | One-state representation | SVD displacement matrix |
|---|---|---|
| ResNet first convolution | 1,000 images × 16 channels × 32 × 32 pixels | 31 updates × 16,384,000 fixed coordinates |
| ResNet pooled representation | 1,000 images × 64 features | 31 updates × 64,000 coordinates |
| ResNet class output | 1,000 images × 10 classes | 31 updates × 10,000 coordinates |
| Transformer embedding/block output | 2,240 token positions × 200 features | 49 updates × 448,000 coordinates |
| Transformer vocabulary output | 2,240 token positions × 33,278 vocabulary entries | 49 updates × 74,542,720 coordinates |
| Correct-label output readout | 1,000 images or 2,240 token positions × 1 value | 31 × 1,000 or 49 × 2,240 coordinates |

The 2,240 token rows are 64 sequences times 35 supervised positions, in the same
order at every update. A right singular vector is a pattern over **all retained
sample/token and feature coordinates**. It is not one class column, not a mean
example, and not a separately refitted direction for each update. A mode can
weight different coordinates differently for different examples. Its coefficient
at time $t$ is one scalar summarizing motion along that fixed full-probe pattern.

Compute the exact $T$-by-$T$ Gram matrix $K_l=D_lD_l^\mathsf T$ in FP64, summing
coordinate blocks when histories are large. Its eigendecomposition gives the
same nonzero singular values and temporal coefficients without materializing
the complete right basis. Check symmetry, PSD within numerical tolerance, trace,
diagonal norms and independently recomputed adjacent products. Dense captures or
replayable checkpoints plus identities must remain available.

The coefficient

$$
\alpha_{t,k}^{(l)}=\langle\Delta A_t^{(l)},v_{l,k}\rangle
=\sigma_{l,k}U_{l,tk}
$$

uses one **fixed pattern** for the whole window. Fix its sign once (largest
absolute temporal coefficient positive in this implementation; a largest spatial
coordinate convention is equally valid). Sign changes indicate reversal along
that pattern. Refitting local SVDs allows the basis to change and does not provide this
test. Near-degenerate singular values make individual modes ambiguous; retain
subspace comparisons and energy coverage rather than overinterpreting mode 1.
Zero coefficients do not count as strictly negative sign products.
Counted flips are descriptive: a tiny coefficient can change sign with negligible
motion. Always inspect its amplitude and the mode's energy alongside the count.
An uncentered leading mode can also describe steady drift rather than the
oscillation of interest; negative full-vector cosine need not coincide with a
sign change in that particular mode.

**Fixed basis does not establish fixed motion.** For example,
$\Delta A_t=\cos(\phi_t)v_1+\sin(\phi_t)v_2$ can rotate within a fixed
two-dimensional span. A concentrated whole-window spectrum does not establish
that local dominant axes or subspaces remain stationary. Testing that requires
explicit local-window SVDs and principal-angle/subspace comparisons, respecting
sign ambiguity and near-degenerate modes. No such rotation test has been run
here. Cross-layer temporal overlap is a separate statistic.

For a new event window, refit the corresponding Gram submatrix; merely cropping
coefficients retains the old basis. The [technical report](../report/instability-tracing-report.pdf)
uses a newly fitted MLP 450-500 window, while the earlier 448-800 analysis remains
available. Keep the preceding displacement for the first cosine but exclude it
from the fitted SVD rows.

## 5. Backtrack from outputs through the computation graph

Start with loss and output motion, then traverse predecessor stages. At each
stage compare reversal timing, displacement magnitude, spectral concentration and
signed coefficient histories. Correlate alignment time series on defined pairs;
these temporally dependent correlations have no IID causal significance.

For two layers of different widths, compare **temporal** subspaces:

$$
O_k(l,m)=\frac{\|U_{l,k}^\mathsf TU_{m,k}\|_F^2}{k},\qquad
E_{l\to m}=\frac{\|U_{l,k}^\mathsf TD_m\|_F^2}{\|D_m\|_F^2}.
$$

Use the same time rows and $k$ no greater than either numerical rank. Equal mode
indices do not imply equal neuron directions. These statistics are invariant to
sign flips and within-subspace rotations. An oscillating early layer and a quiet
input localize the earliest learned motion; they do not demonstrate temporal
causal initiation. Report stage coverage gaps, such as attention's internal
query/key matrices when only attention-module boundaries are captured.

Representation coordinates themselves matter: a function-preserving change of
hidden coordinates can change hidden Euclidean geometry. Global reversal does
not imply every image, token or feature reverses. Identifying output-sensitive
versus output-insensitive hidden components requires an additional spatial
projection or Jacobian analysis, as performed separately for the MLP. Temporal
SVD similarity alone does not establish that distinction.

Window lengths limit the available rank and affect top-k energy and overlap.
Reports with different numbers of updates are standardized descriptive case
studies, not a controlled architectural comparison. Do not infer that one model
is intrinsically more low-dimensional by comparing unmatched-window top10
fractions; retain the complete spectra and rank95/rank99 statistics.

## 6. Transmission and separation of possible mechanisms

For an affine layer $Z=HW^\mathsf T+\mathbf1b^\mathsf T$,

$$
\Delta Z=\Delta H W_t^\mathsf T+H_t\Delta W^\mathsf T+
\Delta H\Delta W^\mathsf T+\mathbf1\Delta b^\mathsf T.
$$

These terms separate upstream motion, local parameter motion and interaction;
they are an accounting identity, not independently causal energy fractions.
For a residual sum $Y=B+S$, verify $\Delta Y=\Delta B+\Delta S$ and retain the
cross term in its squared norm. For elementwise nonlinearities, finite secants
measure attenuation/amplification, with the derivative limit at zero difference.
Normalization and attention mix coordinates: norm ratios alone do not isolate
their local parameters from incoming changes. Use matched frozen-module/Jacobian
probes only as explicitly additional diagnostics.

The standard trace is a first pass. Follow-up decomposition is required before
calling an apparent boundary a causal amplifier. Input covariance, normalization
denominators, neuron saturation and parameter norms provide hypotheses, not an
automatic source label.

## 7. Standardized output and verification

For each model provide: (a) full-objective and probe loss; (b) raw-gradient and
output alignment; (c) all-stage alignment heatmap; (d) per-stage displacement RMS
time series/heatmap; (e) spectral concentration and output temporal overlap;
(f) leading signed coefficients for selected architecture landmarks; and a table
with every stage's dimensions, defined/negative counts and energy statistics.

The observation starts with conclusions, then specifies coverage, exact replay
status, architecture, event window, definitions, interpretation limits, artifact
hashes, scripts and figure captions. Show the fixed-input zero-motion result
separately rather than assigning it cosine 0. Validate observers against an
unobserved forward, unchanged gradients/buffers/RNG and independent products.
Archive a recovery checkpoint, progress log, raw evidence and transfer inventory.
Render publication PDFs and inspect their numerical and visual contents.

## Algorithm in executable order

```text
Inputs: qualified state theta_s and optimizer state; fixed training objective;
        fixed ordered probe Q; time window [a,b]; ordered computational stages.
1. Validate source hashes, model mode, data grouping, masks and snapshot hooks.
2. Continue the original optimizer; log native loss/raw gradient and probe loss.
   Label native coverage: full objective in the current cases; otherwise the
   original training batch. Additional reference evaluations are explicit.
3. For t=a-1,...,b+1, capture each A_l(theta_t,Q) without changing training state.
4. For every stage l:
     build delta rows A_l(t+1)-A_l(t), t=a-1,...,b;
     accumulate the exact FP64 temporal Gram over all probe coordinates;
     calculate consecutive cosine and displacement RMS for t=a,...,b;
     eigendecompose the Gram restricted to t=a,...,b;
     recover singular energies and signed fixed-basis temporal coefficients.
   For raw gradients, use g_t rows directly, never g_(t+1)-g_t.
5. Start at output; visit each predecessor, including both residual branches.
   Compare reversal timing, magnitude, energy concentration and temporal overlap.
6. Report the earliest observed oscillating learned stage and any discontinuity
   in transmission, with probe/window limits. A constant input has undefined
   motion alignment. Do not convert this localization into a causal claim.
7. Validate numerical reductions and raw identities; produce the common report.
8. Put targeted causal tests on a separate TODO; execute only under their own
   matched intervention contract.
```
