# Tracing loss-associated oscillation through network representations

This section consolidates the MLP, ResNet and Transformer diagnostics before
causal interventions. It describes the measured trajectories, not a universal
instability law. [Reusable execution guide](../../../research/INSTABILITY_TRACING.md).

The [subsequent technical report](../report/instability-tracing-report.pdf) refits
MLP on 450-500 for an event-scale comparison. This earlier note retains the
448-800 results; do not mix its MLP spectra/overlaps with the shorter fit.

## 1. Motivation

A loss spike and reversing gradients identify an optimization event but do not
explain what changes inside the network. We trace the associated functional
motion backward, from outputs through intermediate representations, to ask:
where is reversal already visible, where does it become stronger or change
structure, and does it involve growing activation magnitude or movement within
representations whose overall magnitude remains similar?

The method combines direction, magnitude and spectral concentration. Negative
cosine alone is insufficient: tiny displacements can reverse harmlessly, and
oscillation can persist between loss peaks. Backtracking localizes observable
motion along the computation graph; all layers are jointly trained, so the first
oscillating representation is not automatically the causal source.
Backward traversal means inspecting predecessor representations, not computing
an additional backward pass through them.

## 2. Methodology

### Measurement contract and probes

Keep the original training trajectory, optimizer, data order and update policy.
Choose a fixed ordered diagnostic set Q before inspecting layer traces. Use the
whole objective's example set when affordable; otherwise use an explicitly
labeled subset that preserves sequence context and normalization groups.
Freeze diagnostic inputs and stochastic choices, and restore any training
RNG/buffer state affected by observation. Record every representation's shape.

All three experiments use native full-objective training gradients. Their
representation coverage differs:

| Case | Training objective per update | Representation diagnostic set | Outgoing updates |
|---|---|---|---|
| MLP / CIFAR-10 | 5,000 images; half squared error | All 5,000 images | 448-800 (T=353) |
| ResNet / CIFAR-10 | 50,000 images; cross-entropy | First original 1,000-image BatchNorm group | 280-310 (T=31) |
| Transformer / WikiText-2 | 2,088,600 supervised tokens; cross-entropy | First 64 sequences, all 2,240 supervised positions | 552-600 (T=49) |

The subsets reduced capture/storage cost while preserving their original forward
contexts. Their selection is convenient, not a claim of population-wide
representativeness. Probe/full-loss correlations are 0.975 and 0.997 for ResNet
and Transformer. MLP diagnostic loss equals full-objective loss. Outputs are
raw linear values for the MLP and log-probabilities for the other two; their
correct-label log-probabilities provide an additional per-image/token loss readout.
ResNet/Transformer windows were chosen from historical loss before inspecting
their layer traces; MLP retains its existing multi-spike window. Event selection
is retrospective, not prospective validation of a spike predictor.

### At one optimizer update t

State t means t completed optimizer updates. Let $A_t^{(\ell)}$ be representation
ell on Q at that state, with all sample/token and feature coordinates retained.
The model and its buffers are those produced by native training; observation
must not alter them. Flatten only for vector operations; let $P_\ell$ be the
number of scalar entries. For example, 5,000 samples x 200 hidden features gives
$P_\ell=1{,}000{,}000$.

For the actual update from t to t+1, define

$$
\delta_t^{(\ell)}=\operatorname{vec}(A_{t+1}^{(\ell)}-A_t^{(\ell)}),\qquad
\rho_t^{(\ell)}=
\frac{\langle\delta_t^{(\ell)},\delta_{t-1}^{(\ell)}\rangle}
{\|\delta_t^{(\ell)}\|_2\|\delta_{t-1}^{(\ell)}\|_2},
$$
$$
r_t^{(\ell)}=\frac{\|\delta_t^{(\ell)}\|_2}{\sqrt{P_\ell}},\qquad
s_t^{(\ell)}=\frac{\|A_t^{(\ell)}\|_F}{\sqrt{P_\ell}}.
$$

Here rho measures consecutive motion direction, r measures displacement
coordinate RMS, and s measures activation coordinate RMS. Zero-norm motion has
undefined cosine. Loss L_t is evaluated at state t; the movement entering a
loss peak p is labeled p-1. Computing rho_t requires A_(t+1), so it is available
after that update, not as a warning before it. Raw-gradient alignment separately
uses $\cos(g_t,g_{t-1})$; gradients themselves are never differenced in this method.

### Fixed-window decomposition and backward traversal

For t=a,...,b, let $T=b-a+1$ and stack the displacement rows into
$D_\ell\in\mathbb R^{T\times P_\ell}$:

$$
D_\ell=U_\ell\Sigma_\ell V_\ell^\top,\qquad
\alpha_{t,k}^{(\ell)}=\langle\delta_t^{(\ell)},v_{\ell,k}\rangle
=\sigma_{\ell,k}U_{\ell,tk}.
$$

Compute the exact Gram $K_\ell=D_\ell D_\ell^\top$ in coordinate blocks.
Its eigenvalues are $\sigma_{\ell,k}^2$ and its eigenvectors give $U_\ell$;
the full spatial basis need not be materialized. Top-k energy is
$\sum_{j=1}^k\sigma_{\ell,j}^2/\sum_j\sigma_{\ell,j}^2$.

Fit one uncentered basis per representation over the entire window, without
sample averaging or row normalization. Fix each mode's sign once. Coefficient
sign changes indicate reversal along a fixed pattern spanning samples and
features; they do not identify individual neurons. Report squared singular-energy
fractions, top-one/top-ten coverage and modes needed for 95%/99%. The displayed
coefficient alpha/sqrt(P) has the same scale as displacement coordinate RMS.
For a gradient SVD, stack the raw g_t vectors instead of representation differences.

Fixed patterns do not imply fixed instantaneous motion. Changing relative
coefficients can rotate a displacement within their common span. Whole-window
concentration therefore does not establish that local dominant axes/subspaces
remain stationary; that requires a separate sliding-window/principal-angle test,
which has not been performed here. Cross-layer temporal overlap below does not
measure that rotation.

Compare layers of different widths through their temporal bases:

$$O_k(\ell,F)=\frac{\|U_{\ell,k}^{\top}U_{F,k}\|_F^2}{k},$$

where F is the declared output space and k does not exceed either numerical
rank. This measures shared temporal subspaces, not identical neuron directions.
Alignment-series correlations provide a complementary comparison.

```text
Choose a loss-defined window [a,b], ordered stages, fixed Q and capture contract.
Retain states a-1 through b+1, preserving the original training trajectory.
For each t=a,...,b:
    use A_(t-1), A_t, A_(t+1) to compute each stage's rho_t, r_t and s_t;
    retain loss and native raw-gradient alignment with declared coverage.
For each stage:
    build exact FP64 temporal Grams; fit the fixed-window SVD;
    compute spectral coverage, signed coefficients and output temporal overlap.
Start at output and visit predecessors, including both residual branches:
    compare reversal timing, displacement magnitude and shared temporal patterns;
    identify the earliest observed reversal and boundaries where it changes.
Verify source identities, observer invariance and reductions; report limits.
```

The window-wide SVD and heatmap normalization by each row's window maximum use
future information relative to an interior t. They are retrospective diagnostics,
not validated online triggers. Near-degenerate modes favor subspace rather than
individual-mode interpretation. Full definitions and exact transmission
identities are in [METHOD.md](../METHOD.md).

## 3. Results

**MLP:** first-linear reversal occurs at 350/353 updates; its alignment-series
correlation with output is 0.991 and top-ten temporal overlap is 0.996. The
pattern is strongly preserved through both tanh layers, although tanh attenuates
movement. This is similarity across layers, not homogeneous motion across hidden
dimensions. Inputs are exactly fixed; the first learned projection changes.

**ResNet:** first-convolution reversal occurs at 19/31 updates, versus 25/31 at
output. Correlation with output alignment rises from 0.578 at the first convolution
to 0.991 at the last residual stage; temporal overlap rises from 0.598 to 0.969.
Thus early reversal exists, but the output-like pattern is clearer deeper in the
residual network. The main loss peak is 0.4582 at state 293, 1.319 times its
pre-excursion baseline, with recovery at 302.

**Transformer:** at update 558, embedding, first attention-output and final-output
cosines are +0.390, -0.672 and -0.920. Strong downstream reversal is visible while
the embedding still moves forward. Embedding/output temporal overlap is 0.461,
versus 0.982 at the final block. Attention internals were not captured, so this
localizes a measured boundary rather than a specific query/key/value mechanism.
Loss peaks at 1.6942 at state 562, 2.010 times baseline, and recovers at 596.

| Output-change spectrum | MLP | ResNet | Transformer |
|---|---:|---:|---:|
| First-mode energy | 25.17% | 24.03% | 41.65% |
| First-ten energy | 98.16% | 87.97% | 90.83% |
| Modes for 95% | 8 | 16 | 15 |

These are not nearly one-dimensional full-window trajectories. Different windows
and output spaces prevent a controlled intrinsic-dimensionality comparison:
rank cannot exceed T, which is 353, 31 and 49, respectively.
Window-wide concentration also does not establish that rank decreases during a
spike; that would require a matched comparison with quiet windows.

Displacement grows around the loss excursions without a comparable global
activation-norm explosion at several measured stages. ResNet first-convolution
activation RMS stays within 0.7891-0.7927; Transformer first-block norm1 output
stays within 1.0006-1.0009. Output displacement maxima occur on updates 292->293
and 561->562, entering the respective loss peaks; some earlier layers peak later.
There is no common rule that displacement peaks after loss. In the MLP,
reversal also persists between peaks, so its presence alone does not identify
which update will produce a spike.

All cases are selected seed-0 Adam trajectories. MLP replay is exactly qualified;
ResNet matches all 56 historical loss/norm pairs. Transformer continuation is not
bitwise historical replay: maximum relative loss discrepancy is 0.270% over 58
states. Raw-gradient/output alignment correlations are 0.776, 0.705 and 0.879;
the two quantities are related but not interchangeable.

## 4. Findings and conclusions

The supported cross-architecture observation is **growing, reversing movement
in learned representations, organized into dominant fixed patterns, accompanying
selected loss spikes**. Architecture affects where that motion becomes strongly
expressed. Relatively steady activation magnitude does not imply steady
representations, and low-rank temporal motion does not mean ten specific neurons
or a shared spatial basis across layers.

The results do not establish a universal first-layer cause, a universal rank-ten
mechanism, a normalization-denominator failure, or a predictor of future spikes.
Temporal similarity alone also does not isolate output-sensitive hidden
directions; that required a separate spatial projection in the earlier MLP study.
Full-data coverage for MLP does not remove the single-seed/window limitation;
ResNet/Transformer additionally have fixed-probe coverage limits.
Global cosine and RMS do not imply identical behavior for every example or feature.

The method supplies candidate boundaries for the next causal tests. Preserve
matched continuations and separate incoming representation changes, local
parameter changes and interactions before assigning a mechanism. The existing
[MLP intervention TODO](../../062-2026-11-07-mlp-gradient-amplification/TODO.md)
remains pending; no intervention is added by this consolidation.

## Evidence and execution cost

Source-qualified observations: [MLP](003-mlp-standardized-trace.md),
[ResNet](001-resnet-trace.md), [Transformer](002-transformer-trace.md).
Their linked tables, numerical results and three reviewed PDFs contain all-stage
statistics. [Run 243 verification](../../../runs/243-2026-11-07-architecture-instability-tracing/verification.json)
records source/transfer/snapshot checks and costs. Full capture plus reduction on
one RTX 4090 took 24.91 minutes for ResNet and 12.56 minutes for Transformer,
overlapping part of the CPU reduction; derived caches were about 161.81/37.65 GB.
These are whole-worker measurements, not isolated diagnostic overhead or portable
throughput promises. The MLP companion reused existing Grams. The reusable guide
specifies probe selection, cost scaling, calibration and retention for future setups.
