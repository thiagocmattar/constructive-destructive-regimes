> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [008-cross-architecture-interventions.md](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md).

# Representation-informed interventions across architectures

Status: complete. All executed branches were retrieved and independently
verified; all 14 new PDF pages were rendered and reviewed. All five task pods
were deleted after verification. Closeout: 8 November 2026 UTC.

## TLDR

- **The strongest transferable optimization candidate is a distributed local
  affine-displacement cap.** Keep all layers trainable and all Adam state
  updates; scale a unit's proposed weight/bias increment only when its direct
  response on fixed probe inputs exceeds a past-calibrated budget. The same q50
  rule lowers mean training loss by 22.85% for MLP seed 0, 16.22% for ResNet,
  and 26.00% for Transformer in their measured windows. MLP seeds 1/2 improve
  by 19.43%/32.08% without retuning.
- **Successful damping does not require removing reversals.** ResNet and
  Transformer have more negative output-change cosines under the cap while
  their loss excursions shrink. Control movement size and location; do not
  treat a negative cosine count as a substitute for learning performance.
- **The result concerns optimization in selected training setups.** Terminal
  test loss is higher under the distributed cap on all three models and all
  three MLP initializations. Other rules trade slower training for better
  terminal test loss on ResNet. There is no uniformly best policy here.
- **The embedding-LR intuition receives mixed support.** Embedding motion stays
  mostly forward-moving at 2x/4x LR, but average training loss is nearly neutral
  at 2x and worse at 4x. Adding 4x to the distributed cap also worsens the mean
  relative to the same-worker cap-only arm.

## Question

Our tracing analysis showed that consecutive representation changes can reverse
along a small set of functional patterns during loss excursions. This motivates
controlling the movement of representations while keeping feature learning
active. It does not establish that negative cosine itself is harmful, that the
first visibly oscillating layer is a unique cause, or that removing its motion
will improve optimization.

This study tests three distinctions: selecting *where* to damp versus uniformly
reducing a layer's norm; controlling displacement *amplitude* versus suppressing
reversal; and applying a transient gate versus retaining damping across updates.
All interventions are compared with unchanged Adam. No layer is frozen in this
new round. Runs 269-285 belong to H048 and the existing Analysis 065.

## Matched training and probes

| Case | Full objective per optimizer update | Native LR | Intervention fork | Loss window | Fixed diagnostic probe |
|---|---|---:|---:|---|---|
| MLP / CIFAR-10 | Half squared output error, summed over 10 outputs and averaged over 5,000 examples | 0.001 | 448 | 448-800, extended to 1152 | All 5,000 examples, original 1,000-example chunks |
| ResNet-32 / CIFAR-10 | Mean cross-entropy over 50,000 examples | 0.003 | 264 | 264-576 | First intact 1,000-example normalization group |
| Transformer / WikiText-2 | Token-weighted NLL over 2,088,600 supervised tokens | 0.003 | 552 | 552-928 | First 64 sequences, 2,240 supervised positions |

The original data ordering, physical batches, architectures, and losses are
unchanged. Adam uses beta1=0.9, beta2=0.999, epsilon=1e-8, no weight decay or
gradient clipping, and native FP32 with TF32 disabled. All raw gradients,
moments and optimizer clocks advance. Most policies transform the native
stored parameter increment *after* Adam. A multiplier of 0.5 therefore denotes
half of that native FP32 increment, not a claim of bitwise identity to Adam
constructed with half its LR. The embedding experiments instead change its
native Adam parameter-group LR to 0.006 or 0.012; the decoder is untied.

The first screen uses initialization 0. Runs 282/283 repeat the selected MLP
rules with pinned existing initializations 1/2 and no seed-specific tuning.
Each worker reconstructs an untreated prefix and qualifies exact native
two-update versus save/reload continuation before branching. Cross-hardware
historical controls are not substituted for the worker's own control.
The forks and displayed windows were selected from earlier diagnostic runs.
Past-only threshold calibration does not turn this into a prospectively
validated spike-detection alarm.

The two RTX 4090 Transformer controls have identical loss histories, but the
RTX 5090 control follows a different numerical trajectory: mean loss 552-928
is 0.91685 versus 0.87432, and maximum loss is 4.1961 versus 3.0525. Native
FP32 determinism within a worker does not imply identical trajectories across
GPU architectures. Every reported intervention effect therefore uses its own
worker's control. The standalone embedding boost and the boost-plus-cap live
on different GPU architectures; their difference is not a matched factorial
estimate of the cap's incremental contribution. The combined arm is also
compared directly with its same-worker cap-only arm where appropriate.
Likewise, the 4090 half-increment control and the 5090 distributed cap are not
a matched head-to-head ranking of those two policies.

Representation cosines flatten **all sample, spatial/token, and feature
coordinates** in the fixed probe. They are not cosines of averaged samples.
MLP outputs are raw regression outputs; ResNet outputs are log-softmax values;
Transformer outputs are log probabilities at valid supervised positions.
The embedding capture is the raw embedding lookup, before multiplication by
sqrt(200) and addition of positional encoding. Its RMS is therefore in those
raw lookup units; it should not be compared numerically with attention RMS
as if the two had the same scale.
Probe reductions never replace or truncate the full training objective.
For N full-objective examples/tokens and n probe examples/tokens, the mean
loss outside the probe is reconstructed as (N L_full - n L_probe)/(N-n).
It is a measured training complement, not a held-out split, and inherits native
loss-reduction rounding. The MLP has n=N, so no outside-probe loss is defined.

Preserving the ResNet normalization group is essential: its batch normalization
does not track running statistics, so an arbitrary smaller group changes the
function being observed.

## Policies

Let u_t be the actual FP32 parameter increment proposed by native Adam at state
t. All threshold calibration uses the **eight preceding untreated updates**.
The same quantiles are transferred across architectures; thresholds themselves
are calibrated separately for each targeted affine module. This differs from
the earlier H047 screen's 32-update calibration, so its quantile rankings need
not agree with this round.

**Coordinate discard and cap.** Pool absolute native increments across the
eight calibration updates, including weights and biases, and take q90. Discard
sets an above-threshold coordinate to zero. Cap preserves its sign and clips
its magnitude. These are tests of update selection, not raw-gradient clipping.

**Local neuron/channel displacement cap.** Hold the module's current input
fixed and evaluate the direct affine contribution of its proposed weight/bias
increment. For output unit j, score the RMS of this contribution over probe
examples and, where applicable, spatial positions or tokens. Pool these scores
over calibration updates and units within each module. Scale the unit's weight
row and bias by min(1, threshold/score), using q75 in the primary entry screen
and q50 in the distributed follow-up. Zero calibration thresholds disable that
module's cap. The stored increment is checked with an FP32 rounding bound.

Entry scope is MLP linear 1, ResNet conv 1, or the first Transformer layer's QKV
projection. Distributed scope is both MLP hidden affine layers, all 31 ResNet
convolutions, or both Transformer QKV projections and all four feed-forward
affines. ResNet normalization/classifier parameters and Transformer attention
output projections, LayerNorm, embeddings and decoder are outside this local
cap. The QKV score is not the complete attention-output change. In general,
this local score does **not** bound the total coupled representation change
when upstream inputs and other layers also move.

[Observation 010](65-010-local-displacement-cap-geometry.md) gives the matrix
form, its input-second-moment interpretation, and the exact incoming-motion
terms left outside this local constraint. It also distinguishes that algebra
from the earlier temporal SVD of representation changes.

The primary entry screen uses q75, whereas the common distributed comparison
uses q50. Their difference therefore does not isolate scope alone. MLP also
has a matched-q50 entry/distributed comparison, and ResNet has matched-q75
entry/distributed arms. The Transformer entry/distributed comparison additionally
crosses GPU architectures; only effects relative to each worker's own control
are claimed there. No isolated Transformer depth effect is established.

**Diagnostic-triggered damping.** A block's multiplier is previous norm EMA
divided by its current raw-gradient norm when rho<0 and the norm exceeds that
preceding EMA; otherwise it is 1. The EMA has beta=0.9 and advances after the
decision. A global version uses one factor. A separate block heuristic scales
by min(1,2/eta-kappa-hat) above 2. The latter is a secant-based heuristic, not an
Adam curvature bound. These variants are deliberately retained as negative
controls when they fail.

**Actual functional budgets.** Runs 278-281 evaluate the complete candidate
network on the fixed probe and backtrack a scalar multiple of u_t until the
actual output change, or all tracked representation changes, satisfy their
prefix coordinate-RMS budgets. These budgets are quantiles of eight outgoing
native changes, rather than quantiles over individual neurons. Run 278 allows
immediate recovery to a factor of 1. Run 279 adds memory: a factor may decrease
immediately but increase by at most 1% per update, starting from 1. The output-only
q50 version is transferred unchanged to ResNet and Transformer. No loss values
or future-window SVD basis enter this controller. A probe movement bound is
not a guarantee of objective decrease, especially outside the probe.

**Probe-loss backtracking.** Runs 284/285 and the second new arm of Run 281
recompute probe loss in FP64 from the candidate outputs and labels. Try native
increment fractions 1, 1/2, ..., 1/2048 and accept the first strictly decreasing
probe loss. If all twelve positive fractions fail, apply a zero increment;
native moments and clocks still advance. This is a loss-aware line-search
comparator, not a newly claimed SVD optimizer. Only the MLP probe covers the
full objective. For the larger models, report both full loss and the weighted
loss outside the probe. Probe loss is a control target, not independent evidence.

**Control reuse.** Runs 281 and 285 reuse a completed control on their own
worker after checking scientific source/config identities, the full fork state
(including optimizer, gradient history and RNG), exact native two-step
continuation, and matching hardware/software. Copied raw files retain their
hashes and a reuse record. Reused controls are neither new independent
replicates nor newly executed training updates. Mechanics tests use fresh short
controls; efficacy comparisons use the authenticated full-window control.

The MLP also tests a retrospective layer-norm-matched control and removal,
halving or capping of the negative component along the preceding first-affine
change. See [observation 009](65-009-what-the-negative-controls-show.md) for those
definitions, the parameter-norm confound, and the bounded follow-up. Those
directions can change with time; none is a future-fitted fixed SVD axis.

The exploratory allocation was adaptive. Run 271's configured block-shock and
embedding-4x-plus-block-shock options were not executed after the MLP shock
failures; their Transformer efficacy is unknown. All three requested selective
criteria were retained on every model. Additional budget went to distributed
functional caps, slow recovery and the loss-aware comparator. Worker manifests
and launch receipts, rather than the full menu in a config, define the executed
arms. The slow first Run 274 cloud attempt completed mechanics only and was
replaced before scientific training began; its evidence and cost remain recorded.

## Measurements, alignment and interpretation

At state t, g_t is the raw full-objective gradient and rho_t=cos(g_t,g_(t-1)).
The recorded eta-kappa-hat is 1 minus <g_t,g_(t-1)>/||g_(t-1)||². It is not a
measurement of a Hessian eigenvalue under Adam. The plotted reference at 1
marks zero adjacent-gradient dot product, not an Adam stability threshold.
Native/actual update norms,
their gradient dot products, gate values, and optimizer clocks are also saved.

A boundary at state s records the incoming representation difference from
s-1 to s. Figures shift it to outgoing update t=s-1, so its displacement is
A_(t+1)-A_t and its cosine compares that with A_t-A_(t-1). Two untouched
predecessor states supply the first comparison. Coordinate RMS divides the
squared displacement norm by the number of retained coordinates before taking
the square root. No sample means, smoothing, interpolation, or new SVD rank
claim is introduced in these screening figures.

Summary reversal fractions divide by all updates in the stated window and
report the fraction with a defined cosine separately. Zero-norm differences
have undefined cosine, not zero alignment. The first actual-update cosine in
each branch is also undefined; representation cosines have the retained
untreated predecessors needed for their first comparison.

One extra terminal state is retained to measure the last outgoing displacement.
Mean loss and positive loss variation in a stated loss window use only loss
states within that window. The output-space loss accounting in observation 009
also includes the last outgoing transition and explicitly labels its coverage.

Assess mean loss, last 32 mean, maximum loss, and total positive loss variation
sum(max(L_(t+1)-L_t,0)) separately. Maximum loss includes the shared initial
state and is not always the height of a localized spike. Lower final loss
after a large transient is not a uniform improvement. We additionally separate
MLP 448-800 from the forward extension 801-1152. The extension was unseen at
the initial screen, but later exploratory policies also used its observed
failures; it is not a held-out selection set for every later idea.

Held-out evaluation is secondary wherever the retained cache provides a full
split. These are optimization experiments, not evidence of improved population
generalization. Architecture transfer uses one initialization per larger model;
the MLP seed replications give limited replication, not a universal guarantee
or a powered uncertainty estimate.

The completed MLP replications already show this distinction concretely. At
1152, distributed q50 raises test loss by 2.87%, 2.76%, and 6.13% on seeds
0/1/2, despite lowering mean training loss on all three. Slow recovery raises
test loss by 0.87%, 0.52%, and 2.26%; uniform half increments raise it by
1.29%, 1.80%, and 2.81%. The seed-0 loss guard raises it by 0.97%. These are
complete 10,000-example test evaluations using the native half-squared-error
reduction. They preclude describing these MLP results as a generalization
improvement. The terminal evaluation table is retained separately from the
training-window summaries so that a terminal value is not assigned to an
earlier window or treated as a trajectory average.

## MLP replication through update 1152

Changes below are percentages relative to unchanged Adam on the same seed;
negative values mean lower loss or smaller positive loss variation (PV).
The tail is the mean of the last 32 loss states. These are descriptive paired
results, not confidence intervals. Seed 0 selected the candidates; seeds 1/2
used the same rules without retuning.

| Seed | Rule | Mean loss | Tail loss | Maximum loss | PV |
|---:|---|---:|---:|---:|---:|
| 0 | Distributed neuron cap, q50 | -22.85% | +4.36% | -39.04% | -85.49% |
| 0 | Output budget + slow recovery | -19.62% | -8.17% | -26.21% | -43.68% |
| 0 | Uniform half increments | -16.18% | -12.97% | +49.15% | -56.86% |
| 1 | Distributed neuron cap, q50 | -19.43% | -28.52% | -19.96% | -73.92% |
| 1 | Output budget + slow recovery | -16.30% | -32.49% | -19.96% | -36.86% |
| 1 | Uniform half increments | -15.33% | -28.33% | +21.98% | -57.55% |
| 2 | Distributed neuron cap, q50 | -32.08% | -64.49% | -51.62% | -82.56% |
| 2 | Output budget + slow recovery | -24.10% | -54.30% | -50.55% | -80.35% |
| 2 | Uniform half increments | -17.08% | -62.32% | +18.21% | -60.29% |

Both representation-informed candidates improve the mean on all three MLP
initializations. The neuron cap has the stronger average improvement, but its
seed-0 tail is slightly worse. Slow recovery improves both mean and tail on
all three. Uniform halving lowers average loss but postpones a larger maximum
in every seed. MLP replication alone does not establish architecture transfer;
the larger-model comparisons are separate tests.

## The three requested selective criteria

The original entry-only screen gives the following changes in mean training
loss relative to each worker's unchanged Adam control. Windows are 448-800
for MLP, 264-576 for ResNet, and 552-928 for Transformer.

| Model | Coordinate discard, q90 | Coordinate cap, q90 | Local displacement cap, q75 |
|---|---:|---:|---:|
| MLP | +1.45% | -2.19% | -8.26% |
| ResNet | +124.20% | +54.59% | -6.29% |
| Transformer | -6.22% | -9.13% | -8.53% |

The local representation criterion is the only one of these three with a
lower mean on every setup. It is not the best entry rule on every model:
coordinate capping has a slightly lower Transformer mean. Nor does it improve
every metric: its MLP tail is 8.50% higher and its ResNet maximum is 21.44%
higher. The distributed q50 follow-up therefore tests a broader candidate;
it must not be described as an isolated change in depth or as the same q75 arm.

## Distributed local caps across architectures

These are effects of the common distributed q50 rule against each setup's own
unchanged Adam control. All percentages are descriptive paired differences;
negative means smaller. The MLP row uses its longer confirmation window.

| Setup | Loss states | Mean loss | Tail loss | Maximum loss | PV |
|---|---|---:|---:|---:|---:|
| MLP, seed 0 | 448-1152 | -22.85% | +4.36% | -39.04% | -85.49% |
| ResNet | 264-576 | -16.22% | -7.48% | -13.82% | -46.78% |
| Transformer | 552-928 | -26.00% | -73.36% | -79.63% | -98.81% |

The cap improves the mean and positive variation in every listed setup, with
the MLP replication reported above. This is the most consistent candidate in
this exploration. It is not a claim of uniformly lower loss at every step,
optimal quantiles, or superiority to a fully tuned LR schedule. In particular,
the Transformer half-increment comparator ran on a different GPU architecture;
it is not a matched head-to-head comparison with this distributed arm.

The cap's full-model update-norm ratio averages 0.879 for ResNet and 0.99883
for Transformer. The Transformer target contains 402,000 of 13,828,478
parameters (2.91%); its embeddings and decoder remain uncapped. A small change
in a global parameter norm can therefore accompany a substantial training
effect when the modified parameters occupy a particular functional location.
This does not identify a unique causal layer or a singular direction.

For the Transformer, output reversals rise from 36.6% to 97.9% of the observed
updates. Raw-gradient cosine is negative on all capped updates, while none of
the defined actual-parameter-update cosines is negative. At retained state
928, actual-update cosine is +0.797 and output-change cosine is -0.462.
Embedding and decoder increments contribute 99.79% of that update's squared
parameter norm. The targeted affine increments have positive cosine at this
particular state too (+0.259): the block accounting is a parameter-space
decomposition, **not a causal attribution of output reversal**.
[The retained-anchor audit](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/actual_update_geometry.json), generated by
`32_actual_update_geometry.py`, checks additive energies and inner products
at fork+32 and the terminal state for both control and cap on each model.

Held-out performance prevents a stronger recommendation. Terminal test loss
rises by 2.87% / 8.18% / 100.17% for the three rows above; the other MLP seeds
also have higher test loss. End states may occupy different phases of an
oscillation, so these are terminal comparisons, not held-out trajectories or
estimates of a persistent generalization gap. The next validation should use
additional forks/seeds and repeated held-out measurements, including comparisons
at matched training progress. Ordinary stochastic minibatches remain untested.

## ResNet: smoothing and learning are different outcomes

Every row uses the same unchanged Adam trajectory over states 264-576.
The percentages have the same definitions as the MLP table.

| Rule | Mean loss | Tail loss | Maximum loss | PV |
|---|---:|---:|---:|---:|
| Entry coordinate discard, q90 | +124.20% | +1996.71% | +352.43% | +124.15% |
| Entry coordinate cap, q90 | +54.59% | +332.16% | +152.64% | +67.91% |
| Entry displacement cap, q75 | -6.29% | -22.76% | +21.44% | -17.15% |
| Distributed displacement cap, q75 | -17.42% | -8.31% | -13.82% | -17.71% |
| Distributed displacement cap, q50 | -16.22% | -7.48% | -13.82% | -46.78% |
| Uniform half increments | +17.05% | +406.57% | +64.02% | +7.53% |
| Block reversal + norm surge | +255.72% | +4335.72% | +433.26% | +191.40% |
| Output budget + slow recovery | +4.05% | +672.35% | -13.82% | -85.12% |
| Probe-loss backtracking | +36.46% | +1486.43% | -13.82% | -97.17% |

The local representation criterion transfers better than the two coordinate
criteria in this test. Distributed q50 improves both average and tail training
loss while reducing positive variation. It does not remove reversals: the
fraction of output-change cosines below zero rises from 36.1% to 66.5%.
Mean output displacement RMS rises from 0.567 to 0.723, while its maximum falls
from 2.785 to 1.592. Thus the successful branch permits more sustained motion
and more reversals while reducing its largest burst; damping does not mean
making every representation change smaller.

The loss guard makes its probe loss monotone, but full-objective positive
variation remains 0.03681 and outside-probe variation 0.03839. Its much worse
training mean and tail show why satisfying an acceptance rule is not enough
to establish faster learning. The enormous tail percentages also reflect the
baseline's small tail mean, 0.00824; the guard's tail mean is 0.13074.

Terminal test behavior is a separate tradeoff. At state 576, distributed q50
has test loss 1.9952 versus 1.8443 for unchanged Adam, whereas slow recovery
and the loss guard have 1.4624 and 1.2374. These are complete 10,000-example
cross-entropy evaluations at one terminal state. The latter two rules slow
training yet improve that terminal test measurement; the local cap improves
training yet worsens it. None is uniformly best across these objectives.

## Transformer: local caps, slow recovery and loss acceptance

These three policies share the same RTX 5090 control and states 552-928.
The loss guard reuses the authenticated Run 274 control; it is not an additional
independent baseline replicate.

| Rule | Mean loss | Tail loss | Maximum loss | PV | Terminal test loss |
|---|---:|---:|---:|---:|---:|
| Distributed displacement cap, q50 | -26.00% | -73.36% | -79.63% | -98.81% | 19.9727 |
| Output budget + slow recovery | -20.47% | -68.04% | -79.63% | -99.99% | 15.9254 |
| Probe-loss backtracking | -22.51% | -69.08% | -79.63% | -99.83% | 16.3544 |
| Unchanged Adam | reference | reference | reference | reference | 9.9778 |

The shared initial loss is the maximum of all three interventions, which
explains their identical maximum reductions. The local cap has the lowest
training mean. Slow recovery nearly eliminates positive full-objective loss
variation (0.000844 versus 8.07845 for the control), but its ResNet mean is
worse than unchanged Adam: this smoothness result does not transfer uniformly
to faster learning. Probe-loss backtracking makes probe loss monotone while
full-objective positive variation remains 0.01406. It applies one zero update;
slow recovery applies none. Every rule has worse terminal test loss here.

All Transformer terminal evaluations retain the complete cached validation
and test splits, with token weighting and exact coverage recorded in
`transfer_heldout_evaluations.csv`. These comparisons are at one terminal
state and do not establish the ordering of held-out trajectories.

## Transformer embedding rates

Both standalone embedding multipliers use the same RTX 4090 control over
552-928. The remaining parameter groups retain native LR 0.003.

| Embedding LR | Mean training loss | Tail loss | Maximum loss | PV | Terminal test loss |
|---|---:|---:|---:|---:|---:|
| 0.006 (2x) | -0.20% | -14.77% | -22.13% | -1.53% | +3.53% |
| 0.012 (4x) | +11.52% | -16.34% | -9.72% | -0.05% | -1.82% |

The 2x rate is nearly neutral in average training loss. The 4x rate lowers the
tail but raises the full-window mean, with almost unchanged positive variation.
Its slightly better terminal test loss is another tradeoff, not evidence of
uniformly faster learning. A comparatively steady embedding trace therefore
does not, by itself, identify a beneficial larger LR. These conclusions concern
this fork and window; they do not establish an optimal embedding rate.

The embedding itself remains mostly forward-moving: its negative-change-cosine
fraction is 5.31% in the control, 6.63% at 2x, and 6.90% at 4x. Mean embedding
displacement RMS increases by 80.04% and 231.61%, respectively. Thus even the
continued scarcity of embedding reversals does not predict the training-loss
tradeoff. These RMS values use the raw lookup capture defined above.

On the RTX 5090 worker, adding a 4x embedding LR to the distributed q50 cap
still lowers mean loss by 9.38% relative to unchanged Adam. However, relative
to its **same-worker cap-only arm**, it raises mean loss by 22.47%, tail loss
by 25.43%, maximum loss by 89.77%, and positive variation by 906.07%.
The last percentage has a small cap-only denominator. Terminal test loss is
17.8930: better than cap-only's 19.9727 but worse than the baseline's 9.9778.
Thus the embedding boost weakens the cap's training benefit in this comparison.
The standalone 4x result above uses a different GPU architecture and must not
be subtracted from this result as a matched factorial contrast.

## What this tests about the tracing idea

The earlier SVD result is descriptive: a small set of temporal patterns can
explain much of a representation's motion in a selected window. This screen
uses that result to motivate interventions on functional movement and its
location. It does **not** project updates onto the top singular vectors or fit
an SVD controller. The one-direction MLP test uses only the preceding actual
affine change; the neuron tests select output units of an affine module, not
singular directions of the complete network.

Three distinctions are necessary when interpreting a successful branch:

- Lower loss with persistent negative representation cosine means that
  reversals can be compatible with progress; it does not show that their
  existence is beneficial or necessary.
- A lower average with a worse tail or larger maximum is a tradeoff. A smooth
  early trajectory can conceal a delayed excursion, as the MLP half-increment
  controls demonstrate.
- A local affine budget and a complete-network probe budget constrain different
  quantities. Neither, by itself, guarantees enough movement toward the labels.
  The loss-aware comparator adds that missing acceptance condition on its probe.

Raw-gradient, actual-parameter-update, and representation-change cosines need
not share a sign. Adam transforms gradients, selective gates transform its
proposal, and the network maps parameter movement to representation movement.
Even a fixed linear map can change the sign of an inner product when it
amplifies different directions unequally. A positive whole-parameter update
cosine therefore does not establish forward-only motion in output space.

The output metric also changes meaning with the objective. For the MLP, squared
Euclidean output displacement is exactly the positive quadratic term in its
squared-loss change (observation 009). RMS over classification log-probabilities
weights every class/vocabulary coordinate uniformly and has no corresponding
squared-loss identity. Transferring a displacement rule therefore does not
transfer an identical constraint on the loss geometry. We do not extend the
MLP overshoot counts to ResNet or Transformer without the required measurements.

These are transfer tests of complete training setups, not an isolated
architecture ablation: data size, loss, normalization, base LR and selected
windows differ between cases.

These experiments use deterministic full-objective updates. They do not yet
establish behavior under ordinary stochastic minibatches, a different optimizer,
a different fork, or a broader range of architectures. Thresholds are local to
each eight-update calibration; sharing a quantile is not sharing a numerical
threshold. Full, probe, outside-probe and held-out losses remain distinct.

## Figure captions and coverage

**Figure 06: selective entry updates.** Three pages, one per architecture.
Columns compare coordinate discard, coordinate cap, and q75 local displacement
cap with unchanged Adam. Rows show full training loss, eta-kappa-hat, raw
gradient cosine/norm, early representation cosine/RMS, output cosine/RMS,
actual/native update-norm ratio, and mean targeted fraction modified. Early
means the first MLP tanh, ResNet conv 1 output, or first Transformer attention
output. Magnitude and loss axes are independent across columns. The final row
averages coordinate or neuron fractions over targeted modules; it is not a
whole-network fraction of parameters.

**Figure 07: distributed caps and learning controls.** The first page transfers
the same q50 local rule across architectures, displaying MLP 448-800,
ResNet 264-576, and Transformer 552-928. The MLP extension compares
distributed and entry caps, uniform half increments, and the retrospective
per-layer norm-matched control. The shaded region begins at 801. Page 2 maps
consecutive-change cosine and RMS across the captured stages for unchanged Adam
and the common q50 cap. Both RMS maps divide by the same baseline stage maximum
in the displayed window, so treatment amplitudes are not normalized away.
These are scalar stage diagnostics, not a new SVD analysis or a capture of every
individual module. Pages 4-6 show MLP initialization replication, slow-recovery
transfer, and probe-loss backtracking. Every column
retains an unchanged control; comparisons using another run require exact
untreated-trajectory identity.

**Figure 08: adaptive rules, negative controls, and embedding LR.** Retain the
failed diagnostic-triggered, directional, and actual-movement-budget rules,
then show slow recovery, ResNet block/uniform/distributed comparisons, and the
larger Transformer embedding rates. Log loss
axes expose large transients without hiding the baseline. Directional panels
use the first affine projection as their early representation. Embedding
panels additionally show embedding and attention motion and the actual
embedding LR. Their update-norm ratio is relative to their own native proposal,
which already incorporates the changed embedding LR.

All three PDFs are generated by `22_figures_transfer.py` from independently
verified `transfer_comparison*.json`. `transfer_figure_audit.json` pins source
hashes, exact windows, panels, checked plotted values, and preserved earlier
figures. `24_summarize_transfer.py` produces comparable full and forward-window
metrics in `transfer_summary.json` and `.csv`. Final render review and compute
closeout are recorded alongside those artifacts.

## Verification and retention

Each run pins scientific sources, cache identities, initialization and config.
Archive and per-file sizes/SHA-256 values are checked on retrieval. Independent
NumPy audits check raw-gradient reductions, thresholds, gate recurrences,
native/actual increments at three retained audit states, optimizer clocks,
and same-state branch identity. Representation triplets are replayed on their
original hardware and their RMS/cosines recomputed independently. Functional
budget rules also replay before/native/actual parameter states and check the
actual complete-network bounds. The loss-aware comparator independently replays
before/accepted losses and verifies every dyadic trial and acceptance decision.

Full fork/end/recovery states, gradient/parameter audits, scalar histories,
calibration and manifests are retained outside Git. Dense per-update SVD
histories are not collected in this screen. Native save/reload qualification
does not imply arbitrary policy resume: controller/probe history must be
reconstructed from the shared fork and saved history for exact continuation.
Infrastructure retries preserve scientific inputs; changed policies use new
run IDs. Mechanics-only work is excluded from efficacy summaries.

The completed study contains 18 scientific workers, 70 arm records including
two authenticated reused controls, and 68 newly executed continuations with
35,452 updates. Independent dense checks cover 210 arm-state audit records
(204 excluding reused-control records). No mechanics-only attempt is counted
as efficacy evidence. All scientific inputs and retrieved evidence passed
identity checks. Local cap scores are supported by source/mechanics checks and
retained placement audits; we do not claim independent replay of every local
convolution/attention score at every update.

The user approved a total Runpod ceiling of USD 20. The final conservative
elapsed-lease compute/storage estimate is **USD 18.7460**. Provider history
reports USD 16.5926 as of 02:29 UTC on 8 November, but can lag recent usage;
both billing figures remain provisional. The USD 19.60 automatic stop was not
needed. All five task pods are absent from the account, including stopped
pods, and no task cloud storage remains. Two unrelated account resources
were preserved. Download verification and replay preceded deletion.

Retained local raw evidence occupies 41.316 GB of unique files (43.046 GB
logical bytes including authenticated hardlinks). Some original artifact paths
are NTFS junctions to task storage on the R: partition; see `artifact_storage.json`.
Owned Spark retains its original evidence and all task workers have exited.
The [compute record](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_compute_closeout.json),
[cloud closeout](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_cloud_closeout.json), and
[Spark closeout](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_spark_closeout.json) record these checks.

Continuation timing compares each policy with its instrumented unchanged
control. Both carry the diagnostic captures and audit retention, so these
ratios do not measure overhead against uninstrumented Adam. Prefix reconstruction,
mechanics, independent replay, transfers, and provisioning are excluded from
arm-level timing and included separately in resource closeout. Concurrent
Spark timings are descriptive, not isolated throughput benchmarks.
Arm timers include diagnostic work, audit checkpoint writes, and terminal
held-out evaluation, so seconds per update is an amortized experiment cost.

For the distributed q50 comparison, observed seconds per update are
0.01376 / 0.05797 (MLP control / cap), 3.53170 / 3.83282 (ResNet), and
7.80567 / 7.95608 (Transformer). The MLP measurements share Spark with other
experiments and should not be read as an isolated 4.2x algorithmic overhead.
The full-objective passes dominate the larger models' work; their probes are
small fractions of the objective. Slow-recovery output budgets use about two
candidate trials per update (maximum three observed). The loss guard averages
2.62 / 4.35 / 2.89 trials for MLP / ResNet / Transformer, with a maximum of
13 including the zero-update fallback. These counts and timings are retained
per arm in the compute record; production overhead needs a separate benchmark.

## Reproduction and review

From the repository root, run `17_reduce_transfer.py` in this analysis for the
primary comparison and with `--run runs/NNN-YYYY-MM-DD-slug` for each follow-up
run 272 through 285. Run
`24_summarize_transfer.py`, `27_reduce_mlp_loss_terms.py`, and
`32_actual_update_geometry.py` to reproduce the aggregate and explanatory
reductions; then run `22_figures_transfer.py`. The scripts require the retained
raw artifacts at their recorded paths. Cloud replay scripts 18/28 and the
Run 278 audit were completed on the original hardware before deletion;
their hash-pinned records remain available for local verification.

`transfer_figure_audit.json` checks 493,164 plotted values and source identities.
`transfer_visual_review.json` records inspection of all 14 pages rendered at
110 dpi, including legends, units, independent axes, undefined-cosine gaps and
captions. Earlier Figures 01-05 remain byte-identical to the pre-study commit.
The reusable intervention workflow is in
[INSTABILITY_TRACING.md](../../../../../../research/INSTABILITY_TRACING.md).
