# Claim, mathematics, and evidence audit

**Iteration 2, 8 November 2026.** This audit supports the author's fixed
six-section paper and the C1-C8 contribution components in [PLAN.md](../PLAN.md).
The formulation, curvature/proxy, loss and sampling analysis are central;
representation tracing and interventions extend their use within Section V.
This document adds no training evidence and applies no manuscript changes.

## 1. Which manuscript is active?

The actual include graph in [bare_jrnl.tex](../../draft/bare_jrnl.tex) governs this
plan. [manuscript/README.md](../../README.md) records the 6 November restoration;
[research/MANUSCRIPT.md](../../../research/MANUSCRIPT.md) still leads with an older
5 November state. Later revision records describe available material, not the
currently compiled manuscript. Do not restore a previously rejected revision
wholesale merely because some of its evidence is useful.

| Active part | Source under `manuscript/draft/` | Proposed disposition |
|---|---|---|
| I Introduction | [01-introduction.tex](../../draft/sections/01-introduction.tex) | Preserve the author-edited argument and four contribution bullets; map C1-C8 and closest literature into that argument. |
| II Stability Analysis of SGD | [02-related-works.tex](../../draft/sections/02-related-works.tex) | Retain the foundation; revise positioning. |
| III Constructive and Destructive Regimes | [03-methods.tex](../../draft/sections/03-methods.tex) | Preserve the displacement formulation and five subsections: curvature, estimator, noise, normalized displacement, and loss decrease/spike formation. Clarify each relation in place. |
| IV Empirical Characterization | [04-gd-experiments.tex](../../draft/sections/04-gd-experiments.tex) and [05-sgd-noise-experiments.tex](../../draft/sections/05-sgd-noise-experiments.tex) | Keep convergence/curvature, regime/norm/R, backward lags and batch effects in this order, with their four current main figures. The SGD file is a subsection of IV. |
| V Raw-Gradient Diagnostics under Adam | [06-adam-diagnostics.tex](../../draft/sections/06-adam-diagnostics.tex) | Keep the same raw-gradient diagnostic; develop the requested characterization, representation tracing and intervention sequence within V. |
| VI Conclusion | [05-conclusion.tex](../../draft/sections/05-conclusion.tex) | Summarize the central framework and analytical/empirical components, then what the completed tracing/interventions add. |
| Appendices A/B | [06-loss-appendix.tex](../../draft/sections/06-loss-appendix.tex), [07-gd-diagnostics.tex](../../draft/sections/07-gd-diagnostics.tex) | Reuse exact path identities and measurement definitions; retain counterexamples and coverage. |

The four controlled GD/SGD floats are queued by
[04-controlled-figures.tex](../../draft/sections/04-controlled-figures.tex) from
Section III, although their logical home is IV. There are currently nine main
figure labels and one Adam event table. A source/PDF hash snapshot is preserved
in [provenance.json](../provenance.json); equation numbers should be taken from a
future verified build, not inferred permanently from source order.

The title and existing notation remain the baseline. This audit does not
authorize a new title, `z` notation, paired C/D factorization, restoration of
the removed endpoint table, or a structural redesign. Historical evidence can
be incorporated without restoring its rejected presentation.

## 2. Contribution-to-evidence ledger

Each row starts with what the contribution provides. Scope checks below qualify
these contributions rather than supply a replacement organizing thesis.

| Component / manuscript home | Formulation or supported insight | Direct supporting evidence and scope |
|---|---|---|
| C1 / III: constructive and destructive learning updates | The exact SGD displacement decomposition exposes positive and negative interactions; adjacent cosine characterizes their local directional regime. | [Active formulation](../../draft/sections/03-methods.tex), `eq:displacement` through `eq:magnitude-alignment`. All temporal pairs remain in the identity; the local regime is an accessible diagnostic, not an assumed lag truncation. |
| C2 / III-A/B and IV: curvature connection and gradient-only proxy | The boxed relation connects alignment, gradient growth, directional curvature and noise. The proxy is an exact smooth-GD path average, with a bound on its difference from starting-point curvature. | [Appendix A](../../draft/sections/06-loss-appendix.tex), `eq:path-curvature` and `eq:curvature-error-bound`; [024 spectral comparison](../../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/010-curvature-spectral-overlay.md); [027 independent directional-HVP comparison](../../../analyses/027-2026-10-30-minibatch-noise/observations/001-controlled-minibatch-noise.md). Across 1,188 state-matched HVP pairs, all stochastic batch sizes improve after the measured noise correction; at batch32 median absolute discrepancy falls .7467 -> .0030 over 288 pairs. This is an offline full-gradient reference on the smooth MLP, not an online corrected estimator or a top-eigenvalue estimator. |
| C3 / III-E and IV/V: reversal, amplification and loss rises | The local GD loss relation gives the joint cosine/norm criterion and separates oscillatory descent from a loss-increasing step; functional accounting develops this explanation under Adam. | [III-E and Appendix A](../../draft/sections/03-methods.tex); [024 direct finite-step audit](../../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/007-section-iv-critical-review.md); [062 exact output-loss identity](../../../analyses/062-2026-11-07-mlp-gradient-amplification/observations/001-gradient-amplification.md). The quadratic-model criterion and its observed nonlinear residual are reported together. In the selected MLP window, the output-Jacobian approximation matches 85/87 loss-change signs, while exact output accounting resolves the rises without a parameter-space Taylor approximation. |
| C4 / III-C and IV: sampling/batch effects | Finite-population variance and the expected noise projection explain a sampling contribution; frozen and moving experiments separate measurement effects from changes in the training path. | [027 frozen/moving controls](../../../analyses/027-2026-10-30-minibatch-noise/observations/001-controlled-minibatch-noise.md): 184,320 frozen observations, three initializations and three checkpoints. Maximum relative discrepancies are .50% for variance and .74% for noise projection; maximum absolute same-state cosine approximation error is .00745. The approximate cosine target is outside 21/36 stochastic intervals, so its approximation status remains explicit. [037](../../../analyses/037-2026-11-02-sgd-batch-extension/observations/001-three-case-sgd.md) supplies the three-model moving-SGD extension. |
| C5 / IV: regime emergence and accumulation | GD rates reveal destructive regimes, larger norms, smaller path-retention ratios, structured backward-lag oscillation and continued training progress. | [024](../../../analyses/024-2026-10-17-eos-gd-lr-calibration/README.md), [035 exact backward profiles](../../../analyses/035-2026-11-02-eos-backward-anchors/observations/002-manuscript-view.md), and [056 retained all-lag checks](../../../analyses/056-2026-11-05-gd-paired-displacement/observations/001-paired-displacement.md). These support the existing four-figure IV sequence. One GD initialization per model is distinct from the three initialization blocks in the minibatch study. |
| C6 / V opening: adaptive-trajectory interpretation | Raw gradients describe local loss slope across states visited by Adam; the same diagnostic has a mixed-Hessian path interpretation. | [Current V](../../draft/sections/06-adam-diagnostics.tex) and [existing `eq:adam-gradient-response`](../../draft/sections/06-loss-appendix.tex). Adam determines the traversed direction and the preceding gradient determines the projection. The original diagnostic notation can remain. |
| C7 / V tracing: learning-representation interpretation and measurement capability | Internal investigations connect spikes to output-sensitive feature motion; the tracing algorithm follows functional reversals through fixed-probe representations and temporal modes. | [061 static/step probes](../../../analyses/061-2026-11-07-mlp-adam-spike-internals/observations/001-spike-internals.md), [062 amplification/functional accounting](../../../analyses/062-2026-11-07-mlp-gradient-amplification/observations/001-gradient-amplification.md), and [063 formal algorithm](../../../analyses/063-2026-11-07-architecture-instability-tracing/METHOD.md). At the selected MLP 465->474 excursion, restoring parameter tensor norms leaves 99.75% of excess loss, while hidden-feature motion contributes 98.75% of the rise under the stated output decomposition. These are controlled static probes of a particular trajectory, not unique causal attribution or a direct forgetting metric. |
| C8 / V interventions: making the interpretation actionable | Local representation displacement control improves selected training trajectories; LR, clipping, beta2, freezing, norm and directional controls show what each action changes. | [058 comparator consolidation](../../../analyses/058-2026-11-06-intervention-consolidation/README.md), [059 beta2](../../../analyses/059-2026-11-07-mlp-beta2-reduction/README.md), [064 freeze](../../../analyses/064-2026-11-07-mlp-freeze-first-linear/README.md), and [065 cross-architecture/MLP replication](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md). [065 functional-loss controls](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/009-what-the-negative-controls-show.md) explain why motion amplitude, direction and residual correction are evaluated together. Each intervention retains its own unchanged baseline and held-out outcomes. |

## 3. Formal meaning and timing to retain

Use a single state convention in the plan: state `s` means `s` completed updates,
`h_s = grad L(theta_s)` is the raw fixed-objective gradient, and
`u_s = theta_(s+1) - theta_s` is the actual applied increment. The letter `h` here
only avoids conflating existing manuscript and run indices; it is not a proposed
new manuscript symbol. In the manuscript's common indexing, `h_s = g_(s+1)`.

For nonzero consecutive gradients define the dimensionless scalar

\[
q_s=1-\frac{h_{s-1}^{\top}h_s}{\|h_{s-1}\|^2}
   =1-\rho_s\frac{\|h_s\|}{\|h_{s-1}\|}.
\]

Under a fixed sufficiently smooth objective (or an absolutely continuous
gradient along the path),

\[
q_s=-\frac{h_{s-1}^{\top}\int_0^1
H(\theta_{s-1}+a u_{s-1})u_{s-1}\,da}{\|h_{s-1}\|^2}.
\]

For `u_(s-1) = -eta_(s-1) d_(s-1)`, the traversal direction is `d_(s-1)`;
the projection direction is `h_(s-1)`. For GD these coincide, giving the
step-averaged directional curvature. For Adam they generally do not: this is
a mixed Hessian projection. The active appendix already contains this extension.

Use the **previous applied step's** LR when writing `q = eta * kappa_hat`.
Use the actual increment when clipping, weight decay, rounding, parameter groups,
or post-Adam caps alter it. A dimensionless `q` avoids division by a zero LR.
For sampled gradients, write `h_s = grad L(theta_s) + xi_s`; their endpoint
difference includes `xi_s - xi_(s-1)`. Fixed-objective curvature and stochastic
gradient response are not the same observable. Classical Hessian integration
also needs qualification across nonsmooth activation boundaries where gradients
can jump. Do not call it exact for every architecture without those assumptions.

| Observable | Earliest availability | What it describes |
|---|---|---|
| `L_s`, `h_s`, `rho_s`, `q_s` | After the forward/backward at state `s`, before applying `u_s` | Response to the already completed incoming step `u_(s-1)` |
| Proposed Adam increment and its current-input local affine score | After building the optimizer proposal, before committing it | A candidate local parameter effect, with the layer input fixed |
| `A_(s+1)-A_s` and its adjacent cosine | After applying `u_s` and capturing state `s+1` | Outgoing representation motion under that completed update |
| Loss at a future peak, recovery boundary, whole-window SVD, window-maximum normalization | After the corresponding future states have been observed | Retrospective event/window description |

Thus an observable may be available before the **next** update while describing
the **previous** one. Calling this spike prediction requires an independently
specified future target, lead time, all eligible non-events, and held-out evaluation.

Two other boundaries are mandatory:

1. For nonzero gradients, `q > 1` and `rho < 0` are algebraically equivalent.
   Showing both is not two independent validations. For plain GD, with
   `s_step = -eta*h_previous` and `y = h_current-h_previous`,
   `kappa_hat = (s_step' y)/(s_step' s_step)`, reciprocal BB1 when defined.
   This is a direct algebraic relation requiring credit to [L11](LITERATURE.md).
2. The arbitrary-step loss depends on `h' u` and the path curvature along `u`,
   not raw-gradient cosine alone. A smooth quadratic counterexample: with
   `H=diag(100,1)`, `theta=(.01,1)`, `eta=1`, `d=(.05,.5)`, the projected scalar
   is 2.75 but loss decreases .505 -> .205. With `H=I`, `theta=(1,0)` and
   `d=(0,2)`, it is zero while loss increases .5 -> 2.5. These are analytic
   counterexamples to a generic arbitrary-update threshold, not Adam experiments.

For actual increments, the exact identity
`||sum u||^2 = sum ||u||^2 + 2 sum_(i<j) <u_i,u_j>` always separates magnitude
and interaction. Restricting the second sum to adjacent pairs is a truncation,
not equality. Raw-gradient interactions coincide with update interactions only
under the corresponding update/LR assumptions. Representation reversal is a
different vector stream again. None measures semantic forgetting by definition.

## 4. Local scope checks for the contribution claims

Percentages below are taken from the cited local observations, not recomputed
for this planning pass. Different rows have different windows and denominators.
They must not be combined into an unmatched method ranking.

| Claim | Verdict | Evidence and required wording |
|---|---|---|
| Adjacent destruction means destructive cumulative learning | **Contradicted as a general inference.** | [Analysis 056](../../../analyses/056-2026-11-05-gd-paired-displacement/observations/001-paired-displacement.md): all nine final all-pair interactions positive; seven lag-one sums negative. Describe vector stream and horizon. |
| The local GD loss criterion reliably predicts every nonlinear loss change | **The local result needs its finite-step scope.** | [Analysis 024 audit](../../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/007-section-iv-critical-review.md): 26,991 transitions; low-rate MLP/Transformer have zero sign disagreements, while the highest-rate Transformer has 1,325/2,999 (44.2%), or 1,239/2,837 (43.7%) after the declared sensitivity filter. This bounds a retrospective approximation; it does not negate the exact identities, 027's independent noise/HVP support, or observed longer-window regime behavior. |
| Same raw-gradient computation is meaningful under Adam | **Supported with narrower interpretation.** | Active Appendix A and [Analysis 057](../../../analyses/057-2026-11-05-adam-response-memory/observations/001-response-memory.md). Realized-path response, not inherited GD stability or parameter-update cancellation. |
| Spike rises arise because Adam steps uphill at first order | **Contradicted for the measured event population.** | Analysis 057: all 794 rise updates from 129 recovered events have `g' actual_increment < 0`. Finite-step loss can still rise. Two censored >10% events remain in incidence. |
| Raw-gradient cosine robustly predicts Adam spike severity | **Not supported by retained transfer tests.** | Analysis 057: cosine added to the log-norm regression worsens held-out-rate MSE in all seven folds; other specifications are mixed. Rate holdout is not independent-seed replication. Preserve this when simplifying the main figures. |
| Gradient and output oscillation are correlated | **Supported for specified windows.** | [063 consolidated observation](../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/004-consolidated-instability-tracing.md): .776/.705/.879 in original MLP/ResNet/Transformer windows. MLP's shorter 450-500 fit has .6420, not >.7. Temporal observations are dependent. |
| One dominant mode explains all three architectures | **Too strong.** | [063 event report record](../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/005-technical-report-event-window.md): output top-one energy 53.02%/24.03%/41.65%; top-ten 99.48%/87.97%/90.83%; rank95 5/16/15. Fixed-window patterns do not prove stationary local axes. |
| Architectural tracing identifies the initiating layer | **Plausible but unproven.** | [063 method](../../../analyses/063-2026-11-07-architecture-instability-tracing/METHOD.md): descriptive localization; joint parameter motion, residual branches, subset probes and stochastic context constrain causal interpretation. |
| First-layer freezing influences downstream oscillation | **Supported within the intervention.** | [064 observation](../../../analyses/064-2026-11-07-mlp-freeze-first-linear/observations/001-freeze-first-linear.md): output reversals 51/51 -> 4/51 over 450-500, RMS maximum -87.38%; mean loss 448-800 +22.27%. This block holds 93.6% of parameters. |
| Halving LR generally harms progress | **Contradicted as a blanket claim.** | [058 half-LR observation](../../../analyses/058-2026-11-06-intervention-consolidation/observations/001-persistent-half-lr.md): mean loss -25.04%/+27.20%/-9.66% for MLP/ResNet/Transformer. No measured basin-escape or fractality mechanism. |
| Clipping is uniformly beneficial or uniformly harmful | **Effects depend on setting and activity.** | [058 clipping](../../../analyses/058-2026-11-06-intervention-consolidation/observations/005-clipping.md): MLP caps1/.5 mean loss +48.01%/+2.08%; ResNet cap1 mean -28.3%, with worse tail/endpoint. [043 median clipping](../../../analyses/043-2026-11-03-spark-fixed-step-interventions/observations/007-clipping-median-response.md) already includes Transformer at fork1024 for64 updates: clipping activates3/64 times, with mean/final loss -0.058%/-0.178%. The missing comparison is Transformer clipping at the selected tracing/cap fork and window, not all Transformer clipping evidence. |
| Lower beta2 fails in our MLP fork | **Supported for that transition.** | [059](../../../analyses/059-2026-11-07-mlp-beta2-reduction/observations/001-beta2-reduction.md), Run232: .999 -> .95/.9/.5 at state512 gives mean loss +16.78%/+18.16%/+127.02%. Moments/clocks retained; native bias correction changes abruptly. Not a general refutation of the cited paper. |
| Distributed caps improve selected training trajectories | **Supported, bounded.** | [065 observation008](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md): mean -22.85% MLP/-16.22% ResNet/-26.00% Transformer; two MLP replicas -19.43%/-32.08%. Selected full-objective forks, one larger-model seed. |
| Representation-sensitive allocation adds information beyond step norm | **Supported by a specific control.** | [065 observation009](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/009-what-the-negative-controls-show.md): MLP layer-norm-matched control mean -3.43% versus cap -22.85%. Reference-arm schedule is retrospective, not an online competing policy. Directional controls fail. |
| Fewer reversals imply a better intervention | **Contradicted as a general rule.** | Same 065 records: Transformer output reversals rise 36.6% -> 97.9% with lower mean loss. Capped raw gradients reverse throughout while defined actual-update cosines remain nonnegative. |
| Higher LR under a cap speeds learning | **Not supported.** | [065 observation013](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/013-distributed-cap-quantile-lr.md): 54 prespecified continuations, three seeds. Every quantile's across-seed mean is worse at 2x/4x than its own 1x. q50 1x lowest observed mean; not an established optimum. |
| Training improvements imply better held-out performance | **Contradicted for reported terminal comparisons.** | 065 observation008: terminal test loss +2.87% MLP/+8.18% ResNet/+100.17% Transformer. Observation013 adds repeated complete MLP tests; q50 1x final-four test mean +3.37%. Distinguish endpoint from trajectory claims. |

## 5. Functional loss accounting connects C3, C7 and C8

For the MLP's half-squared-error objective, the actual output change obeys

\[
L_{s+1}-L_s=\frac{\langle F_s-Y,\Delta F_s\rangle}{N}
             +\frac{\|\Delta F_s\|_F^2}{2N}.
\]

The first term measures residual correction along the realized functional
change; the second is its nonnegative squared-movement contribution.
[Analysis062](../../../analyses/062-2026-11-07-mlp-gradient-amplification/observations/001-gradient-amplification.md)
measures both terms directly. On update473->474 they are -.0926296 and
+.0985978, giving a +.0059682 loss rise. This supplies a concrete link from
representation motion to loss, beyond interpreting a cosine alone.

[Analysis065's retained accounting](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/009-what-the-negative-controls-show.md)
extends the same identity to controls and interventions. Across outgoing
updates448-1152, all377/371/379 meaningful baseline loss rises in seeds0/1/2
have a negative residual-correction term exceeded by the positive movement
term. The threshold is1e-8; these counts include the extra terminal transition
to1153, whereas the ordinary loss means/PV stop at state1152. Native FP32 loss
rounding is checked separately. The functional linear term uses the actual
nonlinear output change and need not equal the parameter derivative `g' u`.

These results develop the author's explanation within Section V: controlling
movement can alter a finite-step excursion, while the accompanying direction
and residual matter. They are exact accounting for this squared-error loss;
cross-entropy cases need their own loss-specific relation. No inference of
semantic forgetting, unique initiating cause, or universal Adam threshold is
needed for this contribution.

## 6. The cap's actual mathematical object

With the current input held fixed, a layer's native parameter proposal gives
`Delta Z_local = H Delta W' + 1 Delta b'`. For unit/filter `j`, take the RMS
over the declared probe coordinates, `s_j`. For a positive threshold `tau`,
scale the proposed weight-row increment and matching bias increment by
`min(1, tau/s_j)`, with factor one
when the score is zero. A calibrated zero threshold disables that layer's cap
in the frozen implementation. The implemented
threshold is calibrated from untreated pre-fork updates; Adam moments and clocks
continue normally. The input-second-moment form `s_j^2 = u_j' G u_j` explains
its geometry; see [065 observation010](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/010-local-displacement-cap-geometry.md).

This bounds the direct local affine response in exact arithmetic for a positive
threshold; final FP32 storage requires the verified rounding tolerance. It does not
bound the propagated change from preceding layers, the cross term from changing
both input and weights, uncapped outputs, or total loss. The SVD in tracing is
not used by this controller. Both the closest prior method and this limited
guarantee must appear in any novelty or intervention explanation.
