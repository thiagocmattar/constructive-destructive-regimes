# Claim, mathematics, and evidence audit

**8 November 2026.** This is a planning audit of retained sources. It adds no
training evidence and does not apply proposed wording to the manuscript.

## 1. Which manuscript is active?

The actual include graph in [bare_jrnl.tex](../../draft/bare_jrnl.tex) governs this
plan. [manuscript/README.md](../../README.md) records the 6 November restoration;
[research/MANUSCRIPT.md](../../../research/MANUSCRIPT.md) still leads with an older
5 November state. Later revision records describe available material, not the
currently compiled manuscript. Do not restore a previously rejected revision
wholesale merely because some of its evidence is useful.

| Active part | Source under `manuscript/draft/` | Proposed disposition |
|---|---|---|
| I Introduction | [01-introduction.tex](../../draft/sections/01-introduction.tex) | Align claims with accepted C1-C3 and closest literature. |
| II Stability Analysis of SGD | [02-related-works.tex](../../draft/sections/02-related-works.tex) | Retain the foundation; revise positioning. |
| III Constructive and Destructive Regimes | [03-methods.tex](../../draft/sections/03-methods.tex) | Audit horizons, stream semantics, approximation limits, and arbitrary-update identity. |
| IV Empirical Characterization | [04-gd-experiments.tex](../../draft/sections/04-gd-experiments.tex) and [05-sgd-noise-experiments.tex](../../draft/sections/05-sgd-noise-experiments.tex) | The SGD file is a subsection of IV. Reuse existing audits; compress around theory tests. |
| V Raw-Gradient Diagnostics under Adam | [06-adam-diagnostics.tex](../../draft/sections/06-adam-diagnostics.tex) | Replace the future-intervention agenda with the approved tracing/intervention structure. |
| VI Conclusion | [05-conclusion.tex](../../draft/sections/05-conclusion.tex) | Revise jointly with title, abstract and introduction at editing time. |
| Appendices A/B | [06-loss-appendix.tex](../../draft/sections/06-loss-appendix.tex), [07-gd-diagnostics.tex](../../draft/sections/07-gd-diagnostics.tex) | Reuse exact path identities and measurement definitions; retain counterexamples and coverage. |

The four controlled GD/SGD floats are queued by
[04-controlled-figures.tex](../../draft/sections/04-controlled-figures.tex) from
Section III, although their logical home is IV. There are currently nine main
figure labels and one Adam event table. A source/PDF hash snapshot is preserved
in [provenance.json](../provenance.json); equation numbers should be taken from a
future verified build, not inferred permanently from source order.

## 2. Formal meaning and timing to retain

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

## 3. Claim ledger

Percentages below are taken from the cited local observations, not recomputed
for this planning pass. Different rows have different windows and denominators.
They must not be combined into an unmatched method ranking.

| Claim | Verdict | Evidence and required wording |
|---|---|---|
| Adjacent destruction means destructive cumulative learning | **Contradicted as a general inference.** | [Analysis 056](../../../analyses/056-2026-11-05-gd-paired-displacement/observations/001-paired-displacement.md): all nine final all-pair interactions positive; seven lag-one sums negative. Describe vector stream and horizon. |
| The local GD loss criterion reliably predicts loss changes | **Not established.** | [Analysis 024 audit](../../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/007-section-iv-critical-review.md): 26,991 transitions; highest-rate Transformer has 1,325/2,999 opposite signs, 43.7% even after the declared sensitivity filter. This is a retrospective approximation check. |
| Same raw-gradient computation is meaningful under Adam | **Supported with narrower interpretation.** | Active Appendix A and [Analysis 057](../../../analyses/057-2026-11-05-adam-response-memory/observations/001-response-memory.md). Realized-path response, not inherited GD stability or parameter-update cancellation. |
| Spike rises arise because Adam steps uphill at first order | **Contradicted for the measured event population.** | Analysis 057: all 794 rise updates from 129 recovered events have `g' actual_increment < 0`. Finite-step loss can still rise. Two censored >10% events remain in incidence. |
| Raw-gradient cosine robustly predicts Adam spike severity | **Not supported by retained transfer tests.** | Analysis 057: cosine added to the log-norm regression worsens held-out-rate MSE in all seven folds; other specifications are mixed. Rate holdout is not independent-seed replication. Preserve this when simplifying the main figures. |
| Gradient and output oscillation are correlated | **Supported for specified windows.** | [063 consolidated observation](../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/004-consolidated-instability-tracing.md): .776/.705/.879 in original MLP/ResNet/Transformer windows. MLP's shorter 450-500 fit has .6420, not >.7. Temporal observations are dependent. |
| One dominant mode explains all three architectures | **Too strong.** | [063 event report record](../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/005-technical-report-event-window.md): output top-one energy 53.02%/24.03%/41.65%; top-ten 99.48%/87.97%/90.83%; rank95 5/16/15. Fixed-window patterns do not prove stationary local axes. |
| Architectural tracing identifies the initiating layer | **Plausible but unproven.** | [063 method](../../../analyses/063-2026-11-07-architecture-instability-tracing/METHOD.md): descriptive localization; joint parameter motion, residual branches, subset probes and stochastic context constrain causal interpretation. |
| First-layer freezing influences downstream oscillation | **Supported within the intervention.** | [064 observation](../../../analyses/064-2026-11-07-mlp-freeze-first-linear/observations/001-freeze-first-linear.md): output reversals 51/51 -> 4/51 over 450-500, RMS maximum -87.38%; mean loss 448-800 +22.27%. This block holds 93.6% of parameters. |
| Halving LR generally harms progress | **Contradicted as a blanket claim.** | [058 half-LR observation](../../../analyses/058-2026-11-06-intervention-consolidation/observations/001-persistent-half-lr.md): mean loss -25.04%/+27.20%/-9.66% for MLP/ResNet/Transformer. No measured basin-escape or fractality mechanism. |
| Clipping is uniformly beneficial or uniformly harmful | **Neither supported.** | [058 clipping](../../../analyses/058-2026-11-06-intervention-consolidation/observations/005-clipping.md): MLP caps1/.5 mean loss +48.01%/+2.08%; ResNet cap1 mean -28.3%, with worse tail/endpoint. Transformer comparison missing in this cohort. |
| Lower beta2 fails in our MLP fork | **Supported for that transition.** | [059](../../../analyses/059-2026-11-07-mlp-beta2-reduction/observations/001-beta2-reduction.md), Run232: .999 -> .95/.9/.5 at state512 gives mean loss +16.78%/+18.16%/+127.02%. Moments/clocks retained; native bias correction changes abruptly. Not a general refutation of the cited paper. |
| Distributed caps improve selected training trajectories | **Supported, bounded.** | [065 observation008](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md): mean -22.85% MLP/-16.22% ResNet/-26.00% Transformer; two MLP replicas -19.43%/-32.08%. Selected full-objective forks, one larger-model seed. |
| Representation-sensitive allocation adds information beyond step norm | **Supported by a specific control.** | [065 observation009](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/009-what-the-negative-controls-show.md): MLP layer-norm-matched control mean -3.43% versus cap -22.85%. Reference-arm schedule is retrospective, not an online competing policy. Directional controls fail. |
| Fewer reversals imply a better intervention | **Contradicted as a general rule.** | Same 065 records: Transformer output reversals rise 36.6% -> 97.9% with lower mean loss. Capped raw gradients reverse throughout while defined actual-update cosines remain nonnegative. |
| Higher LR under a cap speeds learning | **Not supported.** | [065 observation013](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/013-distributed-cap-quantile-lr.md): 54 prespecified continuations, three seeds. Every quantile's across-seed mean is worse at 2x/4x than its own 1x. q50 1x lowest observed mean; not an established optimum. |
| Training improvements imply better held-out performance | **Contradicted for reported terminal comparisons.** | 065 observation008: terminal test loss +2.87% MLP/+8.18% ResNet/+100.17% Transformer. Observation013 adds repeated complete MLP tests; q50 1x final-four test mean +3.37%. Distinguish endpoint from trajectory claims. |

## 4. The cap's actual mathematical object

With the current input held fixed, a layer's native parameter proposal gives
`Delta Z_local = H Delta W' + 1 Delta b'`. For unit/filter `j`, take the RMS
over the declared probe coordinates, `s_j`. For a positive threshold `tau`,
multiply the weight row and matching bias by `min(1, tau/s_j)`, with factor one
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
