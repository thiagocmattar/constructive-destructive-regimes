# Contribution framing: gaps, evidence, and claim boundaries

Discussion record, 15 November 2026. Based on the current
[introduction](draft/sections/01-introduction.tex) and
[manuscript](draft/manuscript.pdf), including its integrated appendix.

This note consolidates the author's discussion with the assistant. It is a
working record, not an approved rewrite of the introduction or a replacement
for [PLAN.md](PLAN.md). No new experiment is authorized by this note.

## Author's framing principle

Define the scientific gap before claiming a contribution. Work through the
introduction's contributions **one at a time**, establishing:

1. What the closest literature already explains.
2. What remains unresolved and why the existing treatment leaves it unresolved.
3. What question this paper actually investigates.
4. Which method and evidence answer that question.
5. Which broader questions remain unanswered.

Avoid reducing the contribution to a list of diagnostics or equations. The
author's intended organizing question connects optimization stability to what
changes in the function represented by the network: how oscillatory parameter
trajectories appear in internal representations and outputs, and how that
information can motivate control of harmful movement while preserving progress.

Do not claim that the entire stability literature ignores representations or
only restates a spectral threshold. Self-stabilization and Central Flows explain
mechanisms beyond threshold crossing; representation-based update controls also
exist. Identify the specific unresolved connection instead.

## Literature support checked during the discussion

### Central Flows: an explicit architectural and learning-level gap

[Cohen et al., Understanding Optimization in Deep Learning with Central Flows,
ICLR 2025, section 7.1](https://arxiv.org/html/2410.24206v2#S7.SS1)
explicitly discusses its black-box treatment of the loss. Its predictions are
at the loss-landscape level and require further translation to make claims
about architecture or the learning problem. In particular, the theory does not
explain how its implicit sharpness penalty affects learning or how different
layers are mechanistically implicated in progressive sharpening and sharpness
reduction.

This is direct support for the motivation, rather than an absence inferred from
a keyword search. The reason for the gap is the abstraction: parameter
trajectories and loss geometry do not by themselves specify which internal
responses change, on which inputs, at which sites, and with what amplitude.

The present paper addresses the **functional characterization** of this gap.
It does not establish the layer-level causal mechanism of sharpening or
self-stabilization requested by the broader open problem.

### Adam spikes: control depends on consequences, not only onset

[Bai et al., Adaptive Preconditioners Trigger Loss Spikes in Adam, ICML 2026](https://proceedings.mlr.press/v306/bai26o.html)
identifies a mechanism involving decoupling between current squared gradients
and the adaptive second-moment estimate. The inspected
[May 2026 manuscript, section 7 and appendix C](https://arxiv.org/html/2506.04805v2#S7)
explicitly leaves open:

- Separating the interacting contributions of loss geometry and preconditioner
  dynamics in complex models.
- Efficient curvature measurements at larger scales and reliable spike timing
  under stochastic sampling.
- Robust distinction between neutral, benign, malignant, and catastrophic
  spikes, informing whether to suppress or preserve them.
- A deeper understanding of how to exploit or control spikes.

The last two points motivate evaluating subsequent progress alongside spike
suppression. Our interventions provide bounded evidence relevant to this
question; they do not solve the general classification of beneficial and harmful
spikes or establish a universal control policy.

### Recent developments constrain historical gap claims

[Regis and Chewi, Unifying Central Flow, Rod Flow, and the Edge Coupling,
November 2026 preprint](https://arxiv.org/html/2611.08358v1#S10)
extends the flow treatment to heavy ball and Adam. Its conclusion retains open
questions about minibatch extensions, formal trajectory-tracking bounds, and
optimizers outside the Adam family. Consequently, the 2025 Central Flows
limitation concerning momentum must not be presented as an unchanged gap in the
current literature. These broader theory questions are not solved by our paper.

## Proposed bullet: functional tracing

This was the concrete wording proposed in the discussion. The author requested
that the discussion be saved before proceeding to the next bullet; final wording
and manuscript insertion remain pending.

> **Connecting oscillatory optimization to representation dynamics.** Existing
> stability analyses explain oscillatory parameter trajectories through loss
> geometry, but leave open how these dynamics manifest within the network.
> Central Flows explicitly identifies the translation from loss-landscape
> dynamics to layer-level mechanisms as an unresolved problem [Cohen et al.,
> 2025]. We address the functional characterization of this gap by tracking
> representation changes on fixed reference inputs across successive training
> states. Separating reversal, displacement magnitude, and shared temporal
> structure reveals architecture-dependent patterns of functional oscillation:
> reversal spans all measured MLP sites, while earlier ResNet and Transformer
> sites exhibit less persistent reversal than their outputs. These measurements
> motivate testing local displacement control while distinguishing observed
> coordination from causal attribution.

Use the existing bibliography key `cohen2025central` when converting this prose
to TeX. Keep the intervention's results in its own subsequent bullet. This
bullet's contribution is measuring and characterizing the functional expression
of oscillatory dynamics, not establishing the unique origin of an instability.

Evidence resides in [the tracing section](draft/sections/05b-tracing.tex),
Figures 6 and 7, and the representation checks in the
[appendix](draft/sections/appendix.tex). Claims concern the selected windows and
fixed reference sets. The MLP/ResNet/Transformer output changes require 5/20/15
modes for 95% of squared movement, among at most 51/61/49 observable directions.
These results are characterizations of the measured changes, not capacity bounds.

## Mathematical novelty: antecedent requiring explicit positioning

The discussion identified a direct antecedent for the step-direction scalar.
[Tan et al., Barzilai--Borwein Step Size for Stochastic Gradient Descent,
NeurIPS 2016, section 2, equation 6](https://proceedings.neurips.cc/paper/2016/file/c86a7ee3d8ef0b551ed58e354a836f2b-Paper.pdf)
reviews the classical formula

\[
\eta_{\mathrm{BB}} = \frac{s^\top s}{s^\top y},
\qquad s=\theta_t-\theta_{t-1},\quad y=g_t-g_{t-1}.
\]

Our step-direction estimate is

\[
\widehat\kappa_t^{\mathrm{step}}=\frac{s^\top y}{s^\top s}.
\]

The expressions are reciprocals when both are defined. Tan et al. adapt the
classical method to choose learning rates in SGD and SVRG, including blockwise
measurements and noise-handling modifications; their full stochastic algorithms
are not simply our diagnostic inverted at every update. Their theoretical
convergence result is for SVRG-BB under strong-convexity and smoothness conditions,
with experiments on logistic regression and SVM problems.

Therefore, do not claim novelty for this scalar expression alone. Its role in
our framework, the interpretation through consecutive-gradient alignment and
magnitude, and its empirical use need separate positioning. This antecedent
does not by itself settle the novelty of the complete framework. The existing
bibliography already contains `tan2016bbsgd`; its appropriate use in the
introduction and methods remains to be discussed.

Also distinguish this actual-step scalar from Adam's raw-gradient-projected
response `eta * kappa_hat`. In plain constant-step SGD, multiplying the
step-direction estimate by `eta` recovers that response; this equivalence does
not hold for general Adam updates.

## Interpretation boundaries established in the discussion

- **Reversal and spikes:** negative gradient cosine indicates opposition between
  raw gradients. It does not alone imply increasing loss, reversal of Adam's
  applied updates, or erasure of learned information. The quadratic GD loss-rise
  condition requires sufficiently amplified opposition, not merely norm growth.
- **Functional measurement:** use the same inputs across training states. Two
  consecutive activation changes require three states. Tracing records site
  activations through forward evaluations; it does not backpropagate an output
  oscillation to infer its cause.
- **Direction and magnitude:** change cosine and RMS displacement answer
  different questions. A global negative cosine can be dominated by coordinates
  with large changes; it does not imply reversal of every sample or feature.
- **SVD:** rows of the change matrix index updates; columns index fixed
  sample/position-feature coordinates. Right singular vectors are patterns in
  that joint coordinate space. A small number of modes describes most squared
  movement over the window, not fewer actual training steps, fewer active
  features, or low representational capacity. The matrix is uncentered, so the
  decomposition retains drift and measures squared movement rather than the
  variance of centered changes. Concentration alone does not establish reversal.
- **Window dependence:** observable rank is bounded by the smaller of the number
  of changes and the number of coordinates. Extending a window can expose new
  directions, but the number of modes needed for a fixed movement fraction need
  not increase monotonically.
- **Temporal overlap:** `O_k` compares leading left-singular-vector subspaces,
  which share the training-step coordinates across sites. It does not compare
  spatial feature directions, measure movement magnitude, or identify which
  layer contributes causally to output variation. RMS alone does not identify
  that causal contribution either.
- **Displacement cap:** the controller scales per-unit proposed parameter
  increments using direct local displacement on current inputs. It does not
  remove SVD modes or guarantee a bound on propagated output movement. Its
  benefits do not establish that the network spreads learning over more modes.

The author's suggestion that capping might increase the effective dimension of
representation changes remains a hypothesis. A less concentrated spectrum could
also result from attenuation of dominant modes, without new movement directions.
No such spectral intervention comparison was established in this discussion.

## Remaining contribution-by-contribution discussion

| Current introduction bullet | Status after this discussion |
| --- | --- |
| Constructive and destructive regimes | Gap and distinctive result still to be refined against EoS and gradient-product diagnostics. |
| Cheap curvature along the applied step | BB antecedent identified; novelty language needs explicit positioning. |
| Reversal, growth, and loss increases | Preserve exact quadratic GD versus nonlinear approximation and retrospective versus outgoing-step timing; gap formulation still pending. |
| Functional tracing | Explicit Central Flows motivation located; proposed bullet recorded above. |
| Selective displacement control | Discuss next, separately; connect to the unresolved control/progress question and representation-based controls. |

For the last bullet, the closest literature includes
[Kosson et al., Analyzing & Reducing the Need for Learning Rate Warmup in GPT
Training, NeurIPS 2024](https://arxiv.org/abs/2410.23922). Its existence prevents
claiming that representation-informed update control is wholly absent.

The assistant's earlier reviewer suggestions included independent replications,
stochastic testing, and controls with comparable applied-update attenuation to
isolate the value of representation-based selectivity. These are proposals for
discussion, not accepted requirements, approved experiments, or a verdict that
the framing discussion has resolved submission readiness. The author's current
priority is to establish each gap and contribution precisely, one at a time.
