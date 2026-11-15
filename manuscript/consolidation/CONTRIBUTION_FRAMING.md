# Contribution framing: gaps, evidence, and claim boundaries

Discussion record, 15 November 2026. Based on the current
[introduction](draft/sections/01-introduction.tex) and
[manuscript](draft/manuscript.pdf), including its integrated appendix.

This note consolidates the author's discussion with the assistant. It is a
working record, not an approved rewrite of the introduction or a replacement
for [PLAN.md](PLAN.md). No new experiment is authorized by this note.

## Consolidated contribution bullets in manuscript order

At the author's request, the four full formulations are collected here in the
order **framework, diagnostic, representations, intervention**. This is the
consolidated working block; the sections below retain the supporting discussion
and claim boundaries. Insertion into the manuscript remains pending. The
implicit-regularization experiment remains a separate side note.

1. **A framework for interpreting constructive and destructive training dynamics
   under sampling noise.** Existing analyses characterize oscillatory stability,
   gradient-based convergence diagnostics, and optimization path length, but these
   perspectives do not by themselves determine how observed update opposition
   translates into accumulated motion. We connect update magnitudes, temporal
   alignment, and displacement through an exact decomposition of SGD trajectories,
   and explicitly characterize how sampling enters consecutive-gradient
   measurements. This separates local reversal, cumulative interference, and
   displacement retention while distinguishing fixed-state alignment dilution
   from changes along an evolving trajectory. Across the studied GD trajectories,
   negative adjacent interactions coexist with positive accumulated interactions
   and continued loss reduction; matched mini-batch continuations further reveal
   how sampling modifies coherent reversal. Together, these results provide the
   parameter-space foundation for investigating the corresponding changes in
   network representations.

2. **Recovering step-averaged curvature from training gradients for practical
   loss-rise diagnostics.** Curvature-based analyses provide insight into loss
   spikes, but their additional derivative computations remain a recognized
   obstacle to scalable monitoring [Bai et al., 2026]. We connect
   consecutive-gradient measurements to an exact geometric quantity: for a smooth
   objective and full gradients, the gradient difference projected onto the
   applied update recovers directional Hessian curvature averaged along that
   step, independently of the optimizer. Under gradient descent, this identity
   reduces to a combination of gradient cosine and norm ratio, linking reversal
   and amplification to the curvature encountered during training. The diagnostic
   reuses training gradients without additional backward passes or
   Hessian--vector products, requiring linear storage and vector operations. For
   quadratic full-batch GD, the resulting condition exactly characterizes loss
   increases on the completed step. In the three studied Adam trajectories, we
   separately test the combined angle--magnitude signal before each outgoing
   update: it achieves the highest accuracy and warning lift among the evaluated
   warning rules, outperforming previous-loss-rise, reversal-only, and
   norm-growth-only diagnostics. These findings support its usefulness for
   inexpensive training-time monitoring, while distinguishing exact retrospective
   geometry from empirically evaluated next-step prediction.

3. **Connecting oscillatory optimization to representation dynamics.** Existing
   stability analyses explain oscillatory parameter trajectories through loss
   geometry, but leave open how these dynamics manifest within the network.
   Central Flows explicitly identifies the translation from loss-landscape
   dynamics to layer-level mechanisms as an unresolved problem [Cohen et al.,
   2025]. We address the functional characterization of this gap by tracking
   representation changes on fixed reference inputs across successive training
   states. Separating reversal, displacement magnitude, and shared temporal
   structure reveals architecture-dependent patterns of functional oscillation:
   reversal spans all measured MLP sites, while earlier ResNet and Transformer
   sites exhibit less persistent reversal than their outputs. These measurements
   motivate testing local displacement control while distinguishing observed
   coordination from causal attribution.

4. **Testing representation displacement as a target for intervention.**
   Explaining how loss spikes arise leaves open how to control disruptive
   movement while preserving training progress [Bai et al., 2026]. Motivated by
   the observed functional oscillations and prior representation-based controls
   [Kosson et al., 2024], we use local displacement caps to test whether
   selectively limiting movement magnitude can improve training without
   eliminating reversals. In matched continuations across three model--dataset
   cases, all tested cap quantiles reduce mean training loss and accumulated
   upward loss variation relative to unchanged Adam, while functional reversals
   persist. Comparisons with learning-rate reduction, second-moment changes,
   and gradient clipping reveal distinct tradeoffs. These findings identify
   local representation displacement as a promising intervention target, with
   endpoint and computational limitations defining the scope of the evidence.

For bullet 2, the comparison is supported by
[Table II](draft/tables/diagnostic-0p1pct.tex) and
[the characterization section](draft/sections/05a-characterization.tex).
The angle/norm and projected-response rules are algebraically equivalent and
tie; they are not independent evidence. The exact actual-step curvature identity
must not be conflated with Adam's raw-gradient response or extended unchanged
to differently sampled gradients. The BB antecedent is discussed below.

An additional recent comparator found during the discussion is
[Lee and Lee, The Road Taken, September 2026 preprint version, sections 6.2--6.4](https://arxiv.org/html/2608.18415v2).
It computes an exact realized loss-balance quantity from the gradient, applied
direction, and observed loss change without HVPs, and leaves stochastic extension
open. This differs from our gradient-difference step-average quantity, but means
that exactness and absence of HVPs alone are not exclusive novelty claims.
Position the diagnostic through its specific quantity, angle/magnitude
interpretation, sampling analysis, and evaluated prediction task.

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

## Accepted framing and full bullet: intervention target

The author approved the following direction and requested retaining the full
bullet without shortening it. The contribution is evidence that controlling
local representation displacement is a promising intervention direction. The
cap serves as an experimental instrument for testing that direction; proposing
a new optimizer or establishing algorithmic superiority is not the intended
central claim. Insertion into the manuscript remains pending.

> **Testing representation displacement as a target for intervention.**
> Explaining how loss spikes arise leaves open how to control disruptive
> movement while preserving training progress [Bai et al., 2026]. Motivated by
> the observed functional oscillations and prior representation-based controls
> [Kosson et al., 2024], we use local displacement caps to test whether
> selectively limiting movement magnitude can improve training without
> eliminating reversals. In matched continuations across three model–dataset
> cases, all tested cap quantiles reduce mean training loss and accumulated
> upward loss variation relative to unchanged Adam, while functional reversals
> persist. Comparisons with learning-rate reduction, second-moment changes,
> and gradient clipping reveal distinct tradeoffs. These findings identify
> local representation displacement as a promising intervention target, with
> endpoint and computational limitations defining the scope of the evidence.

Use the existing bibliography keys `bai2026spikes` and `kosson2024warmup` when
converting this prose to TeX. The literature motivation is recorded above;
the experimental support is in
[the intervention section](draft/sections/05c-interventions.tex), Figures 8–9,
Table III, and the endpoint and cost checks in the appendix. The claim concerns
the three measured continuations, with one initialization per model and unequal
effective intervention strengths. It does not assert improved generalization,
asymptotic convergence, or time-to-loss performance.

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

## Proposed fourth bullet: the Section III framework

The author requested saving this full bullet after discussing Section III as a
whole, including the regime definitions, cumulative displacement, and sampling
effects. It is a contribution of integrated formulation, interpretation, and
empirical characterization. Manuscript insertion remains pending.

> **A framework for interpreting constructive and destructive training dynamics
> under sampling noise.** Existing analyses characterize oscillatory stability,
> gradient-based convergence diagnostics, and optimization path length, but these
> perspectives do not by themselves determine how observed update opposition
> translates into accumulated motion. We connect update magnitudes, temporal
> alignment, and displacement through an exact decomposition of SGD trajectories,
> and explicitly characterize how sampling enters consecutive-gradient
> measurements. This separates local reversal, cumulative interference, and
> displacement retention while distinguishing fixed-state alignment dilution
> from changes along an evolving trajectory. Across the studied GD trajectories,
> negative adjacent interactions coexist with positive accumulated interactions
> and continued loss reduction; matched mini-batch continuations further reveal
> how sampling modifies coherent reversal. Together, these results provide the
> parameter-space foundation for investigating the corresponding changes in
> network representations.

The gap is our synthesis of the literature comparison, not a claim that an
individual cited paper explicitly asks this exact question. Relevant antecedents
are [Pesme et al., ICML 2020, section 4](https://proceedings.mlr.press/v119/pesme20a/pesme20a.pdf)
on noise-sensitive consecutive-gradient diagnostics;
[Gupta et al., JMLR 2021](https://jmlr.org/papers/v22/19-979.html)
on optimization path length; and
[Andreyev and Beneventano, sections 5 and 9](https://arxiv.org/html/2412.20553)
on distinguishing stochastic oscillations from instability and the open question
of their effects on learned features. The contribution does not claim novelty
for the squared-norm expansion alone or a general stability guarantee.

Evidence and definitions are in [Section III](draft/sections/03-regimes.tex),
[Section IV](draft/sections/04-gd-sgd.tex), and the
[integrated appendix](draft/sections/appendix.tex). All nine GD trajectories have
positive all-pair interactions; seven have negative adjacent-pair interactions.
The exact accumulated-gradient formula assumes plain constant-rate SGD; general
optimizers require actual parameter increments. The fixed-state cosine formula
is an approximation, and moving-trajectory damping is an empirical finding.

## Side note: noise, reversal damping, and implicit regularization

Author's hypothesis, recorded on 15 November 2026 for later discussion:
mini-batch noise weakens coherent negative gradient alignment in the measured
trained-state continuations. If this damping also reflects a reduction in the
curvature encountered along training, it could provide evidence for an implicit
regularization effect of SGD. The intended connection is from sampling-induced
changes in update dynamics to a preference for lower-curvature regions.

This is a hypothesis, not a conclusion established by the current measurements.
Section III separates the explicit sampling contribution to consecutive-gradient
products from an approximate dilution of cosine for independent gradients at a
fixed state. The latter does not establish attenuation of negative cosine along
a moving trajectory, or a reduction in the Hessian of the full objective.
The moving-trajectory damping is an empirical observation in the studied cases.

The relation `rho ~= (norm_previous / norm_current) *
(1 - eta * kappa + epsilon)` makes the identification issue explicit:
changes in cosine can reflect norm ratios and sampling terms as well as
directional curvature. A change in direction can also reduce directional
curvature without lowering the largest Hessian eigenvalue. Consequently, damping
of rho alone does not identify curvature reduction or an implicit regularizer.

To substantiate the author's interpretation, a future matched comparison would
need independent full-objective curvature measurements, distinguishing curvature
along the applied direction from spectral sharpness, and separating sampling
effects at fixed parameters from the geometry reached by different trajectories.
Evidence of a systematic preference for lower-curvature regions would support
implicit regularization; improved generalization would require separate evidence.
These are proposed checks, not an authorized experiment or a new registered
hypothesis/run.

Author's follow-up experimental idea: revisit the batch-size comparison at fixed
saved training states, where rho was measured while varying batch size, and
also measure directional curvature explicitly. Keep this as a side investigation,
not a central paper contribution or a requirement for the current framing.
Before designing new measurements, reconcile this idea with the existing
independent HVP checks in the appendix. At a fixed state the full-objective
Hessian is unchanged by resampling: directional curvature can vary because the
sampled direction changes, whereas mini-batch Hessians measure different
objectives. This comparison could help separate measurement and direction
effects; showing a preference for flatter regions would additionally require
trajectory comparisons. No implementation or execution is requested here.

Relevant antecedents already located in the literature discussion are
[Andreyev and Beneventano, Edge of Stochastic Stability](https://arxiv.org/html/2412.20553)
and [Liao, Kolomvaki, and Kyrillidis, The Stochastic Sharpness Gap,
April 2026 preprint](https://arxiv.org/html/2604.21016). These works address
stochastic stability and suppression of sharpness, so any future claim should
identify the additional evidence or mechanism provided here.

## Mapping from the original introduction to the consolidated block

| Current introduction bullet | Status after this discussion |
| --- | --- |
| Constructive and destructive regimes | Consolidated bullet 1: framework, cumulative displacement, and sampling. |
| Cheap curvature along the applied step | Combined into bullet 2; BB and the recent realized-loss-balance comparator constrain novelty claims. |
| Reversal, growth, and loss increases | Combined into bullet 2; exact quadratic retrospective relation remains distinct from empirical outgoing-step prediction. |
| Functional tracing | Consolidated bullet 3; explicit Central Flows motivation and measurement scope retained. |
| Selective displacement control | Consolidated bullet 4; intervention-target framing retained, rather than a new-optimizer claim. |

All four formulations are now recorded in this file. Manuscript insertion
remains pending.

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
