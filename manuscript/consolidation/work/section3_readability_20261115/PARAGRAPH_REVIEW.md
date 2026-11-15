# Conservative Section III readability edits

15 November 2026. Each diagnosis and proposed solution was recorded before its source edit. Unlisted prose was retained. Equations, notation, labels, and subsection order are protected.

## III opening

**Read**

```tex
Loss spikes and oscillatory updates motivate diagnostics that explain how
training is progressing as it happens. The gradients already computed by
the optimizer contain two complementary signals: how strongly the loss
changes locally, and whether the next update reinforces or opposes the
preceding one. We first show how these interactions determine accumulated
parameter change. We then connect consecutive alignment to curvature,
obtaining an inexpensive estimator and a joint condition on gradient
reversal and growth that explains when the loss can increase.
```

**Diagnosis:** The opening leaves the optimizer scope and measurement timing implicit. Gradient magnitude is presented as a loss change, although finite-step loss change requires more information.

**Proposed solution:** Introduce sizes and directions under plain SGD, then distinguish accumulated motion, curvature, and the loss change over a completed step.

**Rewrite:**

```tex
For plain SGD, gradient sizes and directions determine how parameter
updates combine. We first distinguish opposition between consecutive
updates from cancellation over the full trajectory. We then relate
consecutive gradients to curvature and examine how mini-batch sampling
affects these measurements. Finally, we derive a joint condition on
gradient reversal and growth for a loss increase over the completed step,
exact for full-batch quadratic GD and approximate for nonlinear losses.
```

## III displacement setup

**Read**

```tex
For training from scratch, $\theta_0$ is the initial model and $\theta_T$
contains the parameter changes accumulated while learning from the data.
Expanding their squared difference separates individual step magnitudes
from interactions across steps
```

**Diagnosis:** The endpoint parameters are described as containing parameter changes, rather than distinguishing the endpoint from its displacement.

**Proposed solution:** Name the two parameter states directly and retain the existing expansion.

**Rewrite:**

```tex
Here $\theta_0$ and $\theta_T$ are the initial and final parameter states.
Expanding the squared displacement separates individual step magnitudes
from interactions across steps
```

## III regime definition

**Read**

```tex
Constructive and destructive behavior are thus encoded directly in the
SGD update equations. For fixed norms, positive alignment increases net
displacement, whereas negative alignment limits it through cancellation.
We identify a \emph{constructive regime} by persistent $\rho_t>0$ and
a \emph{destructive regime} by persistent $\rho_t<0$.
```

**Diagnosis:** The paragraph moves from all-pair interactions to consecutive-gradient regime labels without stating that those are different levels of description. Persistent is not operationally qualified.

**Proposed solution:** Preserve both definitions and their qualitative use, explain repeated signs rather than isolated pairs, and explicitly state that the labels do not determine the all-pair sum. Keep normalized displacement in III-D.

**Rewrite:**

```tex
For fixed norms, positive pairwise alignment reinforces displacement,
whereas negative alignment contributes cancellation. We identify a
\emph{constructive regime} by persistent $\rho_t>0$ and a
\emph{destructive regime} by persistent $\rho_t<0$. Persistence describes
a repeated sign pattern over a training interval, rather than one isolated
pair. These labels concern consecutive updates; they do not determine
the sum of interactions across all pairs. A destructive regime can
therefore coexist with reinforcement over the full trajectory.
Section~\ref{subsec:normalized-displacement} measures the resulting
displacement relative to the distance traveled.
```

## III-A step and expansion

**Read**

```tex
We now connect these regimes to the curvature encountered along a step.
For mini-batch SGD, use
$g_t=\nabla\mathcal L(\theta_t)+\xi_t$ from
Eq.~\eqref{eq:sgd_noise_decomposition}. Suppose $\mathcal L$ is twice
continuously differentiable along the preceding step and write
$H_{t-1}:=\nabla^2\mathcal L(\theta_{t-1})$.
Expanding the full gradient around $\theta_{t-1}$ gives
```

**Diagnosis:** The expansion's clock is implicit: the reader must infer which gradient produced the step and which is its response.

**Proposed solution:** Identify the completed transition and its generating gradient before retaining the existing smoothness assumptions and expansion.

**Rewrite:**

```tex
We now examine how the step from $\theta_{t-1}$ to $\theta_t$, generated
by $g_{t-1}$, changes the gradient. For mini-batch SGD, use
$g_t=\nabla\mathcal L(\theta_t)+\xi_t$ from
Eq.~\eqref{eq:sgd_noise_decomposition}. Suppose $\mathcal L$ is twice
continuously differentiable along this step and write
$H_{t-1}:=\nabla^2\mathcal L(\theta_{t-1})$.
Expanding the full gradient around $\theta_{t-1}$ gives
```

## III-A projection

**Read**

```tex
where $\mathcal R_t$ is the Taylor remainder. Neglecting it and projecting onto
the preceding gradient yields
```

**Diagnosis:** Projecting names an operation without saying why it is performed.

**Proposed solution:** Explain that the inner product isolates the component along the old gradient, while keeping the neglected remainder explicit.

**Rewrite:**

```tex
where $\mathcal R_t$ is the Taylor remainder. Neglecting it, we take
the inner product with $g_{t-1}$ to isolate the new gradient's component
along the preceding gradient:
```

## III-A curvature location

**Read**

```tex
Define the curvature along the preceding gradient, $\kappa_t$, and the normalized
noise contribution, $\varepsilon_t$, as
```

**Diagnosis:** The definition specifies direction but does not foreground that the Hessian is evaluated at the starting point, unlike the subsequent step-average.

**Proposed solution:** Name both the starting point and direction without changing either definition.

**Rewrite:**

```tex
Define $\kappa_t$ as the curvature at the starting point
$\theta_{t-1}$ along $g_{t-1}$, and $\varepsilon_t$ as the noise
contribution divided by the preceding gradient's squared norm:
```

## III-A alignment sign

**Read**

```tex
In full-batch GD, $\varepsilon_t=0$ and the norm ratio is positive.
Within this approximation, $\rho_t\propto1-\eta\kappa_t$, so
$\eta\kappa_t<1$ gives constructive alignment, while $\eta\kappa_t>1$
gives destructive alignment. The boundary $\eta\kappa_t=1$ separates
the two regimes. At this boundary the gradients are orthogonal if
both are nonzero; a vanishing gradient leaves the cosine undefined.
```

**Diagnosis:** Proportionality can suggest a constant factor across steps although the norm ratio varies. The conclusion actually concerns the sign within a local approximation.

**Proposed solution:** State the sign implication for nonzero full gradients and retain the approximation and zero-gradient caveats.

**Rewrite:**

```tex
In full-batch GD with nonzero gradients, $\varepsilon_t=0$ and the
norm ratio is positive. Within this approximation, the sign of $\rho_t$
is therefore the sign of $1-\eta\kappa_t$: $\eta\kappa_t<1$ gives
constructive alignment, while $\eta\kappa_t>1$ gives destructive
alignment. At the boundary $\eta\kappa_t=1$, the approximation gives
orthogonal gradients; a vanishing gradient leaves the cosine undefined.
```

## III-A interpretation and timing

**Read**

```tex
This relation has two complementary interpretations. In terms of
parameter changes, a destructive update opposes parameter changes made by its
predecessor. Even with iid data from a fixed distribution, successive
updates can repeatedly undo components of accumulated parameter change.
Whether the corresponding representations also reverse must be measured;
Section~\ref{subsec:tracing} follows these changes inside the network.
From an optimization perspective, negative alignment indicates
that a GD step has passed the minimum along its preceding direction in
the local quadratic model. The optimizer then partially reverses that
step. Such reversals can coexist with convergence; the magnitude of the
returning gradient determines when they instead increase the loss.
```

**Diagnosis:** The interpretation shifts from the just-completed GD step to the returning update, making it unclear which loss change the later criterion explains. It also risks conflating local opposition with cancellation of accumulated displacement.

**Proposed solution:** Keep the parameter and representation interpretations, then distinguish the completed step from the next opposing update. Attribute the quadratic loss balance to the completed step.

**Rewrite:**

```tex
In parameter space, negative alignment means that the next plain-SGD
update opposes a component of the preceding update. Whether the
corresponding representations also reverse must be measured;
Section~\ref{subsec:tracing} follows these changes inside the network.
For full-batch GD, negative alignment indicates that the completed step
has passed the minimum along its direction in the local quadratic model.
The next update then partially reverses that step. Passing this minimum
does not itself imply a loss increase: the completed-step loss balance
also depends on the size of the new gradient, as derived below.
```

## III-B measured curvature

**Read**

```tex
The estimator has a direct geometric meaning. Under smooth full-batch GD,
it equals the directional Hessian curvature averaged over the preceding
step. If the curvature changes little during that step, it also estimates
the curvature at its starting point. This is the curvature encountered
along the direction actually followed; it need not equal the largest
Hessian eigenvalue, which describes the sharpest available direction.
```

**Diagnosis:** The estimator is first derived from an approximation and then declared exact, without naming the change of target clearly. The displayed scaled quantity could also be mistaken for curvature itself.

**Proposed solution:** Explicitly name unscaled kappa-hat as an exact step-average under smooth full-batch GD and distinguish it from starting-point kappa and the largest Hessian eigenvalue.

**Rewrite:**

```tex
Under smooth full-batch GD, $\widehat\kappa_t$ measures directional
curvature averaged over the completed step exactly. This differs from
$\kappa_t$, the curvature at the step's starting point. The two are close
when curvature changes little along the step. Both concern the direction
actually followed, rather than the largest Hessian eigenvalue, which
describes the sharpest available direction.
```

## III-B directional measurement

**Read**

```tex
The numerator measures how much the gradient changes along the applied
step; division by squared step length gives curvature per unit displacement.
For full gradients and a smooth objective, this is exactly
```

**Diagnosis:** Curvature per unit displacement suggests an extra normalization of curvature. The ratio instead measures the directional gradient change per unit step length.

**Proposed solution:** Describe the whole ratio as a rate of directional-gradient change, then state the fixed-objective assumption before the unchanged exact identity.

**Rewrite:**

```tex
This ratio measures the change in the gradient component along the step,
per unit distance traveled. For full gradients of a smooth, fixed
objective, it is exactly
```

## III-C evolving-trajectory noise setup

**Read**

```tex
Fresh, conditionally unbiased batches satisfy
$\mathbb E[g_{t-1}^\top\xi_t]=0$ and
$\mathbb E[\nabla\mathcal L(\theta_{t-1})^\top\xi_{t-1}]=0$.
To evaluate the remaining term, substitute
$g_{t-1}=\nabla\mathcal L(\theta_{t-1})+\xi_{t-1}$ into
$g_{t-1}^{\top}\xi_{t-1}$. Its expectation is
$\mathbb E\|\xi_{t-1}\|^2$, because the full-gradient/noise cross term
vanishes. Subtracting it from the zero expectation of
$g_{t-1}^{\top}\xi_t$ gives
```

**Diagnosis:** The derivation starts with expectation identities before telling readers whether the parameters are evolving or fixed.

**Proposed solution:** State that this first calculation concerns successive states and retain its conditional-unbiasedness argument.

**Rewrite:**

```tex
First consider consecutive gradients along the SGD trajectory.
Fresh batches whose expected gradient equals the full gradient at each
state satisfy $\mathbb E[g_{t-1}^\top\xi_t]=0$ and
$\mathbb E[\nabla\mathcal L(\theta_{t-1})^\top\xi_{t-1}]=0$.
Substituting $g_{t-1}=\nabla\mathcal L(\theta_{t-1})+\xi_{t-1}$ into
$g_{t-1}^{\top}\xi_{t-1}$ leaves expectation
$\mathbb E\|\xi_{t-1}\|^2$, because the full-gradient/noise cross term
vanishes. Subtracting this from the zero expectation of
$g_{t-1}^{\top}\xi_t$ gives
```

## III-C fixed-state experiment

**Read**

```tex
For two independent gradients sampled at the same fixed state $\theta$,
concentration of their norms gives the approximation
```

**Diagnosis:** The text changes the sampling experiment and reuses rho_t without warning. Concentration is named without explaining the random denominator approximation.

**Proposed solution:** Announce that no update is applied between samples, explain the unchanged notation, and state the norm-concentration approximation in words before retaining the equation.

**Rewrite:**

```tex
Now hold the parameters fixed and sample two independent gradients,
without applying an update between them. In the following expression,
$\rho_t$ denotes their cosine at that same state $\theta$. If their norms
concentrate near a common scale, we approximate the product of their norms
by the expected squared norm, giving
```

## III-C fixed-state interpretation

**Read**

```tex
The numerator is the direction shared by both sampled gradients; the
additional variance in the denominator increases their lengths without
providing shared alignment. Smaller batches therefore dilute this common
direction and bring the expected cosine closer to zero. This fixed-state
calculation concerns two estimates of the same full gradient. Along an
evolving trajectory, sampling also changes the next state and the full
gradient there. It can weaken a coherent back-and-forth motion, but the
negative term in Eq.~\eqref{eq:noise-expectation} alone does not prove
attenuation of a negative cosine. Section~\ref{sec:sgd-noise-experiments}
tests that behavior during training.
```

**Diagnosis:** The numerator is incorrectly called a direction rather than a squared magnitude. The expected-cosine interpretation also needs to remain explicitly tied to the approximation.

**Proposed solution:** Name the scalar correctly, explain the denominator as added variability, and preserve the warning that fixed-state sampling does not prove attenuation of negative trajectory cosine.

**Rewrite:**

```tex
The numerator is the squared norm of the full gradient shared in
expectation by both samples. Sampling adds variance to the denominator
without adding shared alignment. Within this approximation, smaller
batches bring the expected cosine closer to zero. Along an evolving
trajectory, sampling also changes the next state and the full gradient
there. It can weaken coherent back-and-forth motion, but neither this
fixed-state calculation nor the negative term in
Eq.~\eqref{eq:noise-expectation} alone proves attenuation of a negative
cosine. Section~\ref{sec:sgd-noise-experiments} tests that behavior during
training.
```

## III-E heading

**Read**

```tex
\subsection{Conditions for Loss Decrease and Spike Formation}
```

**Diagnosis:** Spike formation suggests a multi-step event, whereas the derivation characterizes a single completed-step loss change.

**Proposed solution:** Narrow the heading while preserving its place and labels.

**Rewrite:**

```tex
\subsection{Conditions for Loss Decrease and Increase}
```

## III-E loss interval and assumptions

**Read**

```tex
When does a destructive regime produce a loss increase? Define the
full-objective change over the preceding step as
$\Delta\mathcal L_t:=\mathcal L(\theta_t)-\mathcal L(\theta_{t-1})$.
The second-order approximation derived in Appendix~\ref{app:math}
separates three contributions
```

**Diagnosis:** The preceding interval is defined, but the prose still allows the reader to associate reversal with the next returning step. The exact quadratic case appears only later.

**Proposed solution:** Ask about the step that produced the observed reversal, identify plain SGD, and state the approximation versus quadratic scope before the unchanged equation.

**Rewrite:**

```tex
When does the step that produces gradient reversal also increase the
loss? For plain SGD, define the full-objective change over the completed
step as $\Delta\mathcal L_t:=\mathcal L(\theta_t)-\mathcal L(\theta_{t-1})$.
The second-order relation derived in Appendix~\ref{app:math}, exact for
a quadratic objective, separates three contributions
```

## III-E interpretation of the completed step

**Read**

```tex
Together, these relations distinguish three cases within the full-batch
quadratic model. When $\eta\kappa_t<1$, consecutive gradients are
constructive and the loss decreases. When $1<\eta\kappa_t<2$, they are
destructive but the return step still reduces the loss. When
$\eta\kappa_t>2$, reversal is sufficiently amplified to increase the loss.
The boundaries give zero alignment and zero loss change, respectively,
when the relevant gradients are nonzero. This closes the connection
between curvature, gradient norms, and interference: the angle/norm
condition expresses the same directional loss balance through inexpensive
measurements. It is not a replacement of directional curvature by the
largest Hessian eigenvalue. For nonlinear losses the quadratic relation
has a finite-step remainder, and sampling adds the explicit noise term.
We next examine how these regimes appear in training.
```

**Diagnosis:** The return-step sentence assigns a completed-step loss criterion to the wrong update. The closing interpretation must retain the distinction between directional curvature, largest eigenvalue, and nonlinear remainder.

**Proposed solution:** Refer to the completed step in all three cases, preserve the existing thresholds and boundary qualifications, and keep the nonlinear and sampling limitations.

**Rewrite:**

```tex
Together, these relations distinguish three cases for a completed
full-batch GD step on a quadratic objective. When $\eta\kappa_t<1$,
consecutive gradients are constructive and that step decreases the loss.
When $1<\eta\kappa_t<2$, the gradients are destructive, yet the completed
step still decreases the loss. When $\eta\kappa_t>2$, the completed step
increases the loss and produces sufficiently amplified reversal.
The boundaries give zero alignment and zero loss change, respectively,
when the relevant gradients are nonzero. The angle/norm condition thus
expresses the same directional loss balance through measurements already
available after the step. The curvature here is directional, not the
largest Hessian eigenvalue. For nonlinear losses the relation has a
finite-step remainder, and sampling adds the explicit noise term.
We next examine how these regimes appear in training.
```

## III regime definition (concision review)

**Read**

```tex
For fixed norms, positive pairwise alignment reinforces displacement,
whereas negative alignment contributes cancellation. We identify a
\emph{constructive regime} by persistent $\rho_t>0$ and a
\emph{destructive regime} by persistent $\rho_t<0$. Persistence describes
a repeated sign pattern over a training interval, rather than one isolated
pair. These labels concern consecutive updates; they do not determine
the sum of interactions across all pairs. A destructive regime can
therefore coexist with reinforcement over the full trajectory.
Section~\ref{subsec:normalized-displacement} measures the resulting
displacement relative to the distance traveled.
```

**Diagnosis:** The added bridge states the local-versus-global distinction twice, increasing length without adding meaning.

**Proposed solution:** Retain qualitative persistence and the distinction in fewer sentences, without moving III-D.

**Rewrite:**

```tex
For fixed norms, positive pairwise alignment reinforces displacement
and negative alignment contributes cancellation. We identify a
\emph{constructive regime} by persistent $\rho_t>0$ and a
\emph{destructive regime} by persistent $\rho_t<0$. These qualitative
labels describe repeated signs over an interval, not isolated pairs.
They concern consecutive updates, so a destructive regime can coexist
with positive all-pair interactions.
Section~\ref{subsec:normalized-displacement} measures displacement
relative to the distance traveled.
```

## III-B computational scope (concision review)

**Read**

```tex
No assumption about how the optimizer chose $u_{t-1}$ is needed. Thus the
estimate applies to momentum, Adam, and other parameter updates. It requires
the two consecutive gradients and the applied increment, but no additional
backward pass or Hessian--vector product. The required inner products and
storage scale linearly with the number of parameters.
```

**Diagnosis:** The optimizer-independent scope is repeated in two sentences. Concision here can offset the explanations added elsewhere without losing assumptions.

**Proposed solution:** Combine the optimizer scope and preserve the inputs, linear cost, and absence of extra derivative computations.

**Rewrite:**

```tex
The identity holds regardless of how the optimizer chose the step,
including momentum and Adam. It uses two full gradients and the applied
increment, with storage and vector operations linear in the number of
parameters, but no additional backward pass or Hessian--vector product.
```

## III-C fixed-state experiment (concision review)

**Read**

```tex
Now hold the parameters fixed and sample two independent gradients,
without applying an update between them. In the following expression,
$\rho_t$ denotes their cosine at that same state $\theta$. If their norms
concentrate near a common scale, we approximate the product of their norms
by the expected squared norm, giving
```

**Diagnosis:** The fixed-state setup repeats the lack of parameter movement and includes avoidable introductory wording.

**Proposed solution:** Keep the distinct experiment, reused notation, and explicit approximation in a shorter setup.

**Rewrite:**

```tex
Separately, sample two independent gradients at a fixed state $\theta$,
without updating parameters. Here $\rho_t$ denotes their cosine.
When their norms concentrate near a common scale, approximating their
product by the expected squared norm gives
```

## III regime definition (final precision review)

**Read**

```tex
They concern consecutive updates, so a destructive regime can coexist
with positive all-pair interactions.
```

**Diagnosis:** Positive all-pair interactions could be read as every pair having a positive product, contradicting the consecutive negative pairs. The intended claim concerns the aggregate.

**Proposed solution:** State net reinforcement across all pairs explicitly.

**Rewrite:**

```tex
They concern consecutive updates, so a destructive regime can coexist
with net reinforcement across all pairs.
```
