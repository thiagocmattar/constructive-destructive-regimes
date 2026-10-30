# What the Pythia evidence supports, and the next useful experiment

The existing experiments already support a substantial Adam characterization.
They distinguish the sampled gradient response, retention in the first moment,
adaptive-direction geometry and loss. I would use these results before adding
another broad learning-rate sweep or another model size. A further experiment
becomes valuable when the claim advances from characterization to curvature
interpretation or to improved learning-rate decisions.

Four new figures are in [the analysis overview](README.md). They reuse the
completed LR sweep, matched schedule branches and three-seed size cohort. No
new training has been executed and the manuscript has not been changed.

## 1. The raw-gradient diagnostic does not rank Adam's operating conditions

The LR-response figure shows that rates change when raw cosine becomes negative,
but late negative alignment appears at every tested rate. Its magnitude is
not monotone in LR or validation loss. The highest rate,0.01, has pronounced
raw-norm growth and deteriorating training loss after its initial reduction,
yet its late raw cosine is closer to zero than at several better-performing
rates. The projected-change expression inherits this limitation.

The following comparison stays within the original GB10 cohort. Geometry
columns are unsmoothed medians over updates612--711; validation is the complete
final evaluation at712, so these are different declared reductions.

| Peak LR | Raw consecutive cosine | Projected change | Moment consecutive cosine | Incoming gradient vs. previous moment | Final validation loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0001 | -0.334 | 1.280 | 0.867 | -0.041 | 6.232 |
| 0.001 | -0.432 | 1.425 | 0.916 | 0.049 | 4.764 |
| 0.01 | -0.103 | 1.093 | 0.403 | -0.467 | 6.636 |

The four intermediate RTX4090 runs retain their own hardware-matched comparison.
LR0.004641589 has the lowest final validation loss,4.318091, versus4.410370 at
LR0.002154435, even though its late momentum cosine is lower,0.852 versus0.956.
More directional persistence is therefore not a universal performance target
for either raw gradients or momentum. This is one seed at each rate, not a
robustness or optimal-LR result.

## 2. A useful additional observation is already recoverable

The most informative new panel is **incoming gradient versus previous moment**.
It uses a different comparison from consecutive raw-gradient cosine. With
`h_t` denoting the clipped input, the first-moment recurrence gives

`h_t = (m_t - beta1*m_(t-1))/(1-beta1)`.

Consequently, the recorded moment norms and adjacent cosine determine

`||h_t||^2 = (||m_t||^2 + beta1^2 ||m_(t-1)||^2
             - 2 beta1 <m_t,m_(t-1)>)/(1-beta1)^2`,

and

`<h_t,m_(t-1)> = (<m_t,m_(t-1)> - beta1 ||m_(t-1)||^2)/(1-beta1)`.

Their ratio supplies the input/memory cosine without recovering full gradient
histories. Global positive clipping preserves this cosine, so it also describes
the raw incoming gradient relative to the prior moment. Across4,977 pairs,
reconstructed clipped norms agree with the clipping records to maximum relative
error3.40e-7. Direct-vector checks validate the reconstruction separately.
The reconstructed comparison is additional interpretation of existing scalar
information, not an independent validation of the EMA recurrence.

At LR0.01, the new input opposes stored momentum strongly: late median cosine
-0.467, versus+0.049 at LR0.001. The incoming contribution's norm relative to
the retained contribution,
`(1-beta1)||h_t||/(beta1||m_(t-1)||)`, has late medians1.151 and0.449,
respectively. The new contribution is therefore comparable to the stored one
in the high-rate case, allowing substantial rotation even while adjacent
momentum cosine remains positive. These are separate temporal medians; plugging
them into a nonlinear formula would not reconstruct the median moment cosine.

This refines the proposed norm-propagation story. At LR0.01, the late raw norm
is5.566 but the moment norm is0.097, and517/712 updates are clipped. Large raw
norms do not keep increasing the magnitude entering the EMA once clipping is
active. Opposition to stored history and the relative incoming contribution
help describe why the moment rotates. At LR0.001 the moment grows late while
the incoming signal remains more aligned with its history. These observations
support analyzing the stages jointly; they do not isolate whether changing
clipping or momentum would improve the attained loss.

## 3. The schedule intervention is the strongest existing causal comparison

Unlike comparing independently evolving LR trajectories, the schedule experiment
branches a common trained state within each model size. It shows that the
schedule changes raw-gradient response, rather than merely correlating with it.
It also exposes the limitation of optimizing that response as an end in itself.

| Model | Schedule | Median raw cosine | Median projected change | Final validation loss |
| --- | --- | ---: | ---: | ---: |
| 14M | Original decay | 0.194 | 0.818 | 5.182 |
| 14M | Held rate | -0.358 | 1.373 | 5.067 |
| 14M | Faster decay | 0.331 | 0.668 | 5.218 |
| 70M | Original decay | -0.109 | 1.109 | 4.056 |
| 70M | Held rate | -0.311 | 1.307 | 3.985 |
| 70M | Faster decay | -0.006 | 1.007 | 4.104 |

Medians use affected pairs358--712, including the partial final batch.
The figure's full-sized diagnostic traces stop at711 and its validation
measurements include712. Faster decay makes alignment less negative and
reduces projected change at both sizes. Nevertheless, held-rate branches have
the best final validation loss. Adaptive directions stay positively aligned
under all three schedules. The70M raw recovery is temporary: alignment can
turn negative again while LR stays at its floor. Thus LR helps shape the
response, while the evolving state and sampled batches also matter.

One seed per size and one branch state limit generality. Because LR also scales
decoupled weight decay, the existing experiment identifies the total schedule
effect, not separate mediation by adaptive-step length and shrinkage.

## 4. The size comparison is replicated, but conditional on the recipe

Across all three seed blocks, full-pass median projected change increases with
size. The14M values are0.993,1.003,0.989;31M values1.084,1.031,1.120;70M
values1.153,1.109,1.132. These are711 defined pairs per trajectory. This extends
the visualization of the previously replicated raw-cosine ordering, not an
independent replication test using unrelated information.

The late-window raw cosine is positive for14M and31M but negative for70M in
every seed. Yet70M attains lower validation loss. Adaptive consecutive alignment
and cumulative retention also decrease with size under this recipe. The raw
alignment ordering is phase dependent, particularly in the middle of training;
the seed bands retain that variation. The plots describe different responses
to a common schedule, not a universal dimensionality law or a demonstration
that lower alignment causes the larger model's advantage.

## 5. How to read eta times kappa under Adam

The quantity plotted under the familiar SGD notation is exactly

`1 - rho_t ||g_t||/||g_(t-1)||
 = -<g_(t-1), g_t-g_(t-1)>/||g_(t-1)||^2`.

It combines direction and relative magnitude. Its departure from `1-rho_t`
is precisely `rho_t*(1-||g_t||/||g_(t-1)||)`. For these seven runs, the95th
percentile of that absolute difference over full-sized pairs ranges from0.151
to0.360. The norm factor can matter, but plotting both expressions is not an
independent validation of curvature. Pointwise crossing1 is algebraically
equivalent to a negative raw cosine; there is no independently calibrated
threshold here. Separately smoothed curves need not cross at exactly the same
update, because averaging the norm-weighted cosine differs from averaging cosine.

Under plain Adam, a local expansion substitutes the adaptive direction for the
SGD gradient in the preceding displacement. The deterministic projected term
is then proportional to `g_(t-1)^T H_(t-1) d_(t-1)`, not
`g_(t-1)^T H_(t-1) g_(t-1)`. AdamW adds a decay contribution. Sampling and the
finite-step remainder remain. Even a fixed Hessian can produce a cosine sign
change solely through step length, as the verification example demonstrates.

The adaptive Edge-of-Stability literature instead examines the spectrum of a
preconditioned Hessian under its specified algorithm and assumptions. It does
not justify overlaying an SGD threshold on our projected-change curves.
[Cohen et al., Adaptive Gradient Methods at the Edge of Stability](https://arxiv.org/abs/2207.14484v2).

## 6. The next experiment I recommend

**Question:** Can the joint optimizer diagnostics distinguish circumstances
where reducing LR improves later loss from circumstances where it only restores
directional alignment?

Start with the retained midpoint356 checkpoints for14M and70M at all three
seeds. All six local checkpoint directories exist. Qualify their manifests and
replay before execution. From each state, branch into constant rates
`0.5 x`, `1 x` and `2 x` the current rate for128 updates:18 continuations and
2,304 new optimizer updates. Match weights, moments, scaler/RNG, future data,
batch, precision and evaluation coverage within each checkpoint. Reuse the
pinned source/cache identities; calibration must not alter them.

To isolate the adaptive step-size effect, hold the per-update decoupled shrink
factor fixed across branches. With LR multiplied by `a`, use decay coefficient
`lambda/a` on the same decay parameter groups, so `eta*lambda` is identical.
Keep betas, epsilon and clip norm fixed. This is a deliberate intervention on
step size conditional on common shrinkage, not an unqualified reproduction of
ordinary fixed-decay AdamW tuning. A later practical schedule comparison must
also evaluate the ordinary fixed-decay recipe.

The primary outcome is paired final **full-validation loss**, with all500
documents and target weighting, at matched token budget. Measure loss at the
start and end, and use a declared128-sequence fixed training probe at local
steps0/16/32/64/128 to describe the response. Select the probe deterministically
before seeing these branch outcomes and report its actual valid-target coverage.
Its results concern a subset, not the full training objective.

Capture raw and clipped gradients, first moments, adaptive directions and actual
parameter increments, their norms/adjacent cosines/cumulative retention, plus
incoming-gradient/previous-moment alignment and clipping frequency. Retain the
previous vectors and running sums needed for these metrics at every accepted
update; scalar logs and final recovery state alone cannot recreate missing
histories. Preserve optimizer, scaler, RNG/data position, diagnostic buffers and
recovery states at0/64/128. Define undefined denominators as missing and record
numerical failure or nonfinite loss as an outcome rather than deleting the arm.

Compare early diagnostic changes over local steps1--16 with later loss change
over16--128, alongside the recent-loss/current-rate baseline. Do not treat
adjacent steps as independent samples. With only six checkpoint blocks, this
is a controlled mechanism pilot, not a validated predictor or scheduler.
The LR0.01 checkpoint from Analysis009 can be an additional stress case if
its retained state qualifies; report it separately as one initialization and
a different parent history, not as replication of the size cohort.

**If curvature is part of the intended claim**, add a compact same-state
one-step check at those checkpoints. Compute the gradient on the same fixed
probe before and after each candidate update; predict its projected change
using `-g_probe^T H_probe delta_theta / ||g_probe||^2`, with the Hessian action
measured at the starting state. Compare this independently measured prediction
with the actual finite-step response and show its residual across step sizes.
This tests a local approximation rather than comparing an algebraic definition
to itself. It measures one projected response on a declared subset, not the
largest full-objective eigenvalue or an Adam stability boundary.

This experiment is a proposal only. GPU choice, live price, diagnostic overhead,
measured runtime, spending/duration cap and retention/cleanup must be qualified
before any billable launch. No historical rental price is used as a new estimate.

If the immediate goal is instead to strengthen the existing schedule claim for
publication, the simplest replication is to repeat Analysis004's three schedules
at both sizes for seeds1234 and1236:12 additional356-update continuations,
4,272 updates. That preserves the original question and adds two independent
seed blocks, but does not itself establish diagnostic-informed decisions or
isolate the decay contribution. These are alternative priorities, not two
automatically combined launch requests.

## 7. Positioning the contribution

A defensible current contribution is a practical account of **where directional
information changes between the sampled gradient and the adaptive update**, and
why loss must accompany those diagnostics. The reconstructed input/memory
comparison makes the momentum explanation more concrete without new training.
The schedule experiment supplies an intervention, and the size cohort supplies
replication under a common recipe. These three roles form a coherent argument.

The existing [Analysis025](../025-2026-10-17-pythia-descent-mechanism/REPORT.md)
already provides norm-matched preconditioning ablations; it should be used
before commissioning a duplicate optimizer-ablation study. Its conclusions
are conditional on its fixed probe and finite continuation, not universal
curvature statements.

Alignment-based learning-rate adaptation is already an active prior-art area:
[GALA](https://arxiv.org/abs/2506.08419) uses consecutive gradients and a local
curvature estimate, including empirical Adam comparisons. A later scheduling
contribution therefore needs an appropriate existing alignment-based baseline
as well as held/cosine rates, and evidence that the joint diagnostics improve
held-out decisions. Simply reducing LR whenever raw cosine is negative would
not be a strong novelty claim and is not supported by our loss comparisons.

Work on [Adam preconditioner dynamics and loss spikes](https://arxiv.org/abs/2506.04805v2)
also motivates separating moments, preconditioning and loss response. Its
mechanism must not be assigned to our trajectories from similar-looking curves;
we have not measured the required second-moment/curvature relation here.

The recommended order is therefore: present the existing characterization,
test a targeted counterfactual response, and develop a scheduling rule only
after its additional predictive or decision value is demonstrated.
