# Experiments proposed for the revised paper

Start with the missing measurement that most directly tests the discussion:
raw gradients versus the first moment before preconditioning. Develop the
controlled intuition alongside it, then strengthen the intervention evidence.
The designs below are proposed choices for scientific review. They do not
authorize or launch computation.

## Order and expected contribution

| Priority | Work | Deliverable | Decision it informs |
| --- | --- | --- | --- |
| 1 | Audit saved sketches and reconstruct a clipped-gradient EMA where valid | Paired exploratory \(g,c,m,\hat m\) curves with approximation and checkpoint checks | Whether the requested overlay can be obtained locally, and which predictions it supports |
| 2 | Exact quadratic and Rosenbrock examples | Contour paths with loss, coordinate motion, cosine, and \(A,B,R\) | When zigzagging accompanies descent, inefficient progress, or divergence |
| 3 | Replicate lag profiles and the completed LR intervention | Seed-level contrasts with the existing definitions | Whether the main empirical claims survive the current one-seed limitation |
| 4 | Short LR probes from matched checkpoints | Geometry response and subsequent loss under smaller/reference/larger LR | Whether diagnostics predict a useful local intervention |
| 5 | Evaluate a bounded feedback rule if priority 4 supports it | Matched-budget comparison against schedules and relevant diagnostic baselines | Whether a scheduling contribution belongs in this paper |

First-moment and actual-update capture can be included in future approved
replication runs to avoid repeating a training pass solely for a missing
diagnostic. Freeze the resulting measurement plan before execution.

## First moment comparison and the requested overlay

### Scientific comparison

Hold the trained trajectory fixed and compare transformations of its recorded
gradients. The first question is descriptive: how much of the raw/adaptive
coherence contrast is already visible before preconditioning?

Measure the complete sequences \(g_t,c_t,m_t,\hat m_t,d_t\) using the definitions
in [the mathematical core](03-mathematical-core.md). The primary first-moment
overlay uses \(\hat m_t\) to avoid attributing the early zero-initialization
factor to averaging; a companion panel reports literal \(m_t\), as requested.
Compare \(g\) with \(c\) and \(c\) with each moment so clipping is visible.
An auxiliary EMA of unclipped \(g_t\) can isolate filtering on raw gradients,
but must be labeled as an offline filter rather than Adam's actual moment.

Test the discussion's predictions separately:

1. \(A\) is lower for the moment sequence.
2. \(B\) is higher for the moment sequence.
3. \(R=B/A\) is higher and the relative gap \(1-R\) is smaller.

Compare all full prefixes and the declared early/middle/late windows, retaining
exceptions. A smaller \(A\) and similar or smaller \(B\) can still support
increased coherence. A positive adjacent moment cosine can still coexist with
a substantial cumulative gap.

### What can be recovered locally

[Run 002](../../runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/README.md)
has scalar raw/adaptive diagnostics and sparse recovery states, not a full
first-moment history. Its exact proposed overlay cannot be recovered from
the published \(A,B,R\) curves alone.

[Runs 017–019](../../analyses/005-2026-10-04-temporal-lag-profiles/README.md)
retain every raw-gradient sketch, clipping-norm logs, and optimizer states at
initial/midpoint/final checkpoints. Because each map \(S_r\) is linear and fixed,
an ideal EMA can be reconstructed in projected space:

\[
z_t^{(r)}=S_rg_t,\qquad
\widetilde m_t^{(r)}
=\beta_1\widetilde m_{t-1}^{(r)}
 +(1-\beta_1)a_tz_t^{(r)}.
\]

This is a feasible local route to investigate, not a completed measurement.
Derive \(a_t\) from the frozen clipping implementation and recorded returned
norm, including epsilon, dtype, and rounding. Check that all accepted updates,
moment clocks, and parameter scopes match. Use the same replay's raw vectors
as the comparator; these seed-1235 histories are not Run 002's seed-1234 history.

The reconstructed projected EMA is not automatically the projection of Adam's
actual FP32 stored moment: gradient clipping, moment arithmetic, and sketch
accumulation have rounding effects. Moreover, raw-gradient pair calibration
does not guarantee accurate first-moment norms or ratios after cancellation.

At the retained checkpoints, compare reconstructed sketches with projections
of the actual stored first moment, and compare projected moment norms with
exact norms. Use exact small-vector sequences to check the full reduction.
Report map sensitivity and the bias introduced by taking square roots of
projected squared norms. Sparse checkpoint checks cannot certify every
intermediate norm. If the errors are comparable to the claimed differences,
keep the result exploratory and capture the moments directly in a new run.

### Direct capture if necessary

Start with a paired 14M mechanics qualification, then include exact streaming
moment diagnostics in the next approved scientific cohort. Preserve the
existing pinned initialization, tokenizer/data bytes, ordering, batch,
optimizer, precision, and schedule. Read actual moments after each accepted
optimizer update; evaluate diagnostics in FP64 from their FP32 state.

Capture raw and clipped gradients at their proper boundaries, both moments,
adaptive directions, LR/decay components, and actual parameter differences
if the paper makes a displacement claim. Check the ideal AdamW reconstruction
against actual updates and report floating-point residuals. Retain recovery
states and moment/sketch states before launching; a final checkpoint cannot
recover missing histories.

Use one accepted accumulated optimizer update as the time unit, all unique
trainable parameters in fixed order, and a packed sequence of up to 2,048
tokens as the sequence unit. Preserve the actual supervised-token denominator,
the 1,024-sequence full batch, and the final partial batch. Record scalar
diagnostics every accepted update, null reasons for zero denominators, and
measured time/memory overhead. Exact norm identities should be checked from
the recorded vectors, not inferred from similar loss values.

### Figure specification

Use one shared linear axis for four curves \(A(g),B(g),A(\hat m),B(\hat m)\).
Color identifies the vector stream; solid/dashed identifies \(A/B\). Include
\(c\) as a control panel and \(m\) as a bias-correction sensitivity panel.
Do not normalize each curve separately, because that would erase the proposed
attenuation effect.

Pair the overlay with \(R\) and adjacent cosine. Keep \(d\), whose units and
scales are changed by preconditioning, in a separate panel. Mark clipping,
warmup, LR decay, and final partial-batch coverage. If reconstructed, identify
the approximation prominently and show its calibration. Caption all streams
as measurements along the same trajectory.

This comparison does not establish that momentum causes a better final model.
A causal optimizer ablation would require changing the optimizer and following
the resulting different trajectory with matched budgets.

## Controlled optimization examples

The recalled banana-shaped objective is likely the Rosenbrock function
\(f(x,y)=(1-x)^2+100(y-x^2)^2\), with minimum at \((1,1)\).
Use it after an analytically transparent quadratic, so the example does more
than supply an attractive trajectory.

| Choice | Proposed controlled setting |
| --- | --- |
| Objectives | \(f(x,y)=(x^2+\kappa y^2)/2\), initially \(\kappa=100\); then Rosenbrock |
| Initialization | Same point for every comparison, e.g. \((1,1)\) for the quadratic and \((-1.2,1)\) for Rosenbrock; add a small predeclared starting-point sensitivity |
| Main optimizer | Deterministic GD, no clipping, momentum, or weight decay; exact gradients in FP64 |
| Varying factor | Fixed LR spanning small steps, oscillatory descent, and divergence; quadratic candidates \(\eta\kappa=0.5,1.5,1.9,2.1\) |
| Extension | Momentum/Adam as separate comparisons after the GD mechanism is established; match gradient evaluations and initial state |
| Budget | Fix iteration cap and evaluation count before plotting; propose 1,000 steps for the quadratic and 10,000 for Rosenbrock, with declared failure/stopping criteria |
| Measurements | Objective and gradient norm, coordinates, true displacement and path length, \(\rho\), rolling and cumulative \(R\), lag profiles |
| Independent outcome | Objective decrease at matched gradient evaluations and evaluations to a declared objective tolerance |

For the quadratic, distinguish shallow-axis progress from alternating
sharp-axis motion. Include both decreasing-loss oscillation and divergence.
For Rosenbrock, compare paths on the same contour grid and show objective
values alongside them. Fix its LR grid before collecting the comparison
outcomes; the grid is not specified by a global constant curvature threshold.

An optional second stage adds explicitly specified stochastic gradient noise
with common random draws across LR conditions and multiple noise seeds.
Keep deterministic and noisy results separate. In deterministic examples,
different initial points are sensitivity cases, not stochastic seed uncertainty.

Exact mathematics, finite-difference checks, and vector-sum identities provide
mechanics verification. The toy figure supplies controlled intuition; it is
not evidence that the same curvature mechanism explains Pythia's profiles.

## Replication of the current empirical core

For [Analysis 004](../../analyses/004-2026-10-04-lr-schedule-intervention/README.md),
extend the original/held/accelerated continuation comparison to the remaining
two original seed blocks, 1234 and 1236, at 14M and 70M. This proposed extension
has 12 continuations. Keep update 356 as the shared junction, the remaining
tokens through 712, and complete validation at the existing milestones.
Use a new original-schedule control in every block.

For [Analysis 005](../../analyses/005-2026-10-04-temporal-lag-profiles/README.md),
repeat the same three model sizes at the additional seeds, retaining windows
1–128, 293–420, 585–712, primary lags 1–64, secondary all 127 lags, fixed-origin
coverage control, exact calibration, and tail sensitivity. That is six proposed
full histories. Coordinate any new moment/update capture with this cohort.

Within each comparison, match hardware and software, pinned model/data/tokenizer
identities, loss reduction, accumulation, parameter scope, clipping, and
initialization/data-order seeds. Initial weights are paired within the relevant
model/condition comparison; differently shaped architectures cannot have
identical tensors. Quantify numerical replay differences instead of assuming
bitwise reproducibility.

Report per-seed effects, their mean and spread, and all reversals. Three seeds
remain limited evidence. Gradient pairs and projection maps must not be used
as independent training replicates for narrow confidence intervals.

## Short learning rate probes

The practical hypothesis is that a short response to an LR perturbation helps
choose a useful local LR. A single sign observed on one run is not yet such a
decision rule.

Start from the existing update-356 recovery states. A later test near update
100 requires explicitly retaining or regenerating the state and all recovery
information there; the old sparse checkpoints do not supply it. Update 100
is a proposed probing location motivated by the pilot, not a universal
transition point.

| Design element | Proposed comparison |
| --- | --- |
| Branches | Multiply the common reference LR sequence by \(0.5,1,2\), subject to predeclared finite bounds |
| Matched factors | Weights, Adam moments and clock, scaler, RNG, data position and future batch/token sequence at the branch |
| Initial probe | 64 accepted updates per branch, with an extension to 128 if the ordering is transient; freeze this rule before observing outcomes |
| Diagnostics | Raw and update cosine; \(A,B,R\) over the branch and causal rolling windows; lag profile; clipping, update norms, and LR |
| Loss outcomes | Same-batch training-loss trajectories plus full 500-document token-weighted validation at the junction and endpoint, including tails |
| Prediction test | Relate the initial diagnostic response to later loss benefit on other declared checkpoints and independent seeds |
| Controls | Compare with phase/LR alone and direct trailing loss; include an unchanged-LR branch and the strongest matched fixed schedule |

The first raw gradient after branching is common because it is evaluated
before the changed update; analyze response after that boundary. The 31-update
centered smoother in Analysis 004 is retrospective and cannot be used online.
Use trailing windows or a causal EMA, with an explicit lag and minimum coverage.

Changing LR also changes AdamW decay. First replicate that operational
intervention faithfully. If the claim becomes specifically about adaptive
step size, add a separate control that holds the per-step decay multiplier
fixed across branches, documenting the altered update rule and group treatment.
Do not silently alter decay in the replication.

Short probes share the expensive prefix and can reduce search cost, but all
branch tokens, validation passes, and diagnostic overhead count in the cost.
A local choice at one checkpoint does not identify the best peak LR from
initialization or the best schedule for a longer budget.

## Feedback scheduling as a conditional extension

A simple candidate to test, not a recommended optimizer, is

\[
\log\eta_{t+1}
=\operatorname{clip}\!\left(
\log\eta_t+\gamma(\bar\rho_t-\rho_\star),
\log\eta_{\min},\log\eta_{\max}\right),
\]

where \(\bar\rho_t\) is a causal trailing statistic and \(\rho_\star\) is a
target learned or fixed using development experiments. This formalizes the
user's direction: increasing LR after unusually positive alignment and
decreasing it after unusually negative alignment. The current evidence does
not establish a target of zero, the right vector stream, or the usefulness
of this rule. A target below zero may be preferable; a sign-only controller
may fail.

Compare a raw-cosine rule, a rule using a declared rolling coherence statistic,
and a relevant hypergradient baseline. Prefix \(R\) or its slope can respond
slowly because the early history dominates; define any rolling statistic and
window before tuning. Ablate diagnostic feedback against the same underlying
schedule without feedback. Include warmup handling, update cadence, bounded
LR changes, hysteresis, undefined-case behavior, and loss-based failure
criteria in the eventual configuration.

Adam's raw-gradient cosine is not the exact hypergradient direction for
its LR: the previous adaptive update direction also matters. The
[prior-work comparison](05-positioning-and-references.md) is necessary before
calling the rule new.

Primary success means better loss at matched tokens, or fewer total tokens
and comparable wall time to reach a declared loss, across independent evaluation
seeds. Count tuning and diagnostics. Keep validation data out of online feedback;
declare any validation-based development and reserve an independent final
evaluation before making performance claims.

If higher coherence fails to improve loss, retain that result. A replicated
diagnostic study with a clearly established limitation can still motivate
future control work; it should not be presented as an effective scheduler.

## Execution records when experiments are selected

New analyses and runs receive the next available independent IDs at execution
planning time; do not reserve or reuse executed IDs here. Place novel code,
configs, artifacts and observations in their owning analysis/run and promote
helpers only when reused. Pin input hashes and measurement coverage before
launch.

Before any future billable work, obtain live pricing and measured calibration,
set the total cost/duration and storage envelope, and specify monitoring,
recovery, retrieval verification, and cleanup using the repository workflow
and installed Runpod router skill. This proposal intentionally quotes no stale
price as a new execution estimate.
