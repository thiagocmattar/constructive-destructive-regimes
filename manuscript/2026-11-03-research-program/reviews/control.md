# Secant control design review

Read the five accessible turns of `Define Loss Terms` and the completed `Design spike intervention tests` turn, plus the active project definitions and Analysis038 protocol. No training, infrastructure, manuscript or repository changes were made. The supplied chat contains useful derivations and itself acknowledges close prior work.

## What the proposed optimal step actually estimates

Use `g_t = grad L(theta_t)`, an applied preceding increment `s = theta_t - theta_(t-1)`, and `y = g_t - g_(t-1)`. For a smooth fixed objective,

\[
\widehat\kappa_{\rm sec}=\frac{s^Ty}{s^Ts}
=\int_0^1 u^TH(\theta_{t-1}+\tau s)u\,d\tau,\qquad u=s/\|s\|.
\]

This is a path-average identity, not an exact measurement of the endpoint Hessian. With plain GD it equals the manuscript's raw projection divided by the preceding learning rate. Under Adam, retaining the actual increment makes the directional interpretation unambiguous, including any learning-rate weighting, rounding or decoupled decay.

For a currently fixed descending Adam direction `d_t`, the local quadratic minimizer is

\[
\alpha_t^\star=\frac{g_t^Td_t}{d_t^TH_td_t},
\]

when both numerator and denominator are positive. Replacing the current Rayleigh quotient with the preceding secant gives

\[
\widehat\alpha_t=\frac{g_t^Td_t}{\widehat\kappa_{\rm sec}\|d_t\|^2}
=\underbrace{\frac{s^Ts}{s^Ty}}_{\rm BB1}
\frac{g_t^Td_t}{\|d_t\|^2}.
\]

This is exactly BB1 times a slope/direction correction. It is a transported scalar curvature approximation, not a full Newton step. The missing assumption is that the previous direction/path is representative of current-direction curvature.

Under a full deterministic objective, the completed-step statistic

\[
Q_{\rm step}=-\frac{s^Ty}{s^Tg_{t-1}},\qquad s^Tg_{t-1}<0,
\]

gives `Delta L = (s^T g_(t-1))*(1 - Q_step/2)` exactly for a quadratic. Under nonlinear objectives this is a trapezoidal approximation. Raw `Q > 2` under Adam is a different bilinear projection and cannot inherit this loss-sign criterion.

Both quantities become available after the movement being characterized. Rejecting the next raw gradient does not undo that movement. Feeding zero into Adam decays its moments and can still move parameters. Skipping weights while updating moments is yet another intervention. A rollback/retry controller must restore complete state, rather than silently calling any of these choices equivalent.

## The useful missing question is transport reliability

Assume the Hessian is locally Lipschitz with constant `L_H`, and let `M = ||H(theta_t)||_op`. For current unit direction `v`, a direct bound is

\[
\left|v^TH(\theta_t)v-\widehat\kappa_{\rm sec}\right|
\leq 2M\sqrt{1-(u^Tv)^2}+\frac{L_H}{2}\|s\|.
\]

The first term is the change of one-dimensional subspace; the second is secant-path to endpoint drift. Parallel and antiparallel directions have the same directional curvature. For sampled gradients, add the explicit contamination magnitude `|s^T(xi_t-xi_(t-1))|/||s||^2`. Older secants also need an elapsed-state-distance drift term. The unknown constants mean a gate computed only from cosines is not a certified safeguard.

There is a simple non-identifiability example. For

\[
f_M(x)=x_1^2/2+Mx_2^2/2+x_2,
\]

the segment `(a,0) -> (0,0)` has identical endpoint gradients `(a,1)` and `(0,1)` for every M, and secant curvature one. A current direction along the second coordinate has exact line optimum `1/M`. The transported scalar proposal is one and can increase loss arbitrarily as M grows. More secants help only when their span covers the relevant direction; a long nearly two-cyclic history can still have essentially rank-one steps.

This supports a defensible research question: **can measured temporal/directional coherence identify when cheap historical curvature remains useful for current Adam steps?** It does not establish a new theorem or controller by itself.

## Close prior work verified

- [Krutikov, Tovbis and Kazakovtsev, Algorithms 2025](https://www.mdpi.com/1999-4893/18/5/268): primary indexed text exposes Algorithm4. It uses a unit-normalized moment/preconditioned direction and multiplicative step adjustment according to the endpoint gradient projection. It is a close step-adaptation baseline, not merely standard Adam with a changed scalar LR. Direct page retrieval returned 429, so the exact earlier directional-interpolation equation should be rechecked in the PDF before quoting its algebra.
- [GALA, Jiang, Kavis and Mokhtari](https://arxiv.org/html/2506.08419v1): already combines gradient alignment and local smoothness feedback with Adam. Its practical implementation estimates feedback from same-mini-batch gradients at old/new parameters. The theoretical algorithm uses additional gradients at the old/interpolated points. Its convergence result is for its analyzed normalized-SGD variant; this is distinct from a theorem for practical Adam-GALA. Equal backward-pass/time budgets matter when comparing with a no-extra-backward method.
- [Robles-Kelly and Nazari](https://arxiv.org/abs/2205.13711): combines BB step sizes with adaptive deep-network methods. The arXiv posting is 2022, but the primary record gives a DICTA 2019 publication reference. Cite the correct provenance.

The opportunity is to establish when historical information is actionable and how it avoids damaging intervention while preserving progress. The scalar formula itself has insufficient novelty as the main contribution.

## Proposed candidate and controls

Use the transported spectral proposal as a reference candidate. Retain ordinary Adam moments; scale its complete direction only. Require finite inputs, a nonzero preceding step and current direction, positive current slope and positive secant curvature. A numerical cap, bounded growth and declared fallback are necessary, especially near zero curvature. These restrictions make implementation defined; they do not guarantee loss decrease.

A reliability gate can use current-versus-previous direction agreement and the consistency of several past secants, with thresholds fixed on development data. Compare against the ungated spectral rule to show whether reliability information matters. A cap-only variant tests spike control; a bidirectional variant tests LR adaptation and requires broader progress/tuning comparisons. Do not mix these claims.

The most informative initial experiment is a frozen-state LR-response audit. Sample ordinary constructive, productive-destructive and pre-spike states using criteria independent of the proposed predictor. From each complete state, stage one Adam direction, evaluate candidate scalar steps on the identical objective with full rollback, and compare (i) raw-Q scaling, (ii) actual-step secant prediction, (iii) exact current-direction HVP/quadratic reference at selected states, and (iv) an observed scalar line-response grid. Measure curvature transport error, one-step regret and actual loss increase versus direction rotation, temporal coherence and noise. This tests the missing assumption before paying for a large controller grid.

Then run whole-trajectory policies with new initializations. Minimal informative comparisons are untreated Adam, a fixed conservative LR, norm-triggered action, raw sign/growth action, actual-step spectral action and a reliability-gated version. If claiming competitive adaptive step selection, include GALA and the close H_ADAM baseline with their actual numerical conventions and evaluation costs. Keep the selected Adam moments and initial/data/RNG states matched; changing the gradient input is a separate arm from scaling its final step.

To test timing, calibrate norm/loss/reliability gates to comparable intervention budgets on development histories and freeze them before independent tests. Include ordinary no-spike histories, misses, delayed spikes, censored events, final/mean validation and training loss, time-to-target, actual cumulative step scale, backward/forward counts and measured overhead. Never condition the confirmation cohort on producing a large spike. Use the seed/trajectory as the replication unit.

## A necessary bounded-cycle negative control

Normalized-gradient descent on `f(x)=x^2/2`, step one and initial x=0.25 has the exact cycle `0.25 -> -0.75 -> 0.25`. Gradients alternate `0.25,-0.75`. Raw rho is minus one; adjacent growth is three on every other boundary, so raw Q is four and the sign-plus-growth gate repeatedly acts. The same-parity ratio `||g_t||/||g_(t-2)||` is exactly one, and the two-step loss envelope is constant.

This is a counterexample to equating adjacent amplification with progressive envelope growth. It is a normalized-gradient toy (equivalently Adam beta1=beta2=0 and epsilon=0 with nonzero gradients), not default Adam. Its alternating losses 0.03125/0.28125 DO satisfy the current >10% recovered-excursion definition. Therefore call it a negative control for runaway/envelope claims, not a false positive for that detector. For default Adam, select bounded oscillatory windows using loss-envelope criteria independently of the diagnostic.

The accompanying NumPy script validates the algebra and synthetic controls only. It does not supply neural-network or policy-performance evidence.
