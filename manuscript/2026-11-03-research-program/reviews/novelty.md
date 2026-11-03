# Temporal geometry and novelty assessment

Read-only review, 3 November 2026. No experiments or cloud actions. This assessment reads the current formulation and Analysis032–035 reports, and checks the primary sources linked below. It proposes hypotheses; it does not claim a novel result has already been demonstrated.

## Recommended scientific identity

The ambitious, testable contribution is **a training-native account of when oscillation remains productive, when it develops a growing envelope, and when curvature learned from past motion can safely inform the current adaptive direction**. Its empirical value must be demonstrated through decisions or calibrated geometry prediction. “Longer-lag cosine plots are new” is not a sufficiently defensible claim.

This strengthens rather than discards the existing work. The historical observations motivate a new matched data collection designed around observability and intervention, not another catalogue of curves. The paper can retain cheap one-gradient monitoring as its minimal diagnostic and separately quantify the added cost/value of a short history.

## Important prior overlap

| Source | What it already establishes | Implication |
| --- | --- | --- |
| [Cohen et al., Adaptive Gradient Methods at the Edge of Stability](https://arxiv.org/html/2207.14484v2) | Full/large-batch Adam exhibits adaptive EoS and explosive oscillation/preconditioner feedback cycles. The beta1-dependent threshold is not the plain-GD threshold. | Productive/unstable oscillation and adaptive compensation are established background. |
| [Song and Yun, NeurIPS 2023](https://arxiv.org/html/2307.04204v2) | Trajectory alignment through bifurcation theory, with persistent period-two dynamics in specific models. | Negative odd/positive even patterns are expected manifestations of known two-cycle dynamics. |
| [Bai et al., ICML 2026](https://proceedings.mlr.press/v306/bai26o.html) | Second-moment decoupling, spike stages, and gradient-directional curvature prediction. | Broad “curvature explains Adam spikes” positioning directly overlaps. |
| [Lee and Lee, Greedy Alignment Principle for Optimizer Selection, v3 May 2026](https://arxiv.org/html/2512.06370v3) | The optimizer is treated as a causal gradient filter. Expected gradient/update alignment is a filter–autocorrelation inner product, yielding dynamic momentum choices. | Gradient autocorrelation and history-guided optimizer tuning are already explicit. Compare against this account before claiming a first temporal view. |
| [Regis and Chewi, Rod Flow for Adam, May 2026](https://arxiv.org/html/2605.06821v1) | Separates the center and extent of oscillatory motion in joint parameter/moment space; treats second moment as an auxiliary state. Period doubling beyond two cycles is a stated limitation. | A slow-drift/oscillation decomposition has close prior structure. Potential value lies in cheap observables, validity diagnostics and intervention, not renaming the decomposition. |
| [Liu and Nocedal, L-BFGS](https://link.springer.com/article/10.1007/BF01589116) | Limited-memory curvature updating for large-scale optimization. | Storing old step/gradient-difference pairs to approximate Newton is classical. |
| [Fang and Saad, multisecant methods](https://onlinelibrary.wiley.com/doi/10.1002/nla.617) | Quasi-Newton/multisecant constraints and their relation to Anderson mixing. | A low-rank S,Y update is not a novel method merely because its input comes from the current diagnostic. |
| [Byrd et al., stochastic quasi-Newton](https://users.iems.northwestern.edu/~nocedal/PDFfiles/stochBFGS.pdf) | Noisy gradient differences can damage curvature updates; spaced/subsampled derivative measurements improve robustness. | Sampling qualification and curvature-pair quality must be part of any stochastic extension. |

These are specific overlaps, not an exhaustive novelty review. No source establishes that the proposed reliability/decision experiment will succeed.

## What the existing temporal evidence does and does not show

Analysis032 has 17/17 primary lag-one reversing profiles with negative odd/positive even cosines through lag 10. Analysis033 extends exact anchored profiles, with damping and rebound rather than uniform attenuation. Of 17 initially reversing primary profiles, only four have the terminal local half-gap at most 0.1; 13 remain above it. The early Transformer LR0.2 gap rises from 0.225 at lag512 to 0.600 at1024 and ends0.326 at2048.

These are interesting observations of persistence and nonstationarity. They come from one initialization, one fixed reference per curve, unequal outcome-dependent longer horizons, and sometimes substantial replay drift. An anchored row is not a stationary autocorrelation estimate or a full temporal Gram matrix. Cosine removes amplitude; a persistent angular pattern can coexist with a shrinking oscillation. Neither long-lag attenuation nor low retention identifies useful progress without loss/outcome measurement.

The opportunity is to turn these plots into a falsifiable account that predicts where a cheap curvature observation is useful and where it fails.

## A precise observability boundary

Let actual past steps be S=[s1,...,sk] and full-gradient differences Y=[y1,...,yk]. For a constant symmetric Hessian H, Y=HS. For a proposed current adaptive direction d=Sa+r with r orthogonal to span(S),

```text
d^T H d = a^T S^T Y a + 2 a^T Y^T r + r^T H r.
```

The first two terms are determined by these secants. The residual term is not. Any symmetric A satisfying AS=0 leaves Y unchanged, while changing d^T(H+A)d whenever d has a suitable component outside span(S). This is an elementary identifiability statement, not a claimed new theorem.

Consequences:

- A scalar raw rho does not identify current Adam directional curvature.
- Longer history can improve directional coverage, but a near-period-two history may be almost rank one: many lags can add little independent curvature information.
- Direction coverage `||r||/||d||`, the effective rank/conditioning of S, and the age/inconsistency of repeated secants are natural validity signals.
- For a nonlinear objective, different y_i correspond to different path-averaged Hessians. Remaining in span(S) therefore does not eliminate curvature drift error.
- For sampled gradients, Y also contains sampling differences. Common-probe or complete-objective derivative checks are needed to distinguish geometry from sampling.

A useful paper could empirically establish when these validity signals predict the error in transporting past secant information to the current Adam direction, and whether respecting that boundary improves interventions. This is more defensible than claiming a new approximate Newton formula.

## Three decisive experiment packages

### 1. Oscillation and curvature-validity experiment

Collect new matched trajectories with independent initializations at predeclared benign oscillatory and spike-producing settings. Do not select only histories that develop spectacular peaks. Start with the existing MLP/ResNet/Transformer objectives for controlled measurement, then add a realistic AdamW setting for confirmation.

At multiple predeclared windows capture raw gradients, Adam moment/direction and actual increments. A rolling history of 8–16 vectors is enough for the first test; 256 vectors are not required online. Retain selected exact longer profiles as scientific context. Record norm-weighted geometry, not only normalized cosine:

- lag-one and lag-two alignment;
- energy and growth of `(g_t-g_(t-1))/2`, with paired-step drift `(g_t+g_(t-1))/2`;
- the analogous actual-step quantities, kept distinct;
- short-history Gram rank, directional coverage of the current Adam proposal and secant age/inconsistency.

The two-component arithmetic is an explanatory probe, not proof that a component causes learning. Validate its association with actual paired-step objective progress and separately selected HVP measurements. On a frozen quadratic, use analytic H to test the exact identifiability boundary and deliberately rotate the current direction outside the history subspace.

For neural models, use sparse, independently evaluated current-direction HVP targets. Compare a last secant/BB scalar, raw-gradient response, actual-step secant and a standard small-memory curvature approximation. Test whether temporal validity features improve held-seed curvature-error calibration or failure detection beyond current norm/growth/rho and simple direction cosine. Use complete ordinary-training states, not just events. Keep pointwise curvature, path average and realized loss response as distinct targets.

**Falsifier:** if the extra history adds no stable value beyond the current pair/direction comparison, long lags remain an explanatory figure rather than a new controller component. If error is mostly direction novelty rather than temporal coherence, make directional coverage the account.

### 2. Matched action and mechanism experiment

Use common complete optimizer checkpoints from both growing-envelope states and benign reversal states. Include unchanged Adam, fixed LR reduction, raw-input clipping, actual-step clipping and the proposed curvature-based step rule with declared fallback. An Armijo or short line-search reference supplies an expensive reference for the loss-versus-step tradeoff, not a new algorithmic novelty claim.

The central test is whether suppressing growing excursions preserves productive drift better than simply reducing all steps. Compare the exact same action with a norm-only gate, a current-pair gate and a history/validity gate after development. Match calibration/selection effort. Report subsequent mean/final loss, time-to-target, excursion area/maxima, delayed spikes, intervention counts and full cost.

For a direction-changing or subspace-selective treatment, include a native-direction control rescaled to the treatment's step norm. This is essential because a better result might come entirely from smaller updates. A random/subspace placebo may be useful if selective suppression is the proposed mechanism, but should not expand the pilot before the primary effect is visible.

**Falsifier:** if norm-matched scalar reduction reproduces the gain, report a scale effect. If spikes shrink but subsequent progress worsens, report a tradeoff. Do not call any action useful solely because peaks disappear.

### 3. Frozen prospective policy confirmation

Once one hypothesis survives development, freeze its statistic, window, threshold, step rule and fallback. Run complete new-seed trajectories in stochastic AdamW with schedules/clipping/decay as appropriate. Retain no-spike histories and count unnecessary interventions. Compare fixed recipe, norm/loss policies and the selected geometry policy at comparable tuning and intervention budgets.

The key success is an independently replicated improvement in loss/cost tradeoff or substantially better calibrated curvature/action decisions. A large model with one seed is weaker confirmation than independent moderately sized runs. If the paper claims an efficient method, measure full-step time and memory for the actual history used; the original one-gradient complexity does not cover a 16-vector implementation.

## Suggested argument hierarchy

1. Established secant observation and a clear scope boundary.
2. Observable distinction between stable alternating motion, growth of its envelope and slow progress, using new controlled data.
3. Reliability of transporting curvature across adaptive directions and changing states.
4. Counterfactual intervention showing whether that reliability information helps beyond norm/step-scale controls.
5. Frozen realistic confirmation of the winning claim.

This is ambitious enough to create a distinct paper if the decisive tests succeed. It also has clear stopping decisions if the temporal or curvature-transfer hypotheses fail. Acceptance or freedom from novelty criticism cannot be guaranteed by the design alone.
