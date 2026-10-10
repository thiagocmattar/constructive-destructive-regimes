# Contribution positioning and primary references

The revised paper should make a specific empirical and methodological
contribution: distinguish temporal horizons and optimizer components, measure
them accurately, and test their relevance to a training decision. The basic
norm expansion, adjacent cosine, momentum smoothing, random projection, and
gradient-based LR adaptation each have prior foundations.

## Focused literature comparison

Primary sources checked on 2026-10-07:

| Work | Relevance | What the revision should do |
| --- | --- | --- |
| [Xing et al., A Walk with SGD, 2018](https://arxiv.org/abs/1802.08770) | Studies consecutive-iteration geometry and interprets motion between valley walls. | Explain what lag-resolved and cumulative measurements add. Do not present zigzagging as a newly discovered phenomenon. |
| [Baydin et al., Online Learning Rate Adaptation with Hypergradient Descent, ICLR 2018](https://arxiv.org/html/1703.04782v3) | Derives online LR adaptation from gradient-based hypergradients, including SGD and Adam; credits Almeida et al. for an earlier procedure. | Compare any proposed feedback rule with this baseline and its normalized variants before claiming novelty or reduced tuning. |
| [Pesme, Dieuleveut and Flammarion, On Convergence-Diagnostic based Step Sizes for Stochastic Gradient Descent, ICML 2020](https://proceedings.mlr.press/v119/pesme20a.html) | Examines a classical diagnostic based on successive stochastic-gradient inner products and its limitations, including quadratic cases. | Discuss why a sign-based diagnostic may fail and what the new lag/cumulative or intervention evidence contributes. |
| [Kingma and Ba, Adam, 2014](https://arxiv.org/abs/1412.6980) | Establishes the first/second moment and bias-correction construction. | Credit the optimizer mechanism; distinguish the new component measurements from inventing temporal averaging. |
| [Weinberger et al., Feature Hashing for Large Scale Multitask Learning, revised 2010](https://arxiv.org/abs/0902.2206v5) | Provides the signed feature-hashing foundation used by the measurement approach. | Present the contribution as calibrated use of fixed sketches for temporal histories, not a new sketching algorithm or unproved uniform guarantee. |
| [Li and Liu, High Probability Guarantees for Nonconvex SGD with Heavy Tails, ICML 2022](https://proceedings.mlr.press/v162/li22q.html) | Reviewer-cited nonconvex SGD theory with optimization/generalization guarantees under stated noise assumptions. | Compare the object and assumptions with descriptive trajectory geometry; do not claim that the proposed identity improves their bounds. |
| [Lei, Towards Better Generalization Bounds of Stochastic Optimization for Nonconvex Learning, author manuscript](https://leiyw.github.io/paper/TPAMI2025.pdf) | Reviewer-cited analysis of nonconvex stochastic optimization and generalization. | Audit the relevant iterate-control statements and final bibliographic metadata before adding a theorem-level comparison. The review alone is not authority for the theorem's scope. |
| [SciPy documentation for the Rosenbrock objective](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen.html) | Confirms the standard banana-valley objective recalled in the discussion. | Use the explicit formula and analytic gradients in the controlled illustration; the benchmark function is not a contribution. |

This is a focused set of directly relevant sources, not a completed systematic
novelty review. In particular, the scheduler literature comparison must precede
an algorithmic novelty claim. The current draft already cites some related
work; preserve correct entries and add verified bibliography records only
during the requested manuscript edit.

## Why the scheduler direction needs more than a cosine rule

For ordinary gradient descent, a one-step derivative of the objective with
respect to the preceding learning rate contains a dot product of successive
gradients. Baydin et al. give this derivation and normalized multiplicative
variants. Consequently, “increase LR for positive alignment, decrease it for
negative alignment” is not enough to establish a new algorithmic contribution.
[Baydin et al., Section 2](https://arxiv.org/html/1703.04782v3)

The proposed differentiator is the empirical distinction between local
cancellation and cumulative structure, an explicit optimizer-component
comparison, and a test of when these measurements improve LR decisions in
the measured language-model setting. This is a possible contribution to
demonstrate, not novelty already established by the present plan.

The existing intervention's disagreement between alignment and loss makes
the question sharper: can a diagnostic distinguish useful large steps from
harmful oscillation? A controller that merely makes cosine more positive
would miss the main limitation already visible in the evidence.

## How to present compressed gradient histories

Add a short methods subsection documenting the fixed map, capture boundary,
parameter order, exact norm denominator, storage, estimator, and calibration.
Place the details and error tables in the appendix. Distinguish three costs:
online projection and exact streaming statistics, stored history, and offline
pairwise reductions. Compression reduces history storage; it does not make
all-pair analysis cost-free.

The current maps occupy about 187 MB per 712-update model history. That measured
storage figure and the calibration values belong to
[Analysis 005](../../analyses/005-2026-10-04-temporal-lag-profiles/METHOD.md).
A runtime-overhead claim needs a matched measurement, and moment reconstruction
needs calibration for its own derived quantities. Exact aggregate closure
cannot validate every lag separately.

## Submission readiness

For a strong characterization paper, aim to establish:

1. A precise object and horizon for every constructive/destructive claim,
   with exact aggregate accounting and calibrated lag measurements.
2. Replication of the central empirical findings and honest reporting of
   hardware/replay sensitivity, rather than treating adjacent pairs as seeds.
3. A useful conclusion about optimizer transformations or intervention
   selection that goes beyond the already known behavior of momentum.
4. A direct comparison with the most relevant diagnostic and scheduling
   literature, including the possibility that the proposed control adds no
   benefit.
5. A fully audited retained manuscript: no unsupported stability/forgetting
   inference, no mislabeled lag-one \(S_3\), and no unresolved estimator issue
   hidden in an appendix.

For a paper that additionally claims an effective scheduler, require a frozen
rule, competitive baselines, independent evaluation, and matched accounting
of tokens, wall time, and tuning. A better-looking geometry curve is insufficient.

These are proposed scientific standards for this revision, not official
TNNLS requirements or a guarantee of publication. A characterization paper
need not invent an optimizer, but its new observations and methodological
value must be clear.
