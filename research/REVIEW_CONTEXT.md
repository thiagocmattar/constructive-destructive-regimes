# Supplied review context

Orientation recorded on 2026-10-03 from `manuscript/2026-10-03 review/`:
`feedback.md`, `TNNLS_2026_P_48775_reviews_tnnls.pdf`, and `stab.pdf`.
The supplied editor decision is reject. This document summarizes the feedback;
it is not a decision that the reviewers' mathematical assertions are correct.

| Concern | Source | Revision question, still open |
| --- | --- | --- |
| Stability is insufficiently defined | Editor; first report C1/C2 | What exact notion is claimed, and what connects the norm decomposition to it? |
| Correlations beyond consecutive steps are assumed negligible | Editor; first report C3; second report (3) | Is the approximation justified, qualified, or replaced? |
| Online covariance diagnostics are described too strongly | First report C4 | Which exact quantity is costly, and which stochastic approximations are available? |
| Norm-control literature and contribution need clarification | Editor; second report (1)/(2) | How do the claims differ from existing bounded-iterate and concentration analyses? |
| Batch distance may be negative as written | Second report (4), Eq. (30) | Audit the expression, assumptions, estimator, and claimed interpretation. |

The second report cites Lei (2025) and Li & Liu (ICML 2022). These citations were
read in the review but not independently verified during setup. Check primary
sources when addressing them. Do not copy a review's citation metadata or Fisher
equivalence into the manuscript without validating its assumptions.

No paper text was revised in response to these concerns during bootstrap, and
no experiment to address them was selected.
