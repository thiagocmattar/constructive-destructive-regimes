# Observation001: noisy raw/momentum geometry

H013, sigma=10, one noise seed1234, LR midpoint.
Raw is the stochastic estimator used by AdamW; signal is the exact analytic
gradient on the same adapted path. Final f=0.135554976392; first joint
true loss/gradient threshold hit=None,
sustained final100=False.

| Stream | Final A | Final B | Final B/A | Late mean cosine |
| --- | ---: | ---: | ---: | ---: |
| Raw | 48241.1579 | 24221.9507 | 0.502101354 | -0.00960791553 |
| Momentum | 30421.7775 | 24238.0817 | 0.796734568 | 0.810608218 |

Sources: [summary](../artifacts/attempts/001-local/summary.json),
[inventory](../artifacts/attempts/001-local/inventory.json), full ignored NPZ
history SHA-256 `a59c5d3d4e90f3e39bbf96e61b6cb7ea4d265a094e6cc174e53a1ce73c140633`, [verification](../verification.json).
Coverage: both coordinates; all2,001 positions and2,000 vectors per stream,
with1,999 adjacent pairs. A=sum norms; B=norm vector sum; ratio B/A; undefined
first cosine/zero vectors and A=0 ratios receive no epsilon. Table statistics
are unsmoothed; late means use the final1,000 updates. No clipping or thinning.

The [owning comparison observation](../../../analyses/014-2026-10-10-rosenbrock-gradient-noise/observations/005-sigma-10.md)
links the navigation/metric and identity PDFs and their generating scripts,
sources, reductions and captions. Cosine curves use100-update trailing means;
norm/A/B Y is linear. Every noise draw, true gradient, optimizer direction,
moment and displacement is retained. Identities are checked pathwise including
signal/noise cross terms. One realization supplies no sampling-error estimate,
universal noise/LR ordering, convergence guarantee or LLM conclusion.
