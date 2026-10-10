# Observation001: noisy raw/momentum geometry

H013, sigma=1, one noise seed1234, LR 1e-3.
Raw is the stochastic estimator used by AdamW; signal is the exact analytic
gradient on the same adapted path. Final f=0.0413583044722; first joint
true loss/gradient threshold hit=None,
sustained final100=False.

| Stream | Final A | Final B | Final B/A | Late mean cosine |
| --- | ---: | ---: | ---: | ---: |
| Raw | 78455.4421 | 74136.7327 | 0.944953349 | 0.083546606 |
| Momentum | 77179.1181 | 74138.0158 | 0.960596826 | 0.906363965 |

Sources: [summary](../artifacts/attempts/001-local/summary.json),
[inventory](../artifacts/attempts/001-local/inventory.json), full ignored NPZ
history SHA-256 `838004d833c9e0bcbe09df9f86072a4c6efce97812e255c3d97bdb109d327cbe`, [verification](../verification.json).
Coverage: both coordinates; all2,001 positions and2,000 vectors per stream,
with1,999 adjacent pairs. A=sum norms; B=norm vector sum; ratio B/A; undefined
first cosine/zero vectors and A=0 ratios receive no epsilon. Table statistics
are unsmoothed; late means use the final1,000 updates. No clipping or thinning.

The [owning comparison observation](../../../analyses/014-2026-10-10-rosenbrock-gradient-noise/observations/004-sigma-1.md)
links the navigation/metric and identity PDFs and their generating scripts,
sources, reductions and captions. Cosine curves use100-update trailing means;
norm/A/B Y is linear. Every noise draw, true gradient, optimizer direction,
moment and displacement is retained. Identities are checked pathwise including
signal/noise cross terms. One realization supplies no sampling-error estimate,
universal noise/LR ordering, convergence guarantee or LLM conclusion.
