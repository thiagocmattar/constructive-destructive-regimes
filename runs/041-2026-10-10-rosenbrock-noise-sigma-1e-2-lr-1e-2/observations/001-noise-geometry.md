# Observation001: noisy raw/momentum geometry

H013, sigma=0.01, one noise seed1234, LR 1e-2.
Raw is the stochastic estimator used by AdamW; signal is the exact analytic
gradient on the same adapted path. Final f=0.0011049362662; first joint
true loss/gradient threshold hit=None,
sustained final100=False.

| Stream | Final A | Final B | Final B/A | Late mean cosine |
| --- | ---: | ---: | ---: | ---: |
| Raw | 9036.09021 | 7415.02175 | 0.820600677 | -0.520474605 |
| Momentum | 8054.65868 | 7414.80854 | 0.920561483 | 0.660936679 |

Sources: [summary](../artifacts/attempts/001-local/summary.json),
[inventory](../artifacts/attempts/001-local/inventory.json), full ignored NPZ
history SHA-256 `38c388591f67e642998bdaf6490de986d696fa3e1fecb26f1c69abb29bb301c8`, [verification](../verification.json).
Coverage: both coordinates; all2,001 positions and2,000 vectors per stream,
with1,999 adjacent pairs. A=sum norms; B=norm vector sum; ratio B/A; undefined
first cosine/zero vectors and A=0 ratios receive no epsilon. Table statistics
are unsmoothed; late means use the final1,000 updates. No clipping or thinning.

The [owning comparison observation](../../../analyses/014-2026-10-10-rosenbrock-gradient-noise/observations/002-sigma-1e-2.md)
links the navigation/metric and identity PDFs and their generating scripts,
sources, reductions and captions. Cosine curves use100-update trailing means;
norm/A/B Y is linear. Every noise draw, true gradient, optimizer direction,
moment and displacement is retained. Identities are checked pathwise including
signal/noise cross terms. One realization supplies no sampling-error estimate,
universal noise/LR ordering, convergence guarantee or LLM conclusion.
