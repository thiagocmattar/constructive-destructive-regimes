# Run057: Rosenbrock with Hessian, LR 0.001

Completed and verified, 2026-10-11: 2,000 CPU float64 AdamW updates from (0,1).
Betas(0.9,0.95), epsilon 1e-8, decoupled decay 0.1, exact gradients, no clipping,
constant LR; no data/seed/cloud cost. Final f=1.10471356953e-05;
lambda_max=996.329309.

[Analysis016](../../analyses/016-2026-10-11-rosenbrock-hessian/README.md) owns
the method and both comparison PDFs. All original arrays are bitwise identical
to [040](../../runs/040-2026-10-10-rosenbrock-adamw-new-start-lr-1e-3/README.md).
[Config](config.json), [verification](verification.json),
[summary](artifacts/attempts/001-local/summary.json),
[manifest](artifacts/attempts/001-local/manifest.json),
[inventory](artifacts/attempts/001-local/inventory.json),
[final state](artifacts/attempts/001-local/final_state.json).

The complete ignored history retains 2,001 positions, full Hessians and both
eigenvalues, plus every original vector/geometry diagnostic: 1,039,279 bytes.
Frozen attempt 001-local refuses overwrite. [Observation001](observations/001-curvature.md).
