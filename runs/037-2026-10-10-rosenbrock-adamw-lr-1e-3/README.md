# Run037: Rosenbrock AdamW, LR 1e-3

Completed and verified on 2026-10-10, exactly 2,000 local CPU updates in
`001-local`. Final objective 0.234111864754, point (0.516148946, 0.266394818),
post-gradient norm 0.964627031.

H011: same method as Analysis011 with momentum and decoupled weight decay 0.1.
f(x,y)=(1-x)^2+100(y-x^2)^2, start (-1.2,1), betas (0.9,0.95), epsilon 1e-8,
constant LR 1e-3, CPU float64, analytic gradients and no clipping.
Both coordinates receive shrinkage; the objective and raw gradient contain no
L2 penalty. No data, stochastic seed, validation split, cloud resource or billing.

[Analysis012](../../analyses/012-2026-10-10-rosenbrock-adamw-lr/README.md)
owns the matched method, comparison and single nine-panel figure. [Config](config.json)
and [implementation](../../src/controlled/rosenbrock.py) are frozen at execution.
`01_run.py` calls that unchanged shared implementation; do not overwrite its attempt.

[Observation001](observations/001-trajectory-geometry.md), [verification](verification.json),
[manifest](artifacts/attempts/001-local/manifest.json), [summary](artifacts/attempts/001-local/summary.json),
[inventory](artifacts/attempts/001-local/inventory.json) and [final state](artifacts/attempts/001-local/final_state.json)
retain provenance. Complete ignored NPZ history: 981,323 bytes;
all 2,001 positions and 2,000 raw/momentum/adaptive/displacement vectors are
retained, together with second moments and exact prefix geometry.
