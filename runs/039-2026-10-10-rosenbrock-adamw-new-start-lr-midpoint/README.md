# Run039: Rosenbrock AdamW from (0, 1), LR 0.0031622776601683794

Completed and verified on 2026-10-10, exactly 2,000 local CPU updates in
`001-local`. Final objective 0.000117958118659, point (0.989636698, 0.979705757),
post-gradient norm 0.162892131.

H012 changes only the initial point from Analysis012 to the user's (0,1).
f(x,y)=(1-x)^2+100(y-x^2)^2; betas (0.9,0.95), epsilon 1e-8, decoupled
weight decay 0.1, constant LR 0.0031622776601683794, zero initial moments,
CPU float64, analytic gradients, no clipping. No data/seed or cloud billing.

[Analysis013](../../analyses/013-2026-10-10-rosenbrock-adamw-new-start/README.md)
owns the matched comparison with [036](../../runs/036-2026-10-10-rosenbrock-adamw-lr-midpoint/README.md)
and the single nine-panel figure. [Config](config.json) and
[numerical implementation](../../src/controlled/rosenbrock.py) are frozen;
`01_run.py` calls the unchanged shared implementation. Do not overwrite its attempt.

[Observation001](observations/001-trajectory-geometry.md), [verification](verification.json),
[manifest](artifacts/attempts/001-local/manifest.json), [summary](artifacts/attempts/001-local/summary.json),
[inventory](artifacts/attempts/001-local/inventory.json) and [final state](artifacts/attempts/001-local/final_state.json)
retain provenance. Complete ignored NPZ history: 936,348 bytes;
all 2,001 positions/losses and 2,000 raw/momentum/adaptive/displacement vectors,
second moments and exact prefix geometry are retained.
