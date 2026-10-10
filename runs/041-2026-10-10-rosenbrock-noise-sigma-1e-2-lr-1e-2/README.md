# Run041: Rosenbrock noisy gradients, sigma 0.01, LR 1e-2

Completed and verified on 2026-10-10: 2,000 local CPU float64 updates in
`001-local`. Final f=0.0011049362662, point (0.968648422, 0.937175163), true gradient
norm 0.42689532. Noise seed1234, sigma=0.01,
g=grad f(theta_previous)+sigma*z; z has independent standard-normal coordinates.
Same z sequence across the sweep, zero initial moments and start(0,1).
AdamW: betas(0.9,0.95), epsilon1e-8, decay0.1, constant LR, no clipping.

[Analysis014](../../analyses/014-2026-10-10-rosenbrock-gradient-noise/README.md)
owns the matched comparison, figures and [method](../../analyses/014-2026-10-10-rosenbrock-gradient-noise/METHOD.md).
[Config](config.json), [numerical code](../../src/controlled/stochastic_rosenbrock.py)
and its dependencies are frozen and hash-pinned. Do not overwrite the attempt.
[Verification](verification.json), [summary](artifacts/attempts/001-local/summary.json),
[manifest](artifacts/attempts/001-local/manifest.json), [inventory](artifacts/attempts/001-local/inventory.json)
and [final state](artifacts/attempts/001-local/final_state.json) retain provenance.
Complete ignored history is 1,533,463 bytes. No cloud billing.
