# Run043: Rosenbrock noisy gradients, sigma 0.01, LR 1e-3

Completed and verified on 2026-10-10: 2,000 local CPU float64 updates in
`001-local`. Final f=2.00107862763e-05, point (0.995530874, 0.991062305), true gradient
norm 0.00406639494. Noise seed1234, sigma=0.01,
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
Complete ignored history is 1,529,599 bytes. No cloud billing.
