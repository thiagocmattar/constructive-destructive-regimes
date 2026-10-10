# Run028: controlled Rosenbrock — Momentum, no decay

Complete, 2026-10-10: 10,000 deterministic local CPU updates; exact replay,
full-horizon independent AdamW arithmetic, every prefix and sizes/SHA-256
passed. [Verification](verification.json) and
[manifest](artifacts/attempts/001-local/manifest.json) retain the evidence.
No cloud resources or new billing.

H009: matched2×2 first-moment/decay pilot. This condition uses beta1=0.9,
decay=0; f(x,y)=(1-x)^2+100(y-x^2)^2, start (-1.2,1), LR 0.01 constant,
beta2=0.95, epsilon 1e-8, float64, analytic unclipped gradients, both coordinates.
Scientific [config](config.json) is frozen. The implementation is
[01_run.py](01_run.py); its hash is frozen in the manifest.
No stochastic seed, dataset, sample decomposition or validation split.

Final objective=3.452147280074322e-6; final point=(1.0000148551, 0.9998439167). Run028 reaches the joint convergence thresholds transiently at update 7,678, but fails the final 100 sustained criterion. Its last 1,000 objectives oscillate between2.21917e-9 and1.42761e-4. Raw and stored momentum are frequently adjacent-opposed despite the low objective.

[Trajectory and geometry PDF](figures/trajectory_geometry.pdf) ·
[Local observation](observations/001-trajectory-geometry.md) ·
[Cross-run analysis and explorer](../../analyses/010-2026-10-10-rosenbrock-controlled/README.md) ·
[Full measurement contract](../../analyses/010-2026-10-10-rosenbrock-controlled/METHOD.md).

The owning attempt retains exact history.npz (ignored), summary.json,
resolved_config.json, final_state.json, inventory.json and ignored progress events.
Raw g, uncorrected stored m, adaptive d and actual displacement u are captured
every update with all Analysis009 norm/A/B/B:A/cosine/norm-ratio quantities;
complete pair interaction, lag-one residual and actual loss changes are added.
There is no recovery resume request; every 2D vector and moment remains local.
One start/LR/beta2/function/horizon supports a descriptive controlled outcome,
not universal optimizer ranking, asymptotic convergence or an LLM result.
