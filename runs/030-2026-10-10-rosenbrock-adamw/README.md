# Run030: controlled Rosenbrock — Momentum, decay 0.1

Complete, 2026-10-10: 10,000 deterministic local CPU updates; exact replay,
full-horizon independent AdamW arithmetic, every prefix and sizes/SHA-256
passed. [Verification](verification.json) and
[manifest](artifacts/attempts/001-local/manifest.json) retain the evidence.
No cloud resources or new billing.

H009: matched2×2 first-moment/decay pilot. This condition uses beta1=0.9,
decay=0.1; f(x,y)=(1-x)^2+100(y-x^2)^2, start (-1.2,1), LR 0.01 constant,
beta2=0.95, epsilon 1e-8, float64, analytic unclipped gradients, both coordinates.
Scientific [config](config.json) is frozen. The implementation is
[Run028/01_run.py](../028-2026-10-10-rosenbrock-adam/01_run.py); its hash is frozen in the manifest.
No stochastic seed, dataset, sample decomposition or validation split.

Final objective=0.0009731081065340264; final point=(0.9698208200, 0.9397629598). Run030 never meets the declared joint convergence criterion. Late momentum median cosine is +0.632115509, although its objective floor is higher than the matched no-decay Run028. Raw, adaptive and actual displacement remain separately measured.

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
