# Analysis036: controlled Adam diagnostics from initialization

Prepared November 2, 2026. The user confirmed **start from the same initial
weights as Analysis024**, rather than GD checkpoint forks. This extends the
Analysis029 diagnostic comparison to the controlled CIFAR-10/MLP,
CIFAR-10/ResNet-32 and WikiText-2/Transformer objectives. Every trajectory uses
one constant LR; no schedule or within-trajectory LR intervention.

Status: design and Run213 screening implementation prepared; eleven focused tests
and a two-update full-objective MLP mechanics check pass. No scientific screening
arm, production experiment or cloud job has been launched. The smoke is not
training-regime evidence and does not estimate remote throughput.

[PROTOCOL.md](PROTOCOL.md) defines the comparison, calibration, duration,
measurement semantics, figure plan, limits and remaining execution decision.
[Run213](../../runs/213-2026-11-02-eos-adam-from-init-calibration/README.md)
owns the screening config, code and pinned input manifest. Production rates and
run IDs will be frozen after calibration. Analysis035/Run212 is separate work.

Raw-gradient turning under Adam is observable, but the SGD eta-kappa expression
measures projected gradient change. The full-objective setting removes the
changing-batch sampling term; adaptive step geometry and finite-step effects
still prevent identifying raw Hessian curvature from that scalar. A separately
proposed HVP audit can test the local expansion.

Reproduce preparation:

    python analyses/036-2026-11-02-controlled-adam-diagnostics/01_prepare.py
    python -m pytest runs/213-2026-11-02-eos-adam-from-init-calibration/test_diagnostics.py -q

The input verifier hashes pinned data, initial weights, reference model sources
and scientific dependencies. See [verification.json](verification.json) for
exact checks, mechanics coverage and environment limitations.

The November2 [parallel Runpod estimate](RUNPOD_ESTIMATE.md) uses live pricing,
recorded cloud timings and a separately labeled local Transformer mechanics
profile. Up to seven 4090s project roughly4--8.5h /USD18--36 for screening plus
the proposed main study; suggested aggregate ceiling USD50. Capacity and new
remote throughput remain unqualified; this is an estimate, not a launch.
