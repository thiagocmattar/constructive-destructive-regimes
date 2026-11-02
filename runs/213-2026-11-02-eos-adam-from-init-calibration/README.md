# Run213: Adam-from-initialization LR calibration

H027 / [Analysis036](../../analyses/036-2026-11-02-controlled-adam-diagnostics/README.md).
Completed November 2, 2026: all9 arms /1,152 updates verified. Executed inputs are frozen.
The embedded config/input-manifest status strings describe preparation time;
arm manifests and Analysis036's closeout are the authoritative execution record.

Nine independent full-objective arms, 128 updates each (1,152 total):

| Case | Fixed Adam LRs | Initialization |
| --- | --- | --- |
| CIFAR-10 / tanh200-200 MLP, first5k | 1e-4, 3e-4, 1e-3 | Analysis024 seed0 |
| CIFAR-10 / ResNet-32, all50k | 3e-4, 1e-3, 3e-3 | Analysis024 seed0 |
| WikiText-2 / 2-layer Transformer, all2,088,600 training targets | 3e-4, 1e-3, 3e-3 | Analysis024 seed0 |

Fresh Adam beta=(0.9,0.999), epsilon1e-8, no clipping, decay, warmup or schedule.
No seed sweep, Hessian/HVP work or within-run LR changes. One full gradient is
measured at every boundary0..128; the terminal gradient is unapplied. Four
stream prefixes cover exactly128 updates. Calibration is training-only, with no
validation/test evaluation. Recovery at64 plus final128 retains Adam and all
diagnostic prefix state.

This replaces an unexecuted seven-rate x256 proposal. Prior continuation
evidence motivates the smaller grid; it does not guarantee the same regimes
from initialization. Short-horizon labels do not prove stability or convergence.
See [FAST_CALIBRATION.md](../../analyses/036-2026-11-02-controlled-adam-diagnostics/FAST_CALIBRATION.md)
for decisions, bounds, monitoring and later study limits.

- diagnostics.py records raw, stored-first-moment, adaptive and actual-step
  geometry, bias correction and parameter-motion audits.
- 01_train.py executes one task/LR arm, atomic recoveries, bounded stops and
  final byte/SHA inventory.
- 02_classify.py applies declared training-loss labels, preserves failed
  outcomes, and requires the full expected grid for rate selection.
- test_diagnostics.py covers vector math, zero norms, accumulation, placement,
  startup bias and serialization/resume (eleven passing tests).
- input_manifest.json pins sources, prepared data and initial tensor hashes.

Scientific output is ignored under artifacts/attempt-01. Local MLP arms use one
RTX5070Ti laptop; cloud arms use matched RTX5090 class for each case. Cloud
artifacts are retrieved and checked before deleting task workers and disks.
Analysis036 owns compact results, figures, provenance and final billing receipt.

The separate 00_profile_timing.py and scratch/mlp-fast-calibration-smoke are
mechanics checks, excluded from scientific calibration. The smoke uses two full
MLP updates with no split evaluation. Never rerun preparation to overwrite the
executed config/input manifest. Main rates and production IDs remain undecided.

Final outcomes, two reviewed PDFs and billing/cleanup are in
[Analysis036 report](../../analyses/036-2026-11-02-controlled-adam-diagnostics/REPORT.md).
All task cloud workers/storage are deleted after verified retrieval; USD2.26
compute/storage estimate is provisional. Retained local raw artifacts are4.323GB.
No main run was launched.
