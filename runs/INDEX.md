# Run registry

`000-template/` is a template, not Run 000.

| Run | Question / hypotheses | Status | Attempts | Observations |
| --- | --- | --- | --- | --- |
| [001](001-2026-10-03-pythia14m-adamw-gradient-baseline/README.md) | AdamW full-pass adjacent/cumulative raw-gradient geometry | Completed; 712 updates; retrieved/verified; cloud closed | Native failure; cloud 1–3 removed; 4 completed | [001: Evolution](001-2026-10-03-pythia14m-adamw-gradient-baseline/observations/001-gradient-evolution.md), [002: Temporal interactions](001-2026-10-03-pythia14m-adamw-gradient-baseline/observations/002-temporal-interactions.md) |
| [002](002-2026-10-03-pythia14m-adamw-adaptive-geometry/README.md) | H001: adaptive Adam direction versus raw-gradient geometry and norms | Completed; 712 updates; retrieved/verified; cloud closed | Attempt 1 calibration retired/retrieved; attempt 2 completed/retrieved | [001: Paired comparison](002-2026-10-03-pythia14m-adamw-adaptive-geometry/observations/001-adaptive-comparison.md) |
| [003](003-2026-10-03-pythia31m-adamw-scale/README.md) | H002: 31M raw/adaptive geometry versus scale | Completed; retrieved/verified; cloud closed | Attempt 1 full pass | [001: Geometry](003-2026-10-03-pythia31m-adamw-scale/observations/001-gradient-geometry.md) |
| [004](004-2026-10-03-pythia70m-adamw-scale/README.md) | H002: 70M raw/adaptive geometry versus scale | Completed; retrieved/verified; cloud closed | Attempt 1 full pass | [001: Geometry](004-2026-10-03-pythia70m-adamw-scale/observations/001-gradient-geometry.md) |
| [005](005-2026-10-04-pythia14m-seed1235/README.md) | H002: 14M, seed 1235 replication | Completed; 712 updates; verified; cloud closed | Attempt 1 | [001: Geometry](005-2026-10-04-pythia14m-seed1235/observations/001-seed-geometry.md) |
| [006](006-2026-10-04-pythia31m-seed1235/README.md) | H002: 31M, seed 1235 replication | Completed; 712 updates; verified; cloud closed | Attempt 1 | [001: Geometry](006-2026-10-04-pythia31m-seed1235/observations/001-seed-geometry.md) |
| [007](007-2026-10-04-pythia70m-seed1235/README.md) | H002: 70M, seed 1235 replication | Completed; 712 updates; verified; cloud closed | Attempt 1 | [001: Geometry](007-2026-10-04-pythia70m-seed1235/observations/001-seed-geometry.md) |
| [008](008-2026-10-04-pythia14m-seed1236/README.md) | H002: 14M, seed 1236 replication | Completed; 712 updates; verified; cloud closed | Attempt 1 | [001: Geometry](008-2026-10-04-pythia14m-seed1236/observations/001-seed-geometry.md) |
| [009](009-2026-10-04-pythia31m-seed1236/README.md) | H002: 31M, seed 1236 replication | Completed; 712 updates; verified; cloud closed | Attempt 1 | [001: Geometry](009-2026-10-04-pythia31m-seed1236/observations/001-seed-geometry.md) |
| [010](010-2026-10-04-pythia70m-seed1236/README.md) | H002: 70M, seed 1236 replication | Completed; 712 updates; verified; cloud closed | Attempt 1 | [001: Geometry](010-2026-10-04-pythia70m-seed1236/observations/001-seed-geometry.md) |
| [011](011-2026-10-04-pythia14m-lr-original-seed1235/README.md) | H004: 14M original LR, seed1235 midpoint continuation | Completed; retrieved/verified; cloud closed | Attempt1 | Analysis004 |
| [012](012-2026-10-04-pythia14m-lr-hold-seed1235/README.md) | H004: 14M hold LR, seed1235 midpoint continuation | Completed; retrieved/verified; cloud closed | Attempt1 | Analysis004 |
| [013](013-2026-10-04-pythia14m-lr-accelerated-seed1235/README.md) | H004: 14M accelerated LR, seed1235 midpoint continuation | Completed; retrieved/verified; cloud closed | Attempt1 | Analysis004 |
| [014](014-2026-10-04-pythia70m-lr-original-seed1235/README.md) | H004: 70M original LR, seed1235 midpoint continuation | Completed; retrieved/verified; cloud closed | Attempt1 | Analysis004 |
| [015](015-2026-10-04-pythia70m-lr-hold-seed1235/README.md) | H004: 70M hold LR, seed1235 midpoint continuation | Completed; retrieved/verified; cloud closed | Attempt1 | Analysis004 |
| [016](016-2026-10-04-pythia70m-lr-accelerated-seed1235/README.md) | H004: 70M accelerated LR, seed1235 midpoint continuation | Completed; retrieved/verified; cloud closed | Attempt1 | Analysis004 |
| [017](017-2026-10-04-pythia14m-temporal-sketch-seed1235/README.md) | H005: 14M temporal sketch, seed1235 | Completed; 712 updates; retrieved/verified; cloud closed | Attempt1 | Analysis005 |
| [018](018-2026-10-04-pythia31m-temporal-sketch-seed1235/README.md) | H005: 31M temporal sketch, seed1235 | Completed; 712 updates; retrieved/verified; cloud closed | Attempt2 | Analysis005 |
| [019](019-2026-10-04-pythia70m-temporal-sketch-seed1235/README.md) | H005: 70M temporal sketch, seed1235 | Completed; 712 updates; retrieved/verified; cloud closed | Attempt2 | Analysis005 |
| [020](020-2026-10-08-dgx-spark-calibration/README.md) | DGX Spark SSH/runtime readiness and 14M/31M/70M full-pass ETC | Completed calibration only; nine updates; compact evidence verified; GPU idle | Attempt1 config compatibility failure before updates; attempt2 completed | [Timing report](020-2026-10-08-dgx-spark-calibration/REPORT.md); approximately 5.0/5.7/7.6 h per pass |
| [021](021-2026-10-08-pythia14m-no-decay-base/README.md) | H007: 14M warmup-only LR 1e-3; raw/momentum/adaptive geometry | Completed; 712 updates; retrieved/verified; PDFs reviewed; remote retained | Attempt1 | [001: Three streams](021-2026-10-08-pythia14m-no-decay-base/observations/001-three-stream-geometry.md); Analysis008 final comparison complete |
| [022](022-2026-10-08-pythia14m-no-decay-small/README.md) | H007: matched warmup-only LR 1e-4 | Completed; 712 updates; retrieved/verified; PDFs reviewed; remote retained | Attempt1 | [001: Three streams](022-2026-10-08-pythia14m-no-decay-small/observations/001-three-stream-geometry.md); Analysis008 final comparison complete |
| [023](023-2026-10-08-pythia14m-no-decay-large/README.md) | H007: matched warmup-only LR 1e-2 | Completed; 712 updates; retrieved/verified; PDFs reviewed; remote retained | Attempt1 | [001: Three streams](023-2026-10-08-pythia14m-no-decay-large/observations/001-three-stream-geometry.md); Analysis008 final comparison complete |
| [024](024-2026-10-09-pythia14m-no-decay-intermediate-1/README.md) | H008: warmup-only LR0.000215443469; g/m/d diagnostics | Completed full pass; compact metrics verified; recoveries downloading | Attempt1; prior infrastructure calibrations preserved | Analysis009 |
| [025](025-2026-10-09-pythia14m-no-decay-intermediate-2/README.md) | H008: warmup-only LR0.000464158883; g/m/d diagnostics | Completed full pass; compact metrics verified; recoveries downloading | Attempt1; prior infrastructure calibrations preserved | Analysis009 |
| [026](026-2026-10-09-pythia14m-no-decay-intermediate-3/README.md) | H008: warmup-only LR0.00215443469; g/m/d diagnostics | Completed full pass; compact metrics verified; recoveries downloading | Attempt1; prior infrastructure calibrations preserved | Analysis009 |
| [027](027-2026-10-09-pythia14m-no-decay-intermediate-4/README.md) | H008: warmup-only LR0.00464158883; g/m/d diagnostics | Completed full pass; compact metrics verified; recoveries downloading | Attempt1; prior infrastructure calibrations preserved | Analysis009 |
| [028](028-2026-10-10-rosenbrock-adam/README.md) | H009: exact Rosenbrock; beta1=0.9, decay=0 | Completed; 10,000 local CPU updates; all histories verified; PDF reviewed | Attempt001-local | [001: Trajectory](028-2026-10-10-rosenbrock-adam/observations/001-trajectory-geometry.md); Analysis010 |
| [029](029-2026-10-10-rosenbrock-no-momentum/README.md) | H009: exact Rosenbrock; beta1=0, decay=0 | Completed; 10,000 local CPU updates; all histories verified; PDF reviewed | Attempt001-local | [001: Trajectory](029-2026-10-10-rosenbrock-no-momentum/observations/001-trajectory-geometry.md); Analysis010 |
| [030](030-2026-10-10-rosenbrock-adamw/README.md) | H009: exact Rosenbrock; beta1=0.9, decay=0.1 | Completed; 10,000 local CPU updates; all histories verified; PDF reviewed | Attempt001-local | [001: Trajectory](030-2026-10-10-rosenbrock-adamw/observations/001-trajectory-geometry.md); Analysis010 |
| [031](031-2026-10-10-rosenbrock-no-momentum-decay/README.md) | H009: exact Rosenbrock; beta1=0, decay=0.1 | Completed; 10,000 local CPU updates; all histories verified; PDF reviewed | Attempt001-local | [001: Trajectory](031-2026-10-10-rosenbrock-no-momentum-decay/observations/001-trajectory-geometry.md); Analysis010 |
| [032](032-2026-10-10-rosenbrock-lr-1e-2/README.md) | H010: Rosenbrock momentum/no decay; LR1e-2 | Completed; 2,000 local CPU updates; all histories verified | Attempt001-local | [001: Trajectory](032-2026-10-10-rosenbrock-lr-1e-2/observations/001-trajectory-geometry.md); Analysis011 single figure |
| [033](033-2026-10-10-rosenbrock-lr-1e-3/README.md) | H010: matched Rosenbrock momentum/no decay; LR1e-3 | Completed; 2,000 local CPU updates; all histories verified | Attempt001-local | [001: Trajectory](033-2026-10-10-rosenbrock-lr-1e-3/observations/001-trajectory-geometry.md); Analysis011 single figure |
| [034](034-2026-10-10-rosenbrock-lr-1e-4/README.md) | H010: matched Rosenbrock momentum/no decay; LR1e-4 | Completed; 2,000 local CPU updates; all histories verified | Attempt001-local | [001: Trajectory](034-2026-10-10-rosenbrock-lr-1e-4/observations/001-trajectory-geometry.md); Analysis011 single figure |
| [035](035-2026-10-10-rosenbrock-adamw-lr-1e-2/README.md) | H011: Rosenbrock momentum/weight decay 0.1; LR 1e-2 | Completed; 2,000 local CPU updates; verified | Attempt001-local | [001: Trajectory](035-2026-10-10-rosenbrock-adamw-lr-1e-2/observations/001-trajectory-geometry.md); Analysis012 single figure |
| [036](036-2026-10-10-rosenbrock-adamw-lr-midpoint/README.md) | H011: matched AdamW; geometric LR midpoint 0.003162277660 | Completed; 2,000 local CPU updates; verified | Attempt001-local | [001: Trajectory](036-2026-10-10-rosenbrock-adamw-lr-midpoint/observations/001-trajectory-geometry.md); Analysis012 single figure |
| [037](037-2026-10-10-rosenbrock-adamw-lr-1e-3/README.md) | H011: matched AdamW; LR 1e-3 | Completed; 2,000 local CPU updates; verified | Attempt001-local | [001: Trajectory](037-2026-10-10-rosenbrock-adamw-lr-1e-3/observations/001-trajectory-geometry.md); Analysis012 single figure |
| [038](038-2026-10-10-rosenbrock-adamw-new-start-lr-1e-2/README.md) | H012: matched AdamW start(0,1); LR1e-2 | Completed; 2,000 local CPU updates; verified | Attempt001-local | [001: Trajectory](038-2026-10-10-rosenbrock-adamw-new-start-lr-1e-2/observations/001-trajectory-geometry.md); Analysis013 single figure |
| [039](039-2026-10-10-rosenbrock-adamw-new-start-lr-midpoint/README.md) | H012: matched AdamW start(0,1); geometric LR midpoint | Completed; 2,000 local CPU updates; verified | Attempt001-local | [001: Trajectory](039-2026-10-10-rosenbrock-adamw-new-start-lr-midpoint/observations/001-trajectory-geometry.md); Analysis013 single figure |
| [040](040-2026-10-10-rosenbrock-adamw-new-start-lr-1e-3/README.md) | H012: matched AdamW start(0,1); LR1e-3 | Completed; 2,000 local CPU updates; verified | Attempt001-local | [001: Trajectory](040-2026-10-10-rosenbrock-adamw-new-start-lr-1e-3/observations/001-trajectory-geometry.md); Analysis013 single figure |
| [041](041-2026-10-10-rosenbrock-noise-sigma-1e-2-lr-1e-2/README.md) | H013: Rosenbrock noisy gradients; sigma 0.01, LR 1e-2, seed1234 | Completed; 2,000 local updates; replay/optimizer/noise identities verified | Attempt001-local | [001: Noise geometry](041-2026-10-10-rosenbrock-noise-sigma-1e-2-lr-1e-2/observations/001-noise-geometry.md); Analysis014 PDFs reviewed |
| [042](042-2026-10-10-rosenbrock-noise-sigma-1e-2-lr-midpoint/README.md) | H013: Rosenbrock noisy gradients; sigma 0.01, LR midpoint, seed1234 | Completed; 2,000 local updates; replay/optimizer/noise identities verified | Attempt001-local | [001: Noise geometry](042-2026-10-10-rosenbrock-noise-sigma-1e-2-lr-midpoint/observations/001-noise-geometry.md); Analysis014 PDFs reviewed |
| [043](043-2026-10-10-rosenbrock-noise-sigma-1e-2-lr-1e-3/README.md) | H013: Rosenbrock noisy gradients; sigma 0.01, LR 1e-3, seed1234 | Completed; 2,000 local updates; replay/optimizer/noise identities verified | Attempt001-local | [001: Noise geometry](043-2026-10-10-rosenbrock-noise-sigma-1e-2-lr-1e-3/observations/001-noise-geometry.md); Analysis014 PDFs reviewed |
| [044](044-2026-10-10-rosenbrock-noise-sigma-1e-1-lr-1e-2/README.md) | H013: Rosenbrock noisy gradients; sigma 0.1, LR 1e-2, seed1234 | Completed; 2,000 local updates; replay/optimizer/noise identities verified | Attempt001-local | [001: Noise geometry](044-2026-10-10-rosenbrock-noise-sigma-1e-1-lr-1e-2/observations/001-noise-geometry.md); Analysis014 PDFs reviewed |
| [045](045-2026-10-10-rosenbrock-noise-sigma-1e-1-lr-midpoint/README.md) | H013: Rosenbrock noisy gradients; sigma 0.1, LR midpoint, seed1234 | Completed; 2,000 local updates; replay/optimizer/noise identities verified | Attempt001-local | [001: Noise geometry](045-2026-10-10-rosenbrock-noise-sigma-1e-1-lr-midpoint/observations/001-noise-geometry.md); Analysis014 PDFs reviewed |
| [046](046-2026-10-10-rosenbrock-noise-sigma-1e-1-lr-1e-3/README.md) | H013: Rosenbrock noisy gradients; sigma 0.1, LR 1e-3, seed1234 | Completed; 2,000 local updates; replay/optimizer/noise identities verified | Attempt001-local | [001: Noise geometry](046-2026-10-10-rosenbrock-noise-sigma-1e-1-lr-1e-3/observations/001-noise-geometry.md); Analysis014 PDFs reviewed |
| [047](047-2026-10-10-rosenbrock-noise-sigma-1-lr-1e-2/README.md) | H013: Rosenbrock noisy gradients; sigma 1, LR 1e-2, seed1234 | Completed; 2,000 local updates; replay/optimizer/noise identities verified | Attempt001-local | [001: Noise geometry](047-2026-10-10-rosenbrock-noise-sigma-1-lr-1e-2/observations/001-noise-geometry.md); Analysis014 PDFs reviewed |
| [048](048-2026-10-10-rosenbrock-noise-sigma-1-lr-midpoint/README.md) | H013: Rosenbrock noisy gradients; sigma 1, LR midpoint, seed1234 | Completed; 2,000 local updates; replay/optimizer/noise identities verified | Attempt001-local | [001: Noise geometry](048-2026-10-10-rosenbrock-noise-sigma-1-lr-midpoint/observations/001-noise-geometry.md); Analysis014 PDFs reviewed |
| [049](049-2026-10-10-rosenbrock-noise-sigma-1-lr-1e-3/README.md) | H013: Rosenbrock noisy gradients; sigma 1, LR 1e-3, seed1234 | Completed; 2,000 local updates; replay/optimizer/noise identities verified | Attempt001-local | [001: Noise geometry](049-2026-10-10-rosenbrock-noise-sigma-1-lr-1e-3/observations/001-noise-geometry.md); Analysis014 PDFs reviewed |
| [050](050-2026-10-10-rosenbrock-noise-sigma-10-lr-1e-2/README.md) | H013: Rosenbrock noisy gradients; sigma 10, LR 1e-2, seed1234 | Completed; 2,000 local updates; replay/optimizer/noise identities verified | Attempt001-local | [001: Noise geometry](050-2026-10-10-rosenbrock-noise-sigma-10-lr-1e-2/observations/001-noise-geometry.md); Analysis014 PDFs reviewed |
| [051](051-2026-10-10-rosenbrock-noise-sigma-10-lr-midpoint/README.md) | H013: Rosenbrock noisy gradients; sigma 10, LR midpoint, seed1234 | Completed; 2,000 local updates; replay/optimizer/noise identities verified | Attempt001-local | [001: Noise geometry](051-2026-10-10-rosenbrock-noise-sigma-10-lr-midpoint/observations/001-noise-geometry.md); Analysis014 PDFs reviewed |
| [052](052-2026-10-10-rosenbrock-noise-sigma-10-lr-1e-3/README.md) | H013: Rosenbrock noisy gradients; sigma 10, LR 1e-3, seed1234 | Completed; 2,000 local updates; replay/optimizer/noise identities verified | Attempt001-local | [001: Noise geometry](052-2026-10-10-rosenbrock-noise-sigma-10-lr-1e-3/observations/001-noise-geometry.md); Analysis014 PDFs reviewed |
| [053](053-2026-10-10-pythia14m-high-lr-0p02/README.md) | H014: Spark warmup-only LR0.02; loss and g/m/d geometry | Completed; 712/712 updates; zero overflows; outputs retrieved/hashed | Attempt01; detached queue | [Analysis015 results/review](../analyses/015-2026-10-10-spark-high-lr/observations/002-completed-results-and-review.md); two PDFs reviewed |
| [054](054-2026-10-10-pythia14m-high-lr-0p04/README.md) | H014: matched Spark warmup-only LR0.04 | Completed; 712/712 updates; zero overflows; outputs retrieved/hashed | Attempt01; detached queue | [Analysis015 results/review](../analyses/015-2026-10-10-spark-high-lr/observations/002-completed-results-and-review.md); two PDFs reviewed |
| [058](058-2026-10-13-cifar10-eos-calibration/README.md) | Calibration only: CIFAR-10 5k tanh MLP, GD/Adam, RTX/Spark, full geometry and Hessian spectra | Complete; 96 short updates; mathematical/state/hash checks pass; no production run | Attempt01-rtx / Attempt01-spark | [Setup, memory and ETC](058-2026-10-13-cifar10-eos-calibration/REPORT.md) |

| [059](059-2026-10-13-cifar10-eos-gd-lr-4e-3/README.md) | H018: CIFAR-10 5k tanh MLP,GD LR0.004; full geometry/Hessian spectra | Complete;20,000 fixed updates; retrieved/hashed/state-verified | Attempt01-spark; detached queue | [Analysis019](../analyses/019-2026-10-13-cifar10-eos-lr/REPORT.md); final train loss0.096876; four PDFs reviewed |
| [060](060-2026-10-13-cifar10-eos-gd-lr-2-over-300/README.md) | H018: CIFAR-10 5k tanh MLP,GD LR0.00666667; full geometry/Hessian spectra | Complete;20,000 fixed updates; retrieved/hashed/state-verified | Attempt01-spark; detached queue | [Analysis019](../analyses/019-2026-10-13-cifar10-eos-lr/REPORT.md); final train loss0.056507; four PDFs reviewed |
| [061](061-2026-10-13-cifar10-eos-gd-lr-1e-2/README.md) | H018: CIFAR-10 5k tanh MLP,GD LR0.01; full geometry/Hessian spectra | Complete;20,000 fixed updates; retrieved/hashed/state-verified | Attempt01-spark; detached queue | [Analysis019](../analyses/019-2026-10-13-cifar10-eos-lr/REPORT.md); final train loss0.032042; four PDFs reviewed |
| [062](062-2026-10-13-cifar10-eos-gd-lr-2e-2/README.md) | H018: CIFAR-10 5k tanh MLP,GD LR0.02; full geometry/Hessian spectra | Complete;20,000 fixed updates; retrieved/hashed/state-verified | Attempt01-spark; detached queue | [Analysis019](../analyses/019-2026-10-13-cifar10-eos-lr/REPORT.md); final train loss0.019452; four PDFs reviewed |
| [063](063-2026-10-13-cifar10-eos-adam-lr-5e-5/README.md) | H018: CIFAR-10 5k tanh MLP,ADAM LR5e-05; full geometry/Hessian spectra | Complete;20,000 fixed updates; retrieved/hashed/state-verified | Attempt01-spark; detached queue | [Analysis019](../analyses/019-2026-10-13-cifar10-eos-lr/REPORT.md); final train loss0.00076582; four PDFs reviewed |
| [064](064-2026-10-13-cifar10-eos-adam-lr-1e-4/README.md) | H018: CIFAR-10 5k tanh MLP,ADAM LR0.0001; full geometry/Hessian spectra | Complete;20,000 fixed updates; retrieved/hashed/state-verified | Attempt01-spark; detached queue | [Analysis019](../analyses/019-2026-10-13-cifar10-eos-lr/REPORT.md); final train loss0.00087757; four PDFs reviewed |
| [065](065-2026-10-13-cifar10-eos-adam-lr-2e-4/README.md) | H018: CIFAR-10 5k tanh MLP,ADAM LR0.0002; full geometry/Hessian spectra | Complete;20,000 fixed updates; retrieved/hashed/state-verified | Attempt01-spark; detached queue | [Analysis019](../analyses/019-2026-10-13-cifar10-eos-lr/REPORT.md); final train loss0.00070753; four PDFs reviewed |
| [066](066-2026-10-13-cifar10-eos-adam-lr-4e-4/README.md) | H018: CIFAR-10 5k tanh MLP,ADAM LR0.0004; full geometry/Hessian spectra | Complete;20,000 fixed updates; retrieved/hashed/state-verified | Attempt01-spark; detached queue | [Analysis019](../analyses/019-2026-10-13-cifar10-eos-lr/REPORT.md); final train loss0.00065939; four PDFs reviewed |

| [067](067-2026-10-14-eos-multitask-calibration/README.md) | Three EOS cases: source/input preparation and production-shaped calibration | Complete; mathematics, residuals and retrieval hashes passed | Owned Spark; no optimizer updates | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); warm gradients0.0127/12.7446/30.7291s |
| [068](068-2026-10-14-eos-frozen-gradient-noise/README.md) | H019 frozen real-data gradient-noise control, three seeds/case | Complete;36 conditions,147,456 observations,37.05min; retrieved/hash checked | Owned Spark;0 optimizer updates | [Analysis020 control](../analyses/020-2026-10-14-eos-multitask/observations/001-frozen-real-data-noise.md); publication figure/report reviewed |
| [069](069-2026-10-14-eos-mlp-gd-full-seed1/README.md) | H019 mlp GD, seed1, batch unitsfull, LR0.01 | Complete500 updates; retrieved/hash verified | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [070](070-2026-10-14-eos-mlp-gd-full-seed2/README.md) | H019 mlp GD, seed2, batch unitsfull, LR0.01 | Complete500 updates; retrieved/hash verified | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [071](071-2026-10-14-eos-resnet-gd-full-seed0/README.md) | H019 resnet GD, seed0, batch unitsfull, LR0.2 | Complete500 updates; retrieved/hash verified | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [072](072-2026-10-14-eos-resnet-gd-full-seed1/README.md) | H019 resnet GD, seed1, batch unitsfull, LR0.2 | Complete500 updates on Spark; local retrieval pending | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [073](073-2026-10-14-eos-resnet-gd-full-seed2/README.md) | H019 resnet GD, seed2, batch unitsfull, LR0.2 | Complete500 updates on Spark; local retrieval pending | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [074](074-2026-10-14-eos-transformer-gd-full-seed0/README.md) | H019 transformer GD, seed0, batch unitsfull, LR0.1 | Complete500 updates; retrieved/hash verified | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [075](075-2026-10-14-eos-transformer-gd-full-seed1/README.md) | H019 transformer GD, seed1, batch unitsfull, LR0.1 | Launched in fixed500-update stage; current status in attempt manifest | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [076](076-2026-10-14-eos-transformer-gd-full-seed2/README.md) | H019 transformer GD, seed2, batch unitsfull, LR0.1 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [077](077-2026-10-14-eos-mlp-sgd-b32-seed0/README.md) | H019 mlp SGD, seed0, batch units32, LR0.01 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [078](078-2026-10-14-eos-mlp-sgd-b32-seed1/README.md) | H019 mlp SGD, seed1, batch units32, LR0.01 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [079](079-2026-10-14-eos-mlp-sgd-b32-seed2/README.md) | H019 mlp SGD, seed2, batch units32, LR0.01 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [080](080-2026-10-14-eos-mlp-sgd-b128-seed0/README.md) | H019 mlp SGD, seed0, batch units128, LR0.01 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [081](081-2026-10-14-eos-mlp-sgd-b128-seed1/README.md) | H019 mlp SGD, seed1, batch units128, LR0.01 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [082](082-2026-10-14-eos-mlp-sgd-b128-seed2/README.md) | H019 mlp SGD, seed2, batch units128, LR0.01 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [083](083-2026-10-14-eos-mlp-sgd-b512-seed0/README.md) | H019 mlp SGD, seed0, batch units512, LR0.01 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [084](084-2026-10-14-eos-mlp-sgd-b512-seed1/README.md) | H019 mlp SGD, seed1, batch units512, LR0.01 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [085](085-2026-10-14-eos-mlp-sgd-b512-seed2/README.md) | H019 mlp SGD, seed2, batch units512, LR0.01 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [086](086-2026-10-14-eos-resnet-sgd-b1-seed0/README.md) | H019 resnet SGD, seed0, batch units1, LR0.2 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [087](087-2026-10-14-eos-resnet-sgd-b1-seed1/README.md) | H019 resnet SGD, seed1, batch units1, LR0.2 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [088](088-2026-10-14-eos-resnet-sgd-b1-seed2/README.md) | H019 resnet SGD, seed2, batch units1, LR0.2 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [089](089-2026-10-14-eos-resnet-sgd-b5-seed0/README.md) | H019 resnet SGD, seed0, batch units5, LR0.2 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [090](090-2026-10-14-eos-resnet-sgd-b5-seed1/README.md) | H019 resnet SGD, seed1, batch units5, LR0.2 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [091](091-2026-10-14-eos-resnet-sgd-b5-seed2/README.md) | H019 resnet SGD, seed2, batch units5, LR0.2 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [092](092-2026-10-14-eos-resnet-sgd-b20-seed0/README.md) | H019 resnet SGD, seed0, batch units20, LR0.2 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [093](093-2026-10-14-eos-resnet-sgd-b20-seed1/README.md) | H019 resnet SGD, seed1, batch units20, LR0.2 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [094](094-2026-10-14-eos-resnet-sgd-b20-seed2/README.md) | H019 resnet SGD, seed2, batch units20, LR0.2 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [095](095-2026-10-14-eos-transformer-sgd-b8-seed0/README.md) | H019 transformer SGD, seed0, batch units8, LR0.1 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [096](096-2026-10-14-eos-transformer-sgd-b8-seed1/README.md) | H019 transformer SGD, seed1, batch units8, LR0.1 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [097](097-2026-10-14-eos-transformer-sgd-b8-seed2/README.md) | H019 transformer SGD, seed2, batch units8, LR0.1 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [098](098-2026-10-14-eos-transformer-sgd-b32-seed0/README.md) | H019 transformer SGD, seed0, batch units32, LR0.1 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [099](099-2026-10-14-eos-transformer-sgd-b32-seed1/README.md) | H019 transformer SGD, seed1, batch units32, LR0.1 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [100](100-2026-10-14-eos-transformer-sgd-b32-seed2/README.md) | H019 transformer SGD, seed2, batch units32, LR0.1 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [101](101-2026-10-14-eos-transformer-sgd-b128-seed0/README.md) | H019 transformer SGD, seed0, batch units128, LR0.1 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [102](102-2026-10-14-eos-transformer-sgd-b128-seed1/README.md) | H019 transformer SGD, seed1, batch units128, LR0.1 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |
| [103](103-2026-10-14-eos-transformer-sgd-b128-seed2/README.md) | H019 transformer SGD, seed2, batch units128, LR0.1 | Held by user; no launch after023 before results review | Owned Spark, detached bounded queue | [Analysis020](../analyses/020-2026-10-14-eos-multitask/REPORT.md); pending cohort retrieval/analysis |

| [104](104-2026-10-15-eos-mlp-gd-lr-0.004-seed1/README.md) | H020 mlp full GD, LR0.004, seed1 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [105](105-2026-10-15-eos-mlp-gd-lr-0.004-seed2/README.md) | H020 mlp full GD, LR0.004, seed2 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [106](106-2026-10-15-eos-mlp-gd-lr-0.02-seed1/README.md) | H020 mlp full GD, LR0.02, seed1 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [107](107-2026-10-15-eos-mlp-gd-lr-0.02-seed2/README.md) | H020 mlp full GD, LR0.02, seed2 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [108](108-2026-10-15-eos-resnet-gd-lr-0.1-seed0/README.md) | H020 resnet full GD, LR0.1, seed0 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [109](109-2026-10-15-eos-resnet-gd-lr-0.1-seed1/README.md) | H020 resnet full GD, LR0.1, seed1 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [110](110-2026-10-15-eos-resnet-gd-lr-0.1-seed2/README.md) | H020 resnet full GD, LR0.1, seed2 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [111](111-2026-10-15-eos-resnet-gd-lr-0.4-seed0/README.md) | H020 resnet full GD, LR0.4, seed0 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [112](112-2026-10-15-eos-resnet-gd-lr-0.4-seed1/README.md) | H020 resnet full GD, LR0.4, seed1 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [113](113-2026-10-15-eos-resnet-gd-lr-0.4-seed2/README.md) | H020 resnet full GD, LR0.4, seed2 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [114](114-2026-10-15-eos-transformer-gd-lr-0.05-seed0/README.md) | H020 transformer full GD, LR0.05, seed0 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [115](115-2026-10-15-eos-transformer-gd-lr-0.05-seed1/README.md) | H020 transformer full GD, LR0.05, seed1 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [116](116-2026-10-15-eos-transformer-gd-lr-0.05-seed2/README.md) | H020 transformer full GD, LR0.05, seed2 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [117](117-2026-10-15-eos-transformer-gd-lr-0.2-seed0/README.md) | H020 transformer full GD, LR0.2, seed0 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [118](118-2026-10-15-eos-transformer-gd-lr-0.2-seed1/README.md) | H020 transformer full GD, LR0.2, seed1 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |
| [119](119-2026-10-15-eos-transformer-gd-lr-0.2-seed2/README.md) | H020 transformer full GD, LR0.2, seed2 | Held for replanning; 500 fixed updates after Analysis020 | Owned Spark; matched LR comparison; no rental |

| [120](120-2026-10-15-eos-resnet-gd-smallest-lr-3k/README.md) | H020 ResNet full GD, LR0.1, seed0,3000 updates | Complete3k;32 files/106.36MB retrieved and hashed; three-case PDF reviewed | Owned Spark; unchanged scientific implementation | [Analysis023](../analyses/023-2026-10-15-eos-smallest-lr-calibration/README.md); Hessians every100, full scalar geometry; no rental |
| [121](121-2026-10-15-eos-transformer-gd-smallest-lr-3k/README.md) | H020 Transformer full GD, LR0.05, seed0,3000 updates | Complete3k;32 files/3.02GB retrieved and hash verified; no subsequent training | Owned Spark; complete prepared corpus each update | [Analysis023](../analyses/023-2026-10-15-eos-smallest-lr-calibration/README.md); no manuscript edits |

| [122](122-2026-10-17-eos-mlp-gd-lr-0.04-3k/README.md) | H020 mlp full GD, LR0.04, seed0,3000 updates | Complete3k;32 files/147.27MB retrieved and hash verified | Owned Spark; no old queue continuation | [Analysis024](../analyses/024-2026-10-17-eos-gd-lr-calibration/README.md); no rental |
| [123](123-2026-10-17-eos-resnet-gd-lr-0.025-3k/README.md) | H020 resnet full GD, LR0.025, seed0,3000 updates | Complete3k;32 files/106.38MB retrieved and hash verified | Owned Spark; no old queue continuation | [Analysis024](../analyses/024-2026-10-17-eos-gd-lr-calibration/README.md); no rental |
| [124](124-2026-10-17-eos-resnet-gd-lr-0.05-3k/README.md) | H020 resnet full GD, LR0.05, seed0,3000 updates | Complete3k;32 files/106.36MB retrieved and hash verified | Owned Spark; no old queue continuation | [Analysis024](../analyses/024-2026-10-17-eos-gd-lr-calibration/README.md); no rental |
| [125](125-2026-10-17-eos-transformer-gd-lr-0.1-3k/README.md) | H020 transformer full GD, LR0.1, seed0,3000 updates | Complete3k;32 files/3.017GB retrieved and hash verified; PDF reviewed | Owned Spark; no old queue continuation | [Analysis024](../analyses/024-2026-10-17-eos-gd-lr-calibration/README.md); no rental |
| [126](126-2026-10-17-eos-transformer-gd-lr-0.2-3k/README.md) | H020 transformer full GD, LR0.2, seed0,3000 updates | Complete3k;32 files/3.017GB retrieved and hash verified; final nine-condition PDF reviewed | Owned Spark; queue exited and GPU idle | [Analysis024](../analyses/024-2026-10-17-eos-gd-lr-calibration/README.md); no follow-up training or rental |

| Run | Question / hypotheses | Status | Attempts | Observations |
| --- | --- | --- | --- | --- |
| [127](127-2026-10-17-pythia14m-descent-baseline-seed1234/README.md) | H021 Pythia-14M, seed1234, baseline350 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [128](128-2026-10-17-pythia31m-descent-baseline-seed1234/README.md) | H021 Pythia-31M, seed1234, baseline350 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [129](129-2026-10-17-pythia70m-descent-baseline-seed1234/README.md) | H021 Pythia-70M, seed1234, baseline350 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [130](130-2026-10-17-pythia14m-descent-baseline-seed1235/README.md) | H021 Pythia-14M, seed1235, baseline350 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [131](131-2026-10-17-pythia31m-descent-baseline-seed1235/README.md) | H021 Pythia-31M, seed1235, baseline350 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [132](132-2026-10-17-pythia70m-descent-baseline-seed1235/README.md) | H021 Pythia-70M, seed1235, baseline350 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [133](133-2026-10-17-pythia14m-descent-baseline-seed1236/README.md) | H021 Pythia-14M, seed1236, baseline350 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [134](134-2026-10-17-pythia31m-descent-baseline-seed1236/README.md) | H021 Pythia-31M, seed1236, baseline350 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [135](135-2026-10-17-pythia70m-descent-baseline-seed1236/README.md) | H021 Pythia-70M, seed1236, baseline350 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [136](136-2026-10-17-pythia14m-no-precond-fork175-seed1234/README.md) | H021 Pythia-14M, seed1234, no-preconditioning50 from175 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [137](137-2026-10-17-pythia14m-no-precond-fork225-seed1234/README.md) | H021 Pythia-14M, seed1234, no-preconditioning50 from225 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [138](138-2026-10-17-pythia31m-no-precond-fork175-seed1234/README.md) | H021 Pythia-31M, seed1234, no-preconditioning50 from175 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [139](139-2026-10-17-pythia31m-no-precond-fork225-seed1234/README.md) | H021 Pythia-31M, seed1234, no-preconditioning50 from225 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [140](140-2026-10-17-pythia70m-no-precond-fork175-seed1234/README.md) | H021 Pythia-70M, seed1234, no-preconditioning50 from175 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [141](141-2026-10-17-pythia70m-no-precond-fork225-seed1234/README.md) | H021 Pythia-70M, seed1234, no-preconditioning50 from225 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [142](142-2026-10-17-pythia14m-no-precond-fork175-seed1235/README.md) | H021 Pythia-14M, seed1235, no-preconditioning50 from175 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [143](143-2026-10-17-pythia14m-no-precond-fork225-seed1235/README.md) | H021 Pythia-14M, seed1235, no-preconditioning50 from225 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [144](144-2026-10-17-pythia31m-no-precond-fork175-seed1235/README.md) | H021 Pythia-31M, seed1235, no-preconditioning50 from175 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [145](145-2026-10-17-pythia31m-no-precond-fork225-seed1235/README.md) | H021 Pythia-31M, seed1235, no-preconditioning50 from225 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [146](146-2026-10-17-pythia70m-no-precond-fork175-seed1235/README.md) | H021 Pythia-70M, seed1235, no-preconditioning50 from175 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [147](147-2026-10-17-pythia70m-no-precond-fork225-seed1235/README.md) | H021 Pythia-70M, seed1235, no-preconditioning50 from225 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [148](148-2026-10-17-pythia14m-no-precond-fork175-seed1236/README.md) | H021 Pythia-14M, seed1236, no-preconditioning50 from175 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [149](149-2026-10-17-pythia14m-no-precond-fork225-seed1236/README.md) | H021 Pythia-14M, seed1236, no-preconditioning50 from225 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [150](150-2026-10-17-pythia31m-no-precond-fork175-seed1236/README.md) | H021 Pythia-31M, seed1236, no-preconditioning50 from175 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [151](151-2026-10-17-pythia31m-no-precond-fork225-seed1236/README.md) | H021 Pythia-31M, seed1236, no-preconditioning50 from225 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [152](152-2026-10-17-pythia70m-no-precond-fork175-seed1236/README.md) | H021 Pythia-70M, seed1236, no-preconditioning50 from175 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |
| [153](153-2026-10-17-pythia70m-no-precond-fork225-seed1236/README.md) | H021 Pythia-70M, seed1236, no-preconditioning50 from225 | Complete; local hash and scientific verification passed | Attempt01 full pass; recovery states retained locally; Runpod resources deleted | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) |

| [154](154-2026-10-23-fixed-ce-calibration/README.md) | H022 full-shape mechanics/timing; seed1000 excluded from inference | Complete; two local calibration attempts | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/README.md) |
| [155](155-2026-10-23-fixed-ce-native32-seed0/README.md) | H022 native32,seed0 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [156](156-2026-10-23-fixed-ce-fixed32-seed0/README.md) | H022 fixed32,seed0 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [157](157-2026-10-23-fixed-ce-ce64-seed0/README.md) | H022 ce64,seed0 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [158](158-2026-10-23-fixed-ce-native32-seed1/README.md) | H022 native32,seed1 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [159](159-2026-10-23-fixed-ce-fixed32-seed1/README.md) | H022 fixed32,seed1 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [160](160-2026-10-23-fixed-ce-ce64-seed1/README.md) | H022 ce64,seed1 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [161](161-2026-10-23-fixed-ce-native32-seed2/README.md) | H022 native32,seed2 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [162](162-2026-10-23-fixed-ce-fixed32-seed2/README.md) | H022 fixed32,seed2 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [163](163-2026-10-23-fixed-ce-ce64-seed2/README.md) | H022 ce64,seed2 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [164](164-2026-10-23-fixed-ce-native32-seed3/README.md) | H022 native32,seed3 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [165](165-2026-10-23-fixed-ce-fixed32-seed3/README.md) | H022 fixed32,seed3 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [166](166-2026-10-23-fixed-ce-ce64-seed3/README.md) | H022 ce64,seed3 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [167](167-2026-10-23-fixed-ce-native32-seed4/README.md) | H022 native32,seed4 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [168](168-2026-10-23-fixed-ce-fixed32-seed4/README.md) | H022 fixed32,seed4 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [169](169-2026-10-23-fixed-ce-ce64-seed4/README.md) | H022 ce64,seed4 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [170](170-2026-10-23-fixed-ce-native32-seed5/README.md) | H022 native32,seed5 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [171](171-2026-10-23-fixed-ce-fixed32-seed5/README.md) | H022 fixed32,seed5 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [172](172-2026-10-23-fixed-ce-ce64-seed5/README.md) | H022 ce64,seed5 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [173](173-2026-10-23-fixed-ce-native32-seed6/README.md) | H022 native32,seed6 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [174](174-2026-10-23-fixed-ce-fixed32-seed6/README.md) | H022 fixed32,seed6 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [175](175-2026-10-23-fixed-ce-ce64-seed6/README.md) | H022 ce64,seed6 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [176](176-2026-10-23-fixed-ce-native32-seed7/README.md) | H022 native32,seed7 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [177](177-2026-10-23-fixed-ce-fixed32-seed7/README.md) | H022 fixed32,seed7 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [178](178-2026-10-23-fixed-ce-ce64-seed7/README.md) | H022 ce64,seed7 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [179](179-2026-10-23-fixed-ce-native32-seed8/README.md) | H022 native32,seed8 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [180](180-2026-10-23-fixed-ce-fixed32-seed8/README.md) | H022 fixed32,seed8 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [181](181-2026-10-23-fixed-ce-ce64-seed8/README.md) | H022 ce64,seed8 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [182](182-2026-10-23-fixed-ce-native32-seed9/README.md) | H022 native32,seed9 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [183](183-2026-10-23-fixed-ce-fixed32-seed9/README.md) | H022 fixed32,seed9 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |
| [184](184-2026-10-23-fixed-ce-ce64-seed9/README.md) | H022 ce64,seed9 | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |

| [185](185-2026-10-30-minibatch-noise-calibration/README.md) | H023 mechanics and diagnostic-workload calibration | Complete; local math, remote complete-objective checks and exact recovery replay pass | Evidence retrieved; task Runpod resources deleted; Spark untouched | [Analysis027](../analyses/027-2026-10-30-minibatch-noise/README.md) |
| [186](186-2026-10-30-mlp-minibatch-noise/README.md) | H023 frozen trained-state noise and matched SGD continuations | Complete; 6k reference + 59,400 continuation updates, 184,320 frozen draws; 22.86 min production | All 14.917 GB production verified locally; cloud deleted; whole task USD 0.87 provisional | [Analysis027](../analyses/027-2026-10-30-minibatch-noise/REPORT.md) |

| [187](187-2026-10-30-pythia14m-seed1234-lr0p5/README.md) | H024 14m-1234 LR0.5 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [188](188-2026-10-30-pythia14m-seed1234-lr1p0/README.md) | H024 14m-1234 LR1 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [189](189-2026-10-30-pythia14m-seed1234-lr2p0/README.md) | H024 14m-1234 LR2 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [190](190-2026-10-30-pythia70m-seed1234-lr0p5/README.md) | H024 70m-1234 LR0.5 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [191](191-2026-10-30-pythia70m-seed1234-lr1p0/README.md) | H024 70m-1234 LR1 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [192](192-2026-10-30-pythia70m-seed1234-lr2p0/README.md) | H024 70m-1234 LR2 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [193](193-2026-10-30-pythia14m-seed1235-lr0p5/README.md) | H024 14m-1235 LR0.5 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [194](194-2026-10-30-pythia14m-seed1235-lr1p0/README.md) | H024 14m-1235 LR1 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [195](195-2026-10-30-pythia14m-seed1235-lr2p0/README.md) | H024 14m-1235 LR2 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [196](196-2026-10-30-pythia70m-seed1235-lr0p5/README.md) | H024 70m-1235 LR0.5 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [197](197-2026-10-30-pythia70m-seed1235-lr1p0/README.md) | H024 70m-1235 LR1 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [198](198-2026-10-30-pythia70m-seed1235-lr2p0/README.md) | H024 70m-1235 LR2 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [199](199-2026-10-30-pythia14m-seed1236-lr0p5/README.md) | H024 14m-1236 LR0.5 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [200](200-2026-10-30-pythia14m-seed1236-lr1p0/README.md) | H024 14m-1236 LR1 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [201](201-2026-10-30-pythia14m-seed1236-lr2p0/README.md) | H024 14m-1236 LR2 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [202](202-2026-10-30-pythia70m-seed1236-lr0p5/README.md) | H024 70m-1236 LR0.5 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [203](203-2026-10-30-pythia70m-seed1236-lr1p0/README.md) | H024 70m-1236 LR1 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [204](204-2026-10-30-pythia70m-seed1236-lr2p0/README.md) | H024 70m-1236 LR2 | Complete;128 updates locally verified | Local artifacts; task cloud deleted; cohort USD13.90 posted, provisional | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/README.md) |
| [205](205-2026-11-01-eos-exact-lag10/README.md) | H020: exact lag-10 profiles near retained GD states | Complete; 27 windows / 270 cosines; 25 primary profiles | One Runpod attempt; 147 files / 7.072 GB locally verified; task cloud deleted | [Analysis032](../analyses/032-2026-11-01-eos-exact-lag10/REPORT.md); compute USD 1.05 provisional |

| [206](206-2026-11-01-eos-long-lags/README.md) | H020: exact long-lag continuation of Analysis032 | Complete; 20 canonical profiles in joint 27; maximum lag 2048 | Verified raw evidence; all task cloud deleted; USD 29.48 provisional joint | [Analysis033](../analyses/033-2026-11-01-eos-long-lags/REPORT.md) |
| [207](207-2026-11-01-eos-parallel-resnet-lags/README.md) | H020: parallel placement of six later ResNet profiles | Complete; baseline 256 before staged extensions; 3 canonical final profiles | Reused 4090 workers; verified retrieval; joint cloud closed | [Analysis033](../analyses/033-2026-11-01-eos-long-lags/REPORT.md) |
| [208](208-2026-11-01-eos-checkpoint-handoffs/README.md) | H020: qualified checkpoint handoffs | Complete; 4 canonical final profiles with exact prefixes | Original gradient tolerance; verified retrieval; joint cloud closed | [Analysis033](../analyses/033-2026-11-01-eos-long-lags/REPORT.md) |

<!-- Analysis034 records -->
| Run | Question / hypotheses | Status | Attempts | Observations |
| --- | --- | --- | --- | --- |
| [209](209-2026-11-01-eos-adam-calibration/README.md) | H026 cold-Adam LR calibration | Complete joint coverage with Run211; 125 canonical arms here, failed/partial originals preserved | All raw evidence locally hash verified; task cloud deleted | [Analysis034](../analyses/034-2026-11-01-eos-adam-continuations/REPORT.md); five rates, training-loss selection only |
| [210](210-2026-11-01-eos-adam-long-lags/README.md) | H026 Adam at matched retained GD states | Complete; 27 exact anchored/adjacent profiles through matched 256/512/2048 | Verified recovery/vectors; all task cloud deleted; USD 39.35 provisional joint | [Analysis034](../analyses/034-2026-11-01-eos-adam-continuations/REPORT.md); 25 primary, two flagged |
| [211](211-2026-11-01-eos-parallel-adam-calibration/README.md) | H026 parallel placement of unchanged Adam arms | Complete; 10 arms at two delayed Transformer states, 64 updates each | Reused warm workers; identical scientific inputs, verified retrieval | [Analysis034](../analyses/034-2026-11-01-eos-adam-continuations/REPORT.md); 135 canonical calibration arms jointly |
<!-- End Analysis034 records -->


<!-- Analysis035 records -->
| Run | Question / hypotheses | Status | Attempts | Observations |
| --- | --- | --- | --- | --- |
| [212](212-2026-11-02-eos-backward-anchors/README.md) | Fixed-endpoint backward raw-gradient cosines under GD and cold Adam | Complete; 18 profiles, lags 1-256, exact FP64 products | 13 canonical workers; 18 physical Pods including five retired before training; all retrieved/verified/deleted | [Analysis035](../analyses/035-2026-11-02-eos-backward-anchors/REPORT.md); two reviewed PDFs; identical starts, later replay drift disclosed; USD 9.26 compute provisional |
<!-- End Analysis035 records -->

<!-- Run213 preparation -->
| Run | Question / hypotheses | Status | Attempts | Observations |
| --- | --- | --- | --- | --- |
| [213](213-2026-11-02-eos-adam-from-init-calibration/README.md) | H027 fixed-LR Adam from matched initial tensors | Completed9 x128 updates; one initialization, no Hessians | Attempt01; all files verified, task Pods/storage deleted, USD2.26 provisional | [Analysis036 report](../analyses/036-2026-11-02-controlled-adam-diagnostics/REPORT.md); three calibration PDFs; main completed in Run215 |
| [215](215-2026-11-02-controlled-full-batch-adam/README.md) | H027 controlled Adam full training from frozen initialization | Complete;9 x1,024 updates, all four streams and full splits verified | Attempt01; three local arms, six production/four pre-production Pods; all retrieved/verified/deleted; USD11.62 main provisional | [Analysis036](../analyses/036-2026-11-02-controlled-adam-diagnostics/README.md); two complete figures, no schedules/Hessians/seed sweep |
<!-- End Run213 preparation -->

Next run: `223-YYYY-MM-DD-slug`. Record calibration-only work as such. Preserve
executed IDs and configurations; retry unchanged scientific inputs in a new
attempt within the same run.

## Active batch-size extension

| Run | Question | Status | Design |
| --- | --- | --- | --- |
| [214](214-2026-11-02-sgd-batch-extension/README.md) | H023 / Analysis037: ResNet batch-size continuations | Complete; 150 forks, three fresh references and GB10 runtime bridge; raw/math acceptance passed | L40S/4090 plus GB10 assistance; all task cloud resources deleted; joint USD 26.30 provisional |
| [216](216-2026-11-02-sgd-transformer-microbatch512/README.md) | H023 / Analysis037: Transformer batch-size continuations | Complete; 198 forks and three fresh references; physical 512 fixed before production; raw/math acceptance passed | Three RTX5090 blocks; cloud deleted; joint three-case figure/manuscript reviewed |

## MLP intervention implementation foundation

| Run | Question | Status | Records |
| --- | --- | --- | --- |
| [217](217-2026-11-03-mlp-adam-control-foundation/README.md) | H028: unchanged Adam control, pre-update observer and immutable forks | Prepared; unit/contract checks only, zero neural training | [Analysis039](../analyses/039-2026-11-03-mlp-control-foundation/README.md); no interventions or spending |

## Adam intervention implementation preparation

| Run | Question | Status | Records |
| --- | --- | --- | --- |
| [218](218-2026-11-03-mlp-adam-step-actions/README.md) | H028: matched local MLP interventions against A0; A7 deferred | Complete8 x128 branches plus657 prefix;64.06s, no rental | [Analysis040](../analyses/040-2026-11-03-mlp-step-actions/README.md); explicit parent identity and immutable branch recovery |

## Temporary LR pilot

| Run | Question | Status | Records |
| --- | --- | --- | --- |
| [219](219-2026-11-03-mlp-temporary-lr/README.md) | H028: temporary half LR using available $\eta\widehat\kappa_t\geq1.8$ versus fixed controls | Complete3 x367 canonical updates; control I/O recovery retained; zero cloud cost | [Analysis041](../analyses/041-2026-11-03-mlp-temporary-lr/REPORT.md); temporary rule inferior, no tuning |

## Cooldown and cosine recovery

| Run | Question | Status | Records |
| --- | --- | --- | --- |
| [220](220-2026-11-03-mlp-lr-cooldown/README.md) | H028: eight-update hold and cosine recovery versus one-step/fixed controls | Complete5 x367 updates; exact replay; UTC audit correction retained; zero cloud | [Analysis042](../analyses/042-2026-11-03-mlp-lr-cooldown/REPORT.md); clean four-arm figure, no timing/generalization claim |

## Overnight fixed-step Spark interventions

| Run | Question | Status | Records |
| --- | --- | --- | --- |
| [221](221-2026-11-03-spark-fixed-step-interventions/README.md) | H029 capacity and warm-parent transfer qualification | ResNet cross-GPU gradient gate failed before treatments; exact tensors/close loss retained | Nine mechanics updates; preserved failed gate and source |
| [222](222-2026-11-03-spark-matched-interventions/README.md) | H029 matched Spark interventions from warm state 1024 | Completed/verified: 30 primary + 6 optional branches; 2,304 main + 12 qualification updates, no failure | [Analysis043 findings](../analyses/043-2026-11-03-spark-fixed-step-interventions/FINDINGS.md); all checkpoints retained, worker exited 06:52 BRT, zero cloud cost |

## Run223: multi-spike interventions (2026-11-04)

[Run223](223-2026-11-04-multispike-interventions/README.md) completed six treatments and three fresh controls in frozen windows512..800/256..576/544..928. Exact replay passed;235 files/7 archives verified. All7 task Pods deleted,0GB cloud storage; estimatedUSD7.58 of15, billing provisional. Analysis044: MLP/Transformer mean loss improves; ResNet does not. Single seed; no historical bitwise replay.

## Run224: MLP return to0.75x at650

[Run224](224-2026-11-04-mlp-lr-return/README.md) completed two matched Spark continuations512..800. Exact history through650 verified; raising LR restores large spikes and worsens post-return mean loss versus held-half. H031 / Analysis045. No cloud rental.

## Run225: MLP cosine70 restoration

[Run225](225-2026-11-04-mlp-cosine-rise70/README.md) completed the user-confirmed cosine rise from 0.5x at650 to0.75x at720, held to800. Exact shared state650 verified; the delayed spike is larger than abrupt restoration. Spark only. H032 / Analysis046.

## Run226: MLP cosine32 to0.75x

[Run226](226-2026-11-04-mlp-cosine75/README.md) completed one matched Spark branch512..800, reaching0.75x at544 and holding. Reuses Run223 unchanged and cosine32-to0.5x. H033 / Analysis047; no control rerun or cloud rental.

## Run227: two-stage MLP cosine decay

[Run227](227-2026-11-04-mlp-two-stage-decay/README.md) completed cosine32-to0.75x from512 then cosine32-to0.5x from650. Exact prior-history replay; retained baseline/single-decay controls. Spark only, H034 / Analysis050.

## Run228: MLP global clipping1.0

[Run228](228-2026-11-04-mlp-global-clipping/README.md) completed288 matched Spark updates512..800 with global gradient clipping1.0 before Adam, unchanged LR. Exact baseline replay passed;125 updates clipped. Mean loss+48.01%, maximum3.82x versus unchanged. H035 / Analysis051.

## Run229: MLP global clipping0.5

[Run229](229-2026-11-04-mlp-clipping-half/README.md) completed288 Spark updates512..800 at global gradient cap0.5 before Adam, unchanged LR. Exact control reuse;238 updates clipped. Mean loss31.03% below clipping1.0 but2.08% above unchanged. H036 / Analysis052.

[Run230](230-2026-11-05-mlp-cosine32-clip1/README.md) paired cosine32 to half LR with global clipping at 1 on the MLP. All 288 updates matched cosine alone.

[Run231](231-2026-11-05-resnet-clip1/README.md) tested ResNet clipping at 1 with a fresh same-Spark control and constant LR 0.003.

## Runs232--233: MLP optimizer interventions (7 November 2026)

[Run232](232-2026-11-07-mlp-beta2-reduction/README.md), H039 / Analysis059,
completed three beta2=.95/.9/.5 branches from exact state512 through800.
All864 updates verified; unchanged control and observer replay exact. Native
bias correction changes with beta2, with warm moments/clocks retained.
Mean training loss+16.78%/+18.16%/+127.02%. Worker12.15s; zero cloud cost.

[Run233](233-2026-11-07-mlp-adam-to-gd/README.md), H040 / Analysis060,
completed288 plain-GD updates at LR.02 from the original untreated state512.
Exact retained-Adam replay and GD resume pass. No loss increases or>10% spikes;
mean loss+2.30% and last32 mean+55.74% versus Adam. Worker3.73s; zero cloud cost.
Both workers exited;80 artifact files verified, audit/recovery/final states
retained locally and on owned Spark. Next run:234.

## Runs234--235: MLP GD rate extension (7 November 2026)

[Run234](234-2026-11-07-mlp-gd-learning-rates/README.md), H040 / Analysis060,
ran GD LR 0.04 then 0.08 from the original Adam state512. LR0.04 completed
288 updates; LR0.08 applied35 before infinite loss at547 (finite through546).
Exact replay/resume and all68 artifact hashes pass; overflow independently
reproduced from recovery544. Worker4.16s; scientific outcome retained as failed
for0.08, with no retuning. Mean0.04 loss is394.45% above Adam.

[Run235](235-2026-11-07-mlp-gd-lr0p03/README.md) adds the user-selected
intermediate0.03 rate:288 updates, worker3.84s, all39 artifacts verified.
Mean loss0.0439842 and maximum0.302456; no independent-seed confirmation.
Analysis060 now shows columns0.02/0.03/0.04; earlier0.08 figure archived.
All task workers exited, recovery evidence retained, zero cloud cost.
Next run:236.

## Run236: MLP Adam spike internals (7 November 2026)

[Run236](236-2026-11-07-mlp-adam-spike-internals/README.md), H041 / Analysis061,
replays535 unchanged Adam updates: all536 loss/norm pairs and full state512
model/optimizer/RNG/gradients match retained control. Captures88 internal states,
15 paired snapshots,512 static parameter hybrids and100 incoming-step probes.
Weight/activation magnitude explosion is not observed; hidden-feature changes
and finite-step overshoot dominate these selected peaks. Worker12.895s; all42
artifacts verified, worker exited, zero cloud cost. Next run:237.

## Run237: MLP gradient amplification (7 November 2026)

[Run237](237-2026-11-07-mlp-gradient-amplification/README.md), H042 / Analysis062,
replays535 unchanged Adam updates with536 exact reference pairs and15 exact
anchors. Captures88 backward/per-example gradient states and87 actual-step,
Adam-denominator and directional-curvature probes. No dynamic normalization or
sudden Adam denominator collapse; spikes accompany reduced gradient cancellation
and growing output oscillations. Worker10.458s,28 artifacts verified, worker
exited, about391MB retained on owned Spark including archive, zero cloud cost.
Next run:238.

## Run238: full-window intra-batch coherence (7 November 2026)

[Run238](238-2026-11-07-mlp-intrabatch-coherence/README.md) extends H042 /
Analysis062 with the requested C=norm(sum gradients)/sum gradient norms.
Exact800-update control replay;801 reference pairs and full512/800 states match.
All353 states448..800 include5000 examples;1,765,000 example energies retained.
Worker10.113s,43 inputs and10 artifacts verified, worker exited, about74MB
retained on owned Spark including archive, zero cloud cost. Next run:239.

## Run239: full-window output-change alignment (7 November 2026)

[Run239](239-2026-11-07-mlp-output-change-alignment/README.md), H042 / Analysis062,
exactly replays the Adam control through800, then advances one unchanged update.
Captures all5000-example output arrays447..801 and353 incoming/outgoing change
cosines for448..800. All801 reference pairs and full512/800 states match;
independent CPU reductions and extra-step reconstruction pass. Worker12.297s,
43 inputs and10 artifacts verified, worker exited, about183MB retained on
owned Spark including archive, zero cloud cost. Next run:240.

## Run240: raw-gradient and hidden/output SVD histories (7 November 2026)

[Run240](240-2026-11-07-mlp-hidden-change-svd/README.md), H042 / Analysis062,
replays the unchanged Adam control through801 and retains all raw gradients,
final hidden activations and outputs447..801. User correction: analyze raw
gradients, not gradient changes. Exact native/control/output qualification
passes;45 input and13 artifact hashes verified. Worker13.766s, exited,4.718GB
retained on owned Spark including archive, zero cloud cost. Next run:241.

## Run241: first-hidden oscillation and transmission (7 November 2026)

[Run241](241-2026-11-07-mlp-first-hidden-oscillation/README.md), H042 / Analysis062,
captures first-hidden activations and second preactivations447..801 on an exact
Adam control replay. All355 gradients/H2/output arrays and full512/800/801
states match prior evidence.47 input and12 artifact identities verified.
Worker18.354s, exited;5.457GB retained including archive on owned Spark,
zero cloud cost. Next run:242.

## Run242: all forward layers and fixed input (7 November 2026)

[Run242](242-2026-11-07-mlp-all-layer-oscillation/README.md), H042 / Analysis062,
adds first-linear preactivations and actual W1/b1 histories. Exact equality
of all355 input matrices, downstream representations and gradients verified;
native/full512/800/801 checks pass.49 source and14 artifact identities checked.
Worker17.615s, exited;4.592GB retained including archive on owned Spark, zero
cloud cost. Next run:243.


## Run243: architecture-wide oscillation tracing (7 November 2026)

[Run243](243-2026-11-07-architecture-instability-tracing/README.md), H043 /
Analysis063, completes unchanged ResNet/CIFAR-10 and Transformer/WikiText-2
continuations with full objectives and fixed activation probes. Exact FP64 SVD
traces cover 131/20 captured representation streams, auxiliary correct-label
readouts and raw gradients. ResNet historical replay is exact; Transformer
maximum relative loss discrepancy is 0.270%. Retained states and transfers are
verified, both cloud workers deleted, Spark calibration workers exited. See the
run for timing, provisional cost and retention. No causal intervention.
Next run:244.

## Run 244: persistent first-linear MLP freeze

[Run 244](244-2026-11-07-mlp-freeze-first-linear/README.md), H044 / Analysis 064:
freeze first weights/biases at 448, keep native Adam moments/clocks and LR0.001,
update later layers through 801. Full 5000-example diagnostics; dense 447-501 and
rolling 447-801 records. All 449 prefix pairs and full historical state 512 match;
first-step remaining parameters/moments match native. 54 input and21 downloaded
artifact identities verified. Worker 14.80s including qualification, peak 406.4MB,
owned Spark, 2.067GB retained, no cloud charge, worker exited. Next run 245.

## Runs 245-246: frozen-first-layer learning-rate interventions

[Run245](245-2026-11-07-mlp-frozen-lr-double/README.md) is I2: freeze from448 plus
LR0.002 at500. [Run246](246-2026-11-07-mlp-frozen-lr-0p02/README.md) is I3: same
freeze and LR0.02 at500 (10x I2). Both branch from the same qualified state500,
continue to801, and compare with unchanged Adam in the continuing Analysis064/065.
I2 retains damping; I3 peaks107.099 at504 and ends800 at0.308132. All66/67 source
identities,22/22 retrieved files, LR-only starting changes, exact first-layer
constancy and independent diagnostics pass. Worker runtimes7.76s/7.96s; peak
456.7MB each. Workers exited,2.021/1.911GB retained on owned Spark, zero cloud cost.
Executed inputs and dense evidence preserved. Next run247; no new analysis.

## Runs247-251: adaptive halving of frozen-MLP learning rate

I4-I8 set LR0.01/0.005/0.0025/0.00125/0.000625 at500, each from the same qualified
full frozen state500, continuing through801. See the ongoing
[Analysis065 observation](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/003-lr-halving.md).
Halving stops at0.000625: all losses500-800 decrease. At0.0025 a delayed peak occurs
at707;0.00125 has only a0.006% initial bump. All67 input identities per run,110
retrieved files, complete fork equality and independent diagnostics verified.
Workers exited;40.62s total,456.7MB peak per GPU worker,10.034GB retained on owned
Spark, no cloud cost. One collection retry, no scientific rerun. Next run252.

## Runs252-257: selective MLP learning-rate controls

H046 / ongoing Analysis065: first-layer-only, global, and later-layer-only rates
at0.5x/0.25x from the identical untreated448 state, no freezing or moment reset.
353 updates per branch; all5000 examples and native Adam settings matched.
[Results and new figure](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/005-selective-layer-lr.md).
Each run passes65 boundary/full512 qualification, exact fork448, native first-step
moments,67 input identities,22 retrieved files and independent dense diagnostics.
Run252 attempt01 failed before treatment due to checkpoint CPU clock aliasing;
evidence preserved, attempt02 corrects restore copying with unchanged science.
All workers exited;49.75s successful execution,400.8MB peak,13.720GB retained on
owned Spark with verified local copies, zero cloud cost. Next run258.

## Runs 258-268: selective MLP update gates and matched norm controls

H047 / ongoing Analysis 065. Run 258 calibrates fixed thresholds using untreated
updates 416-447. Runs 259/260 discard/cap large first-layer Adam coordinates;
Runs 261/262/263 cap per-neuron preactivation RMS at prefix q90/q75/q50.
Runs 264-268 uniformly scale their own first-layer proposals to the respective
selective arm's actual norm schedule. Every branch starts at untreated 448,
ends 801, retains native moments/clocks and LR 0.001 in all layers, and compares
with unchanged Adam. [Methods, results and figures](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/006-selective-update-gates.md).
Neuron q75 has the best selective mean(-10.42% versus baseline, -4.63% versus
its paired control), but a higher tail mean and persistent reversals. Uniform
q50 norm matching has the lowest new branch mean; no blanket selection advantage.
All source identities, 270 retrieved files, native qualification, independent
gate/dense checks and three new PDF pages pass. An exact-equality CPU audit
rounding edge case was corrected without changing or rerunning training.
Workers exited; 385.42 s total, 539.4 MB peak, 21.907 GB retained on owned Spark with
verified local copies, zero cloud cost. Next run 269; continue Analysis 065.

## Runs 269-285: representation-informed transfer and controls

H048 / existing Analysis 065. Every intervention uses unchanged Adam as its
baseline, complete native objectives, exact fork/native continuation checks,
past-only calibration and retained gradient/update/representation diagnostics.
All scientific runs are complete and verified. The table lists scope; executed
arms and attempts are defined by the linked run records, not configuration menus.

| Run | Experiment |
|---|---|
| [269](269-2026-11-07-mlp-representation-interventions/README.md) | mlp representation-informed interventions |
| [270](270-2026-11-07-resnet-representation-interventions/README.md) | resnet representation-informed interventions |
| [271](271-2026-11-07-transformer-representation-interventions/README.md) | transformer representation-informed interventions |
| [272](272-2026-11-07-mlp-displacement-confirmation/README.md) | MLP displacement confirmation |
| [273](273-2026-11-07-resnet-distributed-displacement/README.md) | resnet distributed displacement cap |
| [274](274-2026-11-07-transformer-distributed-displacement/README.md) | transformer distributed displacement cap |
| [275](275-2026-11-07-mlp-matched-displacement-norms/README.md) | matched hidden-layer increment norms |
| [276](276-2026-11-07-mlp-functional-axis-damping/README.md) | causal functional-axis damping |
| [277](277-2026-11-07-mlp-bounded-functional-axis/README.md) | bounded functional-axis modifications |
| [278](278-2026-11-07-mlp-functional-step-budget/README.md) | global functional-step budgets |
| [279](279-2026-11-07-mlp-slow-release-damping/README.md) | slow recovery of functional damping |
| [280](280-2026-11-07-resnet-slow-release-damping/README.md) | resnet transfer of slow-release functional damping |
| [281](281-2026-11-07-transformer-slow-release-damping/README.md) | Transformer probe-based damping |
| [282](282-2026-11-07-mlp-intervention-seed1/README.md) | MLP seed 1 intervention replication |
| [283](283-2026-11-07-mlp-intervention-seed2/README.md) | MLP seed 2 intervention replication |
| [284](284-2026-11-07-mlp-probe-loss-backtracking/README.md) | probe-loss backtracking comparator |
| [285](285-2026-11-07-resnet-probe-loss-backtracking/README.md) | ResNet probe-loss backtracking |

[Results, definitions and caveats](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md): common distributed q50
lowers mean loss in all three architectures and three MLP initializations,
but raises terminal test loss. Negative alignment can persist or increase in
successful branches. Failed diagnostic/directional controls remain reported.
18 scientific workers, 70 records including two authenticated reused controls,
68 newly executed continuations and 35,452 updates. Independent source/retrieval,
dense placement/reduction and original-hardware replay checks pass; 14 new PDF
pages reviewed. All five task pods deleted; no cloud storage retained.
41.316 GB unique raw files retained locally, Spark originals preserved.
Conservative compute/storage estimate USD 18.75 (provider billing provisional)
within USD 20. Next run 286; continue Analysis 065.

## Runs 286-288: MLP distributed cap quantile x LR (H049)

| Run | Initialization | Comparison | Status |
| --- | --- | --- | --- |
| [286](286-2026-11-08-mlp-cap-lr-seed0/README.md) | 0 | 18 fixed-budget cap/LR conditions | Completed and verified |
| [287](287-2026-11-08-mlp-cap-lr-seed1/README.md) | 1 | Same 18 conditions | Completed and verified |
| [288](288-2026-11-08-mlp-cap-lr-seed2/README.md) | 2 | Same 18 conditions | Completed and verified |

[Results](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/013-distributed-cap-quantile-lr.md): All 54 continuations (38,070 updates) completed on owned Spark. Fixed q25 caps reduce mean training loss 87.88%/94.75% versus uncapped 2x/4x, but neither LR improves the across-seed mean over q25 at 1x. The lowest observed mean is q50 at 1x (24.79% below original Adam; positive variation 80.66% lower). Tight caps can increase smaller oscillations and worsen test loss. Three initializations, one continuation window; no generalization superiority claim. All 162 retained audits and six historical trajectories pass; 310,660 plot values and four PDF pages checked. Training 34.34 min, zero cloud cost. Full raw evidence remains on Spark (7.978 GB); compact verified outputs are local after the disk-limit adjustment. Next run 289; continue Analysis 065.

## H050: complete conventional-intervention model coverage (active)

| Run | Case | Varied factor | Fixed fork / end | Status |
|---|---|---|---|---|
| [289](289-2026-11-08-resnet-beta2-reductions/README.md) | ResNet-32 | beta2 .95/.9/.5 and unchanged Adam | 256 / 576 | Running; preflight passed |
| [290](290-2026-11-08-transformer-beta2-reductions/README.md) | Transformer | beta2 .95/.9/.5 and unchanged Adam | 544 / 928 | Running; exact cross-worker replay passed |
| [291](291-2026-11-08-transformer-gradient-clipping/README.md) | Transformer | global raw-gradient caps .5/1 | 544 / 928 | Running; exact common control from Run 290 |

Seed 0, original Run 223 checkpoints, full objectives. Continue Analysis 065; maximum USD 20 and verified raw retention on owned Spark. Next unused run 292.
