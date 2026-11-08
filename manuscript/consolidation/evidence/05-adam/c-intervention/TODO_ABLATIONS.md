# Intervention ablation TODO

- **Scope:** execution checklist for V-C; complements the single paper [PLAN](../../../PLAN.md).
- **Requested now:** Transformer clipping and ResNet/Transformer beta2 reductions, Runs 289–291 (H050, Analysis065). **Running/preparing; no results claimed yet.**
- **Baseline rule:** every cell needs unchanged Adam from the same fork, initialization, objective, runtime and horizon. Comparing two interventions alone does not fill a cell.

| Family | MLP evidence already available | ResNet evidence already available | Transformer evidence already available | Missing for the same basic grid |
|---|---|---|---|---|
| LR schedules | Persistent half LR and cosine transition to half, seed0 | Same schedules, seed0 | Same schedules, seed0 | Additional seeds; common timing with the other families |
| Beta2 reductions | .999 → .95/.9/.5, seed0 | Run289 executing these three reductions | Run290 executing these three reductions | Additional seeds; repeat at independent intervention times |
| Gradient clipping | Thresholds .5 and 1, seed0 | Threshold 1, seed0 | Run291 executing .5 and 1 | **ResNet threshold .5**; additional seeds/times |
| Distributed displacement cap | q50 at three seeds; q25/q50/q75/q90/q99 × LR1/2/4 at three seeds | q50, seed0 | q50, seed0 | ResNet/Transformer seeds1–2 and matching quantile × LR grids |

- **Priority 1 — defensible comparison among the four families:**
  - [ ] Finish/verify Runs289–291 and replace the incomplete conventional-intervention figures.
  - [ ] Fill ResNet clipping at .5 if the main comparison uses both clipping thresholds.
  - [ ] Repeat the selected setting of each family at seeds1–2 in all models.
  - [ ] Align forks, calibration history and evaluation horizons across families. Current conventional forks are MLP512 / ResNet256 / Transformer544; selected distributed-cap forks are MLP448 / ResNet264 / Transformer552. Current results support within-family contrasts, not a fair ranking across all four families.
  - [ ] Compare mean/post-fork loss, late-window loss, maximum/excursion severity and full held-out loss; show seed-resolved effects and cost. Do not select a winner using only its most favorable metric.
- **Priority 2 — mechanism and sensitivity, if claiming a fully ablated intervention study:**
  - [ ] LR: reduction magnitude and transition duration, matched to the selected fixed and cosine schedules.
  - [ ] Beta2: timing/duration and recovery to .999; distinguish the present native-bias-correction intervention from any future moment reset/rescaling or history-product correction. Each is a separate intervention.
  - [ ] Clipping: threshold-response curves in each model; report activation frequency and optimizer-input norm alongside raw norm. Thresholds are objective-scale dependent.
  - [ ] Cap: ResNet/Transformer quantile × LR × seed grid; calibration window/probe size and intervention-time sensitivity. Keep q50 as the sole selected representation-inspired method.
  - [ ] Independent windows: pre-excursion, destructive phase and post-excursion, fixed before examining treatment outcomes.
  - [ ] Cost: forward/backward passes, diagnostics/probe passes, wall time and memory for each selected method on matched hardware.
- **Optional, only for an explicit interaction claim:** LR × beta2, LR × clipping, and cap × conventional intervention. A complete Cartesian product is not required for the paper's present characterization-and-control contribution.
- **Execution boundary:** this note records missing experiments; it does not launch or authorize the additional grid.
