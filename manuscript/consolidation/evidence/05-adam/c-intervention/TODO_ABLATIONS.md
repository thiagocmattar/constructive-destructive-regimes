# Intervention ablation TODO

- **Current decision document:** [matched experimental plan](EXPERIMENTAL_PLAN.md), 14 November 2026. It contains the complete setup, measurement contract, ETC/costs, outputs and stopping criteria. This checklist does not authorize a launch.
- **Completed historical coverage:** beta2 .95/.9/.5 in all three models; clipping .5/1 in MLP and Transformer and 1 in ResNet. Runs289-291 and their 20 update audits remain verified; [outcomes](tables/conventional-outcomes.md) retain unfavorable results too.
- **Baseline rule:** every intervention needs unchanged Adam from the same fork, initialization, objective, runtime and horizon. A second intervention is not its baseline.

| Priority | Work | Exact scope | Status |
|---|---|---|---|
| **Current** | Common starts, clocks and horizons | MLP **512 to 800**; ResNet **256 to 576**; Transformer **544 to 928**; initialization **0 only** | Planned; replace the old cap forks in the new comparison, preserving historical evidence |
| **Current** | Immediate LR grid | **0.75x/0.5x/0.1x** original LR in every model; no ramp/cosine | Six missing treatment cells; 0.5x exists historically |
| **Current** | Beta2 grid | **.95/.9/.5**; retained moments/clock and native bias correction | All nine settings exist; qualify/replay for common measurements |
| **Current** | Global raw-gradient clipping | Absolute L2 thresholds **.5/1** in every model | **ResNet .5 missing**; qualify/replay the other five settings |
| **Current** | Distributed cap | **q25/q50/q75** in every model; calibrate on eight preceding untreated updates ending at the shared fork | Nine matched cells needed; q50 remains the main representation-inspired method |
| **Current** | Shared evidence and output | 36 total conditions; full loss histories, stage cosine/RMS, activity, full held-out evaluations, cost; all settings visible | Plan includes one outcome table, loss grid, tradeoff plot, q50 heatmaps and supporting diagnostics |
| **Deferred** | Initializations 1 and 2 | **72 additional continuations**, all 12 conditions/model/initialization | ~$47-67 / 10-14 h on recommended cloud fleet; ~8-12 days Spark-only |
| **Deferred** | Cap quantile x LR x initialization | **3x3x5 capped cells/model**, plus same-LR uncapped controls: **54/model, 162 total** | Gross ~$100-145 / 22-30 h on recommended fleet; ~$102-150 / 11-16 h faster fleet; ~21-30 days Spark-only |

- **Current comparison budget:** approximately **6-9 hours / $25-37**, using Spark + one RTX4090 + three RTX5090s; proposed $45/10-hour limit. Includes full prefix construction, common diagnostics, retrieval and temporary storage; excludes engineering/figure preparation and capacity queues. No work has been launched from this plan.
- **Deferred grid assumptions:** LR multipliers **1x/2x/4x**, caps **q25/q50/q75/q90/q99**, initializations **0/1/2**. This budgets the existing exploratory grid; it does not start it. The 162-cell gross estimate is before any qualified reuse.
- **Estimate caveat:** prices are the live 14 November quote. Transformer-on-Spark ETC extrapolates only three mechanics updates; it requires sustained calibration before a long run. See the decision document for measured anchors, source hashes and exact calculations.
- **Other future sensitivity:** intervention time, probe/calibration-window size, alternative beta2 reset/correction conventions, ramped LR and cross-family combinations. No new initializations or cap x LR ablation is required for the current single-initialization comparison.
- **Claim boundary:** one initialization gives controlled case comparisons, not seed-level uncertainty or general superiority. Keep absent excursions, divergence, inactive clipping and held-out deterioration visible.
