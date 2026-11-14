# Matched intervention figures

- **All 36 conditions are complete and verified.** Initialization 0; common starts 512/256/544 and ends 800/576/928. All families retain unchanged Adam.
- **Review first:** [three-page IEEE-style V-C draft](../../../../draft/intervention-review.pdf). It contains the polished main figures 24/25, a compact 12-row table, Algorithm 2 and precise captions. Numerical and final-size visual checks pass; final full-paper numbering remains open.
- **Supporting pages:** the complete training-loss matrix, every setting's stage motion, full held-out test trajectories, applied-update/controller activity, and detailed outcome tables. Terminal test outcomes and endpoint/cost limitations remain in the main table/text.

| Use | PDF |
|---|---|
| Main: all-setting loss-variation/learning tradeoff | [24-main-intervention-tradeoff.pdf](24-main-intervention-tradeoff.pdf) |
| Main: paired q50 functional motion | [25-main-q50-functional-motion.pdf](25-main-q50-functional-motion.pdf) |
| Supplement: matched intervention training losses | [17-matched-intervention-loss.pdf](17-matched-intervention-loss.pdf) |
| Preserved earlier layout: identical tradeoff values | [18-matched-spike-learning-tradeoff.pdf](18-matched-spike-learning-tradeoff.pdf) |
| Preserved earlier layout: identical q50 histories | [19-matched-q50-stage-heatmaps.pdf](19-matched-q50-stage-heatmaps.pdf) |
| Supporting stage motion for every setting | [20-matched-all-setting-stage-heatmaps.pdf](20-matched-all-setting-stage-heatmaps.pdf) |
| Full held-out test trajectories | [21-matched-heldout-loss.pdf](21-matched-heldout-loss.pdf) |
| Applied movement and controller activity | [22-matched-update-and-controller-activity.pdf](22-matched-update-and-controller-activity.pdf) |
| Detailed supporting outcomes | [23-matched-intervention-outcomes.pdf](23-matched-intervention-outcomes.pdf) |

- **Compact main table:** [editable TeX](../tables/matched-main-outcomes.tex), rendered in the review above. Twelve setting rows cover all 36 conditions, with terminal test outcomes and absolute Adam references.
- **Captions, reductions and limitations:** [original scientific observation](../observations/65-016-matched-intervention-matrix.md), [main-layout observation](../observations/main-intervention-layout-and-draft.md), [composed V-C captions](../../../../draft/05c-interventions.tex).
- **Historical assets:** [background](../background/README.md). The older quantile/LR ablation remains at the section root.
- **Deferred experiments:** [TODO](../TODO_ABLATIONS.md).
