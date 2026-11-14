# Matched intervention figures

- **All 36 conditions are complete and verified.** Initialization 0; common starts 512/256/544 and ends 800/576/928. All families retain unchanged Adam.
- **Main reading order:** training loss matrix, spike/learning tradeoff, then the q50 stage heatmaps. The loss matrix preserves LR -> beta2 -> clipping -> cap order.
- **Supporting pages:** every setting's stage motion, full held-out test trajectories, applied-update/controller activity, and a three-page outcome table.

| Use | PDF |
|---|---|
| Figure 7: matched intervention training losses | [17-matched-intervention-loss.pdf](17-matched-intervention-loss.pdf) |
| Figure 8: spike burden versus learning progress | [18-matched-spike-learning-tradeoff.pdf](18-matched-spike-learning-tradeoff.pdf) |
| Figure 9: matched q50 loss and functional motion | [19-matched-q50-stage-heatmaps.pdf](19-matched-q50-stage-heatmaps.pdf) |
| Supporting stage motion for every setting | [20-matched-all-setting-stage-heatmaps.pdf](20-matched-all-setting-stage-heatmaps.pdf) |
| Full held-out test trajectories | [21-matched-heldout-loss.pdf](21-matched-heldout-loss.pdf) |
| Applied movement and controller activity | [22-matched-update-and-controller-activity.pdf](22-matched-update-and-controller-activity.pdf) |
| All 36 matched intervention outcomes | [23-matched-intervention-outcomes.pdf](23-matched-intervention-outcomes.pdf) |

- **Captions, reductions and limitations:** [current observation](../observations/65-016-matched-intervention-matrix.md).
- **Historical assets:** [background](../background/README.md). The older quantile/LR ablation remains at the section root.
- **Deferred experiments:** [TODO](../TODO_ABLATIONS.md).
