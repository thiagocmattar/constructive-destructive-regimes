# MLP interventions: ongoing analysis (064 / 065)

This is the continuing workspace for the MLP intervention study. Analysis 064
contains its preserved first-stage evidence; 065 continues the same analysis,
not a separate scientific study. Keep subsequent iterations here until the user
requests another analysis. Distinct executed configurations retain distinct run
IDs. Every intervention is compared with the **unchanged Adam baseline**.

| Arm | Complete set of baseline changes | Evidence |
|---|---|---|
| Baseline | None; Adam LR 0.001, all layers updated | Qualified retained control in 064 |
| I1 | Freeze first-linear weights and biases from state 448 | Run 244 |
| I2 | I1 plus Adam LR 0.001 -> 0.002 from state 500 | Run 245 |
| I3 | I1 plus Adam LR 0.001 -> 0.02 from state 500 (10x I2) | Run 246 |

The first doubled-rate update is 500 -> 501. No optimizer reset or unfreezing.
All 5000 CIFAR-10 training examples, ordering, initialization, objective and
remaining optimizer/precision settings match. I2 and I3 branch from the same
state 500; I3 does not wait until the end of I2. Comparisons between interventions
describe incremental LR effects and supplement the baseline comparisons.

## Results and artifacts

Both interventions suppress the baseline's large loss excursions and output
motion. Across 448-800, however, mean training loss is **22.27% higher for I1**
and **20.58% higher for I2** than the unchanged baseline. Endpoint loss at 800 is
10.32% / 13.04% lower, illustrating why the endpoint alone is insufficient.
I2 produces more small output reversals than I1 without restoring large spikes.

**I3 at LR 0.02 causes severe overshoot:** loss rises from 0.028335 at 500 to
107.099 at 504 and recovers only to 0.308132 at 800 (10.19x baseline). First-layer
parameters and representations remain exactly fixed. Later-layer updates can
therefore cause large instability despite that freeze at sufficiently high LR.
The run stays finite; this is degraded training, not numerical overflow.

- [Two-page baseline comparison](figures/01-frozen-first-linear-double-lr-cifar10-mlp.pdf):
  I1, I2 and I3 side by side, each overlaid with the same unchanged baseline.
  Page 1: eight diagnostics over 448-800, including eta kappa hat, gradient norms
  and LR. Page 2: all five learned representations over 500-550.
- [Observation, definitions and caveats](observations/001-frozen-lr-double.md)
- [I3: the 10x LR extension and severe overshoot](observations/002-frozen-lr-0p02.md)
- [Numerical results and provenance](comparison.json), [window summaries](metrics.csv)
- [Run 245 verification](../../runs/245-2026-11-07-mlp-frozen-lr-double/verification.json)
- [Run 246 verification](../../runs/246-2026-11-07-mlp-frozen-lr-0p02/verification.json)
- [First-stage analysis and SVD evidence](../064-2026-11-07-mlp-freeze-first-linear/README.md)

Reproduce with `python analyses/065-2026-11-07-mlp-frozen-lr-double/01_reduce.py`
then `02_figures.py` in the same directory, using the verified retained artifacts.
Native continuation qualification, 66 source identities, 22 retrieved files,
independent dense reductions and baseline duplication checks pass. Both PDF pages
were rendered and reviewed. Run 245 took 7.76 s and Run 246 took 7.96 s on owned
Spark; both workers exited. Their retained directories use 2.021 / 1.911 GB,
respectively, with verified local copies and zero cloud cost. Run 246 adds 67
verified source identities and 22 retrieved files. Plot scales are explicitly
logarithmic or symmetric-log where needed to show I3's much larger excursion.
