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
| I4 | I1 plus Adam LR 0.001 -> 0.01 from state 500 (half I3) | Run 247 |
| I5 | I1 plus Adam LR 0.005 from state 500 | Run 248 |
| I6 | I1 plus Adam LR 0.0025 from state 500 | Run 249 |
| I7 | I1 plus Adam LR 0.00125 from state 500 | Run 250 |
| I8 | I1 plus Adam LR 0.000625 from state 500 | Run 251 |
| I9 | First-layer LR 0.0005 from 448; later layers 0.001; no freeze | Run 252 |
| I10 | First-layer LR 0.00025 from 448; later layers 0.001; no freeze | Run 253 |
| I11 | All-layer LR 0.0005 from 448; no freeze | Run 254 |
| I12 | All-layer LR 0.00025 from 448; no freeze | Run 255 |
| I13 | First-layer LR 0.001; later layers 0.0005 from 448; no freeze | Run 256 |
| I14 | First-layer LR 0.001; later layers 0.00025 from 448; no freeze | Run 257 |

For I2-I8 the first changed-rate update is 500 -> 501. No optimizer reset or unfreezing.
All 5000 CIFAR-10 training examples, ordering, initialization, objective and
remaining optimizer/precision settings match. I2 through I8 branch from the same
state 500; none waits until the end of another intervention. Comparisons between interventions
describe incremental LR effects and supplement the baseline comparisons.

I9-I14 instead branch from the **untreated full state 448**, with all layers
trainable and first changed update 448 -> 449. They test selective damping of
first-layer updates against global and later-layer-only LR controls. Their
complete interventions contain no freezing and no LR change at 500.

## Results and artifacts

I1 and I2 suppress the baseline's large loss excursions and output
motion. Across 448-800, however, mean training loss is **22.27% higher for I1**
and **20.58% higher for I2** than the unchanged baseline. Endpoint loss at 800 is
10.32% / 13.04% lower, illustrating why the endpoint alone is insufficient.
I2 produces more small output reversals than I1 without restoring large spikes.

**I3 at LR 0.02 causes severe overshoot:** loss rises from 0.028335 at 500 to
107.099 at 504 and recovers only to 0.308132 at 800 (10.19x baseline). First-layer
parameters and representations remain exactly fixed. Later-layer updates can
therefore cause large instability despite that freeze at sufficiently high LR.
The run stays finite; this is degraded training, not numerical overflow.

**Halving stopped at 0.000625 (I8):** all 300 loss transitions over 500-800
decrease. At 0.00125, visible spikes are already gone; only a 0.0060% first-step
bump remains. At 0.0025, a delayed peak reaches 0.0724478 at 707. The lower rates
must be judged over the full window, not only immediately after the change.
The recorded strict rule permits no upward step above 1e-8 and no loss above
loss500 + 1e-8. This is a finite-window criterion, not an optimal-LR estimate.

**Selective LR reductions retain feature learning:** I10 (first layer at 0.25x,
later layers unchanged) lowers mean loss 448-800 by 7.06% and positive loss
variation by 99.62%. First-hidden reversals fall 350/353 -> 1/353. Its last-32
mean remains 15.79% above baseline, so it does not win on every learning metric.
First-layer 0.5x develops a larger delayed peak at 673. Global 0.5x has the best
full-window mean (-17.05%) but a rising excursion at 800. Later-layer reductions
improve average loss while retaining most reversals. See observation005 for the
matched controls, sustained-target results and limits.

- [Four-page baseline comparison](figures/01-frozen-first-linear-double-lr-cifar10-mlp.pdf):
  I1-I4 on pages 1/2 and I5-I8 on pages 3/4, each overlaid with unchanged baseline.
  Overview pages: eight diagnostics over 448-800, including eta kappa hat,
  gradient norms and LR. Companion pages: five representations over 500-550.
  Scales match across columns and page groups.
- [Observation, definitions and caveats](observations/001-frozen-lr-double.md)
- [I3: the 10x LR extension and severe overshoot](observations/002-frozen-lr-0p02.md)
- [I4-I8: LR halving and the strict stopping rule](observations/003-lr-halving.md)
- [Separate I6/I5 comparison](figures/02-i6-vs-i5-cifar10-mlp.pdf): I6 left,
  I5 right, each against unchanged Adam over 448-800. Independent linear loss
  axes; shared scales for the other seven diagnostic rows.
  [Observation and reproduction](observations/004-i6-i5-comparison.md).
- [New selective-LR comparison](figures/03-selective-layer-lr-cifar10-mlp.pdf):
  three pages, two columns per page (0.5x / 0.25x), comparing first-layer,
  global, and later-layer-only reductions against unchanged Adam. Ten rows
  include first-hidden motion; loss axes are independent.
  [Results, methods and caveats](observations/005-selective-layer-lr.md).
- [Verified halving summary](halving_summary.json), [recorded rule](halving_plan.json)
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

Runs 247-251 add 1,505 intervention updates in 40.62 s total, 67 verified source
identities per run and 110 verified downloaded files. Their full fork states are
exactly identical to I2's fork. All workers exited; 10.034 GB retained for these
five runs on owned Spark with verified local copies, no cloud cost. Stopping was
confirmed from local raw losses; all four final PDF pages were reviewed.

Runs252-257 complete six prescribed LR-location branches, 353 updates each,
with exact untreated448-512 qualification and full fork/state matches. All
67 source identities and 22 retrieved files per successful run, independent
dense reductions, 50832 plotted values and three new PDF pages pass review.
Workers exited after49.75s total successful execution; peak allocation400.8MB.
13.720GB retained on owned Spark, including one rejected qualification attempt,
with verified local copies and no cloud charge. Regenerate the new results with
`07_reduce_layer_lr.py` and the new PDF with `08_figures_layer_lr.py`.
