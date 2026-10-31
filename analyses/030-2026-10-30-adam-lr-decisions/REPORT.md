# Matched Adam learning-rate pilot

All 18 branches completed and passed local artifact and scientific acceptance. Three seed clusters, two sizes and one fork stage support a pilot, not a validated scheduler.

Doubling LR wins in every starting state, despite negative early raw-gradient
cosines. The diagnostic feature additions do not improve the prespecified
selection comparison: the fixed-LR and loss-only baselines already select the
best observed branch in all six states. These results provide a counterexample
to treating more positive gradient alignment as a performance objective; they
do not validate a diagnostic-driven scheduler.

| State | Lower LR minus held loss | Higher LR minus held loss | Best multiplier |
| --- | ---: | ---: | ---: |
| 14m-1234 | +0.044851 | -0.035666 | 2 |
| 14m-1235 | +0.045785 | -0.050583 | 2 |
| 14m-1236 | +0.044148 | -0.046899 | 2 |
| 70m-1234 | +0.048476 | -0.035937 | 2 |
| 70m-1235 | +0.059663 | -0.023816 | 2 |
| 70m-1236 | +0.056975 | -0.043005 | 2 |

Negative differences mean the LR intervention improved final full-validation loss at matched tokens. Shrink factors were matched; this is not native fixed-decay schedule tuning.

The same LR multiplier wins across all blocks. A common winning multiplier does not demonstrate state-dependent decision value.

Exploratory leave-one-seed-out mean regret (six correlated size/state blocks):

- Hold LR: 0.039318.
- Fixed LR: 0.000000.
- Loss baseline: 0.000000.
- Plus raw cosine: 0.011137.
- Joint diagnostics: 0.016247.

Predictors observe updates 2–16 from every candidate branch. This tests early recognition after intervention, not selection using only the common checkpoint. Probing all three rates for 16 updates and completing the chosen 128-update branch would require 160 total updates, 25% more than one predetermined rate before evaluation overhead. The regret comparison is not an equal-compute scheduler comparison.

No significance test or controller-validation claim is made. All predictions, including failures to select the best arm, are retained in results.json. All three figures passed visual review, and independent source reductions and augmented-least-squares prediction checks passed.

Training finished October 30 at 20:23 BRT. All 180 files (121.692 GB) were locally
verified before task pods/storage were deleted at 21:49 BRT. The report/figures
were generated at 21:50 BRT and subsequently reviewed. ETC is zero. Posted task
charges are USD 13.90 at the billing read, provisional until provider reconciliation;
no task cloud compute or storage remains. See `billing.json`, `closeout.json`,
`result_verification.json` and the local figure observation for evidence and scope.
