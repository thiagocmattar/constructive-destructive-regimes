# Analysis047: cosine32 floors, each versus unchanged LR

Completed 2026-11-04. Run226 / H033 with retained Run223 evidence. [Findings](FINDINGS.md), [paired vector figure](figures/mlp_cosine_floors_paired.pdf), [results](results.json), [source values](plotted_values.csv).

User presentation requirement: compare each intervention only with unchanged baseline LR, placing different interventions in adjacent columns. This figure has cosine32-to0.75x on the left and cosine32-to0.5x on the right; there are no restoration schedules and no intervention-versus-intervention overlay. Keep this baseline-only presentation for subsequent comparisons unless the user requests otherwise.

Both schedules start at exact state512, reach their floor at544 and hold it through799; endpoint800 is observed and unapplied. Same seed0 MLP, full5000-image CIFAR-10 objective, physical1000 partitions, native Adam clocks/precision/model/data/order and Spark hardware. New0.75x branch is Run226; unchanged baseline and0.5x branch are reused from completed Run223, without retraining. Exact fork-gradient/two-update replay verifies reuse. Historical cross-GPU bitwise replay is not claimed.

Figure: display448..800, five rows (log training loss, raw rho, raw gradient norm, normalized projected gradient change, applied/base LR). Curves solid and unsmoothed; ranges independent per panel, including the LR panels. Gray denotes unchanged in both columns; blue/green denote the respective interventions. Shading before512 and the vertical line mark the common prefix/fork. All data are from Spark, so no hardware boundary occurs within the figure. Extending the last LR interval to800 does not imply another update.

Reduction: mean loss513..800, post-decay mean545..800, last32 states769..800 and endpoint800. Excursions are the unchanged Analysis036 loss-only fixed-pre-rise/recovery detector with >10% threshold and peaks after512, including onset/recovery/censoring. Counts alone do not measure all local oscillations or their severity. Single selected seed/window, no general stability guarantee.

Regenerate using01_report.py. The script verifies74 artifact files across Run223/226 and7056 plotted values. The one-page vector PDF contains no raster image objects, was rendered and visually inspected. Verification.json in Run226 independently checks all applied LRs, raw-gradient identity and checkpoint clocks/current gradients.
