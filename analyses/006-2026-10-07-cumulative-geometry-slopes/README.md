# Analysis 006: Cumulative geometry slopes and loss

Status: complete local reanalysis of Run 002, seed 1234. No new training.

The user proposed that A and B acquire equal inclinations, that a flat cumulative
B/A tracks loss convergence, and that this means new gradient contributions are
destroyed without storing additional information in the network. H006 preserves
that hypothesis. This exploratory analysis separates those three claims.

The terminal data do not show equal A/B slopes or a zero ratio slope. A nearly
flat raw ratio occurs earlier while B grows and training loss declines.
Mathematically, a flat B/A means that marginal resultant growth matches the
historical ratio; it does not imply zero resultant growth or absence of learning.

- [Answer and numerical results](REPORT.md)
- [Definitions, derivative identities, and measurement coverage](METHOD.md)
- [Observation and figure captions](observations/001-slopes-and-loss.md)
- [A/B curves and slopes](figures/cumulative_slopes.pdf)
- [Ratio slopes and training loss](figures/ratio_slope_and_loss.pdf)
- [3x2 grid: per-step loss difference versus A/B/ratio slopes](figures/loss_difference_vs_slopes.pdf)
- [Grid data and exact loss differences](loss_difference_grid_data.csv)
- Matched-lag grids: [lag 2](figures/loss_difference_vs_slopes_lag02.pdf),
  [lag 5](figures/loss_difference_vs_slopes_lag05.pdf),
  [lag 20](figures/loss_difference_vs_slopes_lag20.pdf)
- [Matched-lag grid data](lagged_loss_difference_grid_data.csv)
- [All 712 reduced update records](curve_data.csv)
- [Results, source hashes, and sensitivity](results.json)
- [Mathematical and artifact verification](verification.json)

## Inputs and comparison

Use the original paired rows underlying
[raw_adaptive_comparison.pdf](../../runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/figures/raw_adaptive_comparison.pdf),
verified against the original attempt-02 events and the source figure's recorded
SHA-256. Both streams come from the same Pythia-14M / MiniPile / AdamW trajectory.
The varying factor is the diagnostic vector: raw gradient versus bias-corrected
preconditioned Adam direction. The model, initialization, batches, ordering,
optimizer, schedule, seed and measurement coverage are matched within that run.

The Adam vector is the adaptive direction already in the requested figure,
not the first moment alone. Its scale differs from raw gradients and it excludes
LR and weight decay. Neither cumulative vector sum is the actual AdamW parameter
displacement.

Primary plots and terminal comparisons use updates 1-711, matching the source
PDF's full-sized batches. The final partial update 712 remains in the CSV and
full-pass/terminal sensitivity results. Training loss is evaluated on changing
batches. Fixed validation is available only initially and finally.

## Reproduction

From the repository root, with NumPy, Matplotlib and PyMuPDF installed:

~~~powershell
python analyses/006-2026-10-07-cumulative-geometry-slopes/01_analyze.py
python analyses/006-2026-10-07-cumulative-geometry-slopes/02_verify.py
python analyses/006-2026-10-07-cumulative-geometry-slopes/03_loss_difference_grid.py
python analyses/006-2026-10-07-cumulative-geometry-slopes/04_lagged_loss_difference_grids.py
~~~

The design is recorded in [design.json](design.json). Scripts check source
identity and preservation, exact finite-difference algebra, coverage, reduced
data integrity, and synthetic counterexamples. PDFs were rendered and visually
inspected; scratch previews are ignored. One trajectory supplies descriptive
evidence; smoothing sensitivity is not uncertainty across seeds.

The additional user-requested grid uses unsmoothed loss and geometry differences,
with raw gradients left and Adam directions right. Its independent
[design](panel_design.json), [observation](observations/002-loss-difference-grid.md)
and [checks](loss_difference_grid_results.json) preserve the original reduction.

The lag 2/5/20 extensions use identical lagged endpoints for all variables,
without smoothing or division by lag. The
[matched-lag observation](observations/003-matched-lag-grids.md) links the design,
source-preservation checks, data coverage and captions.
