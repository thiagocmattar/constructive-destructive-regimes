# Observation 002: Per-step loss difference versus geometry slopes

The user requested a 3-row by 2-column panel grid of per-step training-loss
differences versus slopes of A, B and cumulative B/A. Left column: raw gradients.
Right column: Adam adaptive directions.

## Sources and reduction

Use the verified Analysis 006 curve_data.csv and original Run002 attempt02
full-pass events. The extension preserves the previous analysis outputs and
source-run evidence. Input identities, output hashes, range checks and coverage
are in [loss_difference_grid_results.json](../loss_difference_grid_results.json);
choices are in [panel_design.json](../panel_design.json).

Every plotted point pairs the unsmoothed backward loss difference
\(L_t-L_{t-1}\) with the corresponding geometric backward difference ending
at t: \(A_t-A_{t-1}\), \(B_t-B_{t-1}\), or
\((B/A)_t-(B/A)_{t-1}\). There is no averaging or regression fit.
Negative y means the recorded batch loss is lower than the previous batch loss.

Coverage: 710 points per panel, updates 2-711, seed 1234, Pythia-14M / MiniPile /
AdamW. Each displayed update has 1024 packed sequences and the same supervised
target count. Update 1 has no prior training-loss measurement, so its difference
is undefined. All 712 rows are retained in
[loss_difference_grid_data.csv](../loss_difference_grid_data.csv), including the
smaller 311-sequence update 712. That tail is excluded from the figure only.

## Figure caption and result

[loss_difference_vs_slopes.pdf](../figures/loss_difference_vs_slopes.pdf):
Rows show slopes of A, B and cumulative B/A on x; y is the per-step training-loss
difference in every panel. Raw-gradient diagnostics are left; the bias-corrected
preconditioned Adam-direction diagnostics are right. Color identifies optimizer
update, using a common scale across panels. A/B and loss axes are linear;
ratio-slope x-axes use identical symmetric logarithmic scales with linear
threshold 1e-4. Dotted horizontal lines indicate zero loss difference; dotted
vertical lines in the last row indicate zero ratio slope. No points are clipped.

The displayed loss differences range from -0.185272 to +0.103356. The same loss
differences appear in both columns because the diagnostics are paired within
one trajectory. Stage-dependent point clouds are visible; no statistical or
causal relationship is estimated.

## Interpretation and verification

Recorded loss_t is measured before optimizer update_t. Thus the loss difference
includes the preceding update's effect as well as changing batches. Pairing it
with geometric increment_t is a contemporaneous association, not a measurement
of the causal benefit of update_t. These losses are supervised-token means on
different batches, not repeated evaluations of a fixed objective.

Raw gradients are measured before clipping. Adam directions are read from actual
optimizer moments and exclude LR and decay; they are not actual parameter
updates. One correlated seed trajectory supplies no independent uncertainty
estimate. Color reveals training phase, which can confound a pooled association.

[03_loss_difference_grid.py](../03_loss_difference_grid.py) checks the source
slopes against original events, exact loss values, telescoping loss differences,
indexing, all 710 coordinates in each panel and source preservation. The one-page
PDF was rendered and visually inspected. All checks passed.
