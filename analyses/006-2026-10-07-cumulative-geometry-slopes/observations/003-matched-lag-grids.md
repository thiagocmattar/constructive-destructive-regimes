# Observation 003: Matched-lag loss and geometry differences

The user requested versions of the existing 3x2 scatter grid at lags 2,5 and 20,
with the same lag on both axes. They clarified that "log differences" meant
lagged differences matching the current figure, rather than logarithmic
differences. This preserves the intended quantity.

## Definition, sources and coverage

At each lag k, pair \(L_t-L_{t-k}\) with \(A_t-A_{t-k}\),
\(B_t-B_{t-k}\) and \((B/A)_t-(B/A)_{t-k}\), separately for raw gradients and
Adam adaptive directions. These are k-update changes without smoothing or
division by k. The first k differences are undefined.

Inputs are the verified Analysis 006 curve_data.csv and original Run002 attempt02
events. Every source level agrees exactly with the original events. Previous
figures/data and source-run evidence remain unchanged, as checked by SHA-256.
The [design](../lagged_panel_design.json),
[results and hashes](../lagged_loss_difference_grid_results.json) and
[combined data](../lagged_loss_difference_grid_data.csv) record the reduction.

Primary display ends at 711, keeping the same full-sized training batches as the
existing figure. Each lag retains all 712 source rows, including undefined initial
differences and the smaller final batch. The combined CSV contains 2136 rows.

| Lag | Displayed endpoints | Points per panel | Training-loss change range |
| --- | --- | ---: | ---: |
| 2 | 3-711 | 709 | -0.331766 to 0.094681 |
| 5 | 6-711 | 706 | -0.706676 to 0.104862 |
| 20 | 21-711 | 691 | -2.332697 to 0.088043 |

## Figure captions

- [Lag 2](../figures/loss_difference_vs_slopes_lag02.pdf)
- [Lag 5](../figures/loss_difference_vs_slopes_lag05.pdf)
- [Lag 20](../figures/loss_difference_vs_slopes_lag20.pdf)

All three figures have identical topology: rows are A change, B change and
cumulative B/A change; columns are raw gradients left and Adam adaptive
directions right. Y is training-loss change over the same lag as X. Color
identifies the pair's ending optimizer update, with the same color normalization
across all figures. Negative y indicates a lower recorded batch loss at the
endpoint than k updates earlier.

A/B and loss axes are linear. Ratio-change x-axes use symmetric logarithmic
scaling with linear threshold 1e-4. Within each PDF, all panels share y limits and
the last-row panels share x limits. Axis limits may differ across lags because
the quantities are accumulated changes, not rates divided by lag. All finite
pairs are shown, with no fitted trend or clipped points.

## Verification and limits

[04_lagged_loss_difference_grids.py](../04_lagged_loss_difference_grids.py)
checks source identity, paired endpoints, coverage, exact scatter coordinates
and agreement of every k-step difference with the sum of k one-step differences.
All three one-page PDFs were rendered and visually inspected; checks passed.

The same one-seed AdamW trajectory supplies both columns. Raw gradients precede
clipping; Adam directions use actual optimizer state and exclude LR and decay.
Loss is measured before the indexed optimizer update on changing batches.
Same-endpoint differences are contemporaneous associations, not causal
step-effect or information-storage measurements. Lagged pairs overlap and
training phase changes both geometry and loss; the figures do not establish
independent-replicate uncertainty or a convergence criterion.
