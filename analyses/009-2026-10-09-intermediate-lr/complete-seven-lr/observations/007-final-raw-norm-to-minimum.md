# Observation007: final raw-gradient norm divided by its trajectory minimum

The user subsequently revised the plotted metric to a difference; the current
version is [Observation008](008-final-raw-norm-minus-minimum.md). This record
preserves the earlier ratio reduction.

Date: 2026-10-10. The user requested the metric from the unsmoothed
[loss/norms-by-LR figure](../figures/loss_and_norms_by_learning_rate.pdf).
For each learning rate, define the dimensionless ratio

`metric = ||g_712|| / min_{1 <= t <= 712} ||g_t||`.

Raw g is the whole-model unscaled accumulated gradient before clipping,
measured at each accepted optimizer boundary. The numerator is the final
recorded raw norm, not a late average. The denominator is the smallest raw
norm in that same run, not a minimum shared across LRs. All 712 unsmoothed
values are included, including the initial zero-LR update. If minima tie,
report the first update. A zero minimum makes this metric undefined; all seven
observed minima are positive, and no epsilon or positive floor is added.

Sources: [exact values behind the source PDF](../loss_and_norms_by_learning_rate_data.json)
and [comparison data](../comparison_data.json). The source PDF/data receipts
are checked before reduction, and all 4,984 source raw norms are independently
matched to original event fields. The [metric data/provenance](../raw_norm_final_to_minimum_data.json)
pin source PDF, source arrays, comparison, generator, output PDF and CSV hashes.
Values are extracted from recorded scalars, without digitizing the PDF image.
Source inputs and all preceding figures are unchanged.

| Run | Peak LR | Final raw norm | Minimum raw norm | Minimum update | Final / minimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 022 | 0.0001 | 1.949151 | 0.330095 | 556 | 5.904819 |
| 024 | 0.000215443469 | 2.385540 | 0.300882 | 283 | 7.928478 |
| 025 | 0.000464158883 | 1.797690 | 0.258894 | 168 | 6.943723 |
| 021 | 0.001 | 1.657252 | 0.183909 | 82 | 9.011244 |
| 026 | 0.002154434690 | 1.099670 | 0.121896 | 60 | 9.021360 |
| 027 | 0.004641588834 | 1.205760 | 0.106833 | 42 | 11.286384 |
| 023 | 0.01 | 8.268583 | 0.106559 | 48 | 77.595960 |

Caption: **Final raw-gradient norm relative to its minimum across seven peak
learning rates in Pythia-14M.** The [PDF](../figures/raw_norm_final_to_minimum.pdf)
uses a logarithmic LR X axis and a linear ratio Y axis. Each point represents
one trajectory; labels show the ratio rounded to two decimals. LR colors
match the source plot's reversed-plasma palette, circles identify Spark and
squares RTX4090. The gray line joins observed points without fitting a response.
The dotted Y=1 reference means final norm equals the trajectory minimum.
Both the exact metric and the partial final batch are stated below the plot.
[CSV](../raw_norm_final_to_minimum.csv) and JSON retain full numerical precision.

The LR1e-2 ratio is 77.596 versus 5.905-11.286 at the other six LRs.
The ordering is not strictly monotone: LR0.000464158883 has a smaller ratio
than LR0.000215443469. This describes an endpoint relative to a selected
sample minimum, rather than a norm slope, cumulative B/A, noise variance,
loss-convergence measure or stability criterion.

Coverage and uncertainty: seven runs, one seed1234, one full MiniPile pass.
Final update712 contains 311 sequences/636,033 supervised targets, versus
1,024 sequences/2,096,128 targets at each preceding full-sized update.
Its batch size differs and affects the interpretation of a single endpoint.
The minimum is also a single selected measurement; there is no smoothing,
token weighting across updates, replicate averaging or uncertainty estimate.
Initialization/order/recipe are matched; hardware differs across old/new groups
and LR scales AdamW decay. No universal LR law or causal performance claim.

Generator: [15_plot_raw_norm_final_to_minimum.py](../../15_plot_raw_norm_final_to_minimum.py).
[Verification](../raw_norm_final_to_minimum_verification.json) checks source
identities, all raw values, every ratio against an independent direct reduction,
plotted point order/values, PDF dimensions/text/bounds and the rendered page.
The final one-page11x7 inch plot was inspected for labels, marker placement,
axes, legends and absence of clipping or overlap.
