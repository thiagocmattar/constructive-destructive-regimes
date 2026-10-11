# Observation008: final raw-gradient norm minus its trajectory minimum

The user revised the preceding ratio to a difference. The current metric is

`difference = ||g_712|| - min_{1 <= t <= 712} ||g_t||`.

This subtracts scalar norms and has raw-gradient norm units. The final update
and minimum window are unchanged: all 712 unsmoothed accepted updates, including
the zero-LR first update and final partial batch. Each run uses its own minimum.
Raw g is the whole-model unscaled accumulated gradient before clipping.
This is not the norm of a difference between gradient vectors.

Sources: [raw norm series](../loss_and_norms_by_learning_rate_data.json),
[preceding reduction](../raw_norm_final_to_minimum_data.json) and
[comparison data](../comparison_data.json). Every final/minimum value is checked
against the complete norm series before subtraction. The
[new data/provenance](../raw_norm_final_minus_minimum_data.json) pin sources,
generator, PDF and CSV hashes. [CSV](../raw_norm_final_minus_minimum.csv) retains
full-precision final norms, minima, minimum-update indices and differences.
The previous ratio evidence remains in Observation007.

| Peak LR | Final minus minimum raw norm |
| ---: | ---: |
| 0.0001 | 1.619056 |
| 0.000215443469 | 2.084658 |
| 0.000464158883 | 1.538796 |
| 0.001 | 1.473342 |
| 0.002154434690 | 0.977773 |
| 0.004641588834 | 1.098927 |
| 0.01 | 8.162023 |

Caption: **Final raw-gradient norm minus its minimum across seven learning
rates in Pythia-14M.** The [PDF](../figures/raw_norm_final_minus_minimum.pdf)
uses log LR X and linear difference Y. One point per trajectory, with values
rounded to three decimals. Colors retain the LR palette; circles are Spark,
squares RTX4090. The gray line connects observations without fitting a response.
The formula and partial endpoint are stated below the plot. No smoothing,
normalization, token weighting across updates or uncertainty bands.

The difference at LR1e-2 is 8.162, versus 0.978-2.085 elsewhere. This is a
descriptive finite-history endpoint increase above a selected sample minimum;
it is not strictly monotone in LR. Final update712 uses 311 sequences/636,033
targets, versus 1,024 sequences/2,096,128 targets at each preceding full update.
That batch-size difference and the selected single minimum affect interpretation.
One seed1234, matched initialization/order/recipe, mixed hardware and LR-scaled
decay: no convergence, stability, noise-variance or causal performance claim.

Generator: [16_plot_raw_norm_final_minus_minimum.py](../../16_plot_raw_norm_final_minus_minimum.py).
[Verification](../raw_norm_final_minus_minimum_verification.json) checks all seven
differences against source values, plotted values/order, source/output hashes,
PDF text/bounds and the rendered 11x7 inch page. The layout was inspected for
legible formula, labels and markers with no clipping or overlap.
