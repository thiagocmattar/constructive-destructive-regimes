# Peak-increase densities across learning rates

Figure: [adam_peak_increase_density.pdf](../figures/adam_peak_increase_density.pdf).
Generator: [06_plot_peak_density.py](../06_plot_peak_density.py).
Sources, exact event membership and checks: [manifest](../peak_density_manifest.json).

Source: verified `spike_dataset/loss_spikes.csv` and its extraction manifest.
All 806 recovered excursions enter, with the six censored events excluded
because their final peak magnitudes are provisional. There is no amplitude,
startup, or diagnostic-based selection. Each event has equal weight within its
own model/LR density. Different event counts do not change the curve's total mass.

Caption: Distribution of peak training-loss increases under controlled
full-batch Adam. Columns show CIFAR-10/MLP, CIFAR-10/ResNet-32 and
WikiText-2/Transformer; colors identify their three constant learning rates,
matching the training-loss overview. Peak increase is
100 times (peak loss / fixed pre-rise loss minus one). The x-axis is logarithmic
in **percent increase**, so 1 denotes +1% and 100 denotes +100% (double loss).
Curves are Gaussian kernel density estimates in log10-percent coordinates.
Density is expressed per log decade; each LR curve integrates to one in that
coordinate, rather than in linear percentage units. The x/y limits are shared
across the three panels.

Estimator: apply a Gaussian kernel to each positive log10 percentage increase.
Within each model, pool its recovered events across LRs and set a common
bandwidth to the pooled sample standard deviation times pooled N^(-1/5).
Thus LR comparisons do not use different smoothing widths within a panel.
Models have different pooled bandwidths; the source rows remain unsmoothed.
Numeric checks evaluate kernels independently with SciPy's normal density and
compare quadrature with the analytic Gaussian-mixture CDF. The common view
contains all observed values and more than 99.99% of each estimated density.

Scale choice: exploratory scratch renders compared a log peak/pre-rise ratio
(where 1 is no change) with log percentage increase. The ratio view compresses
most observations against 1. The percentage-log view is retained to resolve
small excursions and large tails. It has no zero tick because the definition
requires a strictly positive increase. The figure is 7.16 x 2.55 inches with
Analysis024 typography, ordinary numeric ticks and the established LR palette.

Interpretation limits: these are conditional distributions among excursions
that recovered within the observed 1,024 updates. They do not measure spike
frequency or duration. Most events remain small, so increasing LR need not
produce a uniform right shift; overlapping distributions and heavier tails can
coexist. Kernel features depend on smoothing and are descriptive. Events from
each of the nine single-initialization arms may be temporally dependent; no
confidence bands, independence assumption, causal effect or formal stochastic
ordering is inferred. No new training, Hessians, interventions or manuscript edits.
