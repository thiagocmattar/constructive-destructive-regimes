# Peak-increase histograms across learning rates

Figure: [adam_peak_increase_histogram.pdf](../figures/adam_peak_increase_histogram.pdf).
Generator: [12_plot_peak_histogram.py](../12_plot_peak_histogram.py).
Source hashes, event membership, exact bin edges/counts and checks:
[manifest](../peak_histogram_manifest.json).

Source and coverage: the same 806 recovered excursions used in the
[density figure](004-peak-increase-density.md), from the verified event dataset.
Six unrecovered excursions are excluded. There is no amplitude, startup or
diagnostic-based cutoff; this uses the density cohort rather than the >10%
anatomy cohort. Counts in ascending LR order are MLP [94,139,123],
ResNet-32 [80,121,44], Transformer [79,89,37]. Original density figure unchanged.

Caption: Histograms of peak training-loss percentage increase under full-batch
Adam. Columns show CIFAR-10/MLP, CIFAR-10/ResNet-32 and WikiText-2/Transformer;
colors identify three constant LRs within each model, matching the overview
and density figure. Peak increase is 100*(loss_peak/loss_before-1).
The logarithmic x-axis shows percentage increase: 1 means +1%, 100 means +100%.
Shared x limits match the density figure (10^-5..10^4 percent). All nine
histograms use the same half-decade bin edges. Colored step outlines with light
fills overlay the LRs. Each height is 100*bin_count/group_n, so the displayed
heights sum to 100% separately for each LR. The y-axis is probability mass per
bin, not KDE density, raw counts or event frequency per training update.
No kernel smoothing. Layout is 7.16 x 2.55 inches, with Analysis024 typography
and ordinary numeric log-axis tick labels.

Observation: the unsmoothed binned distributions overlap across LRs while retaining
visible model/LR differences in central mass and tails. Bin positions/width affect
the visible shape, so no uniform shift or stochastic ordering is inferred.

Limits and uncertainty: distributions condition on observed recovery within 1024
updates; normalization removes differences in total event counts, not dependence
between events from a trajectory. The nine arms use one frozen initialization
per model. No confidence bands, causal or cross-seed inference, new training,
cloud resources, interventions or manuscript edits.

Verification: source hashes match extraction and density manifests; exact event
IDs agree with each density group. All 806 values are binned exactly once.
Independent direct bin membership counts agree with NumPy histogram counts;
per-LR heights sum to 100%. Artist heights/edges match recorded aggregates.
Shared axes include every observed value. One vector PDF page, embedded fonts
and text within page bounds. Poppler rendering visually reviewed: readable
panels, overlays, legends and labels without clipping or overlap.
