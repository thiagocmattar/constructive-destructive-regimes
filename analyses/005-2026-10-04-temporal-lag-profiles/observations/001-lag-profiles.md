# Observation001: temporal raw-gradient cosine profiles

Question: how does signed cross-step alignment change with lag, training phase
and model size? Do the proposed early/middle/late profile shapes appear?

## Sources and coverage

- [14M Run017](../../../runs/017-2026-10-04-pythia14m-temporal-sketch-seed1235/README.md), `attempt-01/full-pass`: raw sketches, update events, exact pairs/window aggregates, metadata and frozen config. Archive SHA-256: `960f9275c39d642931d93d6d7e0573b78a8d76377c3a004d1f23e05a9383925d`.
- [31M Run018](../../../runs/018-2026-10-04-pythia31m-temporal-sketch-seed1235/README.md), `attempt-02/full-pass`: raw sketches, update events, exact pairs/window aggregates, metadata and frozen config. Archive SHA-256: `63da75f9fcb986668a2b782bf2f4719e6dc11f3cdee869e166a5725786f51850`.
- [70M Run019](../../../runs/019-2026-10-04-pythia70m-temporal-sketch-seed1235/README.md), `attempt-02/full-pass`: raw sketches, update events, exact pairs/window aggregates, metadata and frozen config. Archive SHA-256: `fc18b71737c67981827c14c4ae5f67c40095513f607de4f9fc5221834f663b7b`.

All models use seed1235 and the original 712-update recipe. Revisions, loss
reduction, accumulation boundary, unique trainable parameter scope and source
hashes are retained in the run configs/metadata and `../results.json`.
Both pair endpoints are inside one of three preselected 128-update windows:
1–128, 293–420 and 585–712. Lags1–64 have 127 down to64 pairs per model/window;
the secondary output through127 has only one pair at its endpoint. There are
zero silently truncated training or validation tokens. Update712 is retained.

Reduction: `../03_analyze.py`, with definitions in `../METHOD.md`. Projected dots
are averaged across four maps in FP64 and divided by original exact norms.
Pair cosines are equally weighted. This is an uncentered signed cosine profile,
without a stationarity assumption. Exact lag-one values come from raw-gradient
FP64 dot products. Zero-denominator pairs are undefined and coverage is reported.

## Results

| Size | Window | Exact C(1) | Negative lag1 pairs | C(2) | C(4) | Mean C(32–64) | Unordered reference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14M | early | 0.3630 | 31.5% | 0.3722 | 0.5058 | 0.3157 | 0.3404 |
| 14M | middle | -0.2537 | 92.1% | -0.1118 | 0.0624 | 0.0918 | 0.0846 |
| 14M | late | 0.3291 | 2.4% | -0.0578 | 0.3574 | 0.2540 | 0.2525 |
| 31M | early | 0.1366 | 46.5% | 0.0734 | 0.2393 | 0.1218 | 0.1321 |
| 31M | middle | -0.2722 | 92.9% | -0.0569 | 0.0690 | 0.0934 | 0.0876 |
| 31M | late | 0.2410 | 7.9% | 0.0111 | 0.2776 | 0.2714 | 0.2702 |
| 70M | early | 0.0601 | 44.1% | -0.1434 | 0.1817 | 0.0711 | 0.0696 |
| 70M | middle | -0.2695 | 85.0% | -0.0578 | 0.1431 | 0.0853 | 0.0785 |
| 70M | late | -0.0231 | 52.8% | 0.0129 | 0.1441 | 0.2059 | 0.2001 |

The individual 127-lag curves, four-map values and norm-weighted interactions
are available in `../curve_data.csv` and `../results.json`. Lag1 alone does not
determine the sign at another lag. Descriptive interpretation is in `../REPORT.md`.

## Calibration and uncertainty

| Size | Window | Exact calibration pairs | Cosine RMSE | Max absolute error |
| --- | --- | --- | --- | --- |
| 14M | early | 168 | 0.00566 | 0.01236 |
| 14M | middle | 170 | 0.00417 | 0.01093 |
| 14M | late | 170 | 0.00390 | 0.01401 |
| 31M | early | 168 | 0.00452 | 0.01446 |
| 31M | middle | 170 | 0.00359 | 0.00837 |
| 31M | late | 170 | 0.00382 | 0.01079 |
| 70M | early | 168 | 0.00357 | 0.00962 |
| 70M | middle | 170 | 0.00385 | 0.00990 |
| 70M | late | 170 | 0.00415 | 0.01349 |

Exact pairs include every within-window adjacent pair and randomly anchored
lags2/4/8/16/32/64. Sample counts and errors separately by lag are in
`../results.json`. The planned full-history qualification uses RMSE<0.02 and
maximum absolute error<0.06. Four-map bands are projection sensitivity only,
not training-seed confidence intervals. Errors from a fixed map can be correlated
across pairs; small sign changes below calibration error remain unresolved.
There is one training seed, not four independent training runs.

Excluding the final partial update changes the primary late cosine profile by
at most the following absolute amount over lags1–64:

| Size | Maximum absolute change |
| --- | --- |
| 14M | 0.010268 |
| 31M | 0.009791 |
| 70M | 0.007967 |

This sensitivity does not replace the retained primary evidence. The new replay
shares immutable initialization/order/recipe but may differ numerically from its
earlier reference; full-history differences are explicitly retained:

| Size | Replay validation loss | Reference validation loss | Adjacent-cosine RMSE versus reference |
| --- | --- | --- | --- |
| 14M | 5.182232 | 5.182016 | 0.270230 |
| 31M | 4.624931 | 4.617403 | 0.290626 |
| 70M | 4.009942 | 4.058427 | 0.360398 |

## Figures and captions

- `../figures/lag_profiles.pdf`: Mean raw-gradient cosine at every integer lag
  1–64, for 14M/31M/70M in early/middle/late training. Both endpoints lie within
  the stated 128-update window. Lines show the four-map mean; shading is the
  range across fixed independent maps; markers at lag1 show its exact mean.
  One training seed1235; 128−k valid pairs at lagk.
- `../figures/lag_profiles_above_shuffled.pdf`: The same profiles after each map's
  unordered-window mean is subtracted. The reference is the expected cosine
  under uniform random ordering within the window. This is descriptive, without
  an exchangeability assumption, permutation p-value or stationary-process claim.
- `../figures/lag_profiles_fixed_origins.pdf`: An additional coverage control,
  keeping the same first64 source updates at every lag1–64. Each lag has64 pairs,
  with targets inside the128-update window. Markers show exact lag1 means for
  those source updates; bands are the four-map range. This control was added
  after the early probes, before middle/late spectra were available. It does
  not replace the predeclared all-pair primary profile or assume stationarity.
- `../figures/temporal_gram.pdf`: Four-map mean raw cosine Gram matrices for each
  model/window, normalized by original exact gradient norms. Local row/column
  indices cover all128 updates. The diagonal includes sketch norm error; the
  color scale saturates outside[−1,1], while numerical data are not clipped.

Generating script: `../03_analyze.py`; PDF SHA-256 records and visual QA are in
`../artifact_validation.json`. Scratch PNGs are ignored preview artifacts.

## Limits

These are raw gradients along AdamW training, not AdamW directions or parameter
updates. Phase averages mix changing local behavior. Shared update windows do
not ensure equivalent learning states across architectures. No stability,
generalization, causal dimensionality mechanism or learning-benefit claim is
established. Multi-seed validation remains deferred, as requested.
