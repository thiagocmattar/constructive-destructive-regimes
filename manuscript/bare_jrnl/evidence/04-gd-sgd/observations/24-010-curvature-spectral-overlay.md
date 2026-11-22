> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [010-curvature-spectral-overlay.md](../../../../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/010-curvature-spectral-overlay.md).

# Directional diagnostic and retained Hessian spectra

The author requested Figure 1's spectral comparison in annotation C40 of the
3 November original-argument feedback. This is a presentation of existing
measurements, with no new training or Hessian computation.

## Sources and reduction

- Nine Analysis024 trajectories, one matched initialization per model and
  3,000 GD updates at each rate.
- Source records: cohort.json, results.json, single_rows_figure_manifest.json
  and mlp_learning_rates_figure_manifest.json, plus their NPZ arrays.
- Generator: [11_plot_curvature_comparison.py](../../../../../analyses/024-2026-10-17-eos-gd-lr-calibration/11_plot_curvature_comparison.py).
  It verifies source hashes, reloads raw event/spectral records, reconstructs
  the complete trailing 50-pair arithmetic means and checks plotted arrays.
- [Manifest](../../../../../analyses/024-2026-10-17-eos-gd-lr-calibration/curvature_comparison_manifest.json): 27 curves and 53,838
  values, including all 279 retained spectra. It records file/array hashes,
  coverage, clock mapping and individual spectral ranges.
- Loss is unsmoothed at parameter states 0-3000. The directional diagnostic
  retains the previous complete 50-pair mean, at gradient indices 51-3000.
  Spectra are unsmoothed at states 0,100,...,3000 and plotted at t=state+2,
  matching the starting state of the diagnostic's preceding step.
  The last spectral point at t=3002 has no recorded gradient pair.

## Coverage and interpretation

MLP spectra use all 5,000 images. ResNet spectra use the first five fixed
1,000-image normalization groups of 50. Transformer spectra use the first
2,500 of 59,680 sequences. These are largest algebraic eigenvalues with the
retained relative eigenpair-residual acceptance of 0.005.

Oscillatory conditions have late directional diagnostics near two, whereas
the largest eigenvalue can be higher. Smooth low-rate conditions can have
near-zero projected change despite nonzero largest curvature. Directional
orientation, finite-step averaging, 50-pair display smoothing and the two
subset objectives all matter to this comparison. It is evidence of related
curvature/regime signatures, not a calibration of the diagnostic as the top
Hessian eigenvalue. The single initialization and conditional rate selection
provide no across-initialization uncertainty estimate.

## Figure and caption

[Publication PDF](../../../../../analyses/024-2026-10-17-eos-gd-lr-calibration/figures/gd_loss_projected_change_spectral.pdf), copied
byte-for-byte to manuscript/draft/figures/analysis024/. The previous PDF remains
intact. Times New Roman/STIX, original LR colors and decimal log-loss tick
format are retained; the new metric legend separates solid diagnostic means
from dashed, marked spectral observations. Axes include every spectral value.

Caption: Training loss and curvature diagnostics under full-batch GD.
Columns identify dataset/model; colors identify learning rate. Top:
unsmoothed full-objective loss on logarithmic axes. Bottom: eta*kappa_hat
(solid, complete 50-pair trailing means) and retained eta*lambda_max
measurements (dashed with circles); the dotted line marks two. Spectra use
the complete MLP objective, 5,000/50,000 ResNet images and 2,500/59,680
Transformer sequences. Loss uses parameter state s; a spectrum at state s
is placed at gradient index t=s+2, the starting state for kappa_hat_t.
The final spectrum has no recorded gradient pair.

## Verification

All values match the prior arrays and raw retained records. The final
7.16-by-3.5-inch vector PDF and its manuscript placement on page 5 were
rendered with Poppler and visually reviewed. All three LR legends, both
metric encodings, threshold line, clocks and large spectral values are
visible. The manuscript revision also preserves the original training
inputs and all prior publication assets.
