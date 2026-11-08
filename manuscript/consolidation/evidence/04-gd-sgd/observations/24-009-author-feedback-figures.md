> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [009-author-feedback-figures.md](../../../../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/009-author-feedback-figures.md).

# Author-feedback GD composites, 2026-11-02

The manuscript revision reorganizes existing evidence into
`figures/gd_loss_projected_change.pdf` (loss/projected change, 2x3) and
`figures/gd_reversal_norm_retention.pdf` (cosine/norm/retention, 3x3).
The publication copies live under `manuscript/draft/figures/analysis024/`.

Source: the hash-verified `derived/single_rows_curves.npz` and its passed
`single_rows_figure_manifest.json`; the original event files are independently
checked again by the revision verifier. All nine conditions and the declared
first 3,000-update horizon remain. This includes the declared prefixes of
the two longer historical MLP runs. Loss includes the measured terminal
state 3000; gradient indexing is unchanged. No scientific array is changed.

Reduction: raw loss and complete-prefix B/A; complete 50-observation trailing
arithmetic means of cosine, norm and pointwise projected change. Norms/loss
use logarithmic display after reduction. The projected-change reference is
two, matching the local quadratic loss-change boundary; it is not a measured
spectral threshold. Smoothing does not estimate exact onset. No seed averaging
or uncertainty bands: one shared initialization across rates within each model.

Observation: large-rate reversal and small path retention coexist with lower
endpoint training loss and greater absolute displacement in all three model
cases. Directional projected change near two provides a first-order view of
oscillatory descent without evaluating a Hessian; it is not lambda-max or a
validated pointwise loss predictor. The existing nonlinear loss audit remains
in Appendix C. The comparison does not isolate architecture or establish
slower optimization from low retention.

Table I adds dataset and general T notation, with all nine endpoints and
45 scalar entries independently checked at displayed precision. Only the
nonzero value rounded to four-decimal zero receives a star.

Generator: `manuscript/draft/revisions/2026-11-02-feedback/01_prepare_figures.py`.
Captions: `manuscript/draft/sections/04-gd-experiments.tex`.
Exact sources, reductions, output hashes, numerical and visual verification:
`manuscript/draft/revisions/2026-11-02-feedback/{figure_manifest,verification,review}.json`.
Earlier figures and raw evidence remain unchanged; no training or cloud work.

Follow-up layout correction: top-row panel letters now share the column-title
line, with the three learning-rate entries immediately below. The header is
shorter and no letter overlaps a legend. Loss axes retain their logarithmic
scales but use ordinary decimal tick labels. All plotted arrays are unchanged.
Unused manuscript copies of earlier figures are archived; owning analysis
figures and source evidence remain in place.
