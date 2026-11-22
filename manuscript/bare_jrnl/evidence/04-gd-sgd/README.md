# IV. Empirical Characterization

- **Paper claim:** Retain the four current GD/SGD figures in order; supporting HVP, finite-step and all-lag evidence checks their interpretation.
- **Boundary:** Nine GD conditions use one initialization per task; minibatch evidence is replicated. Equal updates do not mean equal data exposure.
- **Authority:** the [single paper plan](../../PLAN.md) governs selection and closure.
- **Presentation:** Figures 1-4 refreshed on 14 November. Training progress is Training step s; lag and batch axes retain their own units. [Current captions and clock conversions](../FIGURE_NOTES.md).

| Asset | Status | Use / source |
|---|---|---|
| [Current manuscript source: 04-gd-experiments.tex](source/04-gd-experiments.tex) | source | Unmodified manuscript snapshot; its input/figure paths resolve from the original draft root. Use the consolidated snippets and selected displays for integration. |
| [Current manuscript source: 05-sgd-noise-experiments.tex](source/05-sgd-noise-experiments.tex) | source | Unmodified manuscript snapshot; its input/figure paths resolve from the original draft root. Use the consolidated snippets and selected displays for integration. |
| [Figure 1](figures/gd_loss_projected_change_spectral_clean.pdf) | prepared main | Revised: completed-step clock; both curvature curves use the same centered 101-step smoother. Sparse spectrum interpolation is explicit in the caption. [Observation](../../work/figure_revision_20261114/OBSERVATIONS.md). [Generator](../../work/figure_revision_20261114/01_figures.py). |
| [Figure 2](figures/gd_reversal_norm_retention.pdf) | prepared main | Revised state-indexed gradient coordinates; original trailing means and cumulative R values preserved. [Observation](../../work/figure_revision_20261114/OBSERVATIONS.md). [Generator](../../work/figure_revision_20261114/01_figures.py). |
| [Figure 3](figures/gd_backward_compact.pdf) | prepared main | Revised state-indexed notation; backward lag remains a separation axis, with the same endpoint and values. [Observation](../../work/figure_revision_20261114/OBSERVATIONS.md). [Generator](../../work/figure_revision_20261114/01_figures.py). |
| [Figure 4](figures/sgd_batch_size_compact.pdf) | prepared main | Revised state-indexed notation; original batch axis, means and initialization ranges preserved. [Observation](../../work/figure_revision_20261114/OBSERVATIONS.md). [Generator](../../work/figure_revision_20261114/01_figures.py). |
| [Analysis 24: 007-section-iv-critical-review.md](observations/24-007-section-iv-critical-review.md) | record |  |
| [Analysis 24: 009-author-feedback-figures.md](observations/24-009-author-feedback-figures.md) | record |  |
| [Analysis 24: 010-curvature-spectral-overlay.md](observations/24-010-curvature-spectral-overlay.md) | record |  |
| [Analysis 35: 002-manuscript-view.md](observations/35-002-manuscript-view.md) | record |  |
| [Analysis 37: 001-three-case-sgd.md](observations/37-001-three-case-sgd.md) | record |  |
| [Analysis 56: 001-paired-displacement.md](observations/56-001-paired-displacement.md) | record |  |
| [Analysis 27: 001-controlled-minibatch-noise.md](observations/27-001-controlled-minibatch-noise.md) | record |  |
| [Curvature/noise validation: frozen_calibration.pdf](support/frozen_calibration.pdf) | support | Smooth tanh MLP; HVP and frozen-state tests. Not cross-architecture Hessian validation. [Observation](observations/27-001-controlled-minibatch-noise.md). [Generator](../../../../analyses/027-2026-10-30-minibatch-noise/03_analyze.py). |
| [Curvature/noise validation: sgd_proxy_validation.pdf](support/sgd_proxy_validation.pdf) | support | Smooth tanh MLP; HVP and frozen-state tests. Not cross-architecture Hessian validation. [Observation](observations/27-001-controlled-minibatch-noise.md). [Generator](../../../../analyses/027-2026-10-30-minibatch-noise/03_analyze.py). |
| [analysis027_hvp_errors.tex](tables/analysis027_hvp_errors.tex) | table |  |
| [analysis027_contrasts.tex](tables/analysis027_contrasts.tex) | table |  |
| [Support data: section_iv_review_checks.json](data/24-section_iv_review_checks.json) | data |  |
| [Support data: endpoints.csv](data/56-endpoints.csv) | data |  |
| [Support data: lag_summary.csv](data/56-lag_summary.csv) | data |  |
| [Analysis027 full measurement report](observations/27-REPORT.md) | record | Preserved source snapshot; current selection/status is governed by PLAN. |
| [Current controlled-figure captions](source/04-controlled-figures.tex) | source | Unmodified manuscript snapshot; its input/figure paths resolve from the original draft root. Use the consolidated snippets and selected displays for integration. |
| [Historical Figure 1 layout manifest](data/figure1-layout-manifest.json) | historical source record | Authenticates the pre-revision source. Current views use the 14 November figure-revision manifest and captions. |
| [Revised figure contracts, captions and timing observations](../FIGURE_NOTES.md) | current figure record | Covers revised Figures 1-6 and the temporal-structure companion; includes every clock conversion, smoother, event marker and coverage limit. |
| [Editable captions for seven revised figures](../figure_captions.tex) | integration source | LaTeX caption macros; does not edit the active manuscript. [Observation](../FIGURE_NOTES.md). |
| [Initial seven-view figure manifest](../figure_revision_manifest.json) | source record | Authenticates the first revision; Figure 6 now has an additional current manifest in V-B/data. [Observation](../../work/figure_revision_20261114/OBSERVATIONS.md). [Generator](../../work/figure_revision_20261114/01_figures.py). |
| [Initial seven-view verification](../figure_revision_verification.json) | verification | Passed checks for the first revision; the larger ResNet peak has additional current verification in V-B/data. [Observation](../../work/figure_revision_20261114/OBSERVATIONS.md). [Generator](../../work/figure_revision_20261114/02_verify.py). |
