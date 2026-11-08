# IV. Empirical Characterization

- **Paper claim:** Retain the four current GD/SGD figures in order; supporting HVP, finite-step and all-lag evidence checks their interpretation.
- **Boundary:** Nine GD conditions use one initialization per task; minibatch evidence is replicated. Equal updates do not mean equal data exposure.
- **Authority:** the [single paper plan](../../PLAN.md) governs selection and closure.

| Asset | Status | Use / source |
|---|---|---|
| [Current manuscript source: 04-gd-experiments.tex](source/04-gd-experiments.tex) | source | Unmodified manuscript snapshot; its input/figure paths resolve from the original draft root. Use the consolidated snippets and selected displays for integration. |
| [Current manuscript source: 05-sgd-noise-experiments.tex](source/05-sgd-noise-experiments.tex) | source | Unmodified manuscript snapshot; its input/figure paths resolve from the original draft root. Use the consolidated snippets and selected displays for integration. |
| [Figure 1](figures/gd_loss_projected_change_spectral_clean.pdf) | retained main | GD loss, directional proxy and spectral curvature; full/subset coverage and smoothing differ. [Observation](observations/24-010-curvature-spectral-overlay.md). [Generator](../../../draft/revisions/2026-11-03-original-argument-feedback-feedback/01_prepare_figures.py). |
| [Figure 2](figures/gd_reversal_norm_retention.pdf) | retained main | Gradient norms, adjacent alignment and normalized displacement; no inference of learning quality from R alone. [Observation](observations/24-009-author-feedback-figures.md). [Generator](../../../draft/revisions/2026-11-02-feedback/01_prepare_figures.py). |
| [Figure 3](figures/gd_backward_compact.pdf) | retained main | Fixed endpoint 2256, backward lags 1–256, including a shorter-lag view; not a moving adjacent series. [Observation](observations/35-002-manuscript-view.md). [Generator](../../../draft/revisions/2026-11-02-feedback/01_prepare_figures.py). |
| [Figure 4](figures/sgd_batch_size_compact.pdf) | retained main | Three tasks and three initialization blocks; label batch size and exposure. [Observation](observations/37-001-three-case-sgd.md). [Generator](../../../draft/revisions/2026-11-02-feedback/01_prepare_figures.py). |
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
| [Figure 1 clean-legend source/layout manifest](data/figure1-layout-manifest.json) | source record | Final two-entry legend variant; exact source/figure hashes. |
