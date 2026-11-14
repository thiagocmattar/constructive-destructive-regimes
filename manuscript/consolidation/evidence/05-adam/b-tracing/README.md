# V-B. Architectural tracing

- **Paper claim:** Fixed-probe consecutive motion reveals stage-specific reversals and structured temporal directions.
- **Boundary:** SVD is retrospective and permutation invariant; stage association does not identify a unique cause. Original windows/probes must remain explicit.
- **Authority:** the [single paper plan](../../../PLAN.md) governs selection and closure.
- **Presentation:** Figure 6 is assembled: three model columns, loss/cosine/RMS rows, 5/6/6 stage endpoints and synchronized markers. SVD/overlap is a separate companion. Historical all-stage PDFs remain in background/figures.

| Asset | Status | Use / source |
|---|---|---|
| [Historical detailed trace: 01-resnet-instability-trace.pdf](background/figures/01-resnet-instability-trace.pdf) | background | Complete original all-stage report, preserved unchanged. The selected three-model loss/motion and temporal-structure views are now assembled separately. [Observation](observations/63-005-technical-report-event-window.md). [Generator](../../../../../analyses/063-2026-11-07-architecture-instability-tracing/02_figures.py). |
| [Historical detailed trace: 02-transformer-instability-trace.pdf](background/figures/02-transformer-instability-trace.pdf) | background | Complete original all-stage report, preserved unchanged. The selected three-model loss/motion and temporal-structure views are now assembled separately. [Observation](observations/63-005-technical-report-event-window.md). [Generator](../../../../../analyses/063-2026-11-07-architecture-instability-tracing/02_figures.py). |
| [Historical detailed trace: 04-mlp-instability-trace-450-500.pdf](background/figures/04-mlp-instability-trace-450-500.pdf) | background | Complete original all-stage report, preserved unchanged. The selected three-model loss/motion and temporal-structure views are now assembled separately. [Observation](observations/63-005-technical-report-event-window.md). [Generator](../../../../../analyses/063-2026-11-07-architecture-instability-tracing/02_figures.py). |
| [Tracing report: instability-tracing-report.pdf](report/instability-tracing-report.pdf) | report | Original technical report and numbering preserved. The selected Figure 6 is the separate three-model loss/motion overview; spectra use its companion figure. [Observation](observations/63-005-technical-report-event-window.md). [Generator](../../../../../analyses/063-2026-11-07-architecture-instability-tracing/08_report_assets.py). |
| [Tracing report: instability-tracing-report.tex](report/instability-tracing-report.tex) | source | Original technical report and numbering preserved. The selected Figure 6 is the separate three-model loss/motion overview; spectra use its companion figure. |
| [Tracing report: results-table.tex](report/results-table.tex) | source | Original technical report and numbering preserved. The selected Figure 6 is the separate three-model loss/motion overview; spectra use its companion figure. |
| [Report figure: figures/architecture-comparison.pdf](report/figures/architecture-comparison.pdf) | historical support | Original technical report and numbering preserved. The selected Figure 6 is the separate three-model loss/motion overview; spectra use its companion figure. [Observation](observations/63-005-technical-report-event-window.md). [Generator](../../../../../analyses/063-2026-11-07-architecture-instability-tracing/08_report_assets.py). |
| [Report figure: figures/mlp-layer-detail.pdf](report/figures/mlp-layer-detail.pdf) | historical support | Original technical report and numbering preserved. The selected Figure 6 is the separate three-model loss/motion overview; spectra use its companion figure. [Observation](observations/63-005-technical-report-event-window.md). [Generator](../../../../../analyses/063-2026-11-07-architecture-instability-tracing/08_report_assets.py). |
| [Analysis 63: METHOD.md](observations/63-METHOD.md) | record |  |
| [Analysis 63: 001-resnet-trace.md](observations/63-001-resnet-trace.md) | record |  |
| [Analysis 63: 002-transformer-trace.md](observations/63-002-transformer-trace.md) | record |  |
| [Analysis 63: 005-technical-report-event-window.md](observations/63-005-technical-report-event-window.md) | record |  |
| [Analysis 63: 006-spectral-controls.md](observations/63-006-spectral-controls.md) | record |  |
| [Analysis 62: 001-gradient-amplification.md](observations/62-001-gradient-amplification.md) | record |  |
| [Analysis 62: README.md](observations/062-development-summary.md) | record |  |
| [mlp_450_500_results.json](data/mlp_450_500_results.json) | data |  |
| [resnet_results.json](data/resnet_results.json) | data |  |
| [transformer_results.json](data/transformer_results.json) | data |  |
| [mlp_450_500_stages.csv](data/mlp_450_500_stages.csv) | data |  |
| [resnet_stages.csv](data/resnet_stages.csv) | data |  |
| [transformer_stages.csv](data/transformer_stages.csv) | data |  |
| [spectral_controls.json](data/spectral_controls.json) | data |  |
| [Figure 6: loss and stage-level representation motion](figures/00-three-model-loss-representation-motion.pdf) | prepared main | Three model columns and three rows; completed-step alignment, fixed stage endpoints, common cosine scale, per-stage normalized RMS and declared reversal/loss markers. [Observation](../../FIGURE_NOTES.md). [Generator](../../../work/figure_revision_20261114/01_figures.py). |
| [Tracing companion: spectra, stage overlap and output coefficients](figures/01-three-model-temporal-structure.pdf) | prepared support | Original fixed-window fits preserved; SVD and temporal overlap separated from the main loss/motion display. [Observation](../../FIGURE_NOTES.md). [Generator](../../../work/figure_revision_20261114/01_figures.py). |

## Consolidated tables

- [spectral controls](tables/spectral-controls.md)

## Editable integration files

- [equations.md](equations.md)
- [equations.tex](equations.tex)
- [algorithm.md](algorithm.md)
- [algorithm.tex](algorithm.tex)
