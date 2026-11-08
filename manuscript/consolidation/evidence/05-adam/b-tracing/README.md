# V-B. Architectural tracing

- **Paper claim:** Fixed-probe consecutive motion reveals stage-specific reversals and structured temporal directions.
- **Boundary:** SVD is retrospective and permutation invariant; stage association does not identify a unique cause. Original windows/probes must remain explicit.
- **Authority:** the [single paper plan](../../../PLAN.md) governs selection and closure.

| Asset | Status | Use / source |
|---|---|---|
| [Figure 6 component: 01-resnet-instability-trace.pdf](figures/01-resnet-instability-trace.pdf) | assembly component | Use the recorded event window; do not mix with the older 448–800 MLP SVD. [Observation](observations/63-005-technical-report-event-window.md). [Generator](../../../../../analyses/063-2026-11-07-architecture-instability-tracing/02_figures.py). |
| [Figure 6 component: 02-transformer-instability-trace.pdf](figures/02-transformer-instability-trace.pdf) | assembly component | Use the recorded event window; do not mix with the older 448–800 MLP SVD. [Observation](observations/63-005-technical-report-event-window.md). [Generator](../../../../../analyses/063-2026-11-07-architecture-instability-tracing/02_figures.py). |
| [Figure 6 component: 04-mlp-instability-trace-450-500.pdf](figures/04-mlp-instability-trace-450-500.pdf) | assembly component | Use the recorded event window; do not mix with the older 448–800 MLP SVD. [Observation](observations/63-005-technical-report-event-window.md). [Generator](../../../../../analyses/063-2026-11-07-architecture-instability-tracing/02_figures.py). |
| [Tracing report: instability-tracing-report.pdf](report/instability-tracing-report.pdf) | report | The report and its figures retain their original numbering; PLAN Figure 6 is the future combined display. [Observation](observations/63-005-technical-report-event-window.md). [Generator](../../../../../analyses/063-2026-11-07-architecture-instability-tracing/08_report_assets.py). |
| [Tracing report: instability-tracing-report.tex](report/instability-tracing-report.tex) | source | The report and its figures retain their original numbering; PLAN Figure 6 is the future combined display. |
| [Tracing report: results-table.tex](report/results-table.tex) | source | The report and its figures retain their original numbering; PLAN Figure 6 is the future combined display. |
| [Report figure: figures/architecture-comparison.pdf](report/figures/architecture-comparison.pdf) | assembly component | The report and its figures retain their original numbering; PLAN Figure 6 is the future combined display. [Observation](observations/63-005-technical-report-event-window.md). [Generator](../../../../../analyses/063-2026-11-07-architecture-instability-tracing/08_report_assets.py). |
| [Report figure: figures/mlp-layer-detail.pdf](report/figures/mlp-layer-detail.pdf) | assembly component | The report and its figures retain their original numbering; PLAN Figure 6 is the future combined display. [Observation](observations/63-005-technical-report-event-window.md). [Generator](../../../../../analyses/063-2026-11-07-architecture-instability-tracing/08_report_assets.py). |
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

## Consolidated tables

- [spectral controls](tables/spectral-controls.md)

## Editable integration files

- [equations.md](equations.md)
- [equations.tex](equations.tex)
- [algorithm.md](algorithm.md)
- [algorithm.tex](algorithm.tex)
