# V-B. Architectural tracing

- **Paper claim:** Fixed-probe consecutive motion reveals stage-specific reversals and structured temporal directions.
- **Boundary:** SVD is retrospective and permutation invariant; stage association does not identify a unique cause. Original windows/probes must remain explicit.
- **Authority:** the [single paper plan](../../../PLAN.md) governs selection and closure.
- **Presentation:** Figure 6 uses ResNet steps 390--450 and the +311.53% peak at 415. MLP/Transformer remain unchanged. The separate SVD companion retains its original ResNet window 281--311; historical all-stage PDFs remain in background/figures.

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
| [Figure 6: loss and stage-level representation motion](figures/00-three-model-loss-representation-motion.pdf) | prepared main | ResNet now spans completed steps 390--450 around the +311.53% peak at 415; unchanged native control and probe. MLP/Transformer panels remain identical. No new training or SVD. [Observation](observations/63-007-resnet-major-peak.md). [Generator](../../../work/resnet_peak_revision_20261114/01_figure.py). |
| [Tracing companion: spectra, stage overlap and output coefficients](figures/01-three-model-temporal-structure.pdf) | prepared support | Preserved original fixed-window fits: completed steps 451--501 / 281--311 / 553--601. The ResNet SVD describes the earlier small excursion, not the revised Figure 6 window 390--450. [Observation](../../../work/figure_revision_20261114/OBSERVATIONS.md). [Generator](../../../work/figure_revision_20261114/01_figures.py). |
| [Larger ResNet loss-event trace](observations/63-007-resnet-major-peak.md) | current observation | Larger native-control event; original SVD fits and intervention assets preserved. [Observation](observations/63-007-resnet-major-peak.md). [Generator](../../../../../analyses/063-2026-11-07-architecture-instability-tracing/11_resnet_major_peak.py). |
| [ResNet 390--450 measurements and source checks](data/resnet_390_450_motion.json) | current source record | Larger native-control event; original SVD fits and intervention assets preserved. [Observation](observations/63-007-resnet-major-peak.md). [Generator](../../../../../analyses/063-2026-11-07-architecture-instability-tracing/11_resnet_major_peak.py). |
| [Current Figure 6 numerical manifest](data/resnet_peak_figure_manifest.json) | source record | Larger native-control event; original SVD fits and intervention assets preserved. [Observation](observations/63-007-resnet-major-peak.md). [Generator](../../../work/resnet_peak_revision_20261114/01_figure.py). |
| [Current Figure 6 verification](data/resnet_peak_figure_verification.json) | verification | Larger native-control event; original SVD fits and intervention assets preserved. [Observation](observations/63-007-resnet-major-peak.md). [Generator](../../../work/resnet_peak_revision_20261114/02_verify.py). |

## Consolidated tables

- [spectral controls](tables/spectral-controls.md)

## Editable integration files

- [equations.md](equations.md)
- [equations.tex](equations.tex)
- [algorithm.md](algorithm.md)
- [algorithm.tex](algorithm.tex)
