# V-C. Intervention comparisons

- **Paper claim:** Present learning-rate schedules, beta2 schedules, gradient clipping and the distributed q50 displacement cap; each retains its unchanged Adam baseline.
- **Boundary:** Training benefits coexist with held-out deterioration. Historical comparators are not one matched benchmark; direct ranking remains pending.
- **Authority:** the [single paper plan](../../../PLAN.md) governs selection and closure.
- **Presentation:** The figures/ folder contains the four selected families, with distributed q50 as the sole representation-inspired method. The quantile/LR ablation is retained at the section root as supporting evidence.

| Asset | Status | Use / source |
|---|---|---|
| [Figure 7: learning-rate schedule](figures/01-persistent-half-lr-cifar10-mlp-resnet-wikitext2-transformer.pdf) | selected conventional intervention | Historical windows MLP 512–800, ResNet 256–576, Transformer 544–928; not matched to cap forks. [Observation](observations/58-001-persistent-half-lr.md). [Generator](../../../../../analyses/058-2026-11-06-intervention-consolidation/01_figures.py). |
| [Figure 8: beta2 schedule](figures/01-beta2-reductions-cifar10-mlp.pdf) | selected conventional intervention | Late switch at MLP state 512 retaining moments/clock; not lower beta2 from initialization. [Observation](observations/59-001-beta2-reduction.md). [Generator](../../../../../analyses/059-2026-11-07-mlp-beta2-reduction/01_report.py). |
| [Figure 9: gradient clipping](figures/05-gradient-clipping-cifar10-mlp-resnet.pdf) | selected conventional intervention | MLP and ResNet historical clipping; activity and placement matter. [Observation](observations/58-005-clipping.md). [Generator](../../../../../analyses/058-2026-11-06-intervention-consolidation/01_figures.py). |
| [Figure 10 selected base: distributed q50 stage heatmaps](figures/07-distributed-q50-stage-heatmaps.pdf#page=1) | main base | Exact page 2 from the full historical PDF, retaining vector content. Only unchanged Adam and distributed q50. Common baseline RMS maxima; MLP 448-800, ResNet 264-576, Transformer 552-928. [Observation](observations/65-008-cross-architecture-interventions.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/22_figures_transfer.py). |
| [Distributed q50: scalar diagnostics](figures/10-local-cap-distributed-manuscript.pdf) | q50 supporting display | Distributed-q50 scalar diagnostics supporting the selected heatmaps; retain held-out outcomes and the recorded evaluation window. [Observation](observations/65-011-local-cap-technical-report.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/34_local_cap_report_assets.py). |
| [Distributed q50: three-initialization MLP replication](figures/11-local-cap-mlp-replication.pdf) | q50 supporting display | Supporting result; preserve its own windows, scopes and baseline. [Observation](observations/65-011-local-cap-technical-report.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/34_local_cap_report_assets.py). |
| [Supporting ablation: cap quantile by learning rate](14-distributed-cap-quantile-ablation-mlp.pdf) | supporting ablation | Retained explicitly for the paper: q25/q50/q75/q90/q99 at 1x/2x/4x LR, three initializations. Shared log-loss scale; mean +/- sample SD. Main intervention remains distributed q50. [Observation](observations/65-014-cap-quantile-ablation-figure.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/43_plot_cap_ablation.py). |
| [Background source: 09-local-cap-entry-manuscript.pdf](background/figures/09-local-cap-entry-manuscript.pdf) | background only | Preserved historical evidence; excluded from the selected four-part presentation. [Observation](observations/65-011-local-cap-technical-report.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/34_local_cap_report_assets.py). |
| [Background source: 07-distributed-displacement-and-controls.pdf](background/figures/07-distributed-displacement-and-controls.pdf) | background only | Preserved historical evidence; excluded from the selected four-part presentation. [Observation](observations/65-008-cross-architecture-interventions.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/22_figures_transfer.py). |
| [Background source: 12-distributed-cap-quantile-lr-mlp.pdf](background/figures/12-distributed-cap-quantile-lr-mlp.pdf) | background only | Preserved historical evidence; excluded from the selected four-part presentation. [Observation](observations/65-013-distributed-cap-quantile-lr.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/40_plot_cap_lr.py). |
| [Background source: 13-distributed-cap-lr-seed-diagnostics.pdf](background/figures/13-distributed-cap-lr-seed-diagnostics.pdf) | background only | Preserved historical evidence; excluded from the selected four-part presentation. [Observation](observations/65-013-distributed-cap-quantile-lr.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/40_plot_cap_lr.py). |
| [Cap report: local-displacement-cap-report.pdf](report/local-displacement-cap-report.pdf) | report |  [Observation](observations/65-011-local-cap-technical-report.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/35_build_local_cap_report.py). |
| [Cap report: local-displacement-cap-report.tex](report/local-displacement-cap-report.tex) | source | Historical report source; entry-cap dependency path rebased to background, scientific content retained. |
| [Cap report: cap-results-table.tex](report/cap-results-table.tex) | source |  |
| [Cap report: cap-seeds-table.tex](report/cap-seeds-table.tex) | source |  |
| [Cap report: cap-sensitivity-table.tex](report/cap-sensitivity-table.tex) | source |  |
| [Analysis 65: 008-cross-architecture-interventions.md](observations/65-008-cross-architecture-interventions.md) | record |  |
| [Analysis 65: 009-what-the-negative-controls-show.md](observations/65-009-what-the-negative-controls-show.md) | record |  |
| [Analysis 65: 010-local-displacement-cap-geometry.md](observations/65-010-local-displacement-cap-geometry.md) | record |  |
| [Analysis 65: 011-local-cap-technical-report.md](observations/65-011-local-cap-technical-report.md) | record |  |
| [Analysis 65: 013-distributed-cap-quantile-lr.md](observations/65-013-distributed-cap-quantile-lr.md) | record |  |
| [Intervention data: transfer_summary.csv](data/transfer_summary.csv) | data |  |
| [Intervention data: transfer_heldout_evaluations.csv](data/transfer_heldout_evaluations.csv) | data |  |
| [Intervention data: cap_lr_aggregate.csv](data/cap_lr_aggregate.csv) | data |  |
| [Intervention data: cap_lr_per_seed.csv](data/cap_lr_per_seed.csv) | data |  |
| [Intervention data: cap_lr_table.tex](tables/cap_lr_table.tex) | table |  |
| [Analysis 58: 001-persistent-half-lr.md](observations/58-001-persistent-half-lr.md) | record |  |
| [Analysis 58: 005-clipping.md](observations/58-005-clipping.md) | record |  |
| [Analysis 59: 001-beta2-reduction.md](observations/59-001-beta2-reduction.md) | record |  |
| [Historical comparator: clip_median_x1.pdf](background/comparators/clip_median_x1.pdf) | background only | Includes Transformer at 1024–1088; median clipping activates 3/64 times. Different cohort. [Observation](observations/43-007-clipping-median-response.md). [Generator](../../../../../analyses/043-2026-11-03-spark-fixed-step-interventions/01_report.py). |
| [Analysis 43: 007-clipping-median-response.md](observations/43-007-clipping-median-response.md) | record |  |
| [Historical comparator: 01-freeze-first-linear-cifar10-mlp.pdf](background/comparators/01-freeze-first-linear-cifar10-mlp.pdf) | background only | Freezing removes 93.6% of trainable parameters in MLP; not a unique initiating-layer test. [Observation](observations/64-001-freeze-first-linear.md). [Generator](../../../../../analyses/064-2026-11-07-mlp-freeze-first-linear/02_figures.py). |
| [Analysis 64: 001-freeze-first-linear.md](observations/64-001-freeze-first-linear.md) | record |  |
| [Author's annotated cap report](source/author-feedback.pdf) | author input | Unmodified copy; original remains in Analysis 065. |
| [Analysis058 source summary](observations/058-summary.md) | record | Preserved source snapshot; current selection/status is governed by PLAN. |
| [Analysis059 source summary](observations/059-summary.md) | record | Preserved source snapshot; current selection/status is governed by PLAN. |
| [Analysis065 source summary](observations/065-summary.md) | record | Preserved source snapshot; current selection/status is governed by PLAN. |
| [Analysis 65: cap-quantile ablation caption and verification](observations/65-014-cap-quantile-ablation-figure.md) | record | Exact reduction and uncertainty for the shared-axis ablation figure. |

## Consolidated tables

- [distributed cap outcomes](tables/distributed-cap-outcomes.md)

## Editable integration files

- [equations.md](equations.md)
- [equations.tex](equations.tex)
- [algorithm.md](algorithm.md)
- [algorithm.tex](algorithm.tex)
