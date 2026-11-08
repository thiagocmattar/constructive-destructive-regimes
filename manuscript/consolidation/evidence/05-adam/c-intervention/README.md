# V-C. Representation-informed intervention

- **Paper claim:** The local affine cap improves selected training-window outcomes; negative controls distinguish direction, amplitude and within-layer allocation.
- **Boundary:** Training benefits coexist with held-out deterioration. Historical comparators are not one matched benchmark; direct ranking remains pending.
- **Authority:** the [single paper plan](../../../PLAN.md) governs selection and closure.

| Asset | Status | Use / source |
|---|---|---|
| [09-local-cap-entry-manuscript.pdf](figures/09-local-cap-entry-manuscript.pdf) | support | Supporting result; preserve its own windows, scopes and baseline. [Observation](observations/65-011-local-cap-technical-report.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/34_local_cap_report_assets.py). |
| [Figure 7 base: distributed local cap](figures/10-local-cap-distributed-manuscript.pdf) | main base | Training-window comparisons; attach held-out and negative-control outcomes. [Observation](observations/65-011-local-cap-technical-report.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/34_local_cap_report_assets.py). |
| [11-local-cap-mlp-replication.pdf](figures/11-local-cap-mlp-replication.pdf) | support | Supporting result; preserve its own windows, scopes and baseline. [Observation](observations/65-011-local-cap-technical-report.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/34_local_cap_report_assets.py). |
| [07-distributed-displacement-and-controls.pdf](figures/07-distributed-displacement-and-controls.pdf) | support | Supporting result; preserve its own windows, scopes and baseline. [Observation](observations/65-008-cross-architecture-interventions.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/22_figures_transfer.py). |
| [12-distributed-cap-quantile-lr-mlp.pdf](figures/12-distributed-cap-quantile-lr-mlp.pdf) | support | Supporting result; preserve its own windows, scopes and baseline. [Observation](observations/65-013-distributed-cap-quantile-lr.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/40_plot_cap_lr.py). |
| [13-distributed-cap-lr-seed-diagnostics.pdf](figures/13-distributed-cap-lr-seed-diagnostics.pdf) | support | Supporting result; preserve its own windows, scopes and baseline. [Observation](observations/65-013-distributed-cap-quantile-lr.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/40_plot_cap_lr.py). |
| [14-distributed-cap-quantile-ablation-mlp.pdf](figures/14-distributed-cap-quantile-ablation-mlp.pdf) | support | Supporting result; preserve its own windows, scopes and baseline. [Observation](observations/65-014-cap-quantile-ablation-figure.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/43_plot_cap_ablation.py). |
| [Cap report: local-displacement-cap-report.pdf](report/local-displacement-cap-report.pdf) | report |  [Observation](observations/65-011-local-cap-technical-report.md). [Generator](../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/35_build_local_cap_report.py). |
| [Cap report: local-displacement-cap-report.tex](report/local-displacement-cap-report.tex) | source |  |
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
| [Historical comparator: 01-persistent-half-lr-cifar10-mlp-resnet-wikitext2-transformer.pdf](comparators/01-persistent-half-lr-cifar10-mlp-resnet-wikitext2-transformer.pdf) | historical comparator | Historical windows MLP 512–800, ResNet 256–576, Transformer 544–928; not matched to cap forks. [Observation](observations/58-001-persistent-half-lr.md). [Generator](../../../../../analyses/058-2026-11-06-intervention-consolidation/01_figures.py). |
| [Analysis 58: 001-persistent-half-lr.md](observations/58-001-persistent-half-lr.md) | record |  |
| [Historical comparator: 05-gradient-clipping-cifar10-mlp-resnet.pdf](comparators/05-gradient-clipping-cifar10-mlp-resnet.pdf) | historical comparator | MLP and ResNet historical clipping; activity and placement matter. [Observation](observations/58-005-clipping.md). [Generator](../../../../../analyses/058-2026-11-06-intervention-consolidation/01_figures.py). |
| [Analysis 58: 005-clipping.md](observations/58-005-clipping.md) | record |  |
| [Historical comparator: 01-beta2-reductions-cifar10-mlp.pdf](comparators/01-beta2-reductions-cifar10-mlp.pdf) | historical comparator | Late switch at MLP state 512 retaining moments/clock; not lower beta2 from initialization. [Observation](observations/59-001-beta2-reduction.md). [Generator](../../../../../analyses/059-2026-11-07-mlp-beta2-reduction/01_report.py). |
| [Analysis 59: 001-beta2-reduction.md](observations/59-001-beta2-reduction.md) | record |  |
| [Historical comparator: clip_median_x1.pdf](comparators/clip_median_x1.pdf) | historical comparator | Includes Transformer at 1024–1088; median clipping activates 3/64 times. Different cohort. [Observation](observations/43-007-clipping-median-response.md). [Generator](../../../../../analyses/043-2026-11-03-spark-fixed-step-interventions/01_report.py). |
| [Analysis 43: 007-clipping-median-response.md](observations/43-007-clipping-median-response.md) | record |  |
| [Historical comparator: 01-freeze-first-linear-cifar10-mlp.pdf](comparators/01-freeze-first-linear-cifar10-mlp.pdf) | historical comparator | Freezing removes 93.6% of trainable parameters in MLP; not a unique initiating-layer test. [Observation](observations/64-001-freeze-first-linear.md). [Generator](../../../../../analyses/064-2026-11-07-mlp-freeze-first-linear/02_figures.py). |
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
