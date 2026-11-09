# Current artifact map

- **Aligned with:** [PLAN.md](../PLAN.md), 8 November 2026. This replaces the earlier F1a/F1b/C1–C8 artifact assignments.
- **Source details:** every section README links its local observation and original generator. The [manifest](copy_manifest.json) records source, destination, role, byte count and SHA-256.
- **Caption contract:** use each local observation's reduction, coverage, uncertainty and limits. Existing report numbering is not final paper numbering.

## Main figures

| Paper slot | Local asset | Readiness |
|---|---|---|
| Figure 1, IV | [GD loss/proxy/spectral curvature](04-gd-sgd/figures/gd_loss_projected_change_spectral_clean.pdf) | Retained main; original clean legend and coverage preserved |
| Figure 2, IV | [Reversal, norms and R](04-gd-sgd/figures/gd_reversal_norm_retention.pdf) | Retained main |
| Figure 3, IV | [Backward lags](04-gd-sgd/figures/gd_backward_compact.pdf) | Retained main; fixed endpoint 2256 |
| Figure 4, IV | [Batch effects](04-gd-sgd/figures/sgd_batch_size_compact.pdf) | Retained main; three initialization blocks |
| Figure 5, V-A | [High-LR overview](05-adam/a-characterization/figures/00-high-lr-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf) | Existing base; attach compact population summary |
| Figure 6, V-B | [Three-case comparison](05-adam/b-tracing/report/figures/architecture-comparison.pdf) and [full components](05-adam/b-tracing/README.md) | Verified components; final grouped heatmap/spectrum/coefficient composition pending |
| Figure 7, V-C | [Learning-rate schedule](05-adam/c-intervention/figures/01-persistent-half-lr-cifar10-mlp-resnet-wikitext2-transformer.pdf) | Selected: persistent half LR, three architectures, each with its own unchanged baseline |
| Figure 8, V-C | [Beta2 schedule](05-adam/c-intervention/figures/15-beta2-reductions-cifar10-mlp-resnet-wikitext2-transformer.pdf) | Selected: all three models, late .999→.95/.9/.5; native moments/clock and bias-correction semantics retained |
| Figure 9, V-C | [Gradient clipping](05-adam/c-intervention/figures/16-gradient-clipping-cifar10-mlp-resnet-wikitext2-transformer.pdf) | Selected: MLP/Transformer thresholds .5/1, ResNet 1; raw/input norms and clipping activity shown |
| Figure 10, V-C | [Distributed q50 stage heatmaps](05-adam/c-intervention/figures/07-distributed-q50-stage-heatmaps.pdf); [scalar diagnostics](05-adam/c-intervention/figures/10-local-cap-distributed-manuscript.pdf) and [MLP replication](05-adam/c-intervention/figures/11-local-cap-mlp-replication.pdf) | Sole representation-inspired intervention. Heatmaps are exact source Figure 07 page 2; final sizing/caption pending |

## Tables, equations and algorithms

| Element | Local deliverable | Scope |
|---|---|---|
| V-A all-step table | [Next-step signs](05-adam/a-characterization/tables/next-step-signs.md) | Flagged increases/decreases at one-step lead |
| V-A severity table | [≥10% step increases](05-adam/a-characterization/tables/severity-10pct.md) | Four signals; complete false-alarm data retained alongside |
| V-B control table | [Spectral controls](05-adam/b-tracing/tables/spectral-controls.md) | Same-norm orthogonal and unit-row references |
| V-C outcome table | [Distributed-cap outcomes](05-adam/c-intervention/tables/distributed-cap-outcomes.md) | Five baseline/cap pairs; training and full terminal test loss |
| V-C conventional outcomes | [Beta2 and clipping outcomes](05-adam/c-intervention/tables/conventional-outcomes.md), [per-arm CSV](05-adam/c-intervention/data/conventional_summary.csv) | Fourteen treatments against their own unchanged controls; mean, tail, maximum, test loss and clipping activity |
| IV HVP tables | [Directional HVP errors](04-gd-sgd/tables/analysis027_hvp_errors.tex), [contrasts](04-gd-sgd/tables/analysis027_contrasts.tex) | Smooth MLP validation; source TeX preserved |
| V-B stage/result tables | [Original three-case table](05-adam/b-tracing/report/results-table.tex), stage CSVs in its section index | Exact recorded windows and probes |
| V-C supporting ablation | [Quantile/LR figure](05-adam/c-intervention/14-distributed-cap-quantile-ablation-mlp.pdf), [grid table](05-adam/c-intervention/tables/cap_lr_table.tex) | Retained sensitivity analysis of the same cap; main intervention remains distributed q50 |
| V-C historical report tables | [Cap results](05-adam/c-intervention/report/cap-results-table.tex), [replication](05-adam/c-intervention/report/cap-seeds-table.tex), [sensitivity](05-adam/c-intervention/report/cap-sensitivity-table.tex) | Audit background; select results according to the current main/ablation scope |
| III equations | [Readable inventory](03-regimes/equations.md), [TeX](03-regimes/equations.tex) | State clock, all pairs/R, curvature/BB/noise, loss criterion |
| V opening/A equations | [Readable inventory](05-adam/a-characterization/equations.md), [TeX](05-adam/a-characterization/equations.tex) | Mixed projection and three distinct future outcomes |
| V-B equations | [Readable inventory](05-adam/b-tracing/equations.md), [TeX](05-adam/b-tracing/equations.tex) | Motion, Gram/SVD, coefficients, overlap and controls |
| V-C equations | [Readable inventory](05-adam/c-intervention/equations.md), [TeX](05-adam/c-intervention/equations.tex) | Exact MLP loss accounting, cap dimensions/metric/bound |
| Algorithm 1 | [Tracing](05-adam/b-tracing/algorithm.md), [TeX](05-adam/b-tracing/algorithm.tex) | Consolidated state notation; original report retained |
| Algorithm 2 | [Local cap](05-adam/c-intervention/algorithm.md), [TeX](05-adam/c-intervention/algorithm.tex) | Current-input local product and native Adam-state advancement |

- The four original compact summary tables have same-name CSV and TeX siblings. The conventional-outcome table links its per-arm and full-curve CSVs in V-C/data. Absolute paired cap values, worker identities and test units are in [distributed-cap-pairs.csv](05-adam/c-intervention/tables/distributed-cap-pairs.csv).
- Complete timing/severity/horizon/warning-memory tables remain in V-A/data; selected display values are not a replacement for that population.
- Figure 10 heatmaps use each stage's common baseline RMS maximum for both arms. MLP shows 448–800; its outcome table also covers 448–1152. Captured-stage summaries retain all probe coordinates and are not SVD coefficients or measurements of every module.
- The [selected figure folder](05-adam/c-intervention/figures/README.md) contains LR, beta2, clipping and distributed q50. The retained quantile/LR ablation sits at the V-C root; other exploratory displays remain [background](05-adam/c-intervention/background/README.md). Same-looking curves do not make forks/hardware/budgets matched.
- Original PDFs are preserved. The standalone heatmap selects one original page without cropping, refitting or renormalization; extraction is recorded and checked against the original page.
