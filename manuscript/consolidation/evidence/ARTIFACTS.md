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
| Figure 7, V-C | [Distributed cap](05-adam/c-intervention/figures/10-local-cap-distributed-manuscript.pdf) and [motion/control display](05-adam/c-intervention/figures/07-distributed-displacement-and-controls.pdf) | Existing bases; matched comparator ranking remains pending |

## Tables, equations and algorithms

| Element | Local deliverable | Scope |
|---|---|---|
| V-A all-step table | [Next-step signs](05-adam/a-characterization/tables/next-step-signs.md) | Flagged increases/decreases at one-step lead |
| V-A severity table | [≥10% step increases](05-adam/a-characterization/tables/severity-10pct.md) | Four signals; complete false-alarm data retained alongside |
| V-B control table | [Spectral controls](05-adam/b-tracing/tables/spectral-controls.md) | Same-norm orthogonal and unit-row references |
| V-C outcome table | [Distributed-cap outcomes](05-adam/c-intervention/tables/distributed-cap-outcomes.md) | Five baseline/cap pairs; training and full terminal test loss |
| IV HVP tables | [Directional HVP errors](04-gd-sgd/tables/analysis027_hvp_errors.tex), [contrasts](04-gd-sgd/tables/analysis027_contrasts.tex) | Smooth MLP validation; source TeX preserved |
| V-B stage/result tables | [Original three-case table](05-adam/b-tracing/report/results-table.tex), stage CSVs in its section index | Exact recorded windows and probes |
| V-C report/ablation tables | [Cap results](05-adam/c-intervention/report/cap-results-table.tex), [replication](05-adam/c-intervention/report/cap-seeds-table.tex), [sensitivity](05-adam/c-intervention/report/cap-sensitivity-table.tex), [LR/quantile grid](05-adam/c-intervention/tables/cap_lr_table.tex) | Supporting historical tables; preserve their own comparisons |
| III equations | [Readable inventory](03-regimes/equations.md), [TeX](03-regimes/equations.tex) | State clock, all pairs/R, curvature/BB/noise, loss criterion |
| V opening/A equations | [Readable inventory](05-adam/a-characterization/equations.md), [TeX](05-adam/a-characterization/equations.tex) | Mixed projection and three distinct future outcomes |
| V-B equations | [Readable inventory](05-adam/b-tracing/equations.md), [TeX](05-adam/b-tracing/equations.tex) | Motion, Gram/SVD, coefficients, overlap and controls |
| V-C equations | [Readable inventory](05-adam/c-intervention/equations.md), [TeX](05-adam/c-intervention/equations.tex) | Exact MLP loss accounting, cap dimensions/metric/bound |
| Algorithm 1 | [Tracing](05-adam/b-tracing/algorithm.md), [TeX](05-adam/b-tracing/algorithm.tex) | Consolidated state notation; original report retained |
| Algorithm 2 | [Local cap](05-adam/c-intervention/algorithm.md), [TeX](05-adam/c-intervention/algorithm.tex) | Current-input local product and native Adam-state advancement |

- Each regenerated summary table also has same-name CSV and TeX siblings. Absolute paired cap values, worker identities and test units are in [distributed-cap-pairs.csv](05-adam/c-intervention/tables/distributed-cap-pairs.csv).
- Complete timing/severity/horizon/warning-memory tables remain in V-A/data; selected display values are not a replacement for that population.
- Historical comparator figures and observations remain in V-C/comparators and V-C/observations. Same-looking curves do not make forks/hardware/budgets matched.
- Publication PDF sources are preserved; no figure was cropped, refitted or re-normalized in packaging.
