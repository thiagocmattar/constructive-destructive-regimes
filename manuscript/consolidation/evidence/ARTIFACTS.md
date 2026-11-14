# Current artifact map

- **Aligned with:** [PLAN.md](../PLAN.md), updated 14 November 2026. This replaces the earlier F1a/F1b/C1–C8 artifact assignments.
- **Source details:** every section README links its local observation and original generator. The [manifest](copy_manifest.json) records source, destination, role, byte count and SHA-256.
- **Caption contract:** revised Figures 1–6 and their companion use [current figure notes](FIGURE_NOTES.md) and [editable captions](figure_captions.tex). They supersede the old display captions while preserving original source observations. Existing report numbering is not final paper numbering.

## Main figures

| Paper slot | Local asset | Readiness |
|---|---|---|
| Figure 1, IV | [GD loss/proxy/spectral curvature](04-gd-sgd/figures/gd_loss_projected_change_spectral_clean.pdf) | Prepared: completed-step axis, both curvature curves explicitly smoothed with the same centered window |
| Figure 2, IV | [Reversal, norms and R](04-gd-sgd/figures/gd_reversal_norm_retention.pdf) | Prepared: gradient coordinates corrected to completed steps; original reductions retained |
| Figure 3, IV | [Backward lags](04-gd-sgd/figures/gd_backward_compact.pdf) | Prepared: consistent state notation; fixed endpoint 2256 and lag axis preserved |
| Figure 4, IV | [Batch effects](04-gd-sgd/figures/sgd_batch_size_compact.pdf) | Prepared: consistent state notation; three initialization blocks and batch axis preserved |
| Figure 5, V-A | [High-LR overview](05-adam/a-characterization/figures/00-high-lr-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf) | Prepared compact view: Q lower limit zero, shading legend moved to caption; population summary remains separate |
| Figure 6, V-B | [Three-model loss and representation motion](05-adam/b-tracing/figures/00-three-model-loss-representation-motion.pdf) | ResNet revised to steps 390–450 around the +311.53% peak at 415; three rows, 5/6/6 stage endpoints and aligned markers |
| Figure 7, V-C | [Matched four-family loss matrix](05-adam/c-intervention/figures/17-matched-intervention-loss.pdf) | Verified: 4 family rows × 3 model columns, all settings and common forks/horizons |
| Figure 8, V-C | [Spike burden versus learning progress](05-adam/c-intervention/figures/18-matched-spike-learning-tradeoff.pdf) | Verified: every setting against unchanged Adam; labels preserve the entire grid |
| Figure 9, V-C | [Matched q50 stage heatmaps](05-adam/c-intervention/figures/19-matched-q50-stage-heatmaps.pdf) | Verified: loss, paired cosine and RMS; full fixed windows, 5/6/6 stage endpoints |

- **Tracing companion:** [SVD, stage overlap and output coefficients](05-adam/b-tracing/figures/01-three-model-temporal-structure.pdf) now matches Figure 6: completed steps 451–501 / 390–450 / 553–601. All ResNet spectral quantities are freshly fitted from replay-qualified captures; MLP/Transformer are unchanged. The original detailed fits and three all-stage PDFs remain under `05-adam/b-tracing/background/figures/` and their canonical sources.

## Tables, equations and algorithms

| Element | Local deliverable | Scope |
|---|---|---|
| V-A all-step table | [Next-step signs](05-adam/a-characterization/tables/next-step-signs.md) | Flagged increases/decreases at one-step lead |
| V-A severity table | [≥10% step increases](05-adam/a-characterization/tables/severity-10pct.md) | Four signals; complete false-alarm data retained alongside |
| V-B control table | [Spectral controls](05-adam/b-tracing/tables/spectral-controls.md) | Same-norm orthogonal and unit-row references |
| V-C matched outcome table | [36-row CSV](05-adam/c-intervention/tables/matched-outcomes.csv), [editable TeX](05-adam/c-intervention/tables/matched-outcomes.tex), [readable PDF](05-adam/c-intervention/figures/23-matched-intervention-outcomes.pdf) | Every condition; PV, mean/final/late training loss and full terminal test loss |
| V-C cost table | [Editable TeX](05-adam/c-intervention/tables/matched-costs.tex), [all 36 timing rows](05-adam/c-intervention/data/matched-costs.csv) | Synchronized components; process high-water memory; no cross-host speed ranking |
| V-C historical cap outcomes | [Earlier paired outcomes](05-adam/c-intervention/tables/distributed-cap-outcomes.md) | Five older baseline/cap pairs; different windows, preserved as support |
| V-C historical conventional outcomes | [Beta2 and clipping outcomes](05-adam/c-intervention/tables/conventional-outcomes.md), [per-arm CSV](05-adam/c-intervention/data/conventional_summary.csv) | Fourteen treatments against their own unchanged controls; mean, tail, maximum, test loss and clipping activity |
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
- Figure 9 uses the same unchanged-Adam stage RMS maximum for both arms. Windows are MLP 512–800, ResNet 256–576 and Transformer 544–928. Stage summaries preserve every probe coordinate; they are not SVD coefficients or averages over every module.
- The [selected figure folder](05-adam/c-intervention/figures/README.md) has seven PDFs / 20 verified pages. Supporting views include all-setting motion, full test trajectories and actual updates/controller activity. Six superseded selections remain [historical](05-adam/c-intervention/background/README.md); the quantile/LR ablation stays at the V-C root.
- Original PDFs are preserved. The earlier standalone heatmap remains an exact extraction of its original page; new matched heatmaps use the new fixed-window data. [Matched captions](05-adam/c-intervention/captions.tex) accompany Figures 7–9 and supporting pages.
