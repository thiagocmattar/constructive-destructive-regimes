# Evidence freshness and consistency audit

- **Result:** the evidence package is now aligned with the latest PLAN. Several former planning files were stale even though most underlying scientific records remained applicable.
- **Active manuscript:** the 6 November rollback recorded in manuscript/README.md governs; 45 source/assets in the original consolidation manifest remain unchanged. Older 18-page revision records are historical, not the active draft.
- **Scope:** local packaging, metadata reconciliation and reconstruction of compact tables; no new training or manuscript editing.

## Corrections made

- Replaced obsolete C1–C8/M01–M24 planning references with the plan's four contribution cards and I–VI structure.
- Replaced former F1a/F1b/F3… assignments with Figures 1–10; V-C now presents LR, beta2, clipping and distributed q50 in the author's order.
- Marked the Adam timing/severity/horizon audit complete and located it in V-A; it is no longer a proposed peak-only analysis.
- Incorporated the completed spectral magnitude controls in V-B.
- Preserved the correct MLP event window 450–500, correlation .642 and rank95=5; older 448–800 values remain explicitly historical.
- Scoped the 1,188 HVP pairs to the tanh MLP and independent derivative calculations, not statistically independent experiments.
- Specified per-module pooled budgets, per-unit scaling, actual Adam increments, target scopes and unbounded incoming-motion terms.
- Included terminal test loss beside training benefits, and kept historical LR/clipping/beta2 controls separate from the still-pending matched comparison.
- Retained the completed 54-continuation MLP quantile/LR ablation directly in V-C as supporting evidence, following the author's latest selection; the main intervention remains distributed q50.
- Moved other representation-inspired interventions and the mixed six-page Figure 07 out of the selected figure folder; extracted its q50 heatmap page intact.
- Corrected artifact generator references and supplied copied source records and executable package verification.
- Removed the old P/E/W task queue as an alternative authority; TASKS now mirrors PLAN closure.

## Checks and their limits

- Every packaged artifact has its repository source and SHA-256 recorded. Transformations are explicit: Markdown headers/links, one historical report's entry-cap dependency path, and the single-page q50 PDF extraction.
- Compact tables are regenerated from retained CSV/JSON reductions with explicit target, window, baseline, split and coverage filters.
- Intervention percentages are independently recomputed from each cap/baseline pair. Test coverage is 10,000 images for MLP/ResNet and 245,550 valid tokens for Transformer.
- ResNet objective units count ten intact normalization groups; Transformer objective units count 7,020 sequences. They are not the loss-reduction units.
- All PDFs are opened and decoded. Ordinary copies remain byte-identical; the extracted heatmap has identical page dimensions, text and rendered pixels to source page 2. No curves were refit or renormalized.
- The selected figure folder contains exactly six one-page PDFs: LR, beta2, clipping and three distributed-q50 displays. All six have been visually inspected. The cap heatmap now anchors Figure 10; its common baseline RMS scaling and windows remain unchanged. Final sizing/caption work remains. Thirteen editable equation/algorithm/table files pass two LaTeX draft-mode passes without errors or overfull boxes.
- The four recent arXiv records in the literature file had title/author/version metadata rechecked; reading-depth limits remain explicit.
- The package audit does not replace the owning analyses' raw-data checks or confirm every literature theorem. It does not declare pending comparisons complete.
- [verification.json](verification.json) records the executed checks; [copy_manifest.json](copy_manifest.json) records source identity. [ARTIFACTS.md](ARTIFACTS.md) indexes the deliverables.

## Still pending

- Final assembly of Figures 5–10 and corresponding captions.
- Qualification/reuse or execution of the plan's matched intervention matrix, cost reporting and held-out time points.
- Any stronger forecasting, causal-source, spike-specific-mode or method-superiority claim needs its separately specified test.
- TNNLS route/format resolution remains a submission-stage item in PLAN, not evidence of permission to resubmit.
