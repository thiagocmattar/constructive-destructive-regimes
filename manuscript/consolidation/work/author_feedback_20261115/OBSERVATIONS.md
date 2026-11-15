# Author-feedback revision — 15 November 2026

- **Read next:** [manuscript.pdf](../../draft/manuscript.pdf), 18 pages. The supplement remains 33 pages.
- **Scope:** all 62 annotations in the supplied 17-page PDF, including the request to extend the clarity pass through the remainder of the paper.
- **Original:** [archived annotated PDF](../../draft/archive/2026-11-15/manuscript+feedback.pdf). Annotation text, page locations, rectangles, and PDF hash are retained in [annotations.json](annotations.json).
- **No new training:** the new table is a reduction of the same verified Run215 histories. No cloud resources were provisioned. Original evidence, human inputs, and the active manuscript outside consolidation remain unchanged.
- **Reproduction:** `01_extract.py` extracts annotations; `02_diagnostic_table.py` rebuilds the 0.1% table. `03_revise.py` and `04_tracing_revision.py` are historical editorial scripts, not safe entry points after further author edits. Use the existing `readability_revision_20261114/02_figures.py`, `03_build.py`, `04_verify.py`, and `05_review_sheets.py` for figures, compilation, verification, and page rendering.

## Diagnostic table contract and findings

- Same highest-rate MLP/ResNet/Transformer trajectories as Fig. 5; all decisions `t=1..1023`.
- Target: `(L[t+1]-L[t])/L[t] >= 0.001`. Event counts: **446 / 88 / 142**. No target-based decision filtering.
- Previous-loss baseline: `L[t] > L[t-1]`, strictly positive previous change. This information is available before applying the outgoing update.
- Accuracy: `(TP+TN)/N`. Always predicting no increase scores **56.4 / 91.4 / 86.1%**.
- Combined angle/norm condition scores **59.9 / 92.0 / 87.2%**, versus **58.1 / 89.0 / 86.5%** for the loss-history baseline.
- Warning hit rate is precision, not classification accuracy. Its event-rate lift is **1.28 / 6.78 / 3.87×**, versus **1.19 / 4.28 / 3.70×** for loss history.
- Strict angle/norm and non-strict `eta*kappa_hat >= 2` agree because none of the recorded values equals two. They appear as separate rows at the author's request, explicitly labeled equivalent.
- Full confusion matrices and additional accuracy ratios are in [diagnostic_0p1pct.csv](diagnostic_0p1pct.csv); source hashes and measurement contract are in [diagnostic_manifest.json](diagnostic_manifest.json). The supplement's Table S3 exposes the confusion counts.
- Existing 0.1% severity reductions independently match every gradient-condition confusion count. These are retrospective, temporally dependent observations from one initialization per model. The manuscript states the before-update association without presenting a validated prospective forecaster.

## Mathematical and implementation audit

- Read Run215 `01_train.py`, its optimizer config, and `src/eos_tasks/adam_diagnostics.py`. Native Adam uses `(0.9,0.999)`, epsilon `1e-8`, no decay or AMSGrad. The measured quantity is `g_prev dot (g_prev-g_current) / ||g_prev||^2`, using raw full-objective gradients before the next update.
- Restored the requested `eta*kappa_hat` label. Its Adam interpretation is now derived as a gradient-projected Hessian response along the actual Adam direction. The actual-step Rayleigh curvature remains separately defined as `kappa_hat^step`. The two coincide for plain GD but need not coincide for Adam.
- Corrected the proposed Adam identity's state indices to the paper's convention: the preceding update connects `theta[t-1]` to `theta[t]`.
- Added 128 smooth-objective numerical checks of the projected-response identity alongside the existing applied-step and finite-step loss checks.
- Checked the initial raw cosines: MLP's first pair is negative (`-0.27888`), then its early constructive interval appears. The prose includes this exception instead of asserting that every first pair is constructive.
- Preserved the distinction between directional curvature and largest Hessian eigenvalue, finite-step and starting-point curvature, displacement retention and retained information, and observed timing and causality.

## Literature checks

- [Cohen et al., Appendix I.2 and I.4](https://arxiv.org/html/2103.00065v3#A9): costly sharpness calculations use the first 5,000 CIFAR-10 examples / five intact normalization groups and 2,500 WikiText examples. This directly supports the explanation of subset spectral measurements. Reference-set representation tracing has its own coverage and is not presented as the same estimator.
- [Li et al., NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/a41b3bb3e6b050b6c9067c67f663b915-Abstract.html): architecture and skip connections affect loss geometry. Used to motivate a representation-learning view, without calling neural loss surfaces universally fractal or unstructured.
- [Bai et al., ICML 2026](https://proceedings.mlr.press/v306/bai26o.html): adaptive preconditioners can generate spikes; cited in the motivation and Adam interpretation. We do not claim that our gradient plots independently establish their proposed mechanism.
- Applied the author's [research-framing skill](../../human/review-skill/SKILL%20(2).md): five explicit gap-to-contribution bullets, with the tracing and intervention contributions separated and claims bounded by the available evidence.

## Comment-by-comment resolution

Numbers follow PDF annotation order; original text is preserved verbatim in `annotations.json`.

| ID | Original page | Resolution in the updated paper |
|---|---:|---|
| 1 | 1 | Broadened the title to “From Gradient Interactions to Functional Oscillations”; did not imply empirical universality across all optimizers or tasks. |
| 2 | 1 | Introduction heuristics now match schedules, clipping, momentum, and second moments. |
| 3 | 1 | Rebuilt motivation around learning and representation changes; cited architecture-dependent loss geometry; separated spike suppression from later progress. |
| 4 | 1 | Added Bai et al. to the motivation paragraph. |
| 5 | 1 | Added the proposed displacement → regimes → curvature → representations progression; stated accumulated displacement rather than an unproved convergence theorem. |
| 6 | 2 | Split functional tracing and selective control into distinct contributions. |
| 7 | 2 | First contribution now leads with constructive/destructive regimes and explains net displacement in ordinary language. |
| 8 | 2 | Curvature contribution leads with Hessian cost and states the existing-gradient computation. |
| 9 | 2 | Added the angle/norm loss-rise inequality inline; retained its quadratic-GD scope and directional interpretation. |
| 10 | 2 | Added architecture-specific tracing findings to the contribution. |
| 11 | 2 | Added 5/20/15 output patterns and 51/61/49 rank bounds. |
| 12 | 2 | Explained why functional oscillation motivates size control but does not establish ineffective learning. |
| 13 | 2 | Used “functional oscillation” throughout framing and tracing. |
| 14 | 4 | Defined final step T before the displacement sum. |
| 15 | 4 | Named kappa and epsilon alongside their verbal definitions. |
| 16 | 5 | Expanded the conditional-noise expectation derivation; removed colons immediately before displayed equations. |
| 17 | 5 | Explained numerator signal and denominator noise in attenuation, distinguishing fixed-state and evolving-trajectory claims. |
| 18 | 5 | Compacted normalized displacement explanation and removed the redundant equation reference. |
| 19 | 5 | Emphasized retained parameter displacement and repeated undoing; did not interpret it as a fraction of useful learning. |
| 20 | 5 | Changed equation annotation to “Interference.” |
| 21 | 5 | Explained explicitly how positive/negative rho affects the loss interference term. |
| 22 | 5 | Added required norm-growth factors for rho = -1, -0.5, -0.2. |
| 23 | 5 | Restored the three curvature regimes and their relation to the angle/norm condition. |
| 24 | 6 | Merged setup prose and delegated numerical details to Table I and the supplement. |
| 25 | 6 | Explained why curvature near two needs the separate norm/reversal view. |
| 26 | 6 | Consolidated the spectral discussion around Fig. 1. |
| 27 | 6 | Introduced Fig. 2 as the missing magnitude/direction/displacement information. |
| 28 | 6 | Explained subset spectral cost and EoS precedent; full-objective gradient coverage is distinguished from finite-step curvature approximation. |
| 29 | 6 | Expanded interpretation of reversal, shrinking retention, and possible valley traversal; did not assert basin jumps from scalar curves. |
| 30 | 6 | Discussed constructive-to-destructive transition; deferred representation overwriting to direct functional measurements. |
| 31 | 6 | Retained oscillation-envelope discussion and explained recurrence without an explicit GD momentum state. |
| 32 | 6 | Cited the attenuation equation at the start of the sampling discussion. |
| 33 | 6 | Explained equal optimizer-step counts versus unequal data exposure; removed the supplementary-section detour. |
| 34 | 6 | Consolidated attenuation and its stability interpretation, preserving norm/trajectory qualifications. |
| 35 | 7 | Removed the separate curvature-check subsection; kept independent checks in S2 and a short pointer with Fig. 1. |
| 36 | 7 | Split Adam spikes/tracing (V) from interventions (VI). |
| 37 | 8 | Separated Adam moment recurrences from the bias-corrected direction/update display. |
| 38 | 8 | Checked native implementation and used the proposed actual-trajectory argument without Q notation. |
| 39 | 8 | Derived the projected gradient response with corrected state indexing and distinguished actual-step curvature. |
| 40 | 8 | Added rate grid, shared objectives, initialization, horizon, and highest-rate display selection. |
| 41 | 8 | Restored eta*kappa_hat in the overview and aligned supplement labels/prose. |
| 42 | 9 | Expanded spike/recovery discussion, second-moment context, norm/reversal relation, and subsequent progress; qualified proportionality, basin, flatness, and premature-convergence interpretations. |
| 43 | 10 | Replaced the compact severe-increase table with a full-width diagnostic comparison. |
| 44 | 10 | Used 0.1%, event rates, previous-loss baseline, accuracy, warning hit rate, and lift; exposed equivalent condition rows. |
| 45 | 10 | Preserved the bridge from whole-network gradients to internal functional changes. |
| 46 | 10 | Centered characterization on before-update results and the new table; moved detailed counts to S3. |
| 47 | 10 | Explained reference-subset feasibility, normalization preservation, and distinct measurement coverage. |
| 48 | 11 | Defined a stage as a named computation output, with layer, attention, and residual-group examples. |
| 49 | 11 | Defined activation matrix rows and columns, including token/spatial positions. |
| 50 | 11 | Gave delta's dimension P = N*m and its coordinate interpretation. |
| 51 | 11 | Replaced tracing subsubsections by explicit questions; intervention divisions are ordinary subsections of their new section. |
| 52 | 11 | Defined affine as linear map plus bias. |
| 53 | 11 | Expanded architecture-specific functional regimes and timing; marked backward-recruitment interpretation as a hypothesis. |
| 54 | 11 | Defined singular values, spatial patterns, temporal coefficients, and reconstruction. |
| 55 | 11 | Explained one SVD per stage over the entire same-input change window, not an SVD per step. |
| 56 | 11 | Rewrote orthogonal and unit-length controls around the questions they answer. |
| 57 | 11 | Gave pattern counts, rank bounds, percentages of available directions, and Transformer coordinate coverage. |
| 58 | 11 | Explained why uncentered squared movement is not variance; updated figure wording accordingly. |
| 59 | 11 | Applied clarity pass through remaining tracing, cap computation, comparisons, interpretation, and conclusion review. |
| 60 | 11 | Added the fixed-pattern example and formal reconstruction; separated patterns from steps and per-stage subspaces from a shared network basis. |
| 61 | 12 | Replaced “stage maximum” axis text with relative RMS notation and defined the window maximum in the caption. |
| 62 | 12 | Moved and introduced Algorithm 1 at the start of tracing. |

## Figure and document verification

- Numerical curves, event windows, reduction rules, and original evidence files are unchanged. Figure changes affect the Adam notation and tracing labels, including supplemental warning legends.
- Label-only exports preserve numerical artist hashes. Exact array comparisons cover 351,994 values in the GD/overview and tracing variants; intervention plots retain their matched manifests. Four metadata-only intervention PDF regenerations were restored to original bytes after matching text and rendered pixels.
- Full main and supplement compilation; no warnings or unresolved references; embedded fonts and page bounds verified. The 144 intervention table cells remain verified against source curves and held-out results.
- All 18 main and 33 supplementary pages inspected as contact sheets; the new table, tracing figures, revised equations, and title pages also inspected at larger resolution. A malformed algorithm-reference escape caught visually was corrected and added to automated checks.
- Final hashes, page inspections, and mathematical errors are recorded in the existing [verification](../readability_revision_20261114/verification.json) and [visual review](../readability_revision_20261114/visual_review.json). All 416 protected source files pass preservation checks.
