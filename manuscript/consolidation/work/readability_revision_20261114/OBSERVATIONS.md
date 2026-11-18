# Author-guided revision, 14 November 2026

- **Review artifact:** [main manuscript](../../draft/manuscript.pdf), 17 IEEE-style pages; [supplement](../../draft/archive/2026-11-18/supplementary.pdf), 33 pages. The author only needs the main PDF to review the narrative.
- **Authority:** the author's latest comments supersede the broader rewrite in commit `20d783d`. The baseline is the preserved `manuscript/draft/bare_jrnl.pdf` and its TeX sources. The supplied [review skill](../../human/review-skill/SKILL%20(2).md) informs clarity and claim discipline within that structure.
- **Scope:** all edits stay in consolidation. No scientific configuration, observation, run, source figure, active manuscript, or human input is changed. No cloud execution or push.

## What changed

| Author request | Revision |
|---|---|
| Keep the original introduction and stability survey | Restore scaling/efficiency motivation, curvature and stochastic stability, EoS, then constructive/destructive behavior. Keep the four original stability subsections, equations, and practical pretraining examples. |
| Keep the constructive/destructive and empirical sections close | Restore their explanatory order and examples. Retain the all-pair displacement identity, directional alignment, sampling, normalized displacement, and loss condition; retain all four GD/SGD figures. Add the improved setup table and concise validation discussion. |
| Introduce cheap curvature along any optimizer's step | Define the estimate from the applied increment and consecutive gradient difference. Prove the smooth path-average identity, pointwise error bound, and sampled-gradient term. Remove the secant-centered comparison from the paper and plan; make no first-ever novelty claim. |
| Use t for steps | Define t as completed updates and evaluate g_t at theta_t before its outgoing update. Regenerate current publication variants in `draft/figures/`; retain all numerical coordinates and reductions. The supplementary continuation index k is explicitly k=t-t0. |
| Make Adam clear | Explain consecutive gradients first, then the overview panels and severity table, representation motion and its temporal decomposition, and controlled interventions. Replace fixed-probe terminology with reference examples/reference set. Detailed counts and motion percentages move to the supplement. |

## Scientific distinctions retained

- **Actual-step curvature:** `kappa_hat_step = u^T(delta_g)/||u||^2`, with `u=theta_t-theta_(t-1)`. For smooth full gradients this is exactly the directional Hessian curvature averaged over that completed step. It is not a full Hessian reconstruction or a next-step guarantee.
- **Existing Adam measurements:** `Q_t=1-rho_t*||g_t||/||g_(t-1)||` still measures reversal and amplification. Under plain SGD it equals `eta*kappa_hat`; under Adam it generally differs from actual-step curvature. Figure 5 is relabeled Q_t without changing a single observed value. A new empirical Adam actual-step-curvature comparison is not claimed.
- **Evidence scope:** nine GD conditions; full Adam timing/severity audit; revised ResNet 390–450 tracing window in both figures; 36 matched intervention conditions with 11,904 total/10,912 treated updates. The single-initialization scope, endpoint/test/cost tradeoffs, and separate historical MLP replication remain explicit.
- **Display:** current draft uses 17 vector figure PDFs, nine main figures, three main tables, two algorithms, and 36 numbered main equations. Preserved evidence PDFs remain source records and may retain previous notation; PLAN links to current variants where available.

## Verification and reproduction

- [Numerical and document checks](verification.json): 416 protected files unchanged; all 144 intervention metrics independently recomputed and found in the compiled table; 24 severity hit/precision values; spectral controls; output-loss decomposition; 128 smooth SGD/cap cases and 128 arbitrary or Adam-like applied-step cases.
- [Figure checks](figure_verification.json): exact retained-array comparisons for GD/SGD, Adam overview and both current tracing figures; exact main intervention point/heatmap/table manifests; before/after numerical-artist hashes for all regenerated exports. These are presentation changes, not new experiments.
- [Build record](build.json): no compilation warnings, unresolved references, missing embedded fonts, or text outside the pages. [Visual review](visual_review.json) is tied to final PDF hashes. Float placement keeps the intervention displays with their results and before the conclusion.
- Build with `02_figures.py` if figure variants need regeneration, then `03_build.py`, `04_verify.py`, and `05_review_sheets.py`. Auxiliary files and images stay in ignored `scratch/`. `01_restore_core.py` records the initial restoration only; it is not a build prerequisite and must not overwrite subsequent editorial changes.
- The earlier rewrite and its audit remain recoverable in Git. The current author-review draft is not a declaration of TNNLS page-limit compliance or submission approval.

## Rechecked practical-stability sources

- Checkpoint recovery and skipped batches are reported in [PaLM, Section 5.1](https://arxiv.org/pdf/2204.02311#page=11).
- Checkpoint recovery, rate reduction, and tighter clipping are reported in [OPT, Section 2.5](https://arxiv.org/pdf/2205.01068#page=3).
- Initialization, normalization, optimizer choices, and loss/gradient spike monitoring are covered by [OLMo 2, Section 3](https://arxiv.org/html/2501.00656v3).
- Query/key weight rescaling and the reported spike-free K2 run are described in [Kimi K2, Section 2.1](https://arxiv.org/html/2507.20534v2). These motivate the problem; they are not comparative evidence for the present cap.


## Contribution framing follow-up

The author requested stronger contributions in the current draft. Only the introduction's contribution block is rewritten; the preceding introduction, scientific sections, measurements and figures are retained. Each item now states prior knowledge, the specific question, the paper's response, and supporting evidence with section references.

| Contribution | Prior knowledge / remaining question | Evidence used here |
|---|---|---|
| Local reversal and accumulation | EoS permits oscillatory progress; stationarity diagnostics use gradient products. How do adjacent opposition and complete-trajectory accumulation differ? | Exact displacement identity; negative adjacent versus positive all-pair interaction; observed loss progress. |
| Inexpensive applied-step curvature | Spectral sharpness does not determine curvature along a particular applied step. | Optimizer-independent path-average identity and error bound; explicit SGD noise term; independent HVP comparison. |
| Reversal and spike formation | Bai explains adaptive preconditioning and directional curvature. How do consecutive-gradient reversal and growth relate to observed increases across severity and timing? | SGD joint condition; exhaustive Adam audit retaining missed increases and false alarms. No validated forecasting claim. |
| Functional tracing and control | Representation-aware update control and unit-wise clipping already exist. Where does oscillation appear, and does capping local functional movement help? | Shared reference-example tracing; temporal structure; matched four-family comparison. Training improvements retain endpoint/test/cost tradeoffs. |

Primary sources rechecked: [Cohen et al., abstract](https://arxiv.org/abs/2103.00065), [Chee and Toulis, abstract](https://proceedings.mlr.press/v84/chee18a.html), [Bai et al., introduction/contributions and Sections 4-5](https://arxiv.org/html/2506.04805v2), [Kosson et al., Section 6 and Appendix B.6](https://arxiv.org/html/2410.23922v1), and [Brock et al., adaptive clipping](https://proceedings.mlr.press/v139/brock21a.html). Gaps are specific questions and measurement distinctions, not unsupported claims that the literature has ignored oscillation or representations. The supplied review skill's optional reference files were not included in its folder; its core framing and adversarial-audit instructions were applied directly.

Final check: the main PDF remains 17 pages and compiles without warnings. All pages were visually reviewed after reflow; three float-placement options were adjusted to avoid overflow. The supplement and every figure asset are unchanged. Numerical/document verification passes.
