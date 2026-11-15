# Sequential readability revision of Sections I and II

15 November 2026. Editorial revision of the [consolidation manuscript](../../draft/manuscript.pdf), following the user's requested sequence: read each paragraph, diagnose its problems, propose a solution, and only then edit it.

## Review record and scope

The [paragraph review](PARAGRAPH_REVIEW.md) contains the original text, diagnosis, proposed solution, and replacement for all 29 original prose units in reading order: 12 in Section I and 17 in Section II. Contributions count as separate units; displayed equations are protected. Several units were divided into shorter paragraphs. Seven subsequent coherence and concision reviews are recorded at the end, yielding 36 audit entries.

The sequential helper writes and flushes each diagnosis and proposal before changing the corresponding source paragraph. Replaying all recorded replacements against the baseline reproduces the final source files exactly. The review therefore retains intermediate wording as well as the final coherence decisions.

The edited scientific text is confined to [Section I](../../draft/sections/01-introduction.tex) and [Section II](../../draft/sections/02-related-work.tex). The main PDF and draft README are refreshed. The abstract, Section III onward, mathematical displays, bibliography source, figures, tables, supplement, and supplied originals are preserved. No experiments were run or empirical results added.

## Writing principles applied

- Give each paragraph a clear job and make the reason for the next paragraph explicit.
- Begin with the scientific problem before listing techniques or related work.
- Explain a quantity's meaning before relying on its technical name. Examples include curvature, sharpness, Hessian--vector products, covariance, warmup, and warning lift.
- Explain changes of object: gradients versus applied updates, parameters versus representations, and a completed-step identity versus a warning about the next step.
- Prefer concrete actions and comparisons over compressed noun phrases. Name what changes, what is measured, and what the comparison can establish.
- Keep necessary qualifications beside the claim they limit. Separate observation from causality and diagnostic identities from empirical prediction.
- Remove repetition only after the explanatory links are in place.

## Main changes

The introduction now begins with opposing updates coexisting with loss reduction. It develops the intervention question directly, replacing the scaling-law detour. Curvature and Adam's scaling are explained before the functional question. The four contributions retain their agreed order: framework, curvature and diagnostics, representation dynamics, and intervention target.

Section II introduces the meaning of each mathematical object around the unchanged equations. The fixed-quadratic example explicitly distinguishes alternating convergence from growing oscillation. Sampling at fixed parameters is separated from changes in the Hessian during training. The diagnostics discussion explains the cost of Hessian--vector products and the different roles of existing gradient diagnostics before stating this paper's question. Representation measurements and controls are described through what they observe or change.

The scientific boundaries remain explicit: the exact curvature identity requires full gradients of a smooth, fixed objective and the nonzero applied update; the quadratic GD criterion is distinguished from its nonlinear approximation and the next-step Adam evaluation; local opposition does not establish accumulated cancellation; sampled-gradient alignment alone does not establish reduced curvature; representation measurements do not identify causal layers; and the cap limits direct local changes at fixed layer inputs.

## Verification and layout

The [verification record](verification.json) checks that all 29 units were reviewed, all 36 audit entries replay correctly, four contribution bullets and three related-work subsections remain, all five displayed equations and section labels are preserved, and citation and cross-reference keys resolve. File hashes protect 55 other draft files and the five pre-existing modified research records.

The [build record](build.json) reports 22 pages and no LaTeX, undefined-reference, overfull, or underfull warnings. Final pages 1--4 were inspected individually and all 22 pages were inspected in contact sheets. No clipping or overlaps were observed. Added explanations move the start of Section III to page 4. A preliminary build exposed a downstream float overflow; logged concision reviews removed repeated framing and resolved it without changing equations, fonts, margins, or float settings.

This is a clarity revision, not a claim of overall shortening. Whitespace-token counts including TeX changed from 915 to 922 in Section I and from 824 to 1,230 in Section II. The manuscript remains 22 pages, but Section II now devotes more space to explanations. Venue page-limit decisions remain separate.

## Reproduction

From the repository root:

```powershell
python manuscript/consolidation/work/readability_paragraphs_20261115/01_build.py
python manuscript/consolidation/work/readability_paragraphs_20261115/03_verify.py
```

The build script recreates the main PDF and individual page renders. Verification requires this review's local, ignored pre-edit manifests and paragraph state. The committed Markdown audit can be reviewed independently; its baseline corresponds to commit `8449b2d`. Auxiliary files, baseline copies, and page renders remain in ignored scratch storage. A rebuilt PDF requires renewed visual inspection before updating the manual-review statement.
