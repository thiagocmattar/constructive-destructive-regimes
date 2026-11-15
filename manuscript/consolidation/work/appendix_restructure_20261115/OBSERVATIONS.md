# Appendix restructuring and readability

15 November 2026. Implemented the approved surgical proposal in the consolidation draft. No experiments, figures, or numerical results were recomputed.

## Result

The main paper now has three appendices: exact identities and approximation limits; experimental definitions and diagnostic checks; and intervention verification, endpoint sensitivity, and cost. Four mathematical equations remain unchanged, apart from adding a reference label to the propagated-change equation. The independent HVP and spectral-control tables, last-32-state table, and cost table retain their values.

A compact nine-trajectory GD audit exposes the existing evidence for accumulated reinforcement despite adjacent opposition, alongside the existing nonlinear loss-sign disagreements. Its 45 cells are an exact projection of `tables/gd-audit.tex`; no independent dataset or calculation is implied.

Former main Figures 10-12 are Supplement Figures S19-S21 in the new Section S7. The three source PDFs are byte-identical. Captions preserve all measurement conventions; the sole textual adaptation replaces the unresolved cross-document figure reference with a description of the main paper's representation-motion figure. Supplementary methods S1-S6 remain byte-identical. The wrapper includes S7 and updates its roadmap.

Section IV points directly to the compact audit. Two Section VI pointers now identify Supplement S7. Other main scientific sections are unchanged. The README identifies the main paper's retained central evidence and the supplement's extended records.

## Paragraph and transfer record

[PARAGRAPH_REVIEW.md](PARAGRAPH_REVIEW.md) records the original text, diagnosis, proposed solution, and replacement before each mutation. [decisions.json](decisions.json) supplies exact sequential replay. It includes 34 initial editorial/structural decisions and one repair: the first main build found that a replacement matched the new B-C heading boundary; the complete heading was restored before the successful build. No scientific content changed in that repair. A final whitespace-only decision removes a trailing blank line flagged by the staged diff check.

Detailed architecture counts, frozen-draw counts, native replay coverage, bridge error, audit counts, and calibration details removed from repeated main-appendix prose already remain in Supplement S2, S4, or S5. Unresolved scientific qualifications remain visible in the main paper: runtime/batch confounding, fixed-state approximation error, finite-step curvature differences, non-causal representation tracing, intervention endpoint tradeoffs, and computational cost.

## Verification

- [01_edit.py](01_edit.py): one-time approved edits, logging before mutation. The final heading repair is recorded in the decisions and paragraph audit.
- [02_build.py](02_build.py): builds either document, checks the final LaTeX log, and renders every page.
- [03_verify.py](03_verify.py): validates full decision replay, equations, retained tables, exact audit projection, caption transfer, protected hashes, links, builds, and changed-page coverage.
- [manuscript-build.json](manuscript-build.json), [supplementary-build.json](supplementary-build.json), and [verification.json](verification.json): final results.

The main PDF decreases from 22 to 20 pages; the supplement increases from 27 to 31. Both builds have no LaTeX, unresolved-reference, or box warnings. All changed pages were inspected in contact sheets; main appendix pages 17-19 and supplementary gallery pages 28 and 30 were also inspected individually. No clipping or overlap was observed. The appendix-to-bibliography transition works without the former full-width-float column reset.

All existing publication figure assets and all existing table files are unchanged. Five unrelated modified research files retain their pre-edit hashes. Local pre-edit snapshots, intermediate builds, logs, and renders remain ignored under `scratch/`; baseline-dependent verification requires those local snapshots. This is editorial and artifact verification, not a new experimental replication.
