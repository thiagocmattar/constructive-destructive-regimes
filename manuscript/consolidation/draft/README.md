# Consolidation manuscript for author review

- **Section IV clarified, 15 November 2026:** observations are separated from quadratic interpretations; the accumulated-interaction result follows displacement retention; backward lag and evolving mini-batch trajectories are explained explicitly. All figures, captions, numerical results, labels, and subsection headings are preserved. The PDF remains 22 pages and warning-free. [Paragraph review and verification](../work/section4_readability_20261115/OBSERVATIONS.md).

- **Section III clarified, 15 November 2026:** conservative prose edits distinguish consecutive opposition from accumulated interactions, starting-point curvature from its step-average, and evolving-trajectory sampling from fixed-state resampling. The loss interpretation now consistently refers to the completed step. All 19 equations, labels, notation, and subsection order are retained; III-D is unchanged. The final subsection title now refers to loss decrease and increase. The PDF remains 22 pages and warning-free. [Review and verification](../work/section3_readability_20261115/OBSERVATIONS.md).

- **Sections I and II polished, 15 November 2026:** all 29 original prose units reviewed sequentially, with a diagnosis and proposed solution recorded before each edit. Concepts now precede technical terms; transitions and diagnostic timing are explicit. Five displayed equations, all labels, and the remaining scientific sources are preserved. The rebuilt PDF is 22 pages and warning-free. [Paragraph review and verification](../work/readability_paragraphs_20261115/OBSERVATIONS.md).

- **Section II revised, 15 November 2026:** now "Related Work and Conceptual Foundations," organized around oscillatory stability and sampling, training-time diagnostics, and representation dynamics and update control. The five equations and all labels are preserved. Three verified references were added; the introduction roadmap was aligned. The PDF remains 22 pages. [Scope, source audit, and verification](../work/section2_framing_20261115/OBSERVATIONS.md).

- **Introduction revised, 15 November 2026:** four contributions now follow the agreed framework, diagnostic, representation, and intervention order. The functional gap, exact-versus-predictive distinction, and intervention-target framing are explicit. The main PDF remains 22 pages; abstract, Section II onward, bibliography source, evidence, and supplement are unchanged. [Revision scope and verification](../work/introduction_framing_20261115/OBSERVATIONS.md).

- **Read:** [revised manuscript - 22 pages](manuscript.pdf), including the compact reviewer appendix beginning on page 17. This is the file for your next commenting round; the second-round feedback remains incorporated.
- **Extended research record:** [supplement - 27 pages](supplementary.pdf), preserved unchanged, with the exhaustive intervention heatmap grid and historical MLP replication. The main manuscript now contains the necessary appendix and does not require this separate document.
- **Main changes:** definitions before tracing algorithms/results; precise modes and temporal overlap; representation retention computed from saved data; clearer diagnostic timing; selective-cap computation and positioning; absolute intervention metrics and a compact q25 motion figure.
- **Contents:** nine main figures, three main tables, two algorithms, and 44 numbered equations. The appendix adds three figures and four small tables: applied updates and controller activity for all settings, selected conventional-intervention heatmaps, curvature and spectral controls, late losses, and measured costs. Sections I-IV retain the framing and regime development; V characterizes Adam and traces functional oscillations; VI tests interventions; VII concludes.
- **Editable sources:** [main wrapper](manuscript.tex), [sections](sections/), [supplement wrapper](supplementary.tex), [bibliography](references.bib), [figures](figures/), and [tables](tables/). Original evidence remains preserved in the sibling folder.
- **Verified:** warning-free compilation, numerical and rendered-page checks, 108 absolute intervention table cells, full-Gram retention, q25 output-loss accounting, and 416 unchanged protected originals. [Current revision and comment resolutions](../work/author_feedback2_20261115/OBSERVATIONS.md).
- **Appendix checks:** 78 plotted series, 18 heatmaps, and 36 late-mean cells verified against frozen evidence. [Appendix sources and rationale](../work/appendix_20261115/OBSERVATIONS.md).
- **Archived feedback:** [round one](archive/2026-11-15/manuscript+feedback.pdf) and [round two](archive/2026-11-15/manuscript+feedback2.pdf), preserved byte-for-byte with their annotations. New comments belong on the updated manuscript.pdf.
- **Preserved:** active manuscript, evidence, and human originals. The earlier full rewrite is in commit `20d783d`; the older [V-C review excerpt](intervention-review.pdf) is historical and retains its previous notation.

Build from the repository root with Python (NumPy, Matplotlib, PyMuPDF, Pillow) and a TeX distribution containing IEEEtran, newtx, algorithm, and algpseudocode:

```powershell
# To reproduce representation retention:
python manuscript/consolidation/work/author_feedback2_20261115/01_retention.py

# To reproduce the earlier diagnostic table (caption received later editorial changes):
python manuscript/consolidation/work/author_feedback_20261115/02_diagnostic_table.py

# Only needed when regenerating current figure variants:
python manuscript/consolidation/work/readability_revision_20261114/02_figures.py

# Or regenerate only the compact appendix figures and tables:
python manuscript/consolidation/work/appendix_20261115/01_artifacts.py

python manuscript/consolidation/work/readability_revision_20261114/03_build.py
python manuscript/consolidation/work/readability_revision_20261114/04_verify.py
python manuscript/consolidation/work/readability_revision_20261114/05_review_sheets.py
```

Page renders and auxiliary files remain in ignored scratch storage. Inspect rebuilt pages before refreshing the visual-review record. This is the author-review version inside consolidation; the submission route and venue formatting remain as recorded in [PLAN](../PLAN.md).
