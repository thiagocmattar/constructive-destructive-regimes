# Complete consolidation manuscript

- **Start here:** [complete manuscript — 14 IEEE-style pages](manuscript.pdf).
- **Optional detail:** [supplement — 33 pages](supplementary.pdf), containing proofs, measurement methods, curvature checks, the full timing/severity audit, all intervention settings, and historical MLP sensitivity.
- **Structure:** original title and Sections I–VI; four retained GD/SGD figures; Adam characterization → tracing → intervention. The main paper contains nine figures, three tables, two algorithms, and 25 numbered equations.
- **Review priorities:** motivation/contributions on pp. 1–2; curvature and loss interpretation on pp. 3–5; functional tracing on pp. 7–10; the intervention comparison on pp. 10–14. Check whether this expresses the intended central argument before requesting changes to the active manuscript.
- **Editable sources:** [main wrapper](manuscript.tex), [section files](sections/), [supplement wrapper](supplementary.tex), and [bibliography](references.bib). Figure/table paths resolve to the preserved `../evidence/` package; retain that sibling folder when building.
- **Verified:** warning-free compilation; every page inspected; all 144 main intervention table cells independently recomputed and located in the PDF; new mathematical identities checked; all 406 protected source files unchanged. [Audit and reproduction](../work/full_rewrite_20261114/OBSERVATIONS.md), [machine-readable checks](../work/full_rewrite_20261114/verification.json).
- **Scope:** this is the complete author-review draft inside consolidation. The active manuscript, original evidence, human inputs, and [previous V-C excerpt](intervention-review.pdf) remain preserved. No new experiment, cloud spending, push, or submission.
- **Before submission:** author approval, the venue route recorded in [PLAN](../PLAN.md), and current venue formatting/page allowance remain author decisions. The 14-page review layout is not a verified TNNLS page-limit determination.

Build from the repository root with Python (NumPy, PyMuPDF, Pillow) and a TeX distribution containing IEEEtran, newtx, algorithm, and algpseudocode:

```powershell
python manuscript/consolidation/work/full_rewrite_20261114/01_prepare.py
python manuscript/consolidation/work/full_rewrite_20261114/02_build.py
python manuscript/consolidation/work/full_rewrite_20261114/03_verify.py
python manuscript/consolidation/work/full_rewrite_20261114/04_review_sheets.py
```

Auxiliary files and page renders stay in the ignored work-folder scratch directory. A changed PDF requires a new visual review; the saved visual-review record applies to its recorded hash.
