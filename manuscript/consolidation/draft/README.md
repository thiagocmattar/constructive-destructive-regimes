# Consolidation manuscript for author review

- **Read:** [revised manuscript — 17 pages](manuscript.pdf). It returns to the active draft's introduction, stability survey, and core theory/empirical progression, with the completed Adam work explained in sequence.
- **Optional detail:** [supplement — 33 pages](supplementary.pdf), including derivations, controls, timing/severity audits, all intervention settings, and historical MLP replication.
- **Main changes:** clearer explanations; general curvature along the applied optimizer step; training step **t** throughout; reference examples in place of fixed-probe terminology; numerical detail concentrated in artifacts. Existing Adam gradient measurements are labeled **Q_t**, distinct from actual-step curvature.
- **Contents:** nine main figures, three tables, two algorithms, and 36 numbered equations. Review I–II for framing, III–IV for continuity with the original, and V for consecutive gradients → spike characterization → learned-function tracing → interventions.
- **Editable sources:** [main wrapper](manuscript.tex), [sections](sections/), [supplement wrapper](supplementary.tex), and [bibliography](references.bib). Current figure variants live in [figures/](figures/); some tables are read from the preserved sibling `../evidence/` folder.
- **Verified:** warning-free compilation; numerical checks and page review; all 144 intervention table values; general curvature identities; 416 protected source files unchanged. [Reproduction and audit](../work/readability_revision_20261114/OBSERVATIONS.md).
- **Preserved:** active manuscript, evidence, and human originals. The earlier full rewrite is in commit `20d783d`; the older [V-C review excerpt](intervention-review.pdf) is historical and retains its previous notation.

Build from the repository root with Python (NumPy, Matplotlib, PyMuPDF, Pillow) and a TeX distribution containing IEEEtran, newtx, algorithm, and algpseudocode:

```powershell
# Only needed when regenerating current figure variants:
python manuscript/consolidation/work/readability_revision_20261114/02_figures.py

python manuscript/consolidation/work/readability_revision_20261114/03_build.py
python manuscript/consolidation/work/readability_revision_20261114/04_verify.py
python manuscript/consolidation/work/readability_revision_20261114/05_review_sheets.py
```

Page renders and auxiliary files remain in ignored scratch storage. Inspect rebuilt pages before refreshing the visual-review record. This is the author-review version inside consolidation; the submission route and venue formatting remain as recorded in [PLAN](../PLAN.md).
