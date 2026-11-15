# Consolidation manuscript for author review

- **Read:** [revised manuscript — 18 pages](manuscript.pdf), updated against all 62 annotations in the 15 November author-feedback PDF. This is the file for the next commenting round.
- **Optional detail:** [supplement — 33 pages](supplementary.pdf), including derivations, controls, timing/severity audits, all intervention settings, and historical MLP replication.
- **Main changes:** broader title and representation-learning motivation; five explicit contributions; clearer regimes and noise derivation; a 0.1% next-step diagnostic table with accuracy, event rates, loss-history baseline, and warning lift; explicit definitions of patterns/modes, squared movement, dimensionality, and temporal overlap. The Adam figure uses **ηκ̂**, with its gradient-projected interpretation derived separately from actual-step curvature.
- **Contents:** nine main figures, three tables, two algorithms, and 39 numbered equations. Sections I–II frame the problem; III–IV develop and examine the regimes; V characterizes Adam spikes and traces functional oscillations; VI tests interventions; VII concludes.
- **Editable sources:** [main wrapper](manuscript.tex), [sections](sections/), [supplement wrapper](supplementary.tex), and [bibliography](references.bib). Current figure variants live in [figures/](figures/); some tables are read from the preserved sibling `../evidence/` folder.
- **Verified:** warning-free compilation; numerical checks and page review; diagnostic confusion counts; all 144 intervention table values; general curvature and Adam projected-response identities; 416 protected source files unchanged. [Current revision and comment resolutions](../work/author_feedback_20261115/OBSERVATIONS.md).
- **Archived feedback:** [original annotated PDF](archive/2026-11-15/manuscript+feedback.pdf), preserved byte-for-byte with all annotations. New comments belong on the updated `manuscript.pdf`.
- **Preserved:** active manuscript, evidence, and human originals. The earlier full rewrite is in commit `20d783d`; the older [V-C review excerpt](intervention-review.pdf) is historical and retains its previous notation.

Build from the repository root with Python (NumPy, Matplotlib, PyMuPDF, Pillow) and a TeX distribution containing IEEEtran, newtx, algorithm, and algpseudocode:

```powershell
# Only needed to reproduce the updated diagnostic table:
python manuscript/consolidation/work/author_feedback_20261115/02_diagnostic_table.py

# Only needed when regenerating current figure variants:
python manuscript/consolidation/work/readability_revision_20261114/02_figures.py

python manuscript/consolidation/work/readability_revision_20261114/03_build.py
python manuscript/consolidation/work/readability_revision_20261114/04_verify.py
python manuscript/consolidation/work/readability_revision_20261114/05_review_sheets.py
```

Page renders and auxiliary files remain in ignored scratch storage. Inspect rebuilt pages before refreshing the visual-review record. This is the author-review version inside consolidation; the submission route and venue formatting remain as recorded in [PLAN](../PLAN.md).
