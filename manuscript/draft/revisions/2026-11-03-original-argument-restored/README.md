# Original argument restored, 2026-11-03

This revision restores commit `bba6203`, immediately before the full writing
revision, as the main-text baseline. That snapshot already includes the updated
methodology, controlled GD/SGD experiments and complete Adam spike analysis.
It preserves the broader argument about gradient interaction, displacement,
curvature and training dynamics for the author's next review.

The introduction now begins exactly:

> Neural scaling laws have ushered in new paradigms for interpreting deep learning dynamics.

The title, abstract, keywords and all seven main-body source files match that
snapshot byte for byte. The shortened appendices, bibliography, current table
assets and eight figure assets remain unchanged. No removed figure or table
has been reintroduced. As explicitly requested, references were left as they
stand: `tab:gd-temporal-accounting` is unresolved, and historical appendix
pointers have not been rewritten to fit the shortened appendix.

This corrects the preceding restoration to `a4bbf0d`, which was after the broad
writing pass and therefore used the wrong baseline. All previous revision
folders and PDFs remain preserved.

Deliverable: [revised-manuscript.pdf](revised-manuscript.pdf), 15 pages.
The active sources are in `manuscript/draft/`.

[Restoration manifest](restoration_manifest.json),
[verification](verification.json), and [visual review](review.json) record
source equality, preserved assets, the exact opening in the compiled PDF,
evidence checks and the accepted broken reference.

## Reproduction

From the repository root with this revision's draft sources checked out:

    python manuscript/draft/revisions/2026-11-03-original-argument-restored/01_build_and_verify.py

The builder compiles with pdflatex/BibTeX and renders with Poppler into ignored
`tmp/pdfs/manuscript-original-argument-restored-2026-11-03/`. It allows the
documented unresolved label and checks for other build problems. Rebuilding
resets visual acceptance to pending; review the new PDF before accepting it.

No training, analysis, new scientific claim or argumentative rewrite is added.
