# Sections I-IV restored, 2026-11-03

This separate revision restores Sections I-IV exactly from commit `a4bbf0d`,
the concise-appendix version immediately before the curvature-focused pass
(`91fc1d4`). The five restored source files cover the introduction, related
work, methods, controlled GD results and the SGD subsection of Section IV.
The SGD source already matched that baseline.

The title, abstract, keywords, Adam section, conclusion, appendices, bibliography,
tables and figure assets retain their current contents. No further prose
revision is included. The previous revision folder and its PDF remain unchanged.

Deliverable: [revised-manuscript.pdf](revised-manuscript.pdf).
The active TeX sources are in `manuscript/draft/`.

The [restoration manifest](restoration_manifest.json) records the exact baseline,
restored source hashes, retained source hashes and preserved previous revision.
[Verification](verification.json) checks those hashes, byte equality with the
Git baseline, the existing mathematical/evidence checks and the compiled layout.
[Visual acceptance](review.json) records the final PDF hash and page review.

## Reproduction

From the repository root, with this revision's draft sources checked out:

    python manuscript/draft/revisions/2026-11-03-sections-i-iv-restored/01_build_and_verify.py

The builder reuses the preceding revision's evidence checks without changing its
files, compiles with pdflatex/BibTeX, and renders all pages with Poppler. Build
scratch is under `tmp/pdfs/manuscript-sections-i-iv-restored-2026-11-03/`.
Rebuilding resets verification to pending visual review; inspect the new pages
and update the visual acceptance before delivery.

The 14-page PDF passes all checks and visual review. No experiment, measurement
or empirical result is changed.
