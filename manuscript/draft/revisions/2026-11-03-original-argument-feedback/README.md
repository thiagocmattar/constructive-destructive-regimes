# Revision of the author's 41 annotations

This revision addresses the feedback in the preserved
[annotated original-argument PDF](../2026-11-03-original-argument-restored/revised-manuscript+feedback.pdf).
The [revised manuscript](revised-manuscript.pdf) has 13 pages, eight figures
and one table.

- [Completed sequential checklist](TASKS.md).
- [Verbatim annotations and selected passages](feedback.md); machine-readable
  extraction in feedback.json.
- [Critical assessment and all 41 dispositions](REVIEW.md).
- [Mathematical, source and build verification](verification.json).
- [Page-by-page visual review](review.json).
- [Preserved-source hashes](preservation_manifest.json).

## Scope and result

Sections II-IV now develop the requested progression from stability and
practical monitoring to constructive/destructive regimes, Hessian-free
directional-curvature estimation, and reversal plus norm growth as the
local condition for a loss increase. The two central relations are boxed.
The noise and cumulative-retention discussion is shorter, and the EoS
interpretation includes self-stabilization and central-flow work.

Figure 1 adds all 279 existing spectral observations, explicitly recording
the full/subset coverage and the starting-state alignment of gradient pairs.
The nine-run endpoint table is removed from compilation; its source remains
intact. All four controlled figures appear on pages 5-7 before Adam begins
on page 8. The appendices remain concise.

The introduction (including its original first sentence), title, abstract,
keywords, Adam section and conclusion are unchanged. Every previous
figure/table asset and the supplied feedback PDF remains unchanged.
The new figure uses a separate filename. No training or Hessian evaluation
was performed, and no removed appendix figures/tables were restored.

## Verification and reproduction

From the repository root:

    python manuscript/draft/revisions/2026-11-03-original-argument-feedback/01_build_and_verify.py

This checks the 41-comment mapping, protected files, retained evidence,
934 numerical mathematical cases, and Figure 1 provenance. It builds with
pdfLaTeX/BibTeX and renders every page to the ignored scratch directory
tmp/pdfs/manuscript-original-argument-feedback-2026-11-03/. It writes this
revision's PDF and the local manuscript/draft/bare_jrnl.pdf.

To regenerate only the new figure:

    python analyses/024-2026-10-17-eos-gd-lr-calibration/11_plot_curvature_comparison.py

Regeneration resets the figure manifest to awaiting visual review. Inspect
the PDF and record acceptance in that manifest before rebuilding the paper.
The [figure observation](../../../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/010-curvature-spectral-overlay.md)
records sources, reduction, caption, coverage and interpretation limits.

The build passes with no overfull boxes or unresolved/multiply defined
references. The final 13 rendered pages have been inspected. Earlier build
scripts remain historical records of their own revision rather than validators
of this changed text.
