# Full-manuscript writing revision, 2026-11-03

This revision implements the author's spoken writing guidance and the 12
annotations in the preceding Adam-spike review PDF. The clarification
"methodology of writing" is applied throughout: experiments, event selection,
raw measurements and scientific figures remain unchanged.

Deliverable: [revised-manuscript.pdf](revised-manuscript.pdf).
The active sources are in the manuscript draft.

- [Writing principles](WRITING_PRINCIPLES.md): distilled before editing.
- [Task list](TASKS.md): preparation, each active section, every annotation,
  mathematical review, visual verification and commit.
- [Review and comment responses](REVIEW.md): all 12 dispositions, mathematical
  audit, evidence checks and primary-source literature positioning.
- [Paragraph review](PARAGRAPH_REVIEW.md): 135 prose units and 14 captions.
- [Extracted feedback](feedback.md) and [original annotation metadata](feedback.json).
- [Evidence manifest](evidence_manifest.json), [verification](verification.json)
  and [final visual acceptance](review.json).

## Argument and presentation

The paper now develops a consistent sequence: inexpensive gradient diagnostics;
constructive/destructive regimes defined by adjacent raw-gradient cosine;
productive destructive descent; amplification and local loss increase;
controlled GD/SGD characterization; and early signs plus recurring anatomy
of Adam loss spikes. The title now covers neural-network stability, including
the Adam extension. Abstract, contributions, background, results and conclusion
follow the same argument.

The Adam section motivates the diagnostics through the raw-gradient response
along an adaptive displacement. Individual events supply temporal evidence;
the full event population and per-rate regressions support the recurring
sign-and-magnitude structure. The >10% table has centered grouped columns,
scientific LR notation, simpler rounding and a self-contained caption.
Essential scope appears with the relevant mathematical assumptions and the
measurement appendix. Prospective detection and interventions remain future work.

The 17-page PDF places the Adam section on pages 7--12, with the approved
overview and micro-plots on page 10, density and anatomy table on page 11,
and regression scatter on page 12. All nine scientific figure PDFs are
unchanged. Historical review snapshots and the supplied annotated PDF are
preserved.

## Reproduction

From the repository root:

    python manuscript/draft/revisions/2026-11-03-regime-writing-revision/00_extract_feedback.py
    python manuscript/draft/revisions/2026-11-03-regime-writing-revision/01_prepare_table.py
    python manuscript/draft/revisions/2026-11-03-regime-writing-revision/02_build_and_verify.py

The table script reruns the prior independent Adam raw-history audit in
ignored scratch destinations before formatting the new table. The builder
checks the frozen figures and data, retained GD/SGD evidence, 186 previous
mathematical cases and 328 additional mathematical cases, then compiles with
pdflatex/BibTeX and renders every page with Poppler. The added cases check
the Adam projected-response relation, finite-population covariance and
all-pair displacement.

Build/render scratch is under tmp/pdfs/manuscript-regime-writing-2026-11-03/.
Rebuilding resets visual acceptance to pending: inspect the new rendering
and record its exact hash in review.json before delivery.

No new training, cloud execution, intervention, hypothesis, run or analysis
was introduced. The completed changes are committed locally; no push is made.
