# Full-manuscript writing revision, 2026-11-03

This revision implements the author's spoken writing guidance and the 12
annotations in the preceding Adam-spike review PDF. The clarification
"methodology of writing" is applied throughout: experiments, event selection,
raw measurements and scientific figure assets remain unchanged. The author's
follow-ups shorten the appendices and restore online curvature estimation as
the central contribution in this same revision and deliverable.

Deliverable: [revised-manuscript.pdf](revised-manuscript.pdf).
The active sources are in the manuscript draft.

- [Writing principles](WRITING_PRINCIPLES.md): distilled before editing.
- [Task list](TASKS.md): preparation, each active section, every annotation,
  mathematical review, visual verification and commit.
- [Review and comment responses](REVIEW.md): all 12 dispositions, mathematical
  audit, evidence checks and primary-source literature positioning.
- [Paragraph review](PARAGRAPH_REVIEW.md): the initial full-paper pass over
  135 prose units and 14 captions, followed by the
  [appendix review](APPENDIX_REVIEW.md).
- [Curvature review](CURVATURE_REVIEW.md): original-paper comparison, formal
  estimator, online cost, heuristic interpretation and added mathematical checks.
- [Extracted feedback](feedback.md) and [original annotation metadata](feedback.json).
- [Evidence manifest](evidence_manifest.json), [verification](verification.json)
  and [final visual acceptance](review.json).

## Argument and presentation

The paper now develops a consistent sequence: directional-curvature estimation
from training gradients without Hessian computation;
constructive/destructive regimes defined by adjacent raw-gradient cosine;
productive destructive descent; amplification and local loss increase;
controlled GD/SGD characterization; and early signs plus recurring anatomy
of Adam loss spikes and implications for stabilization heuristics. The title covers neural-network stability, including
the Adam extension. Abstract, contributions, background, results and conclusion
follow the same argument.

The main estimator uses one retained gradient and linear reductions. The
appendix gives its exact smooth-GD path-average identity and pointwise error
bound; the sampling term and Adam's adaptive curvature response are explicit.
The Adam section motivates the diagnostics through the raw-gradient response
along an adaptive displacement. Individual events supply temporal evidence;
the full event population and per-rate regressions support the recurring
sign-and-magnitude structure. The >10% table has centered grouped columns,
scientific LR notation, simpler rounding and a self-contained caption.
Essential scope appears with the relevant mathematical assumptions and the
measurement appendix. Prospective detection and interventions remain future work.

The current 14-page PDF (previously 17 pages) places the Adam section on pages 7--12, with the approved
overview and micro-plots on page 10, density and anatomy table on page 11,
and regression scatter on page 12. The appendices now contain one consolidated
loss derivation and essential measurement conventions on pages 12--13, with no
tables or figures. Eight main-text figures and two tables remain compiled;
the three removed supplementary tables and raw-cosine figure remain in the
research materials. The preceding full pass is preserved in commit `f05098a`;
the supplied annotated PDF and earlier review snapshots are preserved.

## Reproduction

From the repository root:

    python manuscript/draft/revisions/2026-11-03-regime-writing-revision/00_extract_feedback.py
    python manuscript/draft/revisions/2026-11-03-regime-writing-revision/01_prepare_table.py
    python manuscript/draft/revisions/2026-11-03-regime-writing-revision/02_build_and_verify.py

The table script reruns the prior independent Adam raw-history audit in
ignored scratch destinations before formatting the new table. The builder
checks the frozen figures and data, retained GD/SGD evidence, 186 previous
mathematical cases and 508 additional mathematical cases, then compiles with
pdflatex/BibTeX and renders every page with Poppler. The added cases check
the Adam projected-response relation, finite-population covariance and
all-pair displacement, and nonlinear path-average identities and their error bound.

Build/render scratch is under tmp/pdfs/manuscript-regime-writing-2026-11-03/.
Rebuilding resets visual acceptance to pending: inspect the new rendering
and record its exact hash in review.json before delivery.

No new training, cloud execution, intervention, hypothesis, run or analysis
was introduced. The completed changes are committed locally; no push is made.
