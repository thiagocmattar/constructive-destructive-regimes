# Manuscript integration of the controlled Adam spike anatomy

The author requested integration of the approved empirical sequence into
`manuscript/draft/`. The [revision record](../../../manuscript/draft/revisions/2026-11-03-adam-spike-anatomy/README.md)
documents exact dispositions, sources, reproduction and interpretation limits.
The [review manuscript](../../../manuscript/draft/revisions/2026-11-03-adam-spike-anatomy/revised-manuscript.pdf)
contains the updated Adam section, abstract, introduction, conclusion and
measurement appendix. Earlier forecasting text is no longer compiled.

## Sources and reductions

Four approved publication figures are copied without altering source bytes:

1. [Loss overview](003-loss-only-overview.md): nine unsmoothed loss curves,
   9225 observations, one initialization per model and 1024 updates per arm.
2. [Spike micro-plots](001-spikes.md): three largest recovered relative peaks
   at the highest tested rates; 18 unsmoothed curves / 1056 plotted values.
   Gray shading covers the +10% crossing to peak, not the entire event.
3. [Peak-increase densities](004-peak-increase-density.md): all 806 recovered
   excursions, per-LR unit mass in log10 percentage coordinates and a common
   pooled bandwidth within each model. No amplitude cutoff in this figure.
4. [Signed product scatter](008-rho-gradient-rise-peak-scatter.md): 127 events
   with defined pre-rise cosine; seven per-LR OLS fits on log10 loss ratio,
   descriptive in-sample R-squared and dashed full-panel extrapolations.

The new manuscript table is generated from the existing 129 recovered >10%
anatomy events, with grouped headers instead of bracketed pairs. Mean/sample
SD and peak maxima are independently recomputed; before/recovery diagnostic
and endpoint norm-ratio meanings are preserved. The source anatomy table is
unchanged. The full density sample and >10% table sample remain distinct.

## Caption, coverage and uncertainty

Captions in the manuscript state the axes, clocks, reductions and scope. The
micro-plot windows end below their first displayed loss, which is not always
the formal event baseline. Largest-spike selection uses no temporal or
diagnostic filter. The 10% cutoff is operational, not an inferred density gap.
Two above-threshold candidates without observed recovery are excluded and
documented. Two initialization events lack pre-rise diagnostics; they remain
in loss/time/norm summaries but not the 127-point scatter.

The narrative separates early gradient response from a proven Adam instability,
repeated descriptive association from prediction/causation, and a proposed
clipping/schedule/data-mixing intervention from a measured improvement.
Negative pre-rise cosine occurs in 117/127 defined events, not all events;
all 129 endpoint gradient norms increase. Only 21/127 pre-rise projected
changes already exceed two. Example recovery alignment is not generalized
to the population. One frozen initialization per model and temporal dependence
between events limit inference; no independent-run uncertainty is estimated.

## Verification

Generating scripts and exact source hashes are in the revision's
[evidence manifest](../../../manuscript/draft/revisions/2026-11-03-adam-spike-anatomy/evidence_manifest.json).
Checks reconstruct all 806 recovered boundaries and validate the 129 event
anchors/phase means against raw histories. Table summaries, largest-event
selection, displayed timing/counts, regression scores and all four asset hashes
pass. Retained GD/SGD evidence and mathematical checks are rerun. Build and
Poppler visual-review results are in the revision's verification/review records.
No source event data, scientific figure, training configuration or cloud resource
is changed by this integration.
