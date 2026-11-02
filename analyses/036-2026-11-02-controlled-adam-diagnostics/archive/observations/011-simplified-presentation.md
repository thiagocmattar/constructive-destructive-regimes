# Observation 011: simpler Adam presentation

The author requested a simpler overview, removal of precision-recall and
onset-overlay figures from the proposed paper, and retention of the selected
spike views. Subsequent feedback increased smoothing, removed the raw overlays
and restored Analysis 024's ordinary numeric loss ticks. This is presentation
work on existing evidence; there is no new training, experiment or cloud work.

## Current overview

`figures/adam_manuscript_overview.pdf` now has three rows and three columns:
log training loss, raw-gradient projected change, and raw-gradient cosine.
Relative norms and their legend are removed. Panels(d)-(f) share limits
[-0.1,2.2] and ticks 0,1,2. This is intentional viewport clipping requested by
the author; the source values are neither clamped nor deleted. There are no
faint raw overlays. The two diagnostic rows use complete 16-observation trailing
arithmetic means, right aligned, increased from 8. Loss remains unsmoothed on
a logarithmic axis with ordinary numeric tick labels (e.g.0.01,0.1,1), matching
the Analysis 024 display standard. The 7.16-inch width, Times New Roman/STIX,
colors, grids and panel letters remain; height is 3.85 inches.

The generator `21_simplify_presentation.py` reads the original Run 215 histories,
checks all 27 curves /27,369 plotted values, independently reconstructs each
mean with math.fsum, checks axes/fonts/text bounds, and confirms the spike PDF
is byte-identical. Maximum mean discrepancy is 4.44e-16. The source/output
identities, intentional out-of-range counts and visual acceptance are in
`paper_evidence/presentation_manifest.json`. The Poppler render was inspected.
Loss arrays remain exactly equal to the preceding overview. No scientific
timing or count is computed from smoothed curves.

## Proposed paper argument

The active analysis proposal uses two figures: the simplified overview and
the unchanged unsmoothed spike examples. The latter retains its relative norm
row. A small descriptive recurrence table supplies the systematic check:
all nine arms, all/large episode counts, earlier fresh-crossing counts, and
conditional lead median/range. It reuses the existing episode census without
a new rule, outcome, classifier or fit. Misses and zero-event arms remain.
The table and candidate section/methods/captions are in `paper_evidence/`.

The onset-overlay and precision-recall figures are withdrawn from the proposed
main and supplementary presentation. Their files and results remain research
history. The draft no longer argues for incremental forecasting information
or calibrated warning probabilities. Its claim is descriptive: related raw
gradient patterns recur before pronounced loss excursions along these measured
Adam trajectories. A crossing is not sufficient for an excursion; the low-rate
zero-episode arms remain counterexamples. Algebraic dependence, loss-only
example selection, dependent episodes and one initialization per case remain
explicit. Interventions remain a deferred TODO.

## Preservation and caption

The prior overview and prose are preserved byte-for-byte under
`paper_evidence/archive/2026-11-02-before-simplification/`, with a source/archive
hash mapping. The earlier four-figure manifest/verification describe that
historical package. Script 20 now resolves superseded output identities through
the archive mapping, so its historical checks remain executable. Script 21 owns
the current presentation. The manuscript and its copied figures are unchanged.

Caption: Controlled full-batch Adam at three constant rates per case. Rows show
training loss (log scale with ordinary numeric ticks), raw projected gradient
change and raw consecutive cosine. Loss is unsmoothed; other curves use complete,
right-aligned 16-observation trailing means without raw overlays. Panels(d)-(f)
restrict the displayed range to[-0.1,2.2]; out-of-range values remain unchanged
in the data. All diagnostics use their parameter-state availability clock.
The reference at two is not an identified Adam stability boundary. One frozen
initialization per case; timing/count analyses use unsmoothed observations.
