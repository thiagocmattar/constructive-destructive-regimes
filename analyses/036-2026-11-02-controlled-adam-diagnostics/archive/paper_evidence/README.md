# Controlled Adam: simplified paper presentation

The current proposal uses **two figures and one descriptive recurrence table**:
trajectory context, a close view of selected spikes, and counts showing how
frequently the sequence recurs. It does not build the paper argument around
classification, precision-recall, average precision, or probability forecasts.
The active manuscript is unchanged. Interventions remain a deferred TODO.

## Proposed argument

Raw-gradient reversal and growth can precede pronounced loss excursions along
an Adam trajectory, even though Adam transforms those gradients before applying
an update. The micro views make that sequence visible. The recurrence table
shows where it repeats and where it is absent.

This is an observational claim about the measured trajectories. The compact
presentation does not establish information beyond recent loss, a universal
alarm threshold, or a causal mechanism. Those stronger questions should not
be inferred from the simpler evidence. We need no replacement classifier or
correlation plot to communicate this narrower result.

## Current figures

1. [Overview](../figures/adam_manuscript_overview.pdf): three rows only: training
   loss, raw-gradient projected change, and raw-gradient cosine. Panels (d)-(f)
   share limits **[-0.1, 2.2]**, with ticks at 0,1,2. Values outside the viewport
   are retained unchanged. Relative norms and their legend are removed.
   Loss is raw, with ordinary numeric tick labels on a logarithmic axis. Other
   lines use complete 16-observation trailing means, without raw overlays.
   The Analysis024 typography,
   three-column structure, colors and 7.16-inch width are preserved.
2. [Spikes](../figures/adam_manuscript_spikes.pdf): unchanged, fully unsmoothed
   examples with loss, projected change, raw cosine, and both relative norms.
   Its three loss-selected cases remain the local illustrations. The request
   to remove relative norms applies to the overview, not this retained figure.

The onset-overlay and precision-recall PDFs are **withdrawn from the proposed
paper presentation**, including supplementary recommendations. Their existing
files and underlying analyses are preserved as research history; they are not
current deliverables or evidence the revised draft asks readers to follow.

## Small systematic table

[RECURRENCE_TABLE.md](RECURRENCE_TABLE.md) and
[table_recurrence.tex](table_recurrence.tex) give all nine arms: episode count,
number with an earlier fresh crossing of two, corresponding large-episode
counts, and lead median/range among large episodes with a crossing. Misses and
zero-event arms stay visible. No classifier is fitted for this table.

At the highest rates, fresh crossings precede 13/17,3/3,7/7 large episodes, with
conditional median leads 5,2,5 updates for MLP, ResNet and Transformer. For all
sizes, the counts are 17/39,5/7,13/13. This distinction prevents presenting only
the most favorable large-episode counts. Lead is measured before the 10% onset,
not before the peak, and conditions on a warning in the preceding eight updates.

The low-rate ResNet and Transformer runs have no qualifying loss episode even
though raw projected change can cross two. Keep that counterexample in the
text: it explains why a recurring precursor is not a sufficient condition.
Likewise, the exact projected-change identity couples cosine and gradient-norm
growth; these are not three independent confirmations of a mechanism.

## Prepared text

- [section_draft.tex](section_draft.tex): concise observational argument.
- [methods_draft.tex](methods_draft.tex): measurement clock, episode selection,
  lead-time definition, display choices and scope.
- [captions.tex](captions.tex): only the two retained figures and recurrence table.

These candidates have not been inserted into the manuscript. No training,
new metric, additional seed, Hessian, cloud work or intervention was introduced.

## Provenance and reproduction

Run `21_simplify_presentation.py`, inspect the overview's Poppler render, then
run it with `--accept-reviewed`. The current source/display checks and intentional
viewport exclusions are in [presentation_manifest.json](presentation_manifest.json).
All 27 curves /27,369 values are checked against the source histories. Loss
values are unchanged; the two diagnostic rows use the longer display window.
The spike PDF is
byte-identical to its previous accepted version.

The previous overview and presentation text are preserved in
[archive/2026-11-02-before-simplification](archive/2026-11-02-before-simplification/).
The earlier `figure_manifest.json`, `verification.json`, protocols, forecast
CSVs and TABLES.md describe the previous evidence package. Its overview hash
now resolves through the archive manifest. Scripts 18-20 belong to that earlier
package; script 19 would recreate its superseded four-figure presentation and
should not be used to regenerate the current overview. No raw evidence or
previous statistical result has been erased or revised to match this edit.

The manuscript's own figure copies still match the older integration, as
recorded in Observation 009; this task has edited the analysis presentation only.
