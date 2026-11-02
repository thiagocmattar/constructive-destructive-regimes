# Adam-section author feedback, 2026-11-02

Source: `source-comments.pdf`, preserved byte-for-byte from the annotated
review in `manuscript/artifacts/draft/`. Six annotations on pages 9--10 are
extracted in `comments.json`. Continue editing in `manuscript/draft/`.
The author explicitly deferred further changes to the SGD batch-size figure.

- [x] Read all six annotations and the complete affected pages.
- [x] A01: remove clipping/AdamW discussion from the Adam section.
- [x] A02: center the section on existing gradient/momentum information and
  its timing around loss spikes; distinguish temporal evidence from a causal
  feedback explanation and a validated control rule.
- [x] A03--A04: delete the old raw-response and momentum/scaling subsections.
- [x] A05: replace Pythia/MiniPile illustrations and their two tables with
  the completed controlled Analysis036 experiments.
- [x] A06: keep the presentation compact and interpretive, like Section IV.
- [x] Prepare compact full-cohort and raw spike figures from existing evidence;
  verify values, averaging, state clocks, event counts and forecast results.
- [x] Align abstract, conclusion and relevant measurement details with the new
  experiments; leave the SGD figure and its measurements unchanged.
- [x] Build, inspect all changed pages and figures, record dispositions and
  verification, and commit only this revision's files. No new training/cloud work.
