# Appendix shortening

The author requested a shorter appendix in the same revision and PDF. The
criterion is reader value: keep the derivation of a central relation and details
that change the meaning of the evidence; leave additional experimental output
and execution records in the research materials.

## Decisions

| Previous material | Decision and reason |
| --- | --- |
| Alignment-based loss derivation (Appendix A) | Keep and condense. The endpoint-gradient integral explains the central loss relation and its remainder. Show the equivalent curvature form once. |
| One-step Taylor derivation (Appendix B) | Merge its useful conditional-noise explanation into A; remove the repeated one-step algebra. |
| Two-step Taylor expansion (Appendix B) | Remove from compilation. No main-text claim uses it, and it adds no necessary definition. |
| Raw-cosine figure | Remove from compilation. The main GD figure already establishes the regimes; preserve the trailing-mean timing qualification in the measurement conventions. |
| Temporal accounting table and extra notation | Remove from compilation. Retain the informative nine-trajectory finding in the main text, tied to the existing all-pair identity. |
| Retrospective loss-audit table and threshold sweep | Remove the table and procedural exposition. Move the highest-rate Transformer's 44.2% sign discrepancy into the main discussion, preserving the important finite-step limitation. |
| Spectral table and measurement exposition | Remove from compilation. The paper's central diagnostic argument does not require this additional, partly subset-based Hessian comparison. |
| Experimental measurement appendix | Condense to essential objectives, sample weighting, precision, clocks, event summaries and material runtime scope. Remove device inventories, repeated captions and additional endpoints. |

The revised appendix has no figures or tables. The main experimental figures
and the two main tables retain their data and assets. The four approved Adam
figures are unchanged.

## Preserved evidence

The preceding full revision is preserved in Git commit `f05098a`. Removed
tables remain in `manuscript/draft/tables/analysis024_temporal_accounting.tex`,
`analysis024_loss_audit.tex` and `analysis024_spectral_context.tex`; the raw
figure remains in `manuscript/draft/figures/analysis024/eos_gd_rho_3k.pdf`.
Analysis024's `section_iv_review_checks.json` retains the temporal accounting,
loss-sign checks and spectral coverage. Analysis035 and Analysis037 retain
the additional endpoint and execution records. No experimental inputs or
measurements were changed.

## Review coverage

Read every retained derivation paragraph and equation, every condensed
measurement paragraph, and the affected main-text prose/captions. Checked
noise signs, index conventions, the nonlinear remainder, objective weighting,
runtime scope and exclusion criteria. The original 135-unit paragraph audit
records the full-paper pass; this note records the subsequent appendix pass.
The build and visual acceptance files describe the current deliverable.

## Outcome

The manuscript decreases from 17 to 14 pages. The two concise appendices
occupy parts of pages 12--13, followed by references. Eight figures and two
tables remain in the main paper; no floats remain in the appendices.
Retained mathematical and numerical checks pass, and every final page was
rendered for review, with detailed inspection of the revised derivations,
measurement conventions and closing reference columns.
