# Revision task list

Scope: complete editorial and mathematical review of the active manuscript for
IEEE Transactions on Neural Networks and Learning Systems. Preserve the supplied
annotated PDF and prior revision. No change to the experiments is requested.

## Preparation

- [x] Read repository instructions, research definitions and the active evidence.
- [x] Read every compiled section, appendix and the main-file derivations.
- [x] Extract the 12 PDF comments and interpret the spoken writing feedback.
- [x] Establish [writing principles](WRITING_PRINCIPLES.md) before revising prose.
- [x] Save annotation extraction, source hash and individually tracked responses.
- [x] Verify primary literature supporting the revised contribution positioning.

## Manuscript review, in reading order

- [x] Title, abstract and keywords: make the diagnostic contribution and scope clear.
- [x] Introduction: problem, regime argument, practical value, contributions, roadmap.
- [x] Background: condense curvature/noise exposition and position prior diagnostics.
- [x] III opening: define regimes explicitly from adjacent raw-gradient cosine.
- [x] III-A: curvature, projected change, overshoot and mathematical assumptions.
- [x] III-B: noise expectations, finite-population variance and evolving-state scope.
- [x] III-C: cumulative retention, all temporal interactions and computation cost.
- [x] III-D: destructive descent versus norm-amplified loss increase.
- [x] IV setup and GD results: connect each figure and table to the central argument.
- [x] IV lag/SGD results: explain what temporal scale and sampling add to regime interpretation.
- [x] V opening: motivate raw diagnostics along Adam's adaptive path; reference prior equations.
- [x] V setup/overview: concise experiment description and informative convergence argument.
- [x] V micro-plots: selection, destructive rise, early warning timing and recovery regimes.
- [x] V spike definition: intuitive fixed-reference excursion, then anchors and formulas.
- [x] V density/table: amplitude, time asymmetry, initial regime and norm amplification.
- [x] V scatter/discussion: quantitative association, mechanism hypothesis and interventions.
- [x] Conclusion: concise positive account of contributions and next scientific step.
- [x] Appendix A and main-file Appendix B: audit every identity and approximation.
- [x] Measurement appendix: retain reproducibility and consolidate interpretation limits.
- [x] Captions/tables: standalone definitions, terminology, legibility and figure clocks.
- [x] Audit each resulting paragraph for purpose, transition, evidence and terminology.

## PDF annotations (original page numbers)

- [x] C01 p8: motivate Adam diagnostics; cite the diagnostic equations; connect raw inputs to adaptive movement.
- [x] C02 p9: replace "interspersed" with plain wording.
- [x] C03 p9: remove the unnecessary final-loss negation.
- [x] C04 p10: remove the selection-rule temporal-filter digression.
- [x] C05 p10: remove the vague momentum-timing qualification.
- [x] C06 p10: sharpen the Transformer warning chronology and its implication.
- [x] C07 p10: describe recovery using constructive/destructive regime terminology.
- [x] C08 p10: explain spike variables intuitively before the definition.
- [x] C09 p10: express the 10% threshold choice affirmatively.
- [x] C10 p10: replace "illustrations" with a precise description.
- [x] C11 p10: replace procedural density/table narration with scientific insight.
- [x] C12 p12: full-width centered table, wider spacing, scientific LR notation,
  dataset/model line breaks, requested rounding and complete caption definitions.

## Verification and closeout

- [x] Check all mathematical statements, indexing, assumptions and zero denominators.
- [x] Recheck unchanged empirical inputs and every new numerical statement.
- [x] Independently verify reformatted table values and rounding.
- [x] Compile with resolved bibliography/references and no overfull boxes.
- [x] Render and visually inspect every final page, with detailed table/figure review.
- [x] Record paragraph audit, comment responses, evidence sources and final PDF hash.
- [x] Update manuscript/research entry points without altering unrelated work.
- [x] Review staged diff and commit only this task's verified changes locally.
