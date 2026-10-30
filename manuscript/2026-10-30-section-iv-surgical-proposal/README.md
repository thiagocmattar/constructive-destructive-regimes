# Surgical change proposal for Section IV

Date: 2026-10-30. Status: applied to the working draft at the user's request.
See the [implementation record](../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/008-applied-section-iv-revision.md).

Refocus Section IV on a result that the existing experiments establish:
**adjacent gradient reversal, cumulative interaction, displacement relative to
path length, and loss progress describe different properties of a trajectory.**
Use the retained all-pair measurements to demonstrate the distinction, and use
the completed retrospective loss audit to delimit the local approximation.

The proposed package keeps the section title, the nine conditions, and all
three main figures. It changes selected paragraphs, one table column, and the
organization of Appendix D. Two small edits in Section III align the meaning of
displacement and cancellation with the experimental evidence.

## Proposed argument

1. The selected learning rates produce different local gradient dynamics while
   all nine trajectories reduce their objectives over the recorded horizon.
2. Seven conditions have negative cumulative adjacent interaction, but every
   condition has positive total cross-step interaction at update 3000.
3. Increasing LR reduces R while increasing absolute displacement within each
   case: accumulated path length grows faster than net displacement.
4. The gradient-derived projected change characterizes consecutive motion.
   Its relation to a local loss approximation has measurable limitations.

This gives the existing figures a specific scientific role. It also addresses
the original review concern about omitted nonadjacent interactions with direct
evidence rather than a lag-one assumption.

## Edit map

Exact anchors and replacement text are in [Proposed edits](01-proposed-edits.md).
All edits below use the current draft following commit `7eb8ddf`; the critical
review and audit were committed in `03aed49`.

| Edit | Location | Action | Scientific purpose |
| --- | --- | --- | --- |
| S1 | IV opening | Replace the opening paragraph | State the empirical question and contribution relative to the cited EOS work |
| S2 | IV-A | Shorten the main protocol; move detailed preparation and eigensolver material to Appendix D | Keep the matched comparison readable while preserving reproducibility |
| S3 | IV-B ending | Replace the claim about supporting Section III-D | Separate qualitative illustration from approximation accuracy |
| S4 | IV-C first two paragraphs | Insert the positive all-pair result and increasing absolute displacement | Demonstrate the distinction between local and cumulative geometry |
| S5 | IV-D | Replace the spectral discussion with the completed loss-approximation audit; narrow the closing claim | Give the theoretical approximation a direct empirical check |
| S6 | Table I | Replace the final spectral column with absolute displacement | Put the key geometric comparison beside R without widening the table |
| S7 | Appendix D | Retain raw cosine; add measurement details, temporal interaction and audit tables, then spectral context | Keep all nine conditions and the interpretation limits inspectable |
| S8 | Section III | Replace two interpretive sentences/blocks; preserve equations | Make information and cancellation terminology consistent with IV |

## Figures and tables

Keep Fig. 1 as unsmoothed log loss, Fig. 2 as smoothed cosine above raw R,
and Fig. 3 as smoothed norms above smoothed projected change. Retain their
current colors, solid LR lines, external labels, and smoothing definitions.
No new main figure is needed for this revision.

Table I keeps its seven-column width. Its last column becomes
`eta * ||sum_{t=1}^{3000} g_t||`, the gradient-sum displacement norm; the
existing late mean q column remains. Move the spectral endpoints to the
appendix with explicit full/subset objective coverage. Generate the new column
from the verified audit, preserving the FP32 update-roundoff qualification.

Appendix D gains two compact evidence tables: all-lag accounting and the
retrospective loss audit. Full/subset spectral endpoints form a separate small
context table. Their column contracts and sources are specified in S7.

## Execution boundary

The recommended edits require local text, table generation and a LaTeX build.
The supporting scalar audit is already complete. No additional training,
gradient reconstruction, HVP computation or cloud execution is part of the
minimum package.

Possible later work should answer a specific additional claim:

| Additional claim | Additional evidence needed |
| --- | --- |
| q estimates pointwise directional curvature accurately | Same-objective directional HVPs at matched states, with finite-step residuals |
| Patterns persist across initializations | Additional paired initialization blocks at a declared budget |
| Diagnostics predict harmful future updates | A prospective criterion evaluated on subsequent outcomes and against relevant baselines |
| Diagnostics have negligible cost | A controlled runtime and memory comparison |

These are separate extensions. They are not needed to report the conditional
geometric result or the limitations of the existing local approximation.
The stronger interpretations in Section V remain a separate revision task;
S8 resolves only the immediate Section III/IV terminology dependencies.

## Implementation and acceptance

Apply S1-S8 together so that citations, moved material and the new table column
remain consistent. Update the existing asset generator rather than hand-editing
its generated table. Extend the manuscript verification to check the new table
values, new labels and retained figure hashes. Compile with resolved references,
render the affected pages and confirm that a sentence or hyphenated word is not
split across the full figure page as currently happens with "Consequently."

Acceptance requires all nine rows, the exact factor-two interaction convention,
2999 correctly indexed loss pairs per condition, explicit threshold denominators,
unchanged source figure arrays, and a clear distinction between temporal
observations and independent initialization replicates. A full paper readiness
claim also requires a later consistency pass on the remaining sections.

## Evidence and status

- [Critical review](../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/007-section-iv-critical-review.md).
- [Verified scalar audit](../../analyses/024-2026-10-17-eos-gd-lr-calibration/section_iv_review_checks.json) and [audit script](../../analyses/024-2026-10-17-eos-gd-lr-calibration/10_review_section_iv.py).
- [Original endpoints](../../analyses/024-2026-10-17-eos-gd-lr-calibration/results.json).
- [Section IV source](../draft/sections/04-gd-experiments.tex), [Section III source](../draft/sections/03-methods.tex), and [Appendix D source](../draft/sections/07-gd-diagnostics.tex).
- [Cohen et al.](https://arxiv.org/abs/2103.00065), the existing EOS reference.

The eight edits have been applied. The [implementation record](../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/008-applied-section-iv-revision.md)
documents the generated tables, independent checks and reviewed 18-page output.
No new run, analysis or hypothesis identifier was required. `verification.json`
preserves the original pre-application anchor, numeric and draft-hash checks;
current verification is in Analysis024's `manuscript_verification.json`.
