# Conservative Section III readability revision

15 November 2026. The user authorized the proposed corrections while explicitly asking to preserve the paper and its formulation. This revision changes selected prose in [Section III](../../draft/sections/03-regimes.tex) and rebuilds the [main manuscript](../../draft/manuscript.pdf).

## Scope

All 19 displayed equations (6--24), all 27 labels, the mathematical notation and definitions, and the order of the five subsections are preserved. Subsection III-D is unchanged, including its examples and its distinction between displacement retention and learned information. The final subsection heading is narrowed from "Conditions for Loss Decrease and Spike Formation" to "Conditions for Loss Decrease and Increase."

Sections I--II and IV onward, the abstract, bibliography source, figures, tables, appendix, and supplement are unchanged. No experiments, new results, new thresholds, or new mathematical formulation are introduced. The only changed draft files are Section III, the rebuilt PDF, and its README.

## Changes and reasons

- The opening identifies plain SGD and the completed-step scope before describing the diagnostics.
- The regime definition now explicitly distinguishes consecutive opposition from the sum of all pairwise interactions. Persistence remains a qualitative description of repeated signs over an interval; no new classifier or window threshold is imposed.
- III-A states which gradient generated the completed step, explains the purpose of the inner product, and locates the defined curvature at the starting point. Its sign argument replaces an ambiguous proportionality statement without changing the underlying equation.
- III-B distinguishes exact directional curvature averaged along the completed step from its approximation to starting-point curvature. Full-gradient and fixed-objective assumptions remain explicit, as do the separate sampled-gradient term and the use of the actual update for Adam.
- III-C announces the switch from an evolving SGD trajectory to independent sampling at fixed parameters before the fixed-state equation. It explains the norm approximation and correctly calls the numerator a squared norm, rather than a direction. The same symbols and formulas are retained.
- III-E consistently assigns the loss criterion to the completed step. The former "return step" wording referred to the wrong update. The original thresholds, numerical examples, and nonlinear/sampling limitations are retained.

## Review sequence and verification

The [paragraph review](PARAGRAPH_REVIEW.md) records the source passage, diagnosis, proposed solution, and replacement before each edit. It contains 20 entries covering 16 prose passages, one heading, and three revisits. A final concision pass retained the added explanations while removing repeated framing and optimizer-scope wording. It resolved a downstream float overflow without changing fonts, margins, equations, or float settings. A final wording check made explicit that reinforcement across all pairs concerns their net contribution.

The [verification record](verification.json) replays the full audit to reproduce the final Section III source. It verifies the unchanged equations and labels, subsection order, intact III-D, all cross-reference keys, 56 protected draft files, and five pre-existing modified research records. Whitespace-token counts including TeX are 1,557 before and 1,573 after.

The [build record](build.json) reports 22 pages with no LaTeX, undefined-reference, overfull, or underfull warnings. Pages 4--6 were inspected individually. All pages with changed rendering (4--14) were checked in contact sheets; no clipping or overlaps were observed. Pages 1--3 and 15--22 render identically to the previously reviewed baseline. Downstream text reflows, but its sources and scientific content are unchanged.

## Reproduction

From the repository root:

```powershell
python manuscript/consolidation/work/section3_readability_20261115/02_build.py
python manuscript/consolidation/work/section3_readability_20261115/03_verify.py
```

Verification uses the local ignored pre-edit manifests, source, and PDF. The committed paragraph audit is independently readable; its baseline is commit `043d694`. The edit helper records decisions before applying exact replacements. Auxiliary files, edit inputs, and rendered pages remain in ignored scratch storage. Any subsequent rebuild needs renewed visual review before updating the recorded manual-review statement.
