# Conservative Section IV readability revision

15 November 2026. Applies the approved review of [Section IV](../../draft/sections/04-gd-sgd.tex), preserving the experiments and formulation in the [main manuscript](../../draft/manuscript.pdf).

## Changes

- The opening names gradient size and alignment instead of unspecified signatures. The results describe observed loss reduction rather than claiming convergence from finite trajectories.
- Smoothed curvature near two is described relative to the quadratic reference threshold. The text separates this averaged observation from individual nonlinear loss changes.
- Final training loss is explicitly compared among the tested learning rates. The accumulated-interaction result now directly follows displacement retention, before the unchanged geometric interpretation. A short explanation distinguishes the squared-step-length comparison from the total path length used by normalized displacement.
- The lag discussion describes increasing backward separation from a fixed endpoint. It retains nonmonotone recurrence and the absence of momentum, without treating the lag axis as training progress or calling the observation apparent memory.
- The mini-batch experiment is distinguished from the fixed-state approximation before its design is described. Its conclusion states what cosine alone cannot explain and identifies gradient magnitudes, the sampling contribution, and the nonlinear remainder as relevant terms.

The [paragraph audit](PARAGRAPH_REVIEW.md) records the read passage, diagnosis, proposal, and replacement before every source edit. Ten entries cover eight prose revisions, the paragraph relocation, and one closing-sentence concision review. The latter keeps the end of Section IV together instead of leaving a three-word tail before Section V. No formatting settings were changed.

## Preserved scope

All four figure blocks and their captions, the setup table, all numerical tokens, labels, citation keys, and three subsection headings and their order are unchanged. The learning rates, initializations, repetition counts, update budgets, displacement percentages, and comparison conditions are preserved. No experimental data, figures, algorithms, equations, or reported results were altered. All other scientific source files, including Sections I--III and V onward, remain unchanged.

The changed draft files are Section IV, its rebuilt PDF, and the draft README. The [verification record](verification.json) checks the protected content, replays the audit to recover the final source, resolves cross-references, and verifies 56 unchanged draft files and five pre-existing modified research records.

## Build and visual review

The [build record](build.json) reports 22 pages with no LaTeX, undefined-reference, overfull, or underfull warnings. The Section IV text was inspected individually; final pages 6--11, whose rendering changed, were reviewed in a contact sheet. No clipping or overlap was observed. All other pages render identically to the previously reviewed baseline. Downstream layout changes are text reflow only.

From the repository root:

```powershell
python manuscript/consolidation/work/section4_readability_20261115/02_build.py
python manuscript/consolidation/work/section4_readability_20261115/03_verify.py
```

Verification requires this task's ignored baseline manifests, source, and PDF. The committed audit is independently readable; its baseline is commit `070e3e0`. Auxiliary files, edit inputs, and page images remain in ignored scratch storage. A later rebuild requires renewed visual review before updating the manual-review statement.
