# Section VII conclusion revision

15 November 2026. Authorized conservative revision of the four conclusion paragraphs.

## Changes and evidence

[PARAGRAPH_REVIEW.md](PARAGRAPH_REVIEW.md) records the original text, diagnosis, proposed solution, and rewrite for each paragraph. Every decision was recorded before its source edit. The structure, heading, and label are retained.

1. Lead with adjacent opposition coexisting with accumulated reinforcement and loss reduction; explain the framework through that distinction.
2. Restore the fixed-objective, full-gradient, smoothness, and nonzero-update conditions of the curvature identity. Distinguish the completed-step identity from the empirical Adam warning rule; state its target, comparative result, and algebraic tie.
3. Specify the architecture contrast and measured-window scope. Name unchanged Adam as the intervention baseline, express both outcomes as reductions, and identify the ResNet q50 endpoint exception and computational cost.
4. Replace the unsupported necessity claim about understanding harmful movement with the observed improvement despite persistent reversal. Keep future validation and scale questions; clarify that the proposed stochastic extension concerns Adam, given the existing SGD experiments.

The revised introduction, characterization subsection, tracing findings, and intervention section support these statements. No new result, citation, equation, experiment, or scientific formulation was introduced. The conclusion summarizes existing evidence; it does not expand the tested population or claim wall-clock gains.

## Verification

- [01_edit.py](01_edit.py) records each decision before applying it.
- [02_build.py](02_build.py) builds and renders the main PDF.
- [03_verify.py](03_verify.py) checks audit replay, preserved files, heading and label, links, build warnings, and changed pages.
- [build.json](build.json) and [verification.json](verification.json) retain the final check results.

All 56 other draft files are byte-identical to their pre-edit versions. Existing unrelated modified files retain their hashes. The PDF remains 22 pages and has no LaTeX, reference, or box warnings. Page 17 was inspected at full page resolution and changed pages 17-19 in a contact sheet; no overlap or clipping was observed. Downstream changes are reflow, with appendix sources unchanged. All other pages render identically to the baseline.

Pre-edit snapshots, decision inputs, build logs, and page renders remain in ignored `scratch/`; the baseline-dependent verification requires those local snapshots. This is an editorial verification, not independent experimental replication.
