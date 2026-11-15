# Section VI readability revision

15 November 2026. Authorized conservative prose revision of the intervention section, preserving its formulation and evidence.

## Paragraph workflow

[PARAGRAPH_REVIEW.md](PARAGRAPH_REVIEW.md) records eleven sequential decisions: original text, diagnosis, proposed solution, and rewrite. Each decision was recorded before the source mutation. The verification replays the decisions against the pre-edit snapshot and reproduces the final source exactly.

The revision explains the local displacement bound and percentile labels beside their definitions, separates intervention settings from optimizer-state handling, and introduces query/key/value terminology. The contribution remains a test of representation displacement as an intervention target. Claims concern loss outcomes over matched updates; elapsed-time superiority is not asserted. The exact squared-loss decomposition is interpreted through both direction and magnitude and is distinguished from a causal explanation of the intervention's benefit.

## Evidence and preservation

Sources checked were the intervention section, matched outcome table, appendix, and extended intervention-method record. No new experiment, analysis, or empirical result was introduced. Five displayed equations, two figures and captions, one algorithm, nine labels, two subsection headings, citation keys, original numerical tokens, and table inputs are unchanged. Added numerical tokens spell out the calibration percentiles. All 56 other draft files are byte-identical to the local baseline, and all five pre-existing modified research files retain their hashes.

## Verification

- [01_edit.py](01_edit.py): records the decision before each source edit; decisions remain in the paragraph audit.
- [02_build.py](02_build.py): rebuilds the manuscript with BibTeX and repeated LaTeX passes, checks warnings, and renders pages.
- [03_verify.py](03_verify.py): checks protected source blocks, references, audit replay, hashes, and rendered-page changes against ignored pre-edit snapshots.
- [build.json](build.json) and [verification.json](verification.json): final build and verification results.

The final PDF has 22 pages, with no LaTeX, reference, or box warnings. Changed pages 12–19 were inspected in contact sheets; the output-loss discussion was also inspected at full page resolution. No clipping or overlap was observed. Other pages render identically to the pre-edit PDF. Downstream page changes are reflow; their source files are unchanged.

The local baseline snapshots and page renders remain ignored in `scratch/`. Verification against that baseline requires these local snapshots. This record describes an editorial check, not independent replication of the experiment. No manuscript sources outside Section VI were scientifically revised.
