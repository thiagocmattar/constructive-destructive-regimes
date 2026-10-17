# Manuscript workspace

The original supplied materials are preserved:

- `latex-zip.zip`: original LaTeX source archive.
- `2026-10-03 review/`: submitted paper PDF, reviewer PDFs, and editor feedback.
- `source_inventory.json`: initial archive hash and per-file byte sizes/SHA-256.

`draft/` is the editable copy extracted from the archive on 2026-10-03. All 22
files initially match the archive bytes. The inventory is an immutable baseline,
not a manifest to refresh after editorial edits.

Entry point: [draft/bare_jrnl.tex](draft/bare_jrnl.tex). Section text is in
`draft/sections/`; the two-step derivation is in the main file. Existing figure
PDFs and `bibliography.bib` are carried over exactly. Keep original assets until
their replacement is explicitly part of an editing task.

When introducing a variable, explain its practical intuition and why it is
useful before or alongside its formal definition. The intended audience uses
the theory to interpret training behavior. Preserve the author's motivation
when polishing, while distinguishing an intuitive interpretation from a proved
quantitative claim.

The archive and submitted PDF are distinct snapshots until their correspondence
is checked. Read the applicable snapshot when addressing a reviewer equation.
The review map is in [research/REVIEW_CONTEXT.md](../research/REVIEW_CONTEXT.md).

Build from `draft/` so relative inputs resolve; direct output into ignored
`manuscript/artifacts/`. The IEEEtran class/bibliography styles and TeX packages
must be available in the selected TeX environment. No manuscript build was run
during bootstrap. New empirical text/figures must trace to this workspace's
run/analysis observations; see [research/MANUSCRIPT.md](../research/MANUSCRIPT.md).

## Revision proposal

[Surgical changes proposal, 2026-10-07](2026-10-07-surgical-changes-proposal/README.md)
organizes the advisor discussion around temporal interference and learning-rate
interventions. It contains an evidence map, section-level changes, mathematical
definitions, an experiment plan, and a focused literature comparison. It is a
planning record; the draft, supplied reflections, and original assets are unchanged.

## GD formulation report

The [restatement report](2026-10-11-restatement/README.md) develops the supplied
formulation in English through the empirical full-batch GD result, stopping
before the optimizer extension. It follows the local writing-style guide and
includes a compact publication view of Analysis019 GD evidence. Editable TeX,
the builder, reviewed PDF and provenance are together in
`2026-10-11-restatement/`. Its source remains a separate reference for the draft revision.

The restatement is now being integrated into `draft/` in user-directed steps.
The background includes a practical pretraining-stability review, and Section 3
uses displacement, directional curvature, sampling noise, normalized
displacement, and a local connection to loss decrease. The older
sample-level algebra is retained in Appendix A for the existing image results.
Appendix B derives the loss relations using the main section's notation.
The scientific interpretation of the image results, the introduction, and the
conclusion still require later alignment with the new formulation. Current
compiled output and mechanical verification records are in ignored
`manuscript/artifacts/draft/`.
