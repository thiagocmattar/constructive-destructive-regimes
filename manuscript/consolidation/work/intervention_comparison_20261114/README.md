# Intervention comparison sheet

Reformats the verified c-intervention experiment inventory into one wide sheet,
with one intervention per row, ten comparison columns, filters and frozen labels.
The distributed-cap quantile/LR ablation is a separate row of the same family.
Missing model/settings coverage and remaining measurements are explicit.

- [Workbook](../../../../outputs/intervention-comparison-20261114/intervention-experiment-comparison.xlsx)
- [Source and output verification](verification.json)
- [Builder](build.mjs)

The sheet summarizes existing records. No training, manuscript, selected figure,
scientific data or ablation TODO is changed. Wider archived screens remain in
their original records. The source hashes and principal numerical checks are
retained in the verification file.

Rebuild with the bundled Codex runtime Node executable after creating a local
`node_modules` junction to its dependency directory. The junction is ignored.
The builder uses `@oai/artifact-tool`, validates source values, exports the XLSX,
and renders three horizontal regions for visual review under ignored `tmp/`.
All table cells are checked before export. Saved freeze panes, filter/table
ranges, representative coverage cells and the absence of exported cell errors
are checked after export. No new statistical aggregation is introduced.
