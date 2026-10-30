# Analysis027: mini-batch noise at matched training states

Completed and verified 2026-10-30. H023; Run185 calibration and Run186 production.
The user approved the [design](../../manuscript/2026-10-30-minibatch-noise-proposal/01-experiment-and-paper-design.md)
and Runpod execution because Spark was busy. Spark and held queues were untouched.

Start with the [report](REPORT.md), [figure review](FIGURE_REVIEW.md),
[scientific observation](observations/001-controlled-minibatch-noise.md) and
[current temporal interpretation and manuscript revision](observations/004-temporal-interpretation.md).

Three CIFAR-10 MLP initializations, three GD checkpoints and five batch sizes
provide frozen sampling controls and matched 200-update SGD continuations at
LR0.02. All full-gradient/loss/covariance/HVP references cover the complete
5,000-example objective. Totals: 184,320 frozen observations, 297 moving forks,
59,400 continuation updates, 6,000 reference updates and 1,188 directional HVP
pairs. Fixed update budgets imply unequal data exposure.

Frozen variance/noise-projection targets agree within 0.50%/0.74%; the cosine
approximation has small but detectable bias. Negative sampled alignment can
coexist with positive complete-gradient alignment along the same SGD path.
Noise correction reduces the batch-32 median directional-HVP discrepancy from
0.7467 to 0.0030. These characterize one objective and LR, not a universal
stability criterion. Historical reconstruction fails qualification; all
comparisons use explicitly identified new matched references.

## Reproduction and evidence

Run186 owns immutable scientific inputs and ignored raw evidence. This analysis
owns preparation/control/retrieval scripts, independent numerical checks,
reductions, five original diagnostic PDFs, three focused figure assets and
manuscript verification. From the root:

```text
python analyses/027-2026-10-30-minibatch-noise/02_verify.py
python analyses/027-2026-10-30-minibatch-noise/03_analyze.py
python analyses/027-2026-10-30-minibatch-noise/06_report.py
python analyses/027-2026-10-30-minibatch-noise/09_paper_figures.py
python analyses/027-2026-10-30-minibatch-noise/11_temporal_paper_figure.py
python analyses/027-2026-10-30-minibatch-noise/12_verify_temporal_revision.py
```

Compile the draft before the last command. Script 12 checks current figures,
new temporal statements and unchanged equations against commit `ed1162e`.
Scripts 07/08 and 10 describe superseded manuscript snapshots. Script 09
generates the frozen calibration and batch-size outcomes; script 11 adds the
temporal view. The current paper includes the outcomes and temporal figures,
while the frozen figure remains in the research record. Re-render and review
whenever figures change. The draft is self-contained for compilation.
Do not replay cloud control scripts as an analysis step: their completed lease
is task-specific and the pod/volume no longer exist.

[verification.json](verification.json), [scientific_summary.json](scientific_summary.json),
[paper_figure_manifest.json](paper_figure_manifest.json),
[temporal_figure_manifest.json](temporal_figure_manifest.json),
[temporal_revision_verification.json](temporal_revision_verification.json) and
[temporal_revision_review.json](temporal_revision_review.json) retain the current
checks and hashes. The original diagnostics, frozen figure and HVP results
remain available in the report. The paper has no added appendix or HVP tables.

## Execution closeout

One RTX 4090 at USD0.89/hour; production took 22.86 minutes, provision-to-cleanup
57.90 minutes. Compute plus storage is approximately **USD0.87 provisional**,
below the conservative USD5 guard. Posted billing was not available at closeout.
All 7,178 inventory-listed files (15.192 GB) were size/SHA-256 verified locally
before the task pod and 30 GB volume were deleted. No cloud storage remains.
Raw evidence and the 10.267 GB transfer archive remain in ignored local storage.
See [retrieval](retrieval.json) and [closeout](closeout.json). Unrelated resources
were preserved. The revised Section V contains the focused sampling argument.
