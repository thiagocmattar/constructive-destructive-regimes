# Run 286: MLP cap quantile x LR, initialization 0

Status: completed and verified. Hypothesis H049; Analysis 065.

This run owns 18 prespecified continuations: distributed cap q25/q50/q75/q90/q99
at LR0.001/0.002/0.004, plus uncapped Adam at each LR. One pinned initialization
(0); all arms share its native state448, eight-update1x calibration, full5000
training objective and probe, Adam moments and data ordering. Continue through
loss state1152 plus one diagnostic outgoing transition. The global LR changes at
update449; output affine is uncapped. Budgets never scale with LR.

The complete scientific/metric/execution contract is
[Analysis065 observation012](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/012-distributed-cap-lr-plan.md).
`config.json` is the resolved scientific recipe; `input_manifest.json` freezes
source/data identities. `01_worker.py` invokes the shared sweep implementation.
Own artifacts live in `artifacts/attempt-01/worker` and are ignored by Git.

All54 continuations across Runs286-288 are explicitly within the user's MLP-only
request. Owned Spark only, zero cloud cost. Raw gradients are never clipped or
differenced. Full10000-image held-out curves every32states are secondary and do
not control this run. Retain recovery/final states, scalar histories and audits.
Record smoke, launch, retrieval, independent verification and closeout below.

## Completed evidence

All 18 conditions completed through state 1153 (705 applied continuation updates
per arm; loss statistics use 448-1152). Independent replay and numerical reduction
pass 54 retained audits, all dense cap/LR/clock/coverage checks, and exact historical
1x control/q50 reproduction. See `reduction_verification.json`.

Complete raw evidence is retained at the remote paths in
`spark_retained_inventory.json`; local compact exports and scalar histories are
verified by `spark_compact_retrieval.json`. Full local checkpoint copies are not
retained after the disk-limit recovery. This changed storage only, not scientific
inputs or execution. Figures, all-condition tables, uncertainty and interpretation
are in Analysis 065 observation 013. Owned Spark only, no cloud charges; task exited.
