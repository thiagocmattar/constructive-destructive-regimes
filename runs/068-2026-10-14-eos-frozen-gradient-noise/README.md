# Run068: real-data frozen-state gradient-noise control

Status: complete. All 36 conditions and 147,456 observations are retrieved and
byte/SHA256 verified. Source snapshots and raw batch indices/metrics are retained
locally and on Spark. The publication figure and formulation report are rendered
and reviewed; see [Analysis020 results](../../analyses/020-2026-10-14-eos-multitask/REPORT.md).
Actual execution: 37.05 minutes, zero optimizer updates, no rental charge.

Three initializations for each selected EOS case, with the complete training
objective held fixed. Fresh independently drawn batches isolate the effect of
gradient estimation noise from trajectory changes. This is a causal sampling
control, **not GD/SGD training evidence**. The owning
[Analysis020](../../analyses/020-2026-10-14-eos-multitask/README.md) records the
protocol and the separate production-training budget decision.

Each batch condition uses 32 independent sequences of 128 gradient draws, with
no parameter updates. We measure the actual batch-gradient error against the
full-objective gradient, adjacent cosine, complete-sequence B/A, gradient-only
projected change and its exact noise correction. Fixed ResNet ghost groups
preserve the BN objective. Complete prepared WikiText-2 is the sampling population;
no corpus subset replaces the training objective.
