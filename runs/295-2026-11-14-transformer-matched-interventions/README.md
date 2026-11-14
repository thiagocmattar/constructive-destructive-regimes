# Transformer: matched four-family interventions

- **Status:** complete and independently verified; H051 / Analysis 065; initialization 0, states 544–928, 12 conditions / 4608 treated updates.
- **Authorization:** the author approved the faster option and complete paper deliverables on 14 November 2026. The whole task stayed within $50 / 8 hours; no extra initializations or cap x LR cells were launched.
- **Frozen scientific inputs:** [config](config.json), [input hashes](input_manifest.json), worker `src/eos_tasks/matched_interventions.py`. Complete native prefix/fork; 8 preceding untreated calibration updates; unchanged Adam plus 11 treatments.
- **Qualification:** exact predecessor replay and identical eight-update fingerprints across this model's workers: `b971d7fd770757b8c8bda370f3e5fbd08405545cc4754ccc815904bd7fc9edda`. Runtime/GPU and full-fork identity are recorded in the result provenance.
- **Checks:** 36 exact gradient/probe/native-update audits; 9 cap audits; 12 terminal full-gradient/training-loss replays and 24 full terminal held-out split replays. Complete boundary/update/evaluation histories and past-only calibration are verified.
- **Retention:** 194 files /31,247,937,611 raw bytes verified on owned Spark. [Durable retention receipt](spark_durable_retention.json); raw tensors/checkpoints are excluded from Git. Compact verified records remain local under ignored artifacts/attempt-01.
- **Results:** [matched observation](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/016-matched-intervention-matrix.md), [36-condition data](../../analyses/065-2026-11-07-mlp-frozen-lr-double/matched/results.json), [publication package](../../manuscript/consolidation/evidence/05-adam/c-intervention/README.md).
- **Compute closed:** all task-owned Pods deleted only after size/SHA-256 verification; no retained cloud volume. [Whole-task provisional billing and cleanup](../../analyses/065-2026-11-07-mlp-frozen-lr-double/matched/compute_closeout.json).
