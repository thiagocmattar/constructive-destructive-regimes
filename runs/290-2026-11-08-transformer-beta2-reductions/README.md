# Run 290: transformer beta2 reductions

- **Status:** completed, independently verified and retained; H050, continuing Analysis 065.
- **Question:** how do the requested conventional interventions change the original full-objective high-LR Adam trajectory?
- **Matched factors:** seed 0; authenticated Run223 state 544; LR .003; beta1 .9; epsilon 1e-8; zero weight decay; fixed dataset/cache/order and full-objective reduction; native FP32 Adam, FP64 diagnostics, deterministic execution.
- **Varied factor:** beta2 reductions; see the frozen [configuration](config.json) for every arm.
- **Coverage:** every loss/raw-gradient boundary 544-928, every outgoing update, first/last full update audits, complete terminal held-out splits.
- **Control:** fresh unchanged Adam. Reuse across workers only after exact fork and eight-update fingerprints and identical GPU/runtime; otherwise retain a separate matched control.
- **Meaning:** beta changes preserve moments and clock, including native new-beta bias correction. Clipping scales the full raw gradient before Adam moments. Raw diagnostics never use the clipped vector.
- **Retention:** full fork/final/recovery and audits on owned Spark; verified compact evidence locally. Input source/cache/config SHA-256 hashes are in [input_manifest.json](input_manifest.json).
- **Execution:** shared `src/eos_tasks/conventional_continuation.py`; Analysis065 scripts45-48 prepare, test, verify and manage task-owned Runpod leases.
- **Limits:** one initialization and one retrospective fixed intervention window; not a seed/generalization claim. Historical cross-device controls are not silently reused.
- **Results:** Four arms / 1,536 updates / eight audits. Beta2 .95/.9/.5 change mean training loss -13.15%/-18.47%/-11.82%, but test loss +24.64%/+33.12%/+27.77%. The unchanged control is also the qualified Run291 comparator.
- **Evidence:** [paired outcomes](../../analyses/065-2026-11-07-mlp-frozen-lr-double/conventional_outcomes.md), [observation/captions](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/015-conventional-model-coverage.md), and [compute/retention closeout](../../analyses/065-2026-11-07-mlp-frozen-lr-double/conventional_closeout.json).
- **Cleanup:** task pods deleted only after complete Spark size/SHA-256 verification; compact results remain local. No manuscript source edits. Remaining experiments are in the [ablation TODO](../../manuscript/consolidation/evidence/05-adam/c-intervention/TODO_ABLATIONS.md).
