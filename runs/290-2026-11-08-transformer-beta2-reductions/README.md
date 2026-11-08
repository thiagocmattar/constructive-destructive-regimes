# Run 290: transformer beta2 reductions

- **Status:** prepared and launching; H050, continuing Analysis 065.
- **Question:** how do the requested conventional interventions change the original full-objective high-LR Adam trajectory?
- **Matched factors:** seed 0; authenticated Run223 state 544; LR .003; beta1 .9; epsilon 1e-8; zero weight decay; fixed dataset/cache/order and full-objective reduction; native FP32 Adam, FP64 diagnostics, deterministic execution.
- **Varied factor:** beta2 reductions; see the frozen [configuration](config.json) for every arm.
- **Coverage:** every loss/raw-gradient boundary 544-928, every outgoing update, first/last full update audits, complete terminal held-out splits.
- **Control:** fresh unchanged Adam. Reuse across workers only after exact fork and eight-update fingerprints and identical GPU/runtime; otherwise retain a separate matched control.
- **Meaning:** beta changes preserve moments and clock, including native new-beta bias correction. Clipping scales the full raw gradient before Adam moments. Raw diagnostics never use the clipped vector.
- **Retention:** full fork/final/recovery and audits on owned Spark; verified compact evidence locally. Input source/cache/config SHA-256 hashes are in [input_manifest.json](input_manifest.json).
- **Execution:** shared `src/eos_tasks/conventional_continuation.py`; Analysis065 scripts45-48 prepare, test, verify and manage task-owned Runpod leases.
- **Limits:** one initialization and one retrospective fixed intervention window; not a seed/generalization claim. Historical cross-device controls are not silently reused.
- **Next:** verify execution, reduce and inspect figures, update consolidation and the [ablation TODO](../../manuscript/consolidation/evidence/05-adam/c-intervention/TODO_ABLATIONS.md).
