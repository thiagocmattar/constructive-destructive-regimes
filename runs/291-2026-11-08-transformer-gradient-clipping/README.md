# Run 291: transformer global gradient clipping

- **Status:** completed, independently verified and retained; H050, continuing Analysis 065.
- **Question:** how do the requested conventional interventions change the original full-objective high-LR Adam trajectory?
- **Matched factors:** seed 0; authenticated Run223 state 544; LR .003; beta1 .9; epsilon 1e-8; zero weight decay; fixed dataset/cache/order and full-objective reduction; native FP32 Adam, FP64 diagnostics, deterministic execution.
- **Varied factor:** global gradient clipping; see the frozen [configuration](config.json) for every arm.
- **Coverage:** every loss/raw-gradient boundary 544-928, every outgoing update, first/last full update audits, complete terminal held-out splits.
- **Control:** fresh unchanged Adam. Reuse across workers only after exact fork and eight-update fingerprints and identical GPU/runtime; otherwise retain a separate matched control.
- **Meaning:** beta changes preserve moments and clock, including native new-beta bias correction. Clipping scales the full raw gradient before Adam moments. Raw diagnostics never use the clipped vector.
- **Retention:** full fork/final/recovery and audits on owned Spark; verified compact evidence locally. Input source/cache/config SHA-256 hashes are in [input_manifest.json](input_manifest.json).
- **Execution:** shared `src/eos_tasks/conventional_continuation.py`; Analysis065 scripts45-48 prepare, test, verify and manage task-owned Runpod leases.
- **Limits:** one initialization and one retrospective fixed intervention window; not a seed/generalization claim. Historical cross-device controls are not silently reused.
- **Results:** Two treatment arms / 768 updates / four audits. Clipping at 1/.5 changes mean training loss -17.74%/-19.02%, but test loss +14.84%/+11.30%. Clipping is active on 243/384 and 349/384 updates, respectively; the exact common control is Run290.
- **Evidence:** [paired outcomes](../../analyses/065-2026-11-07-mlp-frozen-lr-double/conventional_outcomes.md), [observation/captions](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/015-conventional-model-coverage.md), and [compute/retention closeout](../../analyses/065-2026-11-07-mlp-frozen-lr-double/conventional_closeout.json).
- **Cleanup:** task pods deleted only after complete Spark size/SHA-256 verification; compact results remain local. No manuscript source edits. Remaining experiments are in the [ablation TODO](../../manuscript/consolidation/evidence/05-adam/c-intervention/TODO_ABLATIONS.md).
