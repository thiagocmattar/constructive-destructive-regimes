# Run214: three-case SGD batch-size extension

The user resumed this task and approved Runpod execution under US$40/four hours,
then extended the shared wall limit to six hours while keeping US$40 total.
Hard deadline: 2026-11-02 19:34:56 UTC. All three ResNet blocks now execute;
Transformer moved to separately frozen Run216 before its production launch.
Qualified short queues are mechanics checks, not paper evidence.

H023 / Analysis037. The user requested execution of the manuscript's
[three-case measurement extension](../../manuscript/draft/revisions/2026-11-02-feedback/SGD_EXTENSION.md).
The original MLP evidence remains unchanged. This run adds ResNet-32 and
Transformer measurements, with three frozen initializations, states0/2000,
eight stochastic repetitions, one full-batch control, and 200 updates per fork.
ResNet LR0.1; Transformer LR0.2; no rate tuning on the new outcomes.

Spark is occupied and is not used. The reference policy regenerates all three
GD trajectories per case from pinned initial tensors, using a common objective,
precision and accumulation convention within each case. ResNet seed blocks use
L40S/4090; their range does not isolate initialization from GPU/runtime effects.
These are new matched references, not historical bitwise replay claims.

The config defines the scientific inputs. Before production, freeze byte/SHA
identities, verify the reductions/sampling/recovery, record calibrated ETC and
the selected execution envelope. Scientific changes after execution need a new
run ID. Mechanics outputs under scratch are not paper evidence.

Retain per-update raw-gradient and actual-update geometry, sampled loss,
batch indices, complete endpoint training losses, starting/final/recovery
states, RNG and diagnostic accumulators, source identities and audit vectors.
Use detached execution and independent artifact inventories. Undefined cosines
remain null and prevent automatic figure acceptance; never silently omit them.
No changes to held queues, unrelated resources or existing raw evidence.

Seven CPU tests pass. Local production-shaped timing covers every batch size.
Both three-update worker queues now pass reference/full-control/sampled-fork
checks, independent geometry and index reduction, copied-artifact verification
and corruption rejection. Full replay is exact for Transformer; ResNet has small
recorded FP32 cuDNN drift rather than bitwise equality. See Analysis037's
[qualification](../../analyses/037-2026-11-02-sgd-batch-extension/worker_qualification.json)
and [Runpod launch packet](../../analyses/037-2026-11-02-sgd-batch-extension/RUNPOD_LAUNCH.md).
Cloud calibration and actual remote transfer are qualified. The full348-fork
cohort across Runs214/216, joint reducer, figure and manuscript remain pending.
Source data, seeds, rates and horizons are unchanged. Analysis037 owns the shared
guard, incremental retrieval, cost accounting and verified worker cleanup.
