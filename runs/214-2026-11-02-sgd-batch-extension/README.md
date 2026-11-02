# Run214: three-case SGD batch-size extension

**Deferred at the user's request:** "Keep this as a TODO, 8 hours is too long."
No production training or cloud resources were launched. All local mechanics
processes have finished. The current scripts are preparation, not a completed
experiment or fully qualified production pipeline.

H023 / Analysis037. The user requested execution of the manuscript's
[three-case measurement extension](../../manuscript/draft/revisions/2026-11-02-feedback/SGD_EXTENSION.md).
The original MLP evidence remains unchanged. This run adds ResNet-32 and
Transformer measurements, with three frozen initializations, states0/2000,
eight stochastic repetitions, one full-batch control, and 200 updates per fork.
ResNet LR0.1; Transformer LR0.2; no rate tuning on the new outcomes.

Preparation and bounded local mechanics/timing are authorized by the request.
Spark is occupied and is not used. No new paid compute is authorized without a
spending ceiling. The reference policy regenerates all three GD trajectories
per case from pinned initial tensors, ensuring a common runtime within each
case; these are new matched references, not historical bitwise replay claims.

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

Seven CPU tests pass. Local production-shaped timing covers every batch size;
the three-update ResNet reference passed a stop/resume and a sampled fork,
and the three-update Transformer reference completed. Transformer sampled-fork
smoke, independent full-batch replay, complete worker/retrieval qualification
and the result reducer remain to be exercised before production. No smoke
measurement supports a manuscript claim. See Analysis037's preparation receipt.
