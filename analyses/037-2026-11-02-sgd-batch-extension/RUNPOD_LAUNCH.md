# Run214 / Analysis037 Runpod launch packet

The user requested execution of the manuscript TODO on Runpod. The earlier
US$75/eight-hour proposal was declined. No cloud allocation or production update
has started; the new aggregate spending/time limit is awaiting the user's reply.

## Scientific work

Keep the registered config: ResNet-32/CIFAR-10 and Transformer/WikiText-2,
three initializations, states0/2000, eight stochastic repetitions and one full
control per cell, 200 updates per continuation. Regenerate all six 2000-update GD
references on the selected runtime. Reuse Analysis027 MLP evidence unchanged.
Total new work: 348 forks, 69,600 continuation updates and 12,000 reference
updates. The measurement contract remains [SGD_EXTENSION.md](../../manuscript/draft/revisions/2026-11-02-feedback/SGD_EXTENSION.md).

Seven CPU tests and both complete short real-data queues pass. The
[qualification receipt](worker_qualification.json) records sampled-index replay,
CPU NumPy audits, actual displacement, 78 inventory entries, copied-artifact
verification and deliberate corruption rejection. A strict ResNet bitwise test
failed; maximum absolute parameter difference is2.97e-8 and relative total L2
error1.07e-9 over three updates. The recorded FP32 tolerances pass. Transformer
replay is exact. This is mechanics evidence, not a long-horizon replay guarantee.

## Resource and timing

The timestamped [preflight](runpod_preflight.json) uses runpodctl3.0.1's live
catalog and official PyTorch2.8 image. It records the baseline resource inventory
and per-case tar packets with source bytes/SHA256. Six independent workers cover
the two cases by three seeds. Select one GPU model and pinned runtime per case;
do not mix device models across its initialization blocks. Re-read price and
capacity immediately before allocation. Preserve unrelated Pods and volumes.

At preflight, RTX4090 has limited reported capacity: US$0.34/GPU-hour Community
or US$0.89 Secure. RTX5090 lists US$0.69/1.19 respectively but reports no capacity.
Availability is not a placement guarantee. Proposed maximum unit compute rate:
US$1.19/GPU-hour, at most six concurrent GPUs.

Cloud ETC is **unmeasured**. Prior laptop samples project60.83 GPU-hours before
endpoint/I/O allowances; the slowest per-seed queue projects11.52 hours. Neither
number is a cloud estimate. Setup, warmup, complete updates and evaluations,
diagnostics, saving and actual transfers must be measured on the selected GPU.
The prepared profiler covers all registered batch sizes. Run the short queue and
remote transfer check before production; calculate an ETC separately for each
case and its parallel critical path. Launch the full workload only if a
conservative projection, including retrieval margin, fits the remaining envelope.
Retain replication and horizons; do not shrink the design to force acceptance.

## Proposed envelope and guard

The pending proposal is **US$40 total / four hours wall time**, including setup,
calibration, all workers and retries, storage, retrieval and verification. The
clock starts at the first billable allocation. Calibration has a20-minute limit;
its measured results gate production without another permission round within the
approved envelope. At six GPUs, four hours at the maximum unit rate costs
US$28.56 in compute before storage. If the user selects another limit, record it
explicitly before allocation and recalculate the guard.

Use30GB container storage and35GB Pod volume per worker. Running disks cost
US$0.10/GB/month; stopped Pod volume costs US$0.20/GB/month. No network volumes
are created. These are the current [Runpod storage prices](https://docs.runpod.io/pods/pricing).
The Pod volume holds recovery, logs and evidence until verified local retrieval.

Detach both remote execution and an independent local guard. Record Pod IDs and
ownership immediately at creation. Monitor every60seconds: updates/tokens,
sampled or complete-objective loss, throughput, remaining ETC, memory, disk,
event age and aggregate estimated spend. Stop scientific work at the earlier of
30minutes before the wall deadline or US$32 estimated spend. Save recovery and
inventory; stop task compute at the hard deadline. All retained disk and recovery
must remain inside US$40. Do not silently extend the deadline or reserve.

Nonfinite/undefined metrics, mismatched inputs/runtime, stale progress beyond
five full updates or an ETC outside the remaining envelope trigger inspection
and a bounded stop. The current CLI exposes no stop/terminate timer in its live
create help; use the independent guard rather than an unsupported flag.

## Acceptance and closeout

Check exact starting/input identities and sample draws, all199 adjacent pairs,
200-update displacement, complete start/end training losses and saved CPU vector
audits. Verify each worker's relative-path inventory, bytes and SHA256 locally
before deleting its Pod/storage. Incomplete evidence remains labeled partial.
Keep unrelated resources intact; report retained disk and provisional billing.

After the full cohort passes independent reduction, create the requested2x3
figure with initialization/2k-state curves, initialization-range shading and
explicit full-batch markers. Write case-specific observations and exceptions,
update the empirical prose and coverage, rebuild and visually inspect the
manuscript, then commit the completed extension. Production outcomes do not yet
exist; the current review PDF remains the preceding verified version.
