# Runpod experiment playbook

Use the installed `runpod` router skill before Runpod work. Prefer live MCP
schemas for account/catalog/lifecycle operations. Use native SSH/SCP, or current
`runpodctl` help, for terminal execution and file movement. Exact flags and prices
must be checked in that session, not copied from this document.

## Read-only preflight

1. Verify authentication without showing secrets and inventory existing resources.
   Identify task ownership; a name alone is not authority to modify a resource.
2. Inspect live GPU price, memory, capacity, placement, template/image, and storage.
3. Verify a usable SSH key/connection route and artifact transfer path.
4. Resolve production memory fit, then GPU placement, then storage. A network
   volume can constrain data-center placement; choose it deliberately.
5. Record the exact source/config/runtime identity, spending envelope, deadline,
   warnings, and retrieval inventory in the run.

At bootstrap, a read-only MCP Pod query succeeded. Native Windows `ssh` and `scp`
are available; `runpodctl` was not on PATH. MCP plus native SSH/SCP is the working
route. Existing account resources were left untouched. This check is historical,
not proof of future capacity, transport readiness, or a launch permission.

Run001 is a completed local example (see its README and closeout receipts).
Its practical lessons are: inspect the actual storage mount before installing
many package files; its FUSE mount made extraction slow, while a container-local
venv succeeded. Align inherited TorchVision/TorchAudio binaries when upgrading
Torch and import the chosen model class during preflight. Test the actual Flash
Attention backend, not CUDA availability alone. For CLI reads that return an
execution session, collect its completion before treating it as a failed poll.
Check fresh price/capacity/schemas and choose a new deadline for each run; the
historical example's expired wrapper/resource IDs are not reusable launch inputs.

## Provision and prepare within authority

Use a GPU Pod for batch training. Prefer a suitable official PyTorch template and
pin the selected image/environment. Keep source, caches, logs, and outputs under
the actual persistent mount. Verify its lifecycle against the
[storage documentation](https://docs.runpod.io/pods/storage/types).

Container storage, Pod volume, and network volume have different retention
rules. Pod-volume data must be retrieved before Pod deletion; network volumes
can survive the Pod and keep billing. Fast local data staging may help random
access, while a durable master preserves recoverability. Record both locations.

Record the Pod ID even if provisioning/reachability times out. Wait for an actual
SSH probe, then verify GPU/CUDA, Python environment, source/config, input hashes,
and production-shaped smoke. Do not equate `Running` with a usable workload.
Re-query SSH details after restarts. Keep secrets out of tracked commands/logs.

## Launch and monitor

Run with `setsid`, `tmux`, or equivalent detachment, with stdin disconnected and
logs on persistent storage. Record PID/command/paths and publish `manifest.json`
plus `events.jsonl`. Validate at least one real update/artifact; process launch
alone is insufficient.

Monitor in separate bounded calls. Report completed steps/tokens, available
losses/diagnostics, throughput, refreshed ETC, GPU memory/utilization, disk, and
event age. Avoid duplicate starts. Preserve failed setup logs; bounded retries
must use the same authorized inputs and fit the aggregate budget/deadline.

## Retrieve and teardown

1. Confirm terminal scientific verification and required files, including agreed
   checkpoint and resume states.
2. Create a file-size/SHA-256 inventory remotely using `tools/artifact_inventory.py`.
3. Transfer by SCP/resumable transfer/object storage as appropriate.
4. Verify the inventory locally. A successful copy command is not enough.
5. Remove only task-owned compute after acceptance; confirm via a fresh read.
6. Reconcile retained storage and continuing cost. Do not delete shared volumes
   just because this run finished.

Allow retrieval time before any automatic deadline. If integrity fails near the
deadline, preserve durable evidence and communicate the bounded recovery choice.
For multiple workers, verify each worker independently before its teardown, then
verify the retrieved cohort.

Closeout records resource IDs, actual runtime, estimated/posted cost and billing
window, verification receipt, cleanup state, and retained resources. Posted
billing may lag deletion; label incomplete totals provisional.

Optional CLI installation should follow the live
[runpodctl README](https://github.com/runpod/runpodctl). Do not install a second
cloud orchestration stack merely for this blank scaffold.
