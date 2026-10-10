# Artifact envelope

These are conventions for future runs, not files from a completed experiment.
Scientific payloads are defined per run after the metric discussion.

```text
artifacts/attempts/001-<label>/
  resolved_config.json
  manifest.json
  events.jsonl
  environment.json
  summary.json
  transfer_inventory.json
  verification.json
  checkpoints/                 ignored bytes; retain identity in the inventory
```

A manifest should record schema version, run/attempt IDs, status, timestamps,
source/config/input identities, environment, execution command, progress, and
required outputs. Suggested statuses: prepared, running, completed, failed,
cancelled. Mark completion only after the run-specific verifier passes. Write
manifests atomically (temporary sibling file plus replacement).

Events are append-only JSONL with UTC timestamp, event type, update index, token
counts, timing, and defined diagnostics. Readers must tolerate the unfinished
last line of an actively written stream. Missing or undefined values are not
numerical zero; store the reason explicitly.

Record initialization and checkpoint hashes. A resumable checkpoint needs the
model, optimizer/scheduler, random states, data-order position, and relevant
precision/scaler state. A final model alone is not proof of exact resumability
and cannot recreate an unrecorded gradient trajectory.

`tools/artifact_inventory.py create ROOT --output INVENTORY` hashes all regular
files under an agreed artifact root except the output inventory itself. Choose a
root containing only intended transfer files. The helper rejects symlinks and
paths outside the root. `verify ROOT INVENTORY` checks sizes and SHA-256; `--strict`
also rejects extra files except the inventory itself. Verification receipts and
later local analysis should live outside an immutable transfer root.

Commit compact records/inventories and relevant summaries. Keep checkpoint/data
bytes and transfer archives ignored. A transfer inventory verifies bytes; the
run-specific verifier establishes scientific completeness and correctness.
