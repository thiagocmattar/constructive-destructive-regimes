# Operational helpers

All helpers use Python 3.12+ standard-library modules. No training library or GPU
environment is needed. They do not provision resources or run an experiment.

## Completion and cost estimate

Create a calibration JSON from measured timings. Example keys (use actual
measurements, not placeholder values):

```json
{
  "optimizer_step_seconds": [1.0, 1.1, 1.2],
  "full_validation_seconds": [10.0, 11.0],
  "provision_seconds": 0,
  "setup_seconds": 20,
  "diagnostics_seconds": 0,
  "checkpoint_seconds": 5,
  "transfer_seconds": 0
}
```

The numbers above illustrate the format only; they are not Pythia calibration.
`checkpoint_seconds` covers all planned saves; `transfer_seconds` includes local
hash verification. Omitted fixed phases are zero; explicitly measure material
phases. Timing samples may be omitted for a phase with zero planned count.

```text
python tools/estimate_etc.py CALIBRATION.json --steps 100 --validation-passes 2
```

For a cost envelope, add `--gpu-count`, `--hourly-gpu-price`, `--maximum-hours`,
and `--storage-cost` using current price and total bounded non-compute costs.
The p90 scenario is not a probabilistic completion bound. See `research/COMPUTE.md`.

## Read-only monitoring

```text
python tools/watch_run.py --manifest MANIFEST.json --events EVENTS.jsonl --max-polls 1
```

For repeated checks, set `--interval` and `--max-polls`. Omit max-polls only for
an intentionally ongoing monitor. Writers replace manifests atomically and end
each JSONL event with a newline; an unfinished last event is ignored. Completed,
failed, and cancelled states terminate polling. Failed/cancelled CLI runs exit 1.
The helper prints recorded status; scientific validity and stale-event checks
remain the owning run's responsibility.

## Transfer inventory

Choose an artifact root containing only agreed transfer files, then:

```text
python tools/artifact_inventory.py create ARTIFACT_ROOT --output INVENTORY.json
python tools/artifact_inventory.py verify LOCAL_ROOT INVENTORY.json --strict
```

Verification checks file sizes and SHA-256 and exits nonzero on mismatch. Strict
mode also rejects extra files except the inventory itself. Symlinks/junctions
are rejected. Preserve the inventory with the received files; put subsequent
verification receipts/analyses outside the immutable transfer root. Stop writers
before creating a final inventory; it is not a live snapshot tool.
