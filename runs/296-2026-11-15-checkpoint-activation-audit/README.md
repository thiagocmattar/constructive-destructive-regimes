# Saved-state activation audit

**Completed and verified:** 21 full-data passes and 48 reference-state measurements across 48 measured sites. No optimizer updates. Capture took 332.7 seconds in total on Spark; source-qualified losses agree within 3.78e-6 relative error. All 150 downloaded outputs verified against remote sizes and SHA-256 hashes.

- Analysis 066 / H052; user authorized a new activation-distribution investigation on owned notebook/Spark hardware, with no Runpod.
- Frozen [configuration](config.json), [worker](01_measure.py), [synthetic checks](02_test.py).
- Actual model states come from retained matched-intervention Runs 293-295. No training updates or intervention changes; new diagnostic capture at selected states.
- [Single analysis entry point](../../analyses/066-2026-11-15-activation-inactivity/README.md) contains measurement definitions, coverage, and results.
- All source hashes, mode/observer qualification, progress, raw arrays, and retrieval hashes remain with the analysis. Large matrices and checkpoints are excluded from Git.
