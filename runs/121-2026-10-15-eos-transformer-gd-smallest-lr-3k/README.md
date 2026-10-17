# Run121: transformer, smallest-LR 3k calibration

Full-batch GD, LR0.05, seed0, 3,000 fixed updates on owned Spark.
[Analysis023 protocol](../../analyses/023-2026-10-15-eos-smallest-lr-calibration/README.md) defines scope,
diagnostics, retention, priority scheduling and limitations. No cloud rental.
New horizon uses a new ID; old executed configs remain unchanged.

Completed October17 at01:05 Sao Paulo time; all3,000 updates are retained.
The32-file inventory (3,017,412,264bytes) was retrieved and SHA256 verified;
[complete results](../../analyses/023-2026-10-15-eos-smallest-lr-calibration/observations/002-three-case-3k.md)
report final train/test loss6.630097/6.365134. No subsequent training started.
