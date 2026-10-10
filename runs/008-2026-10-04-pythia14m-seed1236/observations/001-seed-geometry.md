# Observation 001: 14M seed 1236 geometry

Status: completed, retrieved, independently verified; owned worker and volume deleted.

## Sources and reduction

[Retrieval receipt](../prelaunch/attempt1-retrieval.json) verifies the complete archive,
55-file inventory, initial tensor hash, independent CPU geometry reconstruction, and
raw gradient norm versus the separate clipping routine. Frozen configuration and source
copies match. Raw evidence lives in `artifacts/attempt-01/full-pass/` (ignored by Git).
Archive SHA-256: `7c2c43c22e540c503cd686934b7e54d723c8d6a0da2ec91c20e810f937529c5a`.
Input-event SHA-256: `0236fda49f7eee499ebedefb4f6d0bac61b9f27e9627d0d14080e3d33218246d`.

[Analysis002](../../../analyses/002-2026-10-04-scale-seed-replication/observations/001-seed-replication.md)
owns the cross-run reduction, uncertainty, publication figures and captions.
Generating scripts: Analysis002 `01_compare.py`, `02_record.py`, and `03_closeout.py`.

## Measurements

All 712 optimizer updates and 1,491,711,416 input tokens are retained. Raw g is measured
before clipping; adaptive d uses bias-corrected actual FP32 moments of clipped gradients
and excludes LR/weight decay. A=sum||x||, B=||sum x||, R=B/A. First cosine is undefined.
Median cosine uses 711 adjacent pairs; other full-pass summaries include update712.

| Vector | Median adjacent cosine | A | B | B/A | Median norm | Negative pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.013478 | 322.305748 | 107.157112 | 0.332470 | 0.409915921 | 348/711 |
| adaptive | 0.966163 | 946363.036 | 664052.59 | 0.701689 | 1246.35636 | 0/711 |

Median paired ||d||/||g||: 3149.895725.
Final token-weighted validation loss: 5.164867; all 500 documents and tail included.
Overflow retries: 0.

## Coverage, limits and cleanup

Evolution plots omit only final partial update712 (311 sequences), as requested. Full
statistics and raw evidence retain it; Analysis002 also records update711 sensitivity.
A single trajectory supplies no independent seed uncertainty. Cross-seed figures use
three trajectories per size, mean/sample SD or seed range, never independent steps.
Initialization and order seeds vary together; order is matched across sizes within seed.
Absolute norms depend on architecture/dimension. Alignment is not a stability, causal
learning benefit or parameter-displacement measure.

[Closeout](../prelaunch/closeout.json): estimated $1.58 including running storage, provisional until billing settles; zero retained task cloud storage.
Hardware status queries returned NVML Unknown Error after completed training.
Packaging omitted that redundant query; captured training/calibration hardware
records and all scientific artifacts passed unchanged acceptance checks.
