# Observation 001: 31M seed 1236 geometry

Status: completed, retrieved, independently verified; owned worker and volume deleted.

## Sources and reduction

[Retrieval receipt](../prelaunch/attempt1-retrieval.json) verifies the complete archive,
55-file inventory, initial tensor hash, independent CPU geometry reconstruction, and
raw gradient norm versus the separate clipping routine. Frozen configuration and source
copies match. Raw evidence lives in `artifacts/attempt-01/full-pass/` (ignored by Git).
Archive SHA-256: `989fa07180b692df94f3f6576e4624725bc36eacc7574efc3f45306aee0b9038`.
Input-event SHA-256: `858d6b8c0de642892435ae165f3cf0441522735f73c28f6417bdbcafef54318e`.

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
| raw | -0.121242 | 314.287822 | 99.0711876 | 0.315224 | 0.403508475 | 459/711 |
| adaptive | 0.953217 | 1207653.75 | 801456.021 | 0.663647 | 1625.99216 | 0/711 |

Median paired ||d||/||g||: 4080.573208.
Final token-weighted validation loss: 4.522797; all 500 documents and tail included.
Overflow retries: 0.

## Coverage, limits and cleanup

Evolution plots omit only final partial update712 (311 sequences), as requested. Full
statistics and raw evidence retain it; Analysis002 also records update711 sensitivity.
A single trajectory supplies no independent seed uncertainty. Cross-seed figures use
three trajectories per size, mean/sample SD or seed range, never independent steps.
Initialization and order seeds vary together; order is matched across sizes within seed.
Absolute norms depend on architecture/dimension. Alignment is not a stability, causal
learning benefit or parameter-displacement measure.

[Closeout](../prelaunch/closeout.json): estimated $1.55 including running storage, provisional until billing settles; zero retained task cloud storage.
