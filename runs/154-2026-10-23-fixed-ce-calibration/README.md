# Run154: H022 mechanics and timing calibration

User approved the Analysis026 protocol and execution on October23.
Seed1000, three loss arms,200 updates each; full-shape model/data and
ordinary/Adam-scaled spectra. Mechanics only, excluded from inference.
The numbered worker and shared scientific core are frozen before execution.
See Analysis026/PROTOCOL.md for measurement, price and retention contracts.

## Completed calibration

Both local attempts are retained: `artifacts/local-attempt-01/` is the initial
mechanics/timing qualification; `artifacts/attempt-01/` is the frozen queue's
admission calibration. Each covers three200-update conditions on seed1000,
with initial/final ordinary and post-training Adam-scaled spectra. Their1,200
total mechanics updates are excluded from all scientific inference.

The approved owned RTX5070Ti Laptop GPU was used. The admission estimate,
including30% allowance and ten-minute closeout, was about73minutes; production
completed within that allowance. No rented resource was created. Initial state,
optimizer/RNG/diagnostic recovery, exact executed source and scalar evidence are
retained locally; the final artifact inventory in Analysis026 includes both
attempts. No calibration claim is presented as paper evidence.
