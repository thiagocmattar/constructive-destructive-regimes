# Analysis033: completed exact long-lag gradient profiles

[Report](REPORT.md), [primary curves](figures/long_lag_profiles.pdf),
[common range](figures/common_lag256.pdf), [odd/even gap](figures/alternating_gap.pdf).

[Largest learning rate, all lags 1–256 connected](figures/largest_lr_lag256.pdf)
shows one curve per architecture/anchor without an odd/even split. See
[Observation002](observations/002-largest-lr-lag256.md) for sources and caption.
The matching [1–64 lag zoom](figures/largest_lr_lag64.pdf) is documented in
[Observation003](observations/003-largest-lr-lag64.md).

Follow-up to [Analysis024](../024-2026-10-17-eos-gd-lr-calibration/README.md)
and [Analysis032](../032-2026-11-01-eos-exact-lag10/REPORT.md): 27 anchored profiles at states 500/2000/3000,
all through lag 256 and up to 2048. Exact streaming products
required no compression. There are 25 primary profiles and two previously
unqualified Run062 starts. Of 17 primary profiles with negative original
lag 1, 4 have terminal local half-gap <=0.1 and 13 remain above it.
Damping can rebound; a crossing does not establish permanent decorrelation.

Runs 206/207/208 own the frozen continuations, parallel placement and qualified
checkpoint handoffs. Canonical profiles by source run: {'206': 20, '207': 3, '208': 4}.
All 20 task Pods were deleted after verified retrieval; no task cloud
storage remains. Compute estimate USD 29.48 is provisional,
within the approved USD 30 envelope. Allocation to last deletion took 197.2 minutes.
Spark and unrelated account resources were preserved.

[Every value](lag_profiles.csv), [results](results.json), [summary](summary.json),
[observation/captions](observations/001-long-lag-profiles.md),
[verification](verification.json), [final review](final_review.json),
[closeout](closeout.json). Numbered scripts preserve preparation, orchestration,
reduction, documentation, reviews and checkpoint captures. Do not rerun completed
allocation; changed scientific inputs need a new run ID.
