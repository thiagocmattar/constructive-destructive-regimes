# Analysis060: warm Adam-to-GD switch

Run233 / H040. Compare plain GD at LR0.02 with continued Adam at LR0.001,
from the identical Run223 MLP state512 through800. Shared original5000-image
CIFAR-10 objective, seed0 and full-gradient partitions. Both optimizer and LR
change; no optimizer-only attribution. User requested this after the beta2
interventions; all branches remain separate from the same untreated parent.

Primary mean513..800, last32, final/max loss and recovered/censored >10%
excursions; endpoint full10000-image test loss secondary. No independent-seed
uncertainty or universal stability claim.

Completed: all288 GD updates decrease training loss, with no>10% excursion.
Mean loss is2.30% above continued Adam and last32 mean55.74% above; maximum
loss is59.92% lower. Exact retained-control replay and GD resume pass.

- [Findings, caption and verification](observations/001-adam-to-gd.md)
- [Paired comparison figure](figures/01-adam-to-gd-cifar10-mlp.pdf)
- [Numerical results and provenance](results.json)

Reproduce with `01_report.py`. All3528 plotted scalars and the vector render
pass checks. Main treatment2.78s, worker3.73s, zero cloud cost. Run233 retains
first-step/recovery/final evidence and39 size/hash-verified artifacts.
