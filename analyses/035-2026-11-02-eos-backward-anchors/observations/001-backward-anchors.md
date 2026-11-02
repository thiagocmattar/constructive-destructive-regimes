# Observation001: fixed endpoint raw gradients compared with their past

Question: display backward anchored cosines through lag 256, separately for GD
and Adam, using the largest GD-rate source histories and existing calibrated
Adam rates. The user chose endpoints reached along the Adam continuations,
so every earlier gradient in the Adam window belongs to that same Adam fork.

Sources: Run212 frozen input manifest/config and complete per-arm results,
scalar histories, CPU/GPU product records, source qualifications, gradient
hashes and six full audit vectors. Inputs are the identical Run205 starts used
by Analysis033/034; the old forward/adjacent scalar curves are not substituted.
Numerical audits are in `verification.json`; figure/source identities and
actual-PDF visual checks are in `figure_review.json`.

Reduction: for each retained start s=500/2000/3000, run exactly 256 updates and
fix t=s+256. For k=1..256 compare g_t with g_(t-k). One pair per lag, all trainable
parameters once in model order, complete original objective, FP32 raw gradients
and FP64 dot/norm reductions. No epsilon; a zero denominator is undefined.
No averaging across endpoints, centering, smoothing, parity split, projection,
envelope or downsampling. Exact scalar arrays are checked independently against
the CSV and each plotted artist. Each grid has 2304 values and shared x=1..256,
y=-1.05..1.05. The plotted endpoint gradient is fixed throughout every panel.

Caption, `figures/gd_backward_lag256.pdf`: Backward anchored whole-model
raw-gradient cosine cos(g_t,g_(t-k)) along GD. Columns: CIFAR-10 MLP (GD LR 0.04,
source Run122), CIFAR-10 ResNet-32 (0.1, Run120), WikiText-2 Transformer (0.2,
Run126). Rows: endpoint t=756/2256/3256 reached from saved starts 500/2000/3000.
Every lag1–256 is connected in order as one curve, with a horizontal zero line.
Increasing lag walks backward toward the retained start.

Caption, `figures/adam_backward_lag256.pdf`: The same backward anchored statistic
along cold Adam from the identical nine GD-trained source states. Adam LRs:
MLP 3e-4, ResNet-32/Transformer 1e-3; both Adam and source GD rates appear in the
column labels. Same endpoints, 256-update budget, axes, styling and coverage as
the GD figure. Equal absolute endpoint indices do not mean equal GD/Adam weights.

Coverage and uncertainty: 18 profiles from nine directly saved primary states,
one seed-0 initialization per architecture, 256 anchored pairs per profile,
no seed uncertainty bands. Full source data/order/objectives are unchanged;
Transformer chunk 512 passes the original64-chunk qualification. Hardware and
FP32 reductions change later trajectories despite matched initial tensors;
`historical_comparison.json` reports substantial differences in some cases.
These are newly recomputed continuations, not identical historical trajectories.
The complete temporary gradient histories are discarded after exact products
and audit-vector extraction; all scalar measurements and gradient hashes,
selected full vectors and optimizer/RNG recovery remain locally verified.
Losses are training only. A cosine curve alone does not establish stability,
generalization, optimizer benefit or asymptotic damping. Backward anchored
cosines, historical forward anchored cosines and moving adjacent cosines answer
different questions and must keep their labels.

Generator: `03_analyze.py`. Publication outputs are the two vector PDFs;
actual-PDF previews are ignored `scratch/gd_backward_lag256.png` and
`scratch/adam_backward_lag256.png`. The report tabulates endpoint lag-one
cosines and losses without ranking optimizers.
