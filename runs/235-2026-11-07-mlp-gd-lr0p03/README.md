# Run235: intermediate GD learning rate0.03

H040 / Analysis060. User requested LR0.03 after observing Run234. This is a
new scientific branch from original untreated Run223 Adam state512 through800,
not from any earlier GD endpoint. Compare with retained GD0.02 and0.04 and
Adam0.001. Same656810-parameter MLP, seed0, pinned5000-image CIFAR-10 full MSE,
physical batches1000, FP32 weights/raw gradients, FP64 diagnostic reductions,
data order and RNG. Native SGD at0.03, momentum/decay zero, no clipping/scaling.
Within-GD comparison varies LR only; Adam comparisons change optimizer and LR.

Reuse the exact frozen shared GD helper from Run234; record its SHA along with
all parent code/cache/checkpoints. Exact fork-gradient and two-update Adam
model/optimizer/RNG/gradient replay qualify reuse of the retained full control.
GD2 versus1+1 resume must match exactly before288 scientific updates. Record
all289 boundaries: loss, raw norm, adjacent raw-gradient rho and projected change
<g_previous,g_previous-g_current>/||g_previous||^2. Zero denominator yields null.
Every outgoing update records LR, actual FP32 increment geometry and residual
from -LR*g. Preserve first-step513, recovery every32, final800, optimizer/RNG
and gradient history. Full10000-image test objective at800 is secondary.

Primary mean513..800, last32 mean, final/max loss and fixed pre-rise >10%
excursions with censoring match previous branches. The LR was selected after
viewing earlier results: this is an exploratory same-seed comparison, without
independent uncertainty or generalization/convergence claims.

Owned Spark only, expected under60s (similar branches about3s);600s worker
limit with detached630s timeout plus30s kill grace. Persist events/log/manifest;
stop on nonfinite loss/gradient, preserve partial evidence. Expected artifacts
below0.2GB retained locally/remotely after size/SHA verification. Zero cloud
cost; no unrelated process changes. Analysis060 updates to columns0.02/0.03/0.04,
while preserving0.08 divergence and its earlier figure in the archive.

## Completed and verified

Completed7 November at13:09:54UTC:288 scientific updates,2.855s main branch
and3.838s worker including6 qualification updates. Exact control and GD resume
pass. All42 input identities and39 artifact files plus archive pass size/SHA
checks. First-step rounding below7.411e-9; full endpoint gradient and10000-image
test coverage verified. Worker exited, zero cloud cost, recovery evidence
retained locally and on Spark. See `completion.json` for storage.

Mean loss0.0439842, final0.0409154, maximum0.302456; mean/tail training loss
111.42%/191.15% above Adam. One unrecovered >10% excursion and125 increasing
updates, despite every step having a negative gradient dot product. This
distinguishes local descent direction from finite-step loss decrease.
[Analysis060 Observation003](../../analyses/060-2026-11-07-mlp-adam-to-gd/observations/003-intermediate-gd-rate.md)
contains the three-column figure, full comparisons, caption and limitations.
