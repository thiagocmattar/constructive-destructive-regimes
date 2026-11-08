# H049: MLP distributed-cap quantile and learning-rate sweep

## Question and fixed comparison

User hypothesis: limiting representation movement may permit a larger Adam
learning rate. Test the existing **distributed local preactivation cap**, changing
only its calibration quantile and the global LR at intervention start. The cap
targets both hidden affine layers, including biases; the output affine remains
uncapped. A local cap is not a bound on total propagated activation movement.

Cross q25/q50/q75/q90/q99 with LR 0.001/0.002/0.004 (1x/2x/4x). Include uncapped
Adam at each LR. Every condition compares with unchanged Adam at 0.001; same-LR
uncapped comparisons isolate the cap, and capped 1x comparisons isolate increasing
LR under the same budget. Use the three existing pinned initializations (0/1/2),
with no seed-specific tuning: 18 conditions per seed, 54 continuations total.
Runs 286/287/288 each own one seed's prespecified factorial comparison.

All arms start from that seed's untreated state 448 and retain moments/clocks.
The first changed update is 449. Continue through loss state 1152, with one
additional outgoing step/state for representation diagnostics. Use all 5,000
training images and the same ordered 5,000-image probe, half-squared error summed
over ten outputs and averaged over examples, physical chunks of 1,000, the
3072-200-tanh-200-tanh-10 MLP, FP32/TF32 off, Adam betas (0.9,0.999), epsilon 1e-8,
no decay or raw-gradient clipping. Data/preprocessing and initializations are
hash-pinned to the existing records.

## Calibration and interpretation

For each seed/module, pool all 200 unit scores across the eight untreated outgoing
updates 440-447 (manuscript update indices 441-448). Compute all five empirical
quantiles with linear interpolation. **Keep these absolute thresholds fixed across
LRs**, calibrated at the original 0.001 LR. Scaling the budgets by LR would change
the question. The native proposal uses the selected global LR; cap each unit's
weight row and bias by min(1, threshold/proposed local RMS). All gradients,
moments and clocks continue normally. Zero score means factor one; zero threshold
disables that module. Verify final FP32 rounding using the existing bound.

Support requires reduced excursions and improved training progress at elevated
LR versus its uncapped counterpart, with comparison to the 1x cap and original
baseline. Lower peaks alone do not show improved learning. Report failures and
tradeoffs for all conditions; do not select the displayed quantiles after seeing
results. This is continuation evidence on three related initializations, not a
from-initialization or architecture-general claim.

## Measurements, retention and statistics

Record each state's full training loss, raw-gradient norm/cosine and
eta-kappa-hat (linear plotting scale), per-layer/output consecutive representation
cosines and coordinate RMS. Record every actual/native step norm, LR, Adam clock,
per-unit factors/scores, fraction modified and cap residual. Cosines with zero
denominator are undefined. Plot loss on log axes; preserve completed-state versus
outgoing-update indexing from the report.

Evaluate all 10,000 test examples at states 448,480,...,1152 (23 evaluations),
without affecting gradients, RNG or training. Held-out curves and statistics are
secondary descriptive evidence; no tuning or early stopping uses them. There is
no independent validation set in this cache. Mean training loss, tail-32 loss,
maximum and total positive loss variation are primary summaries. Add time-sampled
held-out mean, final-four-evaluation mean and terminal test loss. Report per-seed
values, across-seed mean/sample SD, and paired percent changes versus original and
same-LR controls. Three seeds do not support a precise population confidence interval.

Retain shared prefix/calibration, full model/optimizer/RNG recovery and final
states, scalar histories, three gradient/update/optimizer audits per arm, resolved
config/source identity and transfer inventory. No dense SVD tensor history is
needed. Recheck original-LR control and q50 trajectories against H048 for reuse
qualification; rerunning them also supplies the newly requested held-out trajectories.

## Execution envelope

Owned DGX Spark (GB10), checked idle before launch; no billable cloud resources.
Use the existing isolated CUDA runtime and authenticated SSH host key. The `.local`
name did not resolve; the short hostname resolves and matches that same host key.
Provisioning cost: USD 0. Planning range before the new smoke: 20-50 minutes of
training, plus staging/retrieval/analysis; refine from the production-shaped smoke.
Run seeds sequentially with a 90-minute task timeout and 15-minute per-arm limit.
Allow 12 GB raw artifacts on Spark and locally; preserve checkpoint/evidence files.
Detached processes, persistent logs/progress and read-only monitoring every 30-60 s
survive a disconnected client. Stop only task-owned processes, verify retrieval
sizes and SHA-256, and report final usage. The earlier USD20 cloud round is closed.
