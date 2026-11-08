# Distributed local caps at larger MLP learning rates

## Conclusions first

**The distributed cap makes abrupt 2x/4x LR increases much less disruptive, but
this sweep does not show an optimization gain from increasing LR under the same
fixed cap budget.** Every higher-LR cap beats uncapped Adam at that LR in each
of the three initializations. However, for every quantile, the across-seed mean
training loss is higher at 2x/4x than at 1x. This is tolerance of a larger
nominal LR, not evidence of accelerated learning.

- **q25 tolerates higher LR best.** Relative to same-LR Adam, mean training loss
  falls 87.88% at 2x and 94.75% at 4x; maxima fall 98.66% and 99.50%. All nine q25
  branches have their maximum at the common untreated state 448. Relative to
  q25 at 1x, mean loss is instead 1.36% +/- 2.49% and 1.53% +/- 1.63% higher
  (paired mean +/- sample SD across three seeds).
- **q50 at 1x has the lowest observed mean training loss**, 0.01333 +/- 0.00095,
  versus 0.01774 +/- 0.00036 for original Adam. Its paired mean reduction is 24.79%,
  and positive loss variation falls 80.66%. The small mean difference from q25
  is descriptive; three seeds do not establish an optimal quantile.
- **Smaller peaks do not imply fewer oscillations.** q25 increases positive
  loss variation by 34.25%/47.49%/46.58% at 1x/2x/4x relative to the original
  baseline, despite lower maxima and mean loss. q50 at 1x still has an average
  672 negative output-change cosines among 705 updates.
- **Test performance remains a tradeoff.** The final-four test mean rises 3.37%
  for q50 at 1x and 4.52% for q25 at 4x versus original Adam. Looser high-LR caps
  can lower test loss while worsening training loss. Uncapped 4x has the lowest
  final-four test mean in this grid, despite its poor training trajectory. No
  generalization superiority or test-selected recommendation is claimed.

Percent changes above are paired within initialization before averaging.
The same-cap LR comparison and all per-seed values are retained in the CSVs.

## Mean training loss across the full window

Mean +/- sample SD over three initializations; 705 states per continuation.

| Quantile | LR 0.001 (1x) | LR 0.002 (2x) | LR 0.004 (4x) |
| --- | --- | --- | --- |
| Uncapped | 0.01774 +/- 0.00036 | 0.11393 +/- 0.00707 | 0.26281 +/- 0.01898 |
| q25 | 0.01353 +/- 0.00091 | 0.01372 +/- 0.00109 | 0.01374 +/- 0.00099 |
| q50 | 0.01333 +/- 0.00095 | 0.01812 +/- 0.00235 | 0.01832 +/- 0.00260 |
| q75 | 0.01562 +/- 0.00182 | 0.02095 +/- 0.00156 | 0.02532 +/- 0.00488 |
| q90 | 0.01648 +/- 0.00073 | 0.02316 +/- 0.00126 | 0.02713 +/- 0.00164 |
| q99 | 0.01726 +/- 0.00009 | 0.02374 +/- 0.00174 | 0.03186 +/- 0.00105 |

[Full readable table](../cap_lr_statistics.md), [aggregate CSV](../cap_lr_aggregate.csv),
[per-seed CSV](../cap_lr_per_seed.csv), and [manuscript table source](../cap_lr_table.tex)
include late loss, peaks, positive variation, held-out results, and all paired
comparisons. These are descriptive statistics, not population confidence bounds.

The applied-step measurements support the saturation interpretation: q25 caps
72.9%/94.1%/99.0% of hidden units on average at 1x/2x/4x. The global retained-norm
ratio falls 0.469/0.233/0.117, while the mean actual increment norm changes only
0.01997/0.02165/0.02287. Thus 4x nominal LR is not 4x applied movement. This is a
mechanistic interpretation consistent with the cap formula, not proof that all
remaining loss fluctuations originate in the uncapped output layer.

## Scientific question and comparison

H049 tests whether the existing distributed local displacement cap supports
larger Adam learning rates. Runs 286/287/288 use pinned initializations 0/1/2.
Each run contains the complete factorial comparison: q25/q50/q75/q90/q99
crossed with LR 0.001/0.002/0.004, plus uncapped Adam at every LR. All 54
continuations were specified before inspecting outcomes. No additional policy,
architecture, seed selection or test-based early stopping is introduced.

Each seed shares its untreated state 448, including Adam moments and clocks.
The intervention begins with update 449. The original baseline remains uncapped
Adam at 0.001; uncapped 2x/4x isolate the cap at the same LR, and the corresponding
1x cap isolates the LR increase under a fixed local movement budget. The
[prespecified design](012-distributed-cap-lr-plan.md) records the choices before
launch. This extends Analysis 065 without allocating a new analysis.

## Method and coverage

The network is a 3072-200-tanh-200-tanh-10 MLP (656,810 parameters) on the same
ordered 5,000 CIFAR-10 training examples. The objective is half squared error,
summed over ten output coordinates and averaged over all examples; physical
chunks of 1,000 exactly accumulate the full objective. The complete training
set is also the fixed diagnostic probe. Adam uses betas (0.9,0.999), epsilon 1e-8,
no weight decay or gradient clipping, and FP32 parameters with TF32 disabled.
Data, initialization and implementation identities are frozen in each run's
input manifest. Only global LR and the calibration quantile vary.

For hidden affine layer l, capture its current input matrix H with samples
as rows. With the native Adam parameter proposal, compute

\[
 \Delta Z^{l,\mathrm{local}}_t
 =H^{l-1}_{t-1}(\Delta W^{l,\mathrm{Adam}}_t)^\top
  +\mathbf{1}(\Delta b^{l,\mathrm{Adam}}_t)^\top,
 \qquad
 s^l_{t,j}=\sqrt{\frac1N\sum_i(\Delta Z^{l,\mathrm{local}}_{t,ij})^2}.
\]

For each module, calibrate one threshold tau at the requested quantile of the
8 x 200 unit scores from untreated updates 441-448, at LR 0.001. Apply the factor
min(1,tau/s) to each proposed weight row and its corresponding bias; a zero
score retains factor one. Both hidden affine layers are targeted. The output
affine is uncapped. The thresholds are held **identical across LRs within each
seed**. Moments and clocks advance normally, without modification by the cap.

This controls the local preactivation response to each layer's own parameter
change, with its input held fixed. It does not bound total propagated activation
movement, the output-layer proposal, or the loss. In particular, increasing the
LR of the uncapped output layer remains part of this global-LR ablation.

Loss states 448-1152 supply 705 consecutive measurements. One additional state
1153 supplies the outgoing representation change at the last plotted state.
At horizontal coordinate s, gradient measurements describe g_(s+1) at theta_s;
the representation displacement is F_(s+1)-F_s. Raw-gradient alignment and
eta-kappa-hat use consecutive raw gradients, not their differences or the
applied increments. The displayed eta-kappa-hat is the normalized projected
gradient-change diagnostic 1-<g_(s+1),g_s>/||g_s||^2; under Adam it is not a
validated Hessian curvature or stability threshold.

All 10,000 test examples are evaluated at 448,480,...,1152 (23 complete
evaluations). These secondary measurements never alter training or select
conditions. No separate validation set is available. Test curves connect these
sampled values; they cannot resolve intervening excursions.

## Statistics and uncertainty

For each seed and condition, report mean training loss, last 32-state mean,
terminal loss, maximum loss, and positive variation sum(max(L_(s+1)-L_s,0)).
Positive variation counts all upward changes and can grow because of many
small oscillations even when the largest excursion decreases. Also report
time-sampled test mean, last-four-evaluation mean (1056/1088/1120/1152), and
terminal test loss; these are different summaries.
The maximum includes the common untreated state 448; a branch's maximum can
therefore occur before its first intervention update.

Across seeds, report arithmetic mean and sample standard deviation. Relative
changes are computed within each seed against its own reference before
averaging. Three related initializations provide descriptive replication,
not a precise population confidence interval. All conditions remain visible;
this is a continuation study at one fork/window, not training from initialization,
a held-out hyperparameter-selection exercise, or a cross-architecture result.
The LR change is abrupt and retains the previously accumulated Adam history;
it does not test a gradual LR ramp or a larger constant LR from initialization.

For interpretation, consider a fully capped unit at a fixed state and Adam
direction, ignoring FP32 storage rounding. Write its proposed increment as
minus eta times d, and its local score as eta times a. The cap factor is
tau/(eta times a), so the applied increment is minus tau times d/a: the nominal
LR cancels. Uncapped units and the output layer still respond to LR, and later
states/directions need not match. Consequently, absorbing a larger nominal LR
does not by itself establish larger useful representation steps or faster
learning. The retained-norm and fraction-capped measurements help interpret this
distinction; they do not prove a unique cause of any outcome.

## Figures and local sources

Figure 12, `12-distributed-cap-quantile-lr-mlp.pdf`: three columns for LR1x/2x/4x;
full training and test losses in two rows. Each trace is the arithmetic mean
of three seeds. All five cap quantiles, same-LR uncapped Adam and the original
baseline appear. Loss axes are logarithmic and independent between columns.
Averaging can hide seed-specific excursions; Figure 13 retains them.

Figure 13, `13-distributed-cap-lr-seed-diagnostics.pdf`: one page per initialization,
three LR columns. Rows show training loss, eta-kappa-hat, raw-gradient cosine,
raw-gradient norm, outgoing output coordinate RMS, retained parameter-step norm
ratio, and fraction of hidden units capped. All 705 states are plotted, without
smoothing or time decimation. Eta-kappa-hat uses a linear axis. Loss, gradient
norm and output displacement use log axes. Parameter retention and the fraction
capped describe different aspects of the intervention.
Here gamma is the norm of the actual stored parameter increment divided by the
norm of the stored native Adam proposal, over all parameters.

The generating script is `40_plot_cap_lr.py`; `39_reduce_cap_lr.py` produces
`cap_lr_curves.npz`, `cap_lr_summary.json`, `cap_lr_per_seed.csv`,
`cap_lr_aggregate.csv` and the readable `cap_lr_statistics.md`. All are in this
analysis folder. Raw boundaries, proposed/actual per-unit scores, cap factors,
complete evaluations, calibration, recovery states and retained audits belong
to Runs 286-288 under `artifacts/attempt-01/worker/` (ignored by Git).

`37_audit_cap_lr.py` replays native Adam and parameter triplets on the original
GPU, independently checks local affine responses and representation statistics
with NumPy, and repeats complete held-out evaluations at retained audit states.
`38_collect_cap_lr.py` retrieves artifacts with archive and per-file size/SHA-256
verification. The reducer independently checks every recorded cap bound,
quantile, LR, clock, coverage and scalar statistic, plus 162 saved update audits.
It also checks exact reproduction of the six historical 1x control/q50 trajectories.

Local figure reproduction uses the verified compact data:

```powershell
python analyses/065-2026-11-07-mlp-frozen-lr-double/40_plot_cap_lr.py
```

For a full numerical reduction beside the retained Spark checkpoints and a new
verified compact export, run `42_export_cap_lr.py` from this analysis folder
with the existing authenticated Spark connection. It invokes `39_reduce_cap_lr.py`
in the retained remote workspace. A complete local raw-checkpoint copy is no
longer assumed; local scalar histories and reduced curves remain available.

## Completed execution and verification

All 54 continuations and 38,070 updates completed successfully on owned Spark in
2060.24 s (34.34 min), within the 90-minute limit. Peak allocated GPU memory was
593,003,520 bytes. Original-hardware replay took 28.76 s; the full independent
numerical reduction took 19.62 s. All 162 retained audits, six exact historical
trajectory reproductions, dense cap/LR/clock/coverage checks, 310,660 plotted
values and four rendered PDF pages pass. No cloud resources were created;
cloud cost is USD 0. Task training and replay processes have exited.

A full local retrieval hit the Windows disk limit. After rechecking remote
sizes and SHA-256 identities, only redundant local binary copies were removed.
The numerical reduction then ran beside the complete Spark evidence; 330 compact
exports (26.85 MB transfer archive) passed archive and individual-file identity
checks. This infrastructure adjustment did not change scientific inputs,
training, coverage or outcomes. Monitoring uses IPv4 after an IPv6 banner timeout.

Complete raw evidence and recovery states remain on owned Spark: 7.978 GB including
mechanics and inventories. Each run has a `spark_retained_inventory.json` with
exact remote paths, sizes and hashes. Verified compact data, boundary histories,
held-out measurements and audit reports are local; complete binary checkpoints
and per-unit streams remain available on Spark. No task cloud storage exists.
Transport archives were removed after verification. All eleven preceding figures
and the existing technical report remain byte-identical.

Receipts: [reduction](../cap_lr_summary.json), [compact transfer](../cap_lr_compact_retrieval.json),
[storage adjustment](../cap_lr_local_storage_adjustment.json), [compute/process closeout](../cap_lr_compute_closeout.json),
[plot values](../cap_lr_plot_verification.json), and [visual review](../cap_lr_visual_review.json).
