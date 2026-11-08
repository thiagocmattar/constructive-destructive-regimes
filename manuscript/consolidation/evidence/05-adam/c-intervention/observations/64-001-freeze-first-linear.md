> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [001-freeze-first-linear.md](../../../../../../analyses/064-2026-11-07-mlp-freeze-first-linear/observations/001-freeze-first-linear.md).

# Freezing the first projection strongly suppresses downstream oscillation

## Conclusion

The intervention supports a causal contribution of first-layer updates to this
particular MLP oscillation. Holding the first linear weights and biases fixed
from state 448 strongly reduces reversals and displacement in the **still-trained
second hidden layer and output**, without changing their Adam learning rates.
The control peak at 474 is suppressed. This is stronger evidence than the prior
descriptive layer trace, but does not establish a unique first-layer initiating
mechanism or literal one-way propagation: all layer gradients are coupled.

Stability and optimization progress separate. The frozen run has smaller spikes
and lower mean loss over 450-500, but **22.27% higher mean training loss over
448-800**. Freezing also prevents feature learning; smoother motion does not
uniformly improve training progress.
This block contains 93.57% of model parameters; there is no matched later-layer
freeze or equal-update-norm control here.

[Two-page comparison figure](../background/comparators/01-freeze-first-linear-cifar10-mlp.pdf).

## Matched intervention

- Source: the exactly qualified seed-0 MLP Adam control from Runs 223/239-242.
  Fork at state 448, before the dominant474 event. Run 244 continues through 801,
  which supplies the outgoing displacement at 800. Primary interval 450-500;
  extended follow-up 448-800. No stopping based on observed improvement.
- Same5000 CIFAR-10 images, fixed ordering, physical batch 1000, input normalization,
  tanh MLP 3072-200-200-10, mean per-image half squared error, and all raw outputs.
  The representation probe equals the full training objective's example set.
- Same Adam LR 0.001, betas (0.9,0.999), epsilon 1e-8, no clipping/decay, FP32 native
  arithmetic and TF32off. No optimizer reset, new initialization or data change.
- At each update, native Adam processes the original raw gradients; only the
  first linear weight/bias tensors are restored to their fork values before the
  next forward. Thus the **actual first-layer increment is exactly zero**, all
  other parameter updates remain native, and Adam moments/clocks advance normally.
  This is persistent parameter freezing with continuing diagnostic gradients;
  it is not a `requires_grad=False`/paused-moment implementation. No unfreeze occurs.
- Frozen parameters: 614600; updated parameters: 42210; total: 656810. Full raw
  gradient statistics include derivatives with respect to the frozen parameters.
  Updated-layer gradient statistics explicitly restrict to the same 42210 later
  parameters in both arms. Neither stream is an actual Adam update statistic.

## Results

| Quantity, 450-500 | Continue Adam | Freeze linear 1 | Change |
|---|---:|---:|---:|
| Loss at the control peak 474 | 0.0536730 | 0.0287550 | -46.43% |
| Maximum loss in window | 0.0536730 | 0.0312337 | -41.81% |
| Mean loss | 0.0330245 | 0.0289833 | -12.24% |
| Loss at 500 | 0.0287051 | 0.0283354 | -1.29% |
| Sum of positive loss changes | 0.0691958 | 0.00271393 | -96.08% |
| Largest upward loss change | 0.00930841 | 0.00136229 | -85.37% |
| Output-change negative cosines | 51/51 | 4/51 | Strongly suppressed |
| Maximum output displacement RMS | 0.145889 | 0.0184175 | -87.38% |
| Sum of squared output RMS | 0.271271 | 0.00261337 | -99.04% |

The frozen maximum is an early relaxation at 451, not a relocated spike near474.
Loss still has10 upward transitions among the 50 transitions within 450-500,
so the intervention does not make the immediate trajectory perfectly monotone.

| Quantity, 448-800 | Continue Adam | Freeze linear 1 |
|---|---:|---:|
| Mean training loss | 0.0228078 | 0.0278867 |
| Maximum loss | 0.0550626 (522) | 0.0322876 (448) |
| Loss at 800 | 0.0302502 | 0.0271296 |
| Positive loss variation | 0.364412 | 0.00271393 |
| Output-change negative cosines | 351/353 | 12/353 |
| Maximum output displacement RMS | 0.163076 | 0.0222974 |

The frozen loss decreases after its last upward transition into 493. Its last
negative output-change cosine is at 542; by then motion is already small.
The control repeatedly reaches lower losses between spikes, explaining its
better mean despite its worse endpoint at 800. The endpoint alone is therefore
not a reliable summary of learning progress.

## What changed inside the network?

Z1 and H1 are exactly constant after the fork. Their zero displacement is a
necessary consequence of freezing, not the substantive causal discovery.
Their motion cosine and normalized SVD energy fractions are **undefined**, not
zero-valued alignment or evidence of a special singular direction.

The useful result is downstream: over 450-500, Z2 and H2 reverse only 3/51 times
instead of 51/51, and output reverses4/51 rather than51/51. Their displacement
amplitudes fall sharply while their parameters continue changing under native
Adam. This supports first-layer motion as an important contributor to sustaining
the large oscillation in this continuation. It does not distinguish an intrinsic
first-layer instability from a feedback loop involving later layers and Adam state.

Low-dimensional concentration survives damping. Output top-one/top-ten energy is
60.58%/99.86% after freezing, versus 53.02%/99.48% in control. Thus a concentrated
SVD spectrum is not specific to the harmful oscillation: it can also describe
small, mostly forward motion. Leading output-coefficient sign changes decline
from 42 to 18 over 50 adjacent pairs, but component sign changes are not the same
as full-vector reversal and must be interpreted with their amplitude.

Each stream/arm has its own fixed uncentered SVD over 450-500. Modes span all
sample-feature coordinates; no sample averaging, centering or row normalization
is applied. These are fixed fitted patterns, not measurements that local axes
stay stationary. No local-window subspace rotation test is added here.

## Definitions, figure caption and provenance

State t means t completed optimizer updates. Loss is L_t; outgoing representation
change is delta_t=A_(t+1)-A_t, and rho_t=cos(delta_t,delta_(t-1)). Coordinate RMS
is ||delta_t||/sqrt(P), where P includes all 5000 sample rows and all features:
1,000,000 hidden/preactivation coordinates and 50000 output coordinates.
Raw-gradient alignment uses g_t itself, not gradient differences. The 51 outgoing
updates 450-500 require representation states 449-501; the 50 loss transitions
within 450-500 are a different count. Positive variation is
sum_{t=a+1}^b max(L_t-L_(t-1),0), with no spike-detector threshold.

The requested optimizer diagnostic is

$$\eta\widehat\kappa_t=1-\frac{\langle g_t,g_{t-1}\rangle}{\|g_{t-1}\|_2^2}
=1-\rho_t^g\frac{\|g_t\|_2}{\|g_{t-1}\|_2}.$$

Under Adam this is a projected raw-gradient-change diagnostic, not a measured
curvature eigenvalue or stability threshold. Both this quantity and the added
full gradient norm use all 656810 original parameters, including the frozen
block. The updated-layer norm remains a separate row. Both arms' values over
448-501 are independently recovered from their retained raw gradient vectors;
the full 448-800 histories use qualified native boundary records. No new training
or inference from a batch norm to per-example gradient coherence is involved.

**Figure 01, page 1:** full 448-800 and event 450-500 columns, each showing training
loss, eta kappa hat, full raw-gradient cosine, full raw-gradient norm,
updated-layer gradient norm, output-change cosine and output displacement RMS.
Gray shading identifies 450-500 in the full
window; the dotted line identifies control loss peak 474 in the zoom. Values are
unsmoothed. **Page 2:** all five learned representations, with consecutive-change
cosine, coordinate RMS and cumulative SVD energy. The fixed streams are labeled
undefined where appropriate; no artificial zero cosine or normalized spectrum.

Sources: [Run 244](../../../../../../runs/244-2026-11-07-mlp-freeze-first-linear/README.md),
its retained raw tensors/streaming diagnostics, and [Analysis 063](../../../../../../analyses/063-2026-11-07-architecture-instability-tracing/README.md)
control reductions. [01_reduce.py](../../../../../../analyses/064-2026-11-07-mlp-freeze-first-linear/01_reduce.py) verifies source identities,
builds exact FP64 event Grams and independently matches dense reductions to every
streamed event cosine/RMS. [02_figures.py](../../../../../../analyses/064-2026-11-07-mlp-freeze-first-linear/02_figures.py) generates the PDF from
[comparison.json](../../../../../../analyses/064-2026-11-07-mlp-freeze-first-linear/comparison.json) and [metrics.csv](../../../../../../analyses/064-2026-11-07-mlp-freeze-first-linear/metrics.csv).

## Verification, uncertainty and execution

All 54 input identities and 21 retrieved artifact identities pass. Prefix 0-448
matches 449 native loss/norm pairs; a short native qualification from 448 to 512
matches the full historical model, optimizer, RNG and gradients. At the first
intervention update, all later-layer parameters and every Adam state exactly
match the native update; only first-layer parameters differ by suppression.
Independent local checkpoint checks confirm both frozen tensors are bitwise
unchanged, all four later parameter tensors changed, optimizer groups remained
identical, and every clock reaches 801. First representations/inputs are checked
at every captured state. Dense and streaming cosines/RMS and independent Gram
SVD/reconstruction checks pass, including exact zero-energy handling.

One seed and one selected fork; no statistical uncertainty interval, new probe,
validation/test evaluation, or generalization claim. No parameter-count-matched
later-layer intervention or rescue/unfreeze control was run. Those are possible
follow-ups, not outcomes of this experiment.

Spark worker time 14.80s including replay/qualification and diagnostics; actual
frozen continuation 7.43s. Peak allocated CUDA 406.4MB. Source staging, compression,
transfer and local analysis are additional. About2.067GB retained on owned Spark;
download archive 981.1MB and raw evidence 1086.2MB verified. Worker exited, no task
process remains, unrelated workloads preserved, and cloud cost is zero.
