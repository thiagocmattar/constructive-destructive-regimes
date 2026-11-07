# Selective learning rates: retaining feature learning while damping oscillation

## Conclusions

**Reducing the first layer's LR to 0.25x is a useful compromise in this window.**
All layers remain trainable. Compared with unchanged Adam, mean training loss
448-800 falls **7.06%**, cumulative positive loss variation falls **99.62%**, and
first-hidden reversals fall from **350/353 to 1/353**. Output reversals fall from
351/353 to 45/353. Unlike freezing, this intervention retains first-layer motion
and avoids the freeze arm's 22.27% increase in mean loss.

**It does not uniformly improve optimization.** Its last-32-state mean loss is
15.79% higher than baseline, and its first ten-state interval below loss 0.02
starts at 648 instead of 585. Conversely, later-layer-only reductions improve
average loss while retaining almost all output reversals. Oscillation suppression
and optimization progress are distinct outcomes; neither cosine sign nor the
loss at one endpoint is an adequate objective by itself.

The weaker first-layer reduction (0.5x) only postpones a large excursion, which
peaks at 673. Global 0.5x has the lowest full-window mean, but its loss is rising
sharply at the right boundary, 800. Its late excursion is censored by the fixed
analysis horizon. These results support a special role for first-layer update
size in the observed representation reversals, not a unique initiating source
or a universal stable LR.

## Design and operational definitions

Runs 252-257 / H046 extend the same Analysis064/065 study. Each branch starts
from the **same untreated full state 448**; there is no inherited freeze and no
change at 500. First changed update: 448 -> 449. All layers update through 801,
which supplies the outgoing representation displacement at 800. The unchanged
Adam control is reused only after exact native replay qualification.

| Arm | Run | First-layer LR | Later-layer LR |
|---|---:|---:|---:|
| Baseline | Retained control | 0.001 | 0.001 |
| I9: first layer, half | 252 | 0.0005 | 0.001 |
| I10: first layer, quarter | 253 | 0.00025 | 0.001 |
| I11: global, half | 254 | 0.0005 | 0.0005 |
| I12: global, quarter | 255 | 0.00025 | 0.00025 |
| I13: later layers, half | 256 | 0.001 | 0.0005 |
| I14: later layers, quarter | 257 | 0.001 | 0.00025 |

Matched: seed-0 tanh MLP 3072-200-200-10; all original 5000 CIFAR-10 training
examples and order; mean per-example half squared error; physical batch 1000;
native Adam betas (0.9,0.999), epsilon 1e-8, no clipping/decay; FP32, TF32 off,
foreach/fused false. Weights and biases both receive their layer's rate.
Moments, clocks, previous/current gradients, RNG, model tensors and data are
preserved at the fork. Temporary native Adam parameter groups apply the rates;
the original group layout is restored between updates. Checkpoint metadata and
immutable configs retain the applied LR policy. There is no moment reset.

Every intervention is compared primarily with unchanged Adam. Global and
later-layer-only changes are supplementary location controls. Equal LR
multipliers do **not** match total applied increment norms, and the first layer
contains 93.57% of parameters. This is not an equal-step-length causal comparison.

Before execution, the plan fixed the primary window 448-800, original event
window 450-500, and last 32 states 769-800. It also fixed a sustained target: the
first ten consecutive measured states with loss strictly below 0.02. This is
reported as an exploratory target, not an independently selected benchmark or
an optimal stopping criterion. All six planned branches were completed.

## Results

All columns below use full training-objective losses, with no smoothing.
Maximum loss can be the common starting value at 448; it is not automatically a
post-intervention peak. I1 is the preserved freeze-only supplementary reference.

| Arm | Mean loss448-800 | Change vs baseline | Maximum loss (state) | Largest upward step | Mean loss769-800 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.0228078 | -- | 0.0550626 (522) | 0.0103766 | 0.0134585 |
| I1: freeze | 0.0278867 | +22.27% | 0.0322876 (448) | 0.0013623 | 0.0271836 |
| I9: first half | 0.0200084 | -12.27% | 0.0861471 (673) | 0.0254193 | 0.0130908 |
| I10: first quarter | 0.0211967 | -7.06% | 0.0322876 (448) | 0.0003976 | 0.0155829 |
| I11: global half | 0.0189189 | -17.05% | 0.0601813 (800) | 0.0194046 | 0.0156249 |
| I12: global quarter | 0.0224891 | -1.40% | 0.0322876 (448) | 0.0015262 | 0.0174434 |
| I13: later half | 0.0204267 | -10.44% | 0.0532532 (517) | 0.0100108 | 0.0112110 |
| I14: later quarter | 0.0194737 | -14.62% | 0.0582721 (559) | 0.0123955 | 0.0134648 |

At the same quarter-rate multiplier, I10 has **5.75% lower mean loss than I12**,
with smaller largest upward loss change. Leaving later-layer rates unchanged
therefore improves this particular smooth-training compromise. I10 still has
11 upward transitions, so it is not strictly monotone. Its positive variation
is 0.00137828 versus 0.364412 for baseline.

The first ten-state intervals below 0.02 start at 585 for baseline,568 forI9,
648 forI10,582 forI11,697 forI12,569 forI13, and536 forI14. I1 does not reach
this target during the window. Confirmation occurs nine states after each start.
The result depends on the specified target and horizon; it is not a universal
ranking of optimization speed.

## What changes inside the network?

| Arm | Negative first-hidden change cosines /353 | Negative output-change cosines /353 |
|---|---:|---:|
| Baseline | 350 | 351 |
| I9 | 149 | 199 |
| I10 | 1 | 45 |
| I11 | 33 | 68 |
| I12 | 1 | 2 |
| I13 | 347 | 348 |
| I14 | 329 | 344 |

First-layer quartering suppresses the reversal pattern already in the first
hidden representation while preserving nonzero movement. Its maximum
first-hidden coordinate displacement RMS falls from 0.055471 to 0.0066713, and
maximum output RMS falls from 0.163076 to 0.0303608. In contrast, reducing only
later-layer rates leaves the first-hidden and output reversal counts near the
baseline's counts, despite improved mean loss. This strengthens the location
evidence beyond the original freeze, while leaving global network coupling and
the unequal update magnitudes as interpretation limits.

I9 and I11 have prolonged quiet intervals with small gradients and representation
changes before renewed reversal and growth. Those intervals do not establish
permanent stabilization. Per-layer actual increment norms, uncorrected first
moment norms and an FP64 reconstruction of bias-corrected Adam denominator RMS
are retained in `updates.jsonl`. These scalar summaries do not identify the
directional cause of the late excursions or establish preconditioner causality.

## New figure and metric contract

[Figure03](../figures/03-selective-layer-lr-cifar10-mlp.pdf) is a new three-page
artifact. Page 1 showsI9/I10, page 2I11/I12, and page 3I13/I14. Every column overlays
the identical unchanged baseline. Existing Figures01 and02 remain unchanged.

**Caption.** Selective and global LR reductions from untreated MLP state 448.
Columns compare0.5x and0.25x changes in the specified parameter group. Rows:
(a)full training loss; (b)eta kappa hat; (c)full and later-layer raw-gradient
cosine; (d)full raw-gradient norm; (e)later-layer raw-gradient norm; (f)first-hidden
change cosine; (g)first-hidden displacement RMS; (h)output-change cosine;
(i)output displacement RMS; (j)applied first-layer and later-layer LRs. Training
loss has independent linear axes; other rows share scales across all pages.
All 353 states448-800 are plotted; log/symmetric-log scales are labeled.

For a fixed representation matrix $A_t$, outgoing differences are
$\Delta A_t=A_{t+1}-A_t$. Cosines compare flattened $\Delta A_t$ and
$\Delta A_{t-1}$. Coordinate RMS is $\|\Delta A_t\|_F/\sqrt P$, where
$P=5000\times200$ for first-hidden activations and$P=5000\times10$ for outputs.
Samples are never averaged before these measurements. Full gradients have 656810
coordinates; later-layer gradients have 42210. All are raw objective derivatives,
not Adam directions, parameter increments, or gradient differences. The diagnostic
$\eta\widehat\kappa_t=1-\langle g_t,g_{t-1}\rangle/\|g_{t-1}\|^2$ is a projected
gradient change under Adam, not a curvature estimate or stability threshold.

## Verification, reproduction, compute and limitations

- All six runs match 65 untreated boundary states448-512 and the complete
  historical model/optimizer/RNG/previous/current-gradient state 512 exactly.
  Every fork matches the original448 checkpoint; all first-step moments/clocks
  remain native, and prescribed scaled increments pass rounding-qualified checks.
- Seven LR combinations, including identity, match independently grouped native
  Adam for three warm steps. Regression checks verify that the source checkpoint's
  CPU step counters are not mutated. Capture preserves gradients, RNG and inputs.
- Each successful run verifies67 source identities and22 downloaded files.
  All five dense representations independently reproduce streamed cosine/RMS
  over448-500; raw/full/first/later gradient reductions and eta are checked over
 448-501. Full rolling records cover through the outgoing update800.
- All 50832 plotted scalars match the verified arrays. The three pages are
  rendered and reviewed; baseline duplication, scale policy, text bounds and
  original-figure hashes are checked.

Run252 attempt01 failed before any intervention update: loading Adam state
aliased CPU step counters, which qualification advanced. The restore guard caught
the mismatch. Its source, logs, manifest and size/hash-verified evidence are
preserved. Attempt02 deep-copies restored optimizer state, with unchanged
scientific configuration; all qualification and independent checks pass. No
failed-attempt intervention trajectory enters these results.

Successful workers took **49.75s total** for 2118 intervention updates plus their
qualification, with peak allocated GPU memory 400.8 MB each. The rejected attempt
took 1.09 s. These timings exclude staging, transfer and local reduction and overlap
some I/O; they are not a controlled throughput benchmark. All task workers exited.
Owned Spark retains **13.720GB** including archives and the failed attempt, with
verified local copies. No cloud resources or charges.

Use [07_reduce_layer_lr.py](../07_reduce_layer_lr.py), then
[08_figures_layer_lr.py](../08_figures_layer_lr.py). The shared native worker is
[layer_lr_continuation.py](../../../src/eos_tasks/layer_lr_continuation.py).
[Plan](../layer_lr_plan.json), [verified results/provenance](../layer_lr_comparison.json),
[metrics](../layer_lr_metrics.csv), [figure audit](../figure03_verification.json),
[render audit](../figure03_visual_verification.json), and
[compute closeout](../layer_lr_compute_closeout.json) retain exact definitions
and identities. Run-local receipts document source and retrieval integrity.

One seed, one selected fork, fixed training subset and horizon. There are no
uncertainty intervals, validation/generalization conclusions, new SVD fits,
input-covariance interventions, or claims about a fixed versus rotating unstable
subspace. Longer horizons and replication are needed before calling any rate
generally preferable. The input-geometry causal test remains a TODO.
