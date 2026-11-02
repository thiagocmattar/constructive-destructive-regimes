# Analysis036: short Adam calibration from initialization

Nine arms completed 128 full-objective updates each (1,152 accepted updates).
All use the same frozen Analysis024 seed0 weights per case and fresh Adam.
No seed sweep, Hessian/HVP, LR intervention or schedule; no validation/test passes.

| Case | LR | Final loss | Tail loss rises | Tail raw cosine | Tail moment cosine | Tail adaptive cosine | Loss label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mlp | 0.0001 | 0.304545 | 0.0% | 0.9995 | 0.9999 | 0.9999 | smooth_progress |
| mlp | 0.0003 | 0.263448 | 0.0% | 0.9933 | 0.9997 | 0.9997 | smooth_progress |
| mlp | 0.001 | 0.222867 | 0.0% | 0.9828 | 0.9996 | 0.9992 | smooth_progress |
| resnet | 0.0003 | 2.068170 | 0.0% | 1.0000 | 1.0000 | 0.9989 | smooth_progress |
| resnet | 0.001 | 1.315642 | 1.6% | -0.1318 | 0.9760 | 0.9187 | smooth_progress |
| resnet | 0.003 | 0.853716 | 17.5% | -0.5116 | 0.5106 | 0.6342 | unclassified |
| transformer | 0.0003 | 5.981771 | 0.0% | 0.9988 | 0.9998 | 0.9995 | smooth_progress |
| transformer | 0.001 | 4.593133 | 0.0% | -0.6602 | 0.9273 | 0.9959 | smooth_progress |
| transformer | 0.003 | 2.943088 | 3.2% | -0.4132 | 0.7559 | 0.9754 | smooth_progress |

All three MLP tails are smooth; its largest rate has a startup loss spike,
not sustained tail oscillation. Continuation evidence narrowed the grid but
did not transfer a from-initialization regime boundary. ResNet and Transformer
already show raw-gradient reversals while adaptive directions remain more
coherent. Interpret loss and geometry separately; negative raw cosine does
not establish harmful training or Hessian curvature.

The last64 boundary losses define labels (63 adjacent changes); tail stream
medians use the last64 accepted updates. Those windows differ by one boundary.
Labels and all pointwise values are preserved in fast_calibration_results.json
and fast_calibration_plot_values.json. Missing requested regimes are reported
in regime_candidates; no rate was changed or automatically added.

## Scientific coverage and next stage

The three fixed objectives are first5k CIFAR10 MLP images, all50k CIFAR10
ResNet images with50 fixed BN groups1000, and all2,088,600 prepared WikiText2
causal targets per update. Gradients cover states0..128; four complete prefixes
cover exactly128 applied updates. Stored m and bias-corrected mhat norms remain
distinct. Actual displacement is measured directly from parameters.

fast_calibration_results.json also records first pointwise and trailing64-mean
attainment of every frozen GD loss milestone. Unattained targets are censored
at128. For example, Transformer Adam at3e-3 first passes Run126's best
tested Transformer GD3000-state loss (6.069747) at update23; its trailing64
mean passes at state63 (the first complete window). ResNet Adam at3e-3
passes Run124's GD3000 loss (0.967229) at update107, while its trailing64
mean has not passed by128. These comparisons support shorter loss-matching
horizons in these cases. Matching an earlier GD loss is not convergence, matched location,
generalization, or a rate-independent optimizer benefit.

The main study remains provisional: up to three fixed LR regimes per case,
initially1,024 updates each (at most9 arms /9,216 updates). Calibration does
not establish convergence or authorize automatic main runs. Any new LR or
horizon requires a new frozen scientific run ID; the one-initialization and
no-Hessian constraints persist.

## Verification and execution

Eleven focused diagnostic tests pass. All manifests and artifact bytes/SHA256
are verified. Independent CPU checks reconstruct final moments/adaptive
directions, q at boundaries1/128, all prefix lengths/energies, parameter
telescoping and frozen initial/final tensor hashes. Cross-host initial gradient
qualification is recorded per arm; FP64 CPU/GPU norm reductions are compared
with relative1e-12 and absolute1e-10 tolerances. Smoke results are excluded.

MLP ran on the owned RTX5070Ti laptop. All ResNet/Transformer rates ran on
RTX5090 class at the live catalog rate USD1.19/GPU-hour, pinned PyTorch2.11/cu130
image, TF32 disabled. Different host CPU/runtime throughput prevents an
unqualified rate-by-rate wall-time comparison.

Two containers never became ready and received no scientific input; their
unchanged arms moved to warm matched workers. A Windows SSH newline issue
and isolated-package setup were corrected before scientific execution.
All preparation/retry/transfer time counts toward the same envelope.

Task compute plus temporary-storage estimate: USD2.26, provisional.
Every retrieved file was verified before deleting its worker and disks.
All five task pods are deleted; retained task cloud storage0GB. Unrelated
stopped resources were preserved. See execution_receipt.json and per-worker
retrieval receipts. The independent USD20 /90-minute guard did not trigger.

## Figures

07_reduce_plot_calibration.py generates both PDFs and exact plotted arrays.
See observations/001-fast-calibration.md for captions, reduction, coverage,
uncertainty and caveats. No manuscript was changed.
