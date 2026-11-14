> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [008-resnet-major-peak-temporal.md](../../../../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/008-resnet-major-peak-temporal.md).

# Matching temporal structure for the larger ResNet excursion

- **Decision:** both selected tracing figures now describe completed steps **390--450** for ResNet. The temporal figure uses a fresh fit over all **61** incoming changes. The original 281--311 fit and reports remain historical evidence.
- **Central message:** the larger excursion has concentrated representation motion, shared temporal structure across depth, and repeated reversal along a fixed output pattern. The result supports a structured functional expression of the oscillation; it does not identify a unique causal source.
- **Output energy:** top one **27.26%**, top ten **87.06%**, rank95 **20/61**, rank99 **35/61**. Raw-gradient top ten is **94.74%**, with rank95 **11/61**. These are energy concentrations, not explained loss or exact low rank; the numerical rank is 61.
- **Shared timing:** first-convolution/output top-ten overlap is **.567**, increasing through the selected residual stages to **.985** at stage 3 and **.994** at pooling. This is temporal-subspace similarity, not identity of spatial/neuron directions or evidence of causal propagation.
- **Signed motion:** the leading output coefficient changes sign **33 times across 60 adjacent fitted pairs**. Inspect its amplitude: small tail sign changes contribute little movement. Full-vector negative cosine occurs at **21/61** displayed steps, including comparison with the predecessor outside the fit; the two counts measure different objects.

| Measured endpoint | Top1 energy % | Top10 energy % | Rank95 | Top10 output overlap |
|---|---:|---:|---:|---:|
| First convolution | 30.78 | 95.75 | 10 | .567 |
| Residual stage 1 | 50.96 | 92.17 | 14 | .630 |
| Residual stage 2 | 28.50 | 85.72 | 20 | .822 |
| Residual stage 3 | 23.58 | 85.88 | 21 | .985 |
| Pooling | 25.09 | 88.37 | 18 | .994 |
| All-class log probabilities | 27.26 | 87.06 | 20 | 1.000 |

- **Magnitude/direction controls:** observed output top-mode energy **27.26%** exceeds **9.95%** for orthogonal rows with the same individual step norms. Unit-row normalization gives top-mode energy **10.85%** and rank95 **33**. Both direction and amplitude affect concentration. These deterministic references are not a randomized null, quiet-window contrast, or a comparison of intrinsic architecture dimensionality.

## Source, capture and fit

- **Selection:** match the already selected loss/motion window in [observation 007](63-007-resnet-major-peak.md), without looking at spectral outcomes. Full loss is 0.1070888 at 404, rises to 0.4407024 at 415 (+311.53%), and recovers at 439.
- **New diagnostic capture:** [Run 292](../../../../../../runs/292-2026-11-14-resnet-major-peak-tracing/README.md) reconstructs the unchanged Run 273 seed-0 control from its full state 264. Adam LR 0.003, betas (.9,.999), no clipping/decay/reset, FP32 native computation, TF32 off, deterministic original runtime, full 50,000-image objective with the original 1,000-image BatchNorm groups.
- **Probe:** the same fixed first 1,000-image group, in training mode with the original non-mutating BatchNorm contract. Keep every sample/channel/spatial/class coordinate. It is a convenient fixed subset, not a new representative sample or held-out set. Full-objective and probe losses are separately verified.
- **Rows:** capture states 388--450. Fit representation rows A_s-A_(s-1) for s=390,...,450. Fit raw gradients g_(s-1), states 389--449, used by those updates. The predecessor displacement into 389 and gradient 388 are only for the first adjacent cosine; neither enters the SVD. Raw gradients are not differenced and are not Adam's applied increments.
- **Reduction:** exact FP64 Grams, accumulated over coordinate blocks; no temporal centering, example averaging, random projection or primary row normalization. Refit eigenvalues, stage energies, top-ten overlaps and coefficients for the whole new window. The fixed leading coefficient is alpha_1/sqrt(P), with P=10,000 output coordinates and one sign convention: largest absolute coefficient positive.
- **Qualification:** all **187 loss states 264--450 match exactly**. The initial full gradient and state 296 full parameters/gradient match bitwise. All six stage cosine/RMS histories agree with the existing figure; maximum absolute cosine discrepancy is **1.34e-15**. This is the qualified native control, not a claim of bitwise reproduction of Run 215's original full-run overview.
- **Checks:** source/download hashes, unchanged observer RNG/gradients/buffers, plain-forward parity, PSD/trace/reconstruction, independent raw-gradient Gram from retained full vectors, independent Gram singular values, and permutation invariance. All pass; [numerical results](../data/resnet_390_450_results.json), [stage table](../data/resnet_390_450_stages.csv), [current three-case controls](../data/spectral_controls.json), [run verification](../../../../../../runs/292-2026-11-14-resnet-major-peak-tracing/verification.json).
- **Cost/retention:** worker replay/capture/reduction 685.99 seconds, peak allocated CUDA 3.53 GB. All 75 retained files verified (266.79 MB), including 63 model snapshots, full gradients, fixed probe, Grams and recovery state. Large derived activation caches were inventoried before worker deletion. Estimated compute USD0.412; final billing may lag. No retained cloud storage. This replay supplies missing diagnostics, not an independent repetition of the experiment.

## Figure, caption and reproduction

- **Current PDF:** [three-model temporal structure](../figures/01-three-model-temporal-structure.pdf). MLP/Transformer curves and panel pixels are unchanged; Figure 6 and intervention assets remain unchanged.
- **Caption:** Temporal structure of the same loss/motion windows: completed steps 451--501, 390--450 and 553--601, with 51/61/49 fitted displacement rows. Top: cumulative squared-singular-value energy for output changes and the corresponding raw gradients; dotted line marks 95%. Middle: top-one/top-ten stage energy and top-ten temporal overlap with output. Bottom: leading signed output coefficient divided by sqrt(P), using one fixed pattern over each stated window. The ResNet fit is freshly computed from exact-replay-qualified captures around peak 415 and recovery 439. Different probes/windows prevent a controlled architecture-rank comparison; energy concentration alone is time-permutation invariant and does not establish causality or spike specificity.
- **Generate:** `12_resnet_major_svd.py`, then the [figure workflow](../../../../work/resnet_temporal_revision_20261114/01_figure.py), [verification](../../../../work/resnet_temporal_revision_20261114/02_verify.py), and [packaging](../../../../work/resnet_temporal_revision_20261114/03_package.py). Rebuild the package with `manuscript/consolidation/work/package_evidence.py --build --verify`.
- **Uncertainty:** one selected event, one initialization and one fixed probe. No confidence interval, prospective alarm, quiet-window contrast, basis-rotation test, output-sensitive hidden projection, or causal intervention is inferred from this SVD.
