# Analysis032: exact anchored gradient cosine through lag 10

All 27 requested windows completed: nine Analysis024 conditions at states
500–510, 2000–2010 and 3000–3010. Each profile has 11 complete-objective gradients
and ten original-rate GD updates. All 270 requested cosines use full FP32
gradient vectors and FP64 reductions, without projection or smoothing.

25/27 profiles qualify for the primary comparison.
Two Run062 MLP LR 0.02 reconstructions, at 2000 and 3000, fail original-history
qualification and are retained in the separately flagged companion. Its early
window and all three Run059 LR 0.004 reconstructions qualify. The other 21
profiles start from exact saved model tensors. Saved-state results are short
cloud continuations, rather than an assertion of bitwise historical replay.
At 3000, qualification for the seven saved parents checks the original endpoint
loss only; gradients 3001–3010 are new and have no original scalar reference.
Scalar qualification does not prove tensor equality for reconstructed starts.
The first failed Run062 replay check is at model state
1134 (gradient update 1135).

Among the 25 primary profiles, 17 have negative lag 1.
Of those, 17 have positive lag 2 and
17 have positive lag 10;
17 have positive cosine at every even
lag through 10, and 17 are negative at every
odd lag. These conditional measurements distinguish consecutive reversal
from alignment at longer update separations. They do not establish stationarity,
independence of distant gradients, optimization stability or generalization.
The three anchors are chronological locations, not matched dynamical regimes.
Rates were selected by the earlier calibration and only seed 0 is represented.
Within each model case, learning rate varies with the original initialization,
data/order and objective matched. Cross-case columns also change architecture,
objective and data volume; they do not isolate a causal model-size effect.
Coverage ends at lag 10. One anchored Gram-matrix row does not identify the
complete norm-weighted all-pair interaction or whole-path displacement.

Even-lag positivity does not imply an unchanged gradient direction. At anchor
3000, the highest-rate MLP, ResNet and Transformer curves have lag-10 cosines
0.3025, 0.1453 and 0.9898,
respectively. The visible sign pattern therefore coexists with substantial
direction changes over even steps in the MLP and ResNet cases. These are
case-specific examples, with the cross-case confounds stated above.

[Primary figure](figures/exact_gradient_lag10.pdf) and
[all profiles with reconstruction flags](figures/all_profiles_with_reconstruction_flags.pdf).
The complete 270 values are in [lag_profiles.csv](lag_profiles.csv), including
eligibility and provenance labels. Each point is one anchored pair, unlike the
window-averaged CountSketch profiles in Analysis005. No uncertainty bands or
training-seed replication are inferred from these points.

The window at 3000 extends the Analysis024 horizon by ten updates, in separate
Run205 outputs. Runs059/062 originally continued to 20000, so their scalar
references also cover this extension. Original runs/configs/artifacts and the
manuscript remain unchanged. Inputs, exact objective coverage, token weighting,
fixed BN groups and numerical choices are in [README.md](README.md) and Run205's
frozen config. No Hessians, sampling-noise probes or held-out evaluation were
added. Every gradient includes the complete training objective.
Run205 also retains byte-identical source input manifests as `source_inputs.json`
and `source_cifar_inputs.json`, including pinned revisions, ordering, normalization
and token exclusions. Preparation's `source_commit` identifies its base HEAD;
the per-file SHA256 inventory freezes the new code and config before execution.

Independent verification reads all 297 stored vectors and recomputes norms,
anchored cosines and adjacent cosines using chunked NumPy FP64 reductions.
All 270 values pass, with maximum absolute cosine error
4.13e-14. It also checks all 21 saved starts
against every tensor of their extracted source checkpoint, and checks actual
parameter displacement against the ten recorded applied gradients with an
explicit FP32 rounding bound. Fixed-state repeated gradients have maximum
relative discrepancy 1.74e-07.
Original/extracted checkpoint and input/source hashes remain recorded locally.
See [verification.json](verification.json), [checkpoint provenance](checkpoint_provenance.json)
and [figure manifest](figure_manifest.json).

Three Secure RTX 4090 workers were read at USD 0.89/GPU-hour. Aggregate production
time was 23.36 GPU-minutes; allocation to
last deletion took 24.58 minutes, including setup, staging and retrieval.
Per-worker median recorded gradient times, excluding fixed-state repeats, were
2.56–3.94 s for ResNet and 8.76–9.71 s for Transformer.
These include this recorder's diagnostics/I/O and are workload measurements,
rather than an isolated hardware benchmark.
Estimated compute cost is USD 1.047, provisional
before final posted billing/storage reconciliation, below the USD 6 cap.
All 147 inventory-listed files (7.072 GB) are locally
size/SHA-256 verified. All three task pods were deleted after acceptance; task
cloud storage is zero. Unrelated resources and Spark were untouched. The base
image reports an unused torchvision dependency mismatch after pinning PyTorch;
the experiment never imports torchvision, and its pinned Torch/NumPy/SciPy
runtime and real gradients pass checks. Raw logs preserve the warning.

Numerical endpoints below are rounded for reading; CSV/JSON retain full values.

| Model | LR | Anchor | Lag 1 | Lag 2 | Lag 10 | Primary eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mlp | 0.004 | 500 | 0.999999 | 0.999996 | 0.999914 | Yes |
| mlp | 0.004 | 2000 | 1.000000 | 1.000000 | 0.999994 | Yes |
| mlp | 0.004 | 3000 | 1.000000 | 1.000000 | 0.999997 | Yes |
| mlp | 0.02 | 500 | 0.999999 | 0.999996 | 0.999898 | Yes |
| mlp | 0.02 | 2000 | -0.993951 | 0.994614 | 0.706600 | No: replay failed |
| mlp | 0.02 | 3000 | -0.990356 | 0.984430 | 0.757118 | No: replay failed |
| mlp | 0.04 | 500 | -0.978939 | 0.968107 | 0.285239 | Yes |
| mlp | 0.04 | 2000 | -0.980221 | 0.979351 | 0.324006 | Yes |
| mlp | 0.04 | 3000 | -0.979053 | 0.958486 | 0.302538 | Yes |
| resnet | 0.025 | 500 | 0.999996 | 0.999986 | 0.999687 | Yes |
| resnet | 0.025 | 2000 | -0.976627 | 0.999996 | 0.999976 | Yes |
| resnet | 0.025 | 3000 | -0.979101 | 0.999252 | 0.989659 | Yes |
| resnet | 0.05 | 500 | -0.873351 | 0.999296 | 0.999005 | Yes |
| resnet | 0.05 | 2000 | -0.952463 | 0.972376 | 0.022948 | Yes |
| resnet | 0.05 | 3000 | -0.980591 | 0.962781 | 0.410422 | Yes |
| resnet | 0.1 | 500 | -0.962485 | 0.996436 | 0.901579 | Yes |
| resnet | 0.1 | 2000 | -0.977632 | 0.970645 | 0.068026 | Yes |
| resnet | 0.1 | 3000 | -0.967186 | 0.954212 | 0.145290 | Yes |
| transformer | 0.05 | 500 | 0.999998 | 0.999994 | 0.999860 | Yes |
| transformer | 0.05 | 2000 | 0.999999 | 0.999998 | 0.999970 | Yes |
| transformer | 0.05 | 3000 | 0.999996 | 0.999988 | 0.999980 | Yes |
| transformer | 0.1 | 500 | -0.946971 | 0.999991 | 0.999832 | Yes |
| transformer | 0.1 | 2000 | -0.958493 | 0.999020 | 0.973514 | Yes |
| transformer | 0.1 | 3000 | -0.987164 | 0.999530 | 0.991736 | Yes |
| transformer | 0.2 | 500 | -0.978964 | 0.999997 | 0.999896 | Yes |
| transformer | 0.2 | 2000 | -0.994017 | 0.999106 | 0.975065 | Yes |
| transformer | 0.2 | 3000 | -0.995214 | 0.999775 | 0.989782 | Yes |
