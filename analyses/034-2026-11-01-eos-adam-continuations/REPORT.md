# Analysis034: Adam from the same EOS checkpoints

Completed the authorized calibration and matched-horizon continuation at all 27
retained states from Analysis032/033. Runs209/211 test five cold-Adam rates;
Run210 restarts at the identical original model states with the selected rate.
There are 25 primary states and two previously unqualified Run062 reconstructions.
Production records 72,192 anchored/adjacent cosine values at positive lags
through the same individual GD horizons, up to lag 2048.

| Model | Selected constant Adam LR | Primary calibration states |
| --- | ---: | ---: |
| mlp | 0.0003 | 7 |
| resnet | 0.001 | 9 |
| transformer | 0.001 | 9 |

The terminal 64-update median adjacent raw-gradient cosine is positive in
10/25 primary profiles;
17/25 contain at least one negative adjacent pair in that window.
The terminal anchored odd/even half-gap is at most 0.1 in
22/25 primary profiles.
These statistics answer different questions: anchored similarity measures
orientation relative to the initial gradient, while adjacent similarity measures
how the raw-gradient direction turns between successive states. Positive adjacent alignment can coexist
with a small or negative anchored cosine. Neither sign establishes loss benefit,
stability, or permanent/asymptotic damping.

Positive terminal median adjacent cosines occur in
0/7 primary MLP profiles,
8/9 ResNet-32 profiles and
2/9 Transformer profiles.
All 9 primary Transformer profiles contain negative adjacent pairs
in the final window. These counts describe dependent forks at the individual
256/512/2048 horizons listed below; they do not rank optimizers or models.

The same seed 0 model, full data/order and original 500/2000/3000 states are matched.
There are 21 directly saved historical starts, four already-qualified legacy
reconstructions and two previously unqualified reconstructions. Every new fork
uses the identical retained Run205 model tensors used by the previous analysis;
the latter two starts remain outside primary summaries.
The three parent GD learning rates identify different histories, not three Adam
rates. Adam starts with zero moments and step 0 at each fork, beta1=0.9,
beta2=0.999, epsilon 1e-8, constant LR, no decay/clipping/schedule/AMSGrad.
This is an optimizer-reset intervention from GD-trained parameters; it is not
Adam training from initialization or continuation of pre-existing Adam moments.
The intervention changes the optimizer and its calibrated numerical LR together;
GD/Adam actual step norms are not matched. Curve differences therefore do not
separate the causal roles of momentum, preconditioning and step scale.

Calibration uses 64 full-objective updates for each rate
1e-5/3e-5/1e-4/3e-4/1e-3 and each of 27 starts. Exclude the two unqualified starts
from selection. Require every primary state to remain finite, peak loss no more
than 4 times initial and final 16 mean loss no more than 1.05 times initial (plus 1e-6).
Minimize equal-checkpoint mean log(final 16 mean/initial), separately per model.
This is conditional short-horizon training-loss tuning, not a globally optimal
rate or independent evidence of optimizer superiority. Cosines and test data
never select the rate; no validation/test evaluation is added.
The ResNet-32 and Transformer selections are at the upper grid boundary, so this
calibration does not establish that larger rates would be worse.
The 64-update calibration does not constrain later training-loss excursions;
the complete production loss histories are retained alongside the cosines.

The MLP objective is half squared error summed over outputs and averaged over
the same first 5000 CIFAR-10 images. ResNet-32 uses all 50000 images with the same
50 fixed 1000-image train-mode BN groups and no running-stat mutation. Transformer
uses all 59680 fixed sequences and 2,088,600 valid targets, token-weighted NLL.
All trainable parameters contribute once in model order; FP32 parameters/gradients,
TF32 disabled, FP64 diagnostic reductions. Transformer chunk 512 qualifies against
the original 64 at every start within 1e-5 relative gradient and 1e-5 absolute loss.
It changes the FP32 accumulation partition, not the objective or token weighting.

Each lag k is one optimizer boundary; g[k] is the complete raw gradient at
theta[k] before the Adam update. Record cos(g0,g[k]) and cos(g[k-1],g[k]) every
boundary, with first adjacent/zero-denominator pairs undefined. No gradient
projection or quantization is used; archive compression is lossless.
The last sampled gradient is unapplied, so optimizer clocks equal the final lag.
Every GPU FP64 cosine and gradient norm is independently checked against CPU NumPy FP64;
maximum cosine disagreement 3.55e-15.
The first three and terminal two full gradients, model/Adam/Torch CPU/CUDA RNG states, source
checkpoints and complete scalar histories are retained locally. All
162 arm-level vector audits pass. Selected calibration and
production prefixes are independently repeated; 18/27
have loss, norm and cosine differences below 1e-5 at every initial 64-update
boundary. All matched initial tensors and boundary checks pass; the full
64-update numerical drift is retained without a claim of bitwise trajectory replay.

Run211 distributes the unchanged 10 calibration arms at Run121 states 2000/3000
over already calibrated warm workers. This placement is canonical by a frozen
infrastructure plan before its outcomes are observed; partial/redundant original
Run209 arms remain visible and never replace a worse outcome. Production places
those two short-horizon continuations on warm ResNet workers. No data, LR grid,
objective, initialization, precision or qualification threshold changes.
Two initial 5090 workers failed the declared chunk qualification before Adam
updates. Their failure logs are retained; qualified 4090 placements were used.

| Task / parent GD run | Anchor | Last lag | Adam LR | First adjacent | Last 64 median adjacent | Last 64 negative pairs | Anchored half-gap | Final train loss | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mlp / 059 | 500 | 256 | 0.0003 | 0.380824 | -0.698996 | 75.00% | 0.004611 | 0.113075 | primary |
| mlp / 059 | 2000 | 256 | 0.0003 | 0.594862 | -0.823088 | 76.56% | 0.004001 | 0.110394 | primary |
| mlp / 059 | 3000 | 256 | 0.0003 | 0.581843 | -0.869587 | 95.31% | 0.001992 | 0.114160 | primary |
| mlp / 062 | 500 | 256 | 0.0003 | 0.604538 | -0.808724 | 85.94% | 0.004201 | 0.107113 | primary |
| mlp / 062 | 2000 | 256 | 0.0003 | -0.874085 | -0.825566 | 90.62% | 0.010698 | 0.101291 | unqualified |
| mlp / 062 | 3000 | 2048 | 0.0003 | -0.885990 | -0.966356 | 100.00% | 0.260371 | 0.006337 | unqualified |
| mlp / 122 | 500 | 2048 | 0.0003 | -0.798430 | -0.971704 | 100.00% | 0.228774 | 0.003124 | primary |
| mlp / 122 | 2000 | 2048 | 0.0003 | -0.878278 | -0.966118 | 100.00% | 0.235790 | 0.002427 | primary |
| mlp / 122 | 3000 | 2048 | 0.0003 | -0.915561 | -0.937124 | 100.00% | 0.220298 | 0.002552 | primary |
| resnet / 123 | 500 | 256 | 0.001 | 0.985793 | -0.601680 | 87.50% | 0.011624 | 0.756458 | primary |
| resnet / 123 | 2000 | 2048 | 0.001 | -0.913504 | 0.984899 | 0.00% | 0.026892 | 0.000708 | primary |
| resnet / 123 | 3000 | 2048 | 0.001 | -0.783975 | 0.975698 | 0.00% | 0.014913 | 0.000539 | primary |
| resnet / 124 | 500 | 2048 | 0.001 | -0.868686 | 0.952189 | 0.00% | 0.039013 | 0.000922 | primary |
| resnet / 124 | 2000 | 2048 | 0.001 | -0.225092 | 0.991279 | 0.00% | 0.015195 | 0.000471 | primary |
| resnet / 124 | 3000 | 2048 | 0.001 | -0.792772 | 0.994525 | 0.00% | 0.007444 | 0.000541 | primary |
| resnet / 120 | 500 | 2048 | 0.001 | -0.749799 | 0.983400 | 0.00% | 0.007970 | 0.000562 | primary |
| resnet / 120 | 2000 | 2048 | 0.001 | -0.867396 | 0.997722 | 0.00% | 0.003324 | 0.000205 | primary |
| resnet / 120 | 3000 | 2048 | 0.001 | -0.854006 | 0.998270 | 0.00% | 0.000975 | 0.000078 | primary |
| transformer / 121 | 500 | 256 | 0.001 | 0.295069 | -0.761750 | 85.94% | 0.009855 | 2.535486 | primary |
| transformer / 121 | 2000 | 256 | 0.001 | 0.388010 | -0.473766 | 90.62% | 0.002808 | 2.457855 | primary |
| transformer / 121 | 3000 | 256 | 0.001 | 0.526560 | -0.699871 | 82.81% | 0.005084 | 2.462658 | primary |
| transformer / 125 | 500 | 2048 | 0.001 | -0.822202 | -0.914919 | 100.00% | 0.034415 | 0.493286 | primary |
| transformer / 125 | 2000 | 2048 | 0.001 | -0.679687 | 0.480547 | 7.81% | 0.027046 | 0.506127 | primary |
| transformer / 125 | 3000 | 2048 | 0.001 | -0.673257 | 0.247096 | 45.31% | 0.062888 | 0.544862 | primary |
| transformer / 126 | 500 | 2048 | 0.001 | -0.695284 | -0.937621 | 98.44% | 0.041800 | 0.446309 | primary |
| transformer / 126 | 2000 | 512 | 0.001 | -0.679435 | -0.365121 | 90.62% | 0.037817 | 1.155872 | primary |
| transformer / 126 | 3000 | 512 | 0.001 | -0.693547 | -0.218975 | 73.44% | 0.050884 | 1.143074 | primary |

The GD anchored reference in the figures is reused without new training.
Historical GD does not supply a complete adjacent-gradient series after lag 10;
the newly measured adjacent curves are Adam only. Horizons are matched to the
prior GD observation, not selected by the new Adam outcomes. There is one
initialization and dependent checkpoint forks; no seed uncertainty, stationary
autocorrelation, curvature interpretation of Adam, or generalization claim.

Four vector PDFs show [calibration](figures/calibration.pdf),
[anchored Adam/GD](figures/anchored_profiles.pdf),
[adjacent Adam](figures/adjacent_profiles.pdf) and its
[first 128-update view](figures/adjacent_first128.pdf). All raw points are retained;
no smoothing or path simplification. [Observation/captions](observations/001-adam-continuations.md),
[every production value](lag_profiles.csv), [results](results.json),
[summary](summary.json), [verification](verification.json).

The additional requested [largest-source-GD view through lag 256](figures/largest_lr_lag256.pdf)
matches Analysis033's `largest_lr_lag256.pdf`: three architecture columns,
three anchor rows, one curve connecting every consecutive positive lag,
and the same linear axis ranges and styling. Source GD LRs are 0.04 / 0.1 / 0.2;
Adam LRs are 3e-4 / 1e-3 / 1e-3 for MLP / ResNet-32 / Transformer.
The user chose these existing full production curves; the MLP 1e-3 candidate
has only 64-update calibration data and is excluded. All nine starts are primary.
All 2304 plotted anchored cosines match the JSON, CSV and plotted arrays exactly;
the vector PDF and its actual render pass review. This view reuses existing
measurements and adds no training or cloud cost.
[Observation/caption](observations/002-largest-lr-lag256.md),
[generator](13_plot_largest_lr_lag256.py), [review](largest_lr_lag256_review.json).

All 21 physical task Pods were deleted after per-file
byte/SHA-256 and archive verification. Retrieved 1903 files /
28.628 GB. No task cloud storage remains. Allocation through
last deletion took 192.6 minutes; aggregate compute
estimate USD 39.35, provisional pending complete
provider billing. The task stayed under its USD 50/4h30 bounds. Spark and unrelated
resources were preserved. [Closeout](closeout.json), [billing](billing.json).
