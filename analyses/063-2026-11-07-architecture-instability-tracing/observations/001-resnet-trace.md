# ResNet / CIFAR-10: standardized architecture oscillation trace

## TLDR

The first learned representation has negative consecutive motion in **19/31** updates; final log-probability motion is negative in **25/31**. Their alignment-series correlation is **0.5775**. This localizes observed motion on the fixed probe and does not identify a unique causal origin.

Full-objective loss over the diagnostic window ranges from **0.299402 to 0.45824**; fixed-probe loss ranges from **0.266152 to 0.4321**. Their temporal correlation is **0.9748**; this checks the probe's loss timing, not population-wide activation representativeness. Loss-event details below distinguish an actual excursion from a merely negative cosine. All architecture/optimizer interventions remain TODO.

### What the architectural trace adds

**The reversal is already present at the first convolution, but the output-like
temporal structure becomes clearer deeper in the residual network.** The
alignment-series correlation with output is 0.5775
at the first convolution, 0.5427 at the end
of residual stage 1, 0.9305 at the end of
stage 2, and 0.9907 at the end of stage 3.
The top-10 temporal-subspace overlap rises from
0.5975 at the first convolution to
0.9691 at the final residual stage. Thus an
early reversal and an output reversal coexist, but they are not one identical
trajectory transmitted unchanged through every layer.

At update 287, first-convolution cosine is -0.0911,
whereas final log-probability cosine is -0.7400.
At 302 they are 0.4123 and
0.1692, respectively. The full loss rises from
0.3473 at state 286 to 0.4582 at 293 (1.319 times the baseline) and recovers by
302. The outgoing displacement with label 292 enters the peak state 293.

The first-convolution activation coordinate RMS remains between
0.789094 and 0.792680, while its
step-displacement RMS spans 0.002616 to
0.009793. A representation can remain similar in overall
size while its small successive movements grow and reverse. This is not evidence
of the whole representation exploding in norm. These global RMS values do not
exclude rare sample/channel excursions.

There is no nearly one-dimensional full-window output explanation: the leading
output mode contains 24.03% of displacement energy, the top ten
contain 87.97%, and 16 modes are needed for 95%.
The first mode changes sign 23 times across 30 adjacent
coefficient pairs. The stem BatchNorm is more spectrally concentrated (top ten
99.01%), but its overlap with the output's top
ten temporal modes is only 0.5925. Concentration and
transmission of the same temporal patterns are different questions.

Correct-label log-probability changes reverse in
29/31 updates and their
alignment-series correlation with all-output changes is
0.9425. The observed functional
oscillation therefore also appears in the per-example loss readout, rather than
being supported only by unrelated class coordinates. This does not measure
per-example gradient coherence.

**Next discriminating diagnostic:** separate incoming representation changes,
local parameter changes and cross terms at residual-stage transitions. Test
the first convolution with a matched intervention before naming it the cause.
Neither a BatchNorm norm ratio nor the first negative cosine is a causal test.


## Coverage and trajectory identity

- Run 243, unchanged Adam from the saved Run 223 seed-0 checkpoint; LR 0.003, betas (0.9, 0.999), eps 1e-8, no clipping or decay. FP32 native, FP64 diagnostics, TF32 off.
- Actual execution: NVIDIA GeForce RTX 4090, PyTorch 2.11.0+cu130, CUDA 13.0.
- Full objective: 50,000 images per update; mean cross-entropy over all loss units.
- Fixed probe: 1000 images in one original BatchNorm group. No coordinates within that probe are sampled or averaged away. This is not a census of all training activations.
- Displayed outgoing-update indices 280--310; 31 SVD time rows, with one extra preceding/following captured state for adjacent motion and endpoint differences.
- Historical exact agreement: **True**. 0/56 state pairs differ in loss or raw-gradient norm; maximum relative loss discrepancy 0 (0.0000%). Same saved starting tensors and policy do not establish bitwise historical continuation across GPU models/runtime kernels.

## Architecture and measurement points

Input 1000 x 3 x 32 x 32; stem Conv (16)-BatchNorm-ReLU; three stages with five residual blocks each, widths 16/32/64 and spatial sizes 32/16/8; average pooling to 64; linear output 10. Each block has Conv-BN-ReLU-Conv-BN, a skip path, addition and ReLU. Downsampling skips use average pooling, BN and zero channel padding. BN uses its original fixed 1000-image group; running statistics are disabled. Inputs use fixed full 50k per-channel normalization.

The standard trace contains 132 representation streams plus the full raw-gradient stream. Pre-ReLU tensors are cloned before in-place mutation. Reused module calls have separate call identifiers. Residual-sum inputs and the separate residual branch outputs are distinguished; no local causal contribution is assigned from an output norm ratio alone. Captured fixed inputs are exactly equal across the measurement states; their displacement is zero and motion cosine undefined.

## Standardized landmark table

Rows of a captured representation index fixed images or supervised token positions; trailing dimensions are feature channels/spatial positions or vocabulary coordinates. The SVD matrix instead has optimizer updates as rows and all flattened probe coordinates as columns.

| Stage | One-state captured shape | Negative/defined cosine | Top1 energy | Top10 energy | Alignment r with output | Top10 temporal overlap |
|---|---|---:|---:|---:|---:|---:|
| `conv1.call0.output` | [1000, 16, 32, 32] | 19/31 | 28.35% | 95.30% | 0.5775 | 0.5975 |
| `bn1.call0.output` | [1000, 16, 32, 32] | 21/31 | 40.06% | 99.01% | 0.5703 | 0.5925 |
| `relu.call0.output` | [1000, 16, 32, 32] | 21/31 | 42.58% | 98.90% | 0.5794 | 0.5925 |
| `layer1.4.relu.call1.output` | [1000, 16, 32, 32] | 21/31 | 54.04% | 92.59% | 0.5427 | 0.6645 |
| `layer2.4.relu.call1.output` | [1000, 32, 16, 16] | 25/31 | 23.63% | 86.91% | 0.9305 | 0.8141 |
| `layer3.4.relu.call1.output` | [1000, 64, 8, 8] | 25/31 | 21.63% | 86.60% | 0.9907 | 0.9691 |
| `avgpool.call0.output` | [1000, 64, 1, 1] | 24/31 | 23.58% | 88.47% | 0.9908 | 0.9798 |
| `fc.call0.output` | [1000, 10] | 25/31 | 26.42% | 88.76% | 0.9967 | 0.9846 |
| `output.log_probabilities` | [1000, 10] | 25/31 | 24.03% | 87.97% | 1.0000 | 1.0000 |
| `output.correct_label_log_probability` | [1000] | 29/31 | 24.01% | 88.14% | 0.9425 | 0.8708 |
| `raw_gradient` | [464250] | 26/31 | 34.81% | 91.98% | 0.7051 | 0.8517 |

The raw-gradient row decomposes $g_t$, not gradient differences; its RMS column in the accompanying CSV is gradient RMS. All other rows decompose outgoing activation changes. The reference output is log-probabilities; raw logits remain a separate captured stage. A common class-logit shift can affect raw-logit geometry while canceling from probabilities.

## Loss excursions in this window

The unchanged Analysis 036 recovery detector is applied to the new full-objective continuation, not copied from the historical worker. Events whose peaks lie inside the diagnostic window are:

```json
[
  {
    "before_state": 283,
    "onset_state": 284,
    "peak_state": 284,
    "recovery_state": 285,
    "end_state": 285,
    "right_censored": false,
    "ratio": 1.0050106761480881
  },
  {
    "before_state": 286,
    "onset_state": 287,
    "peak_state": 293,
    "recovery_state": 302,
    "end_state": 302,
    "right_censored": false,
    "ratio": 1.3193211721806362
  },
  {
    "before_state": 302,
    "onset_state": 303,
    "peak_state": 303,
    "recovery_state": 304,
    "end_state": 304,
    "right_censored": false,
    "ratio": 1.0084561404936978
  },
  {
    "before_state": 304,
    "onset_state": 305,
    "peak_state": 305,
    "recovery_state": 307,
    "end_state": 307,
    "right_censored": false,
    "ratio": 1.0292903895097145
  }
]
```

Indices are completed optimizer updates. The increment entering a peak $p$ is labeled $p-1$. Right-censored events are not claimed recovered. Event timing was selected from the original control before inspecting these representation diagnostics, so this remains a retrospectively selected window rather than an independent discovery sample.

## Algorithm, figures and limits

Apply [the common method](../METHOD.md): fixed probes; unchanged trajectory; outgoing differences; exact uncentered FP64 temporal Grams; fixed-window SVD; signed coefficients; backward architectural comparison of cosine, magnitude and temporal subspaces. No per-step changing SVD basis, sample averaging, random sketch or gradient differencing is used. Temporal-coefficient signs are oriented once by the largest absolute temporal coefficient. Correlations use dependent time points and do not supply independent causal evidence.

**Figure 1, page 1:** full training/probe loss, raw-gradient/output alignment, every captured representation's consecutive-change cosine, displacement coordinate RMS divided by that stage's own window maximum, and top1/top10 SVD energy with top10 output temporal overlap. Row-wise scaling in the RMS heatmap shows timing, not absolute cross-layer magnitude. Raw zero-input motion is stated separately; no zero-cosine curve is invented.

**Figure 1, page 2:** architecture landmarks with consecutive cosine, absolute displacement coordinate RMS and the leading fixed SVD coefficient divided by the square root of the number of captured coordinates. Independent layer bases do not identify the same neuron directions; coefficient signs across different layers are arbitrary after their separate fixed orientation.

Strong resemblance across layers can reflect coupled updates driven by the same downstream loss. Even an earliest-layer reversal is not a causal initiating source. Probe selection, one seed, short selected windows, near-degenerate SVD modes and numerical hardware differences limit extrapolation. Normalization/attention mixing cannot be explained solely by elementwise secant arguments. Follow-up causal tests and deeper attention/normalization accounting are separate from this first-pass trace.

## Sources and verification

- [Numerical results and source hashes](../resnet_results.json); [every-stage table](../resnet_stages.csv).
- [Standardized PDF](../figures/01-resnet-instability-trace.pdf).
- Scripts 01/02/03 in this analysis generate reductions, figures and this observation; Run 243 owns capture and source identities.
- Observer verifies plain-forward parity, unchanged gradients/RNG/buffers and fixed probe inputs. Independent FP64 contractions check all Gram diagonal norms and adjacent products. Local reduction verifies PSD, trace, full Gram reconstruction and full-gradient boundary norms/cosines. Model checkpoints, input/target identities, exact Grams and raw-cache hashes are retained with transfer verification. Dense activation caches are regenerable from these snapshots and the pinned computation contract.

Rendering and final artifact integrity checks are recorded in the figure and execution verification records. No manuscript edits or causal interventions are part of this result.
