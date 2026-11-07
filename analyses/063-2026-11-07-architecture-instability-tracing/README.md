# Analysis 063: architecture-wide oscillation tracing

Start with the [consolidated motivation, per-update algorithm, results and conclusions](observations/004-consolidated-instability-tracing.md).
The [reusable workflow guide](../../research/INSTABILITY_TRACING.md) specifies
probe choices, computational costs and adaptation to future training setups.

## Conclusions first

The ResNet and Transformer show loss-associated representation reversals, but **the location
and temporal structure differ**. The first learned layer is not a universal
explanation of the output oscillation, and these measurements do not establish a
unique causal source.

| Diagnostic | ResNet-32 / CIFAR-10 | Transformer / WikiText-2 |
|---|---|---|
| Displayed outgoing updates | 280--310 | 552--600 |
| Main loss excursion | 0.3473 to 0.4582; peak 293, recovery 302 | 0.8431 to 1.6942; peak 562, recovery 596 |
| Fixed representation probe | 1,000 images, complete original BatchNorm group | 64 sequences, all 2,240 supervised positions |
| Full training objective per update | 50,000 images | 2,088,600 supervised tokens |
| First learned representation reverses | 19/31 updates at first convolution | 4/49 updates at embedding |
| Final log-probability changes reverse | 25/31 updates | 12/49 updates |
| First learned stage / output alignment-series correlation | 0.578 | 0.805 |
| First learned stage / output top-10 temporal overlap | 0.598 | 0.461 |
| Last residual/block representation / output temporal overlap | 0.969 | 0.982 |
| Historical continuation qualification | All 56 recorded loss/norm pairs exact | Not bitwise exact; maximum relative loss discrepancy 0.270% over 58 states |

For **ResNet**, early convolutional changes already reverse, but the output's
pattern becomes much clearer in the second and third residual stages. The
first-convolution/output alignment correlation is 0.578; the final residual
stage/output correlation is 0.991. This is compatible with a coupled network
oscillation that changes as it passes through the architecture.

For the **Transformer**, at update 558 embedding-change cosine is **+0.390**,
first attention-output cosine is **-0.672**, and final output-change cosine is
**-0.920**. Strong output-like reversal is visible after attention while the
embedding still moves forward. Attention's changing parameters and incoming
representations have not yet been separated, so attention is a candidate
boundary for further diagnosis, not an established initiating cause.

Activation RMS can remain nearly constant while successive displacements grow
and reverse. The Transformer first normalization output stays near RMS 1 even
during strong oscillation. This explains why bounded or normalized activation
size alone does not exclude unstable functional motion. Per-token normalization
denominators and local Jacobians were not measured here.

## Read and reproduce

- [Formal method](METHOD.md): fixed probes, tensor/matrix dimensions, outgoing
  differences, raw gradients without differencing, exact temporal Grams, fixed
  SVD coefficients, architectural backtracking and causal limits.
- [ResNet standardized observation](observations/001-resnet-trace.md) and
  [two-page vector figure](figures/01-resnet-instability-trace.pdf).
- [Transformer standardized observation](observations/002-transformer-trace.md)
  and [two-page vector figure](figures/02-transformer-instability-trace.pdf).
- [MLP companion observation](observations/003-mlp-standardized-trace.md) and
  [matching two-page figure](figures/03-mlp-instability-trace.pdf), reusing all
  5,000 examples over steps 448-800 from Analysis 062. Its diagnostic set equals
  its full training objective; raw outputs retain the original squared-error
  contract. No new training is required.
- [ResNet every-stage table](resnet_stages.csv) and
  [Transformer every-stage table](transformer_stages.csv), with complete spectra
  and histories in the corresponding `*_results.json` files.
- [Deferred MLP causal tests](../062-2026-11-07-mlp-gradient-amplification/TODO.md).
- [Run 243 execution and integrity record](../../runs/243-2026-11-07-architecture-instability-tracing/README.md).

`01_reduce.py` produces the numerical tables; `02_figures.py` generates both
ResNet/Transformer PDFs (and the MLP companion after adaptation);
`03_observations.py` regenerates the ResNet/Transformer observations;
`04_verify_mlp_reference.py` checks the common method against the prior MLP
analysis; `05_target_readout.py` extracts correct-label log-probabilities from
captured full outputs before discarding the large derived caches.
`06_mlp_trace.py` adapts the previously audited MLP Grams; reproduce the companion
with that script followed by `02_figures.py --case mlp`.

The ResNet/Transformer figures have the same contract: full/probe loss, gradient/output alignment,
all measured architecture boundaries, displacement magnitude, SVD concentration
and temporal overlap, followed by landmark coefficient histories. The correct-
label readout retains individual examples/tokens; its negative mean reproduces
probe loss. No sample averaging or randomized projection precedes the SVD.

These are descriptive single-seed case studies. Different window lengths and
probe sizes prevent a controlled comparison of intrinsic dimensionality between
architectures. Temporal subspace similarity does not identify the same neuron
directions or isolate output-sensitive hidden components. No architecture,
optimizer or manuscript intervention was performed.
