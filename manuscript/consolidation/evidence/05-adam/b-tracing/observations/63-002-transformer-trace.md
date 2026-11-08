> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [002-transformer-trace.md](../../../../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/002-transformer-trace.md).

# Transformer / WikiText-2: standardized architecture oscillation trace

## TLDR

The first learned representation has negative consecutive motion in **4/49** updates; final log-probability motion is negative in **12/49**. Their alignment-series correlation is **0.8049**. This localizes observed motion on the fixed probe and does not identify a unique causal origin.

Full-objective loss over the diagnostic window ranges from **0.82348 to 1.69423**; fixed-probe loss ranges from **0.906665 to 1.69516**. Their temporal correlation is **0.9966**; this checks the probe's loss timing, not population-wide activation representativeness. Loss-event details below distinguish an actual excursion from a merely negative cosine. All architecture/optimizer interventions remain TODO.

### What the architectural trace adds

**Strong output reversal is present while the embedding still moves mostly
forward.** At update 558, embedding cosine is 0.3902,
first attention-output cosine is
-0.6721, and final
log-probability cosine is -0.9204. The embedding
reverses on only 4/49 updates; final output reverses on
12/49. This rules out the descriptive picture of an identical
embedding reversal merely passed unchanged to every later representation.
It does not prove that attention is an independent causal source: its parameters
and its inputs both evolve, and its gradients depend on the downstream network.

| Outgoing update | Embedding | First attention output | First block output | Second block output | Log-probabilities |
|---|---:|---:|---:|---:|---:|
| 556 | 0.8171 | -0.0093 | -0.2493 | -0.6597 | -0.6495 |
| 558 | 0.3902 | -0.6721 | -0.8025 | -0.9186 | -0.9204 |
| 559 | -0.0160 | -0.8430 | -0.9111 | -0.9325 | -0.9400 |
| 560 | -0.3847 | -0.9025 | -0.9398 | -0.8813 | -0.8722 |
| 562 | -0.1945 | -0.7523 | -0.8083 | -0.7680 | -0.7898 |
| 563 | 0.4405 | -0.0062 | 0.0613 | -0.1583 | -0.3783 |

The top-10 temporal overlap with output is 0.4615 at
the embedding, 0.7369
at the first attention output, and
0.9823 at the second
block output. The first attention boundary is the first measured location where
the output's strong reversal is already visible while the embedding is still
forward-moving. Internal query/key/value projections and attention weights
were not captured, so this trace cannot localize the transition inside attention.

The embedding's leading uncentered mode changes sign only
2 times and contains 18.02% of energy;
the final output's leading mode changes sign 22 times
and contains 41.65%. The top ten output modes contain
90.83% and 15 modes cover 95%. Embedding motion
includes a substantial forward component; a leading-mode flip count alone
would miss reversals carried by its other modes. Small late-window coefficient
flips should not be equated with the large spike-associated oscillation.

Embedding activation RMS stays within 0.099001 to
0.099870. First-block norm1 output RMS stays within
1.000599
to 1.000899,
despite displacement RMS reaching
0.098917.
Normalization can constrain representation size while the representation still
rotates or reverses between updates. These global values do not test whether a
few token-wise normalization variances become small; that requires denominator
and Jacobian measurements. This model uses LayerNorm, not RMSNorm.

The full loss rises from 0.8431 at state 557 to 1.6942 at 562 (2.010 times the
baseline) and recovers at 596. Correct-label log-probability changes reverse in
18/49 updates and correlate
0.9462 with all-vocabulary
alignment. The result is therefore visible in the token-level loss readout,
not just the many vocabulary coordinates that have tiny predicted probability.

The position-encoding displacement Gram equals 200 times the embedding Gram
to relative error 1.2e-07,
as required by the fixed positional encoding and embedding multiplier sqrt(200).
That boundary rescales motion without introducing its own temporal pattern.

**Next discriminating diagnostic:** decompose the first attention computation
into changing embeddings, local query/key/value/output parameters and attention
mixing. Measure per-token normalization denominators before assigning an
amplification mechanism. These are candidate follow-ups, not executed causal tests.


## Coverage and trajectory identity

- Run 243, unchanged Adam from the saved Run 223 seed-0 checkpoint; LR 0.003, betas (0.9, 0.999), eps 1e-8, no clipping or decay. FP32 native, FP64 diagnostics, TF32 off.
- Actual execution: NVIDIA GeForce RTX 4090, PyTorch 2.11.0+cu130, CUDA 13.0.
- Full objective: 2,088,600 supervised tokens in 59,680 sequences per update; mean cross-entropy over all loss units.
- Fixed probe: 64 sequences, 2240 supervised token positions. No coordinates within that probe are sampled or averaged away. This is not a census of all training activations.
- Displayed outgoing-update indices 552--600; 49 SVD time rows, with one extra preceding/following captured state for adjacent motion and endpoint differences.
- Historical exact agreement: **False**. 58/58 state pairs differ in loss or raw-gradient norm; maximum relative loss discrepancy 0.0027004 (0.2700%). Same saved starting tensors and policy do not establish bitwise historical continuation across GPU models/runtime kernels.

## Architecture and measurement points

Input 64 x 35 fixed token IDs; learned embedding width 200, scaling by sqrt(200), fixed positional encoding; two post-normalization Transformer blocks with two attention heads, feed-forward width 200 and ReLU; linear decoder to 33278 vocabulary coordinates; log-softmax. Dropout is 0. All 2240 probe targets are supervised. Captured attention boundaries exclude internal query/key/value and attention-weight matrices; this is not a complete attention-internals decomposition.

The standard trace contains 21 representation streams plus the full raw-gradient stream. Pre-ReLU tensors are cloned before in-place mutation. Reused module calls have separate call identifiers. Residual-sum inputs and the separate residual branch outputs are distinguished; no local causal contribution is assigned from an output norm ratio alone. Captured fixed inputs are exactly equal across the measurement states; their displacement is zero and motion cosine undefined.

## Standardized landmark table

Rows of a captured representation index fixed images or supervised token positions; trailing dimensions are feature channels/spatial positions or vocabulary coordinates. The SVD matrix instead has optimizer updates as rows and all flattened probe coordinates as columns.

| Stage | One-state captured shape | Negative/defined cosine | Top1 energy | Top10 energy | Alignment r with output | Top10 temporal overlap |
|---|---|---:|---:|---:|---:|---:|
| `encoder.call0.output` | [2240, 200] | 4/49 | 18.02% | 82.10% | 0.8049 | 0.4615 |
| `pos_encoder.call0.output` | [2240, 200] | 4/49 | 18.02% | 82.10% | 0.8049 | 0.4615 |
| `transformer_encoder.layers.0.self_attn.call0.output` | [2240, 200] | 9/49 | 39.31% | 87.66% | 0.9509 | 0.7369 |
| `transformer_encoder.layers.0.norm1.call0.output` | [2240, 200] | 8/49 | 43.70% | 88.85% | 0.9593 | 0.7370 |
| `transformer_encoder.layers.0.norm2.call0.output` | [2240, 200] | 8/49 | 47.72% | 90.24% | 0.9675 | 0.7624 |
| `transformer_encoder.layers.1.self_attn.call0.output` | [2240, 200] | 13/49 | 49.64% | 90.83% | 0.9820 | 0.9667 |
| `transformer_encoder.layers.1.norm2.call0.output` | [2240, 200] | 12/49 | 45.17% | 89.76% | 0.9902 | 0.9823 |
| `decoder.call0.output` | [2240, 33278] | 12/49 | 44.15% | 89.72% | 0.9941 | 0.9900 |
| `output.log_probabilities` | [2240, 33278] | 12/49 | 41.65% | 90.83% | 1.0000 | 1.0000 |
| `output.correct_label_log_probability` | [2240] | 18/49 | 36.23% | 93.49% | 0.9462 | 0.8603 |
| `raw_gradient` | [13828478] | 15/49 | 64.40% | 98.52% | 0.8789 | 0.8622 |

The raw-gradient row decomposes $g_t$, not gradient differences; its RMS column in the accompanying CSV is gradient RMS. All other rows decompose outgoing activation changes. The reference output is log-probabilities; raw logits remain a separate captured stage. A common class-logit shift can affect raw-logit geometry while canceling from probabilities.

## Loss excursions in this window

The unchanged Analysis 036 recovery detector is applied to the new full-objective continuation, not copied from the historical worker. Events whose peaks lie inside the diagnostic window are:

```json
[
  {
    "before_state": 557,
    "onset_state": 558,
    "peak_state": 562,
    "recovery_state": 596,
    "end_state": 596,
    "right_censored": false,
    "ratio": 2.009557559770217
  }
]
```

Indices are completed optimizer updates. The increment entering a peak $p$ is labeled $p-1$. Right-censored events are not claimed recovered. Event timing was selected from the original control before inspecting these representation diagnostics, so this remains a retrospectively selected window rather than an independent discovery sample.

## Algorithm, figures and limits

Apply [the common method](63-METHOD.md): fixed probes; unchanged trajectory; outgoing differences; exact uncentered FP64 temporal Grams; fixed-window SVD; signed coefficients; backward architectural comparison of cosine, magnitude and temporal subspaces. No per-step changing SVD basis, sample averaging, random sketch or gradient differencing is used. Temporal-coefficient signs are oriented once by the largest absolute temporal coefficient. Correlations use dependent time points and do not supply independent causal evidence.

**Figure 2, page 1:** full training/probe loss, raw-gradient/output alignment, every captured representation's consecutive-change cosine, displacement coordinate RMS divided by that stage's own window maximum, and top1/top10 SVD energy with top10 output temporal overlap. Row-wise scaling in the RMS heatmap shows timing, not absolute cross-layer magnitude. Raw zero-input motion is stated separately; no zero-cosine curve is invented.

**Figure 2, page 2:** architecture landmarks with consecutive cosine, absolute displacement coordinate RMS and the leading fixed SVD coefficient divided by the square root of the number of captured coordinates. Independent layer bases do not identify the same neuron directions; coefficient signs across different layers are arbitrary after their separate fixed orientation.

Strong resemblance across layers can reflect coupled updates driven by the same downstream loss. Even an earliest-layer reversal is not a causal initiating source. Probe selection, one seed, short selected windows, near-degenerate SVD modes and numerical hardware differences limit extrapolation. Normalization/attention mixing cannot be explained solely by elementwise secant arguments. Follow-up causal tests and deeper attention/normalization accounting are separate from this first-pass trace.

## Sources and verification

- [Numerical results and source hashes](../data/transformer_results.json); [every-stage table](../data/transformer_stages.csv).
- [Standardized PDF](../figures/02-transformer-instability-trace.pdf).
- Scripts 01/02/03 in this analysis generate reductions, figures and this observation; Run 243 owns capture and source identities.
- Observer verifies plain-forward parity, unchanged gradients/RNG/buffers and fixed probe inputs. Independent FP64 contractions check all Gram diagonal norms and adjacent products. Local reduction verifies PSD, trace, full Gram reconstruction and full-gradient boundary norms/cosines. Model checkpoints, input/target identities, exact Grams and raw-cache hashes are retained with transfer verification. Dense activation caches are regenerable from these snapshots and the pinned computation contract.

Rendering and final artifact integrity checks are recorded in the figure and execution verification records. No manuscript edits or causal interventions are part of this result.
