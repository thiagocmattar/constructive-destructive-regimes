# Observation010: model-size overlays for Adam adaptive directions

The user requested the same split-range figure using Adam quantities and then
explicitly selected d. The [new PDF](../figures/training_size_adam_direction_split_n5_n21.pdf)
shows natural-log loss change, projected direction change q_d, adjacent adaptive
direction cosine rho_d and adaptive direction L2 norm. All three model sizes
share each panel. Columns cover updates 1-100 with complete trailing N=5 means
and 100-400 with N=21. The prior raw-gradient figure and its sources are preserved.
There is no new training or cloud execution.

## Sources and comparison

Pythia-14M/31M/70M use measurement replays Runs017/018/019, respectively, seed1235.
Read accepted `full-pass/events.jsonl` from Run017 attempt-01 and Runs018/019
attempt-02. Per-update `adaptive.adjacent_cosine` and `adaptive.direction_norm`
are dense-vector scalar reductions, not temporal CountSketch estimates.
Original Analysis005 source hashes, accepted retrieval receipts, artifact inventory
sizes/hashes, resolved configurations and input identities are checked again.
The initial/midpoint/final recovery states remain locally retained; the generator
does not need to load checkpoint tensors or reconstruct missing moment histories.

Across sizes, seed, MiniPile token order, batch/accumulation, update coverage,
AdamW betas (0.9,0.95), epsilon 1e-8, global clipping at 1, LR schedule and decay
recipe are matched. Architecture, parameter count and initial tensors vary.
Original model/data/tokenizer revisions and hashes remain in the resolved configs
and input identities linked through the manifest. This is one seed, not three
independent replications. Hardware/precision details follow observation003.

All 712 accepted updates per model are present. This figure selects 1-400, all
full batches of 1024 packed 2048-token sequences, with token-mean next-token loss
over each actual optimizer update. The final partial batch at update712 is
outside the visible ranges, as in the preceding versions. Update400 is retained.
No validation loss or fixed-minibatch evaluation is substituted for training loss.

## Definitions and reduction

Let m_t and v_t denote the actual FP32 Adam moments updated with the globally
clipped gradient at accepted update t. Bias correction and direction arithmetic
are evaluated in FP64 after the optimizer step, over all unique trainable
parameters in the recorded order:

```text
d_t = [m_t / (1-beta1^t)] / [sqrt(v_t / (1-beta2^t)) + epsilon]
rho_d[t] = <d_(t-1), d_t> / (||d_(t-1)|| ||d_t||)
q_d[t] = 1 - rho_d[t] * ||d_t|| / ||d_(t-1)||
       = -<d_t-d_(t-1), d_(t-1)> / ||d_(t-1)||^2
loss_change[t] = ln(L_t) - ln(L_(t-1))
```

d excludes LR, decoupled weight decay and parameter-subtraction rounding; it is
distinct from the first moment m and actual parameter displacement. The initial
zero-LR update still advances the Adam state and has a nonzero direction norm.
q_d is a dimensionless normalized projection of direction change. Substituting d
into this identity does not establish the plain-SGD eta*kappa curvature relation;
the second row is therefore explicitly labelled q_d rather than eta*kappa.

First-pair loss change, q_d and rho_d are undefined at t=1, not zero. All measured
norms here are positive, and no zero-denominator substitution is needed. Each
scalar, including q_d computed from unsmoothed adjacent records, is then averaged
arithmetically with a complete trailing window. Smoothing precedes range selection
and spans the update100 boundary. Norms are averaged before applying the log axis:
this is a mean of norms, not a norm of mean directions. The loss row starts with
raw log differences, never with already-smoothed means.

Left-column norm means begin at update5; other means begin at6. Right-column
means are defined at every update100-400. Update100 appears twice, using different
N. The loss means are exactly equal to the prior raw-gradient figure, including
NaN placement, and satisfy mean_N[delta ln L_t] = ln(L_t/L_(t-N))/N.

All model curves are solid blue/orange/green. Each column has its own Y limits;
the norm row uses log Y. Cosine axes are narrowly scaled around their observed
positive values to reveal size differences. Reference levels outside the data
range do not expand the direction-change/cosine axes.

## Descriptive differences and limits

Over displayed updates200-400, the arithmetic averages of the N=21 plotted
means are:

| Model | Mean log-loss change | Mean q_d | Mean rho_d | Mean direction norm |
| --- | ---: | ---: | ---: | ---: |
| 14M | -0.00042345 | 0.03251 | 0.96810 | 1301.86 |
| 31M | -0.00047645 | 0.05221 | 0.94822 | 1577.65 |
| 70M | -0.00102676 | 0.08054 | 0.91962 | 2005.95 |

These averages describe overlapping smoothed windows and are not independent
samples or uncertainty estimates. In this trajectory, larger models have larger
direction norms, lower positive adjacent alignment and larger projected direction
change over this late displayed interval. The 70M loss also falls faster on
average there. Early size ordering of alignment is less uniform. No plotted
smoothed adjacent direction cosine is negative in either column.

Norms are unnormalized whole-vector norms across different parameter counts;
part of a size difference can arise from dimensionality and Adam preconditioning.
Larger direction norms do not imply larger raw gradients, larger first moments,
or a causal explanation of the loss decrease. Different minibatches, smoothing,
time-varying second moments, LR and decay prevent reading this association as a
same-objective descent, measured-curvature, stability or generalization result.
There are no seed error bars; the source reductions' finite precision and the
N-dependent temporal blur remain. No manuscript claim is added.

**Caption.** Adam adaptive-direction dynamics for Pythia-14M/31M/70M on MiniPile,
seed1235. Columns display updates1-100 (N=5) and100-400 (N=21), with complete
trailing arithmetic means for every metric. Rows show adjacent natural-log
training-loss change, normalized projected direction change q_d, adjacent
direction cosine rho_d and direction L2 norm. d is the bias-corrected,
preconditioned Adam direction excluding LR and decay. Model sizes share each
panel as solid coloured curves. Y limits vary by column; norms use log Y.
Update100 is included with both smoothing windows. q_d is descriptive direction
change, not measured directional curvature.

## Reproduction and checks

Generate with `python 16_plot_adam_direction.py`. The
[CSV](../training_size_adam_direction_split_n5_n21_data.csv) contains 1,203 rows
keyed by size, column, window and update, with four plotted means and blanks for
undefined pairs. The [manifest](../training_size_adam_direction_split_n5_n21_manifest.json)
records source/output hashes, capture definitions, curve ranges and array hashes.
All 4,755 finite means are checked against independent sliced-window means and
the Matplotlib line arrays. All 1,188 finite loss points and their undefined
placements match the prior figure exactly. Log telescoping, CSV roundtrip,
accepted coverage, full-batch selection and preserved-file hashes also pass.

Render with Poppler to `scratch/training_size_adam_direction_split_n5_n21-1.png`,
inspect all eight panels, then record acceptance using `--accept-reviewed`.
