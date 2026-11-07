# Gradient coherence, prediction reversals, and intra-batch dispersion

Analysis062 / Run237. Explanation of the measured quantities, with plain-text
formulas for reading in a terminal. No new training experiment is involved.

The main distinction: absolute gradient dispersion increases at the peaks,
while dispersion relative to the mean gradient decreases. Both can happen
simultaneously.

## 1. Gradient coherence

For each training example i, define its loss and raw parameter gradient:

```text
loss_i = 0.5 * ||f_theta(x_i) - onehot(y_i)||^2
g_i    = gradient of loss_i with respect to all model parameters
g_mean = (1/N) * sum_i g_i
E      = (1/N) * sum_i ||g_i||^2

coherence C = ||g_mean||^2 / E
```

Here `||v||` denotes the Euclidean vector norm. All parameter coordinates are
included: the three weight matrices and their biases. N = 5,000, covering the
entire fixed training batch. Physical batches of 1,000 are only used to perform
the computation; the statistic is aggregated over all 5,000 examples.

These are raw gradients, before Adam's transformation. Each g_i is the gradient
of one example's loss, without a 1/N factor. The mean gradient includes that
factor.

- C = 0 means the gradients cancel exactly, provided E is nonzero.
- C = 1 means all gradients are identical.
- Increasing C means a larger fraction of individual-gradient energy survives
  averaging.

This is not the average pairwise cosine, nor a cosine between consecutive
training steps. Coherence depends on gradient magnitudes as well as directions.
Even gradients with identical directions can have C below one if their
magnitudes differ. A zero E would make C undefined; all measured states here
have positive E.

The denominator is measured using per-example backpropagation. At a linear
layer, let a_i be the input features and delta_i the derivative of the example
loss with respect to the layer's preactivation:

```text
weight gradient for example i = outer_product(delta_i, a_i)
squared weight-gradient norm  = ||delta_i||^2 * ||a_i||^2
squared bias-gradient norm    = ||delta_i||^2
```

Sum these contributions over all layers, then average over examples. This
avoids materializing a full parameter-gradient vector for every example while
still measuring its squared norm. It is not inferred from the batch gradient
alone. Analytically aggregated gradients were checked against native autograd.

Implementation: [Run237 diagnostics, observe()](../../../runs/237-2026-11-07-mlp-gradient-amplification/diagnostics.py).

## 2. Consecutive prediction changes

At state t, concatenate the 10 raw network outputs for every example, keeping
the same 5,000 examples in the same order:

```text
F_t = concatenate(f_theta_t(x_1), ..., f_theta_t(x_5000))

F_t contains 50,000 values.
delta_F_t = F_(t+1) - F_t

prediction-reversal statistic B_t = cosine(delta_F_t, delta_F_(t-1))

cosine(u, v) = dot(u, v) / (||u|| * ||v||)
```

A value near -1 means the latest output displacement nearly reverses the
preceding displacement. It compares changes in outputs, not predicted class
labels or the output vectors themselves. These are raw outputs of the final
linear layer, not softmax probabilities.

For the update entering peak 474:

```text
cosine(F_474 - F_473, F_473 - F_472) = -0.98505
```

For the update entering peak 522:

```text
cosine(F_522 - F_521, F_521 - F_520) = -0.98999
```

This is a global statistic over all examples and output coordinates. Large
changes contribute more strongly. It does not imply that every example or
coordinate reverses equally, and is not an average of per-example cosines.

The retained arrays contain residuals, `f_theta(x_i) - onehot(y_i)`. Since labels
are fixed, subtracting residuals at consecutive states gives exactly the same
prediction changes as subtracting outputs. The plotted value at state s
compares the incoming change s-1 -> s with the outgoing change s -> s+1.
All measured changes have nonzero norm.

Implementation: [Analysis062 report generator](../01_report.py), the `flat`,
`dr`, and `cosine` reductions.

## 3. Does intra-batch gradient dispersion increase?

Define absolute dispersion around the mean gradient:

```text
V = (1/N) * sum_i ||g_i - g_mean||^2
  = E - ||g_mean||^2

absolute RMS spread = sqrt(V)
relative RMS spread = sqrt(V) / ||g_mean||
```

V is the trace of the population covariance, using divisor N rather than N-1.
It is the average squared distance of the individual gradients from their mean.
The following values are reduced from the retained full-example measurements:

| Quantity | State 465 | Peak 474 |
|---|---:|---:|
| Mean gradient norm | 0.488 | 2.018 |
| Mean individual squared-gradient norm E | 16.794 | 44.431 |
| Absolute dispersion V | 16.555 | 40.360 |
| Absolute RMS spread sqrt(V) | 4.069 | 6.353 |
| Relative RMS spread | 8.335 | 3.149 |
| Coherence C | 1.42% | 9.16% |

Absolute RMS spread increases about 56%, while the mean gradient norm increases
about 313%. The gradients spread farther apart in absolute parameter-gradient
space, but their collective mean grows much faster.

The connection between coherence and relative dispersion is exact:

```text
V / ||g_mean||^2 = 1/C - 1

relative RMS spread = sqrt(1/C - 1)
```

The next large event shows the same distinction:

| Quantity | State 512 | Peak 522 |
|---|---:|---:|
| Mean gradient norm | 0.410 | 2.184 |
| Absolute RMS spread | 3.699 | 6.643 |
| Relative RMS spread | 9.032 | 3.042 |
| Coherence C | 1.21% | 9.75% |

Thus the measured answer is:

- Absolute gradient dispersion increases.
- Dispersion relative to the mean gradient decreases.
- Angular dispersion among normalized gradients `g_i / ||g_i||` was not
  measured directly. Coherence alone cannot establish how it changes, because
  coherence also depends on gradient magnitudes.

## Sources and limits

The numerical inputs are Run237's retained
`artifacts/spark/payload/worker/backward.jsonl`, fields `gradient_norm` and
`unit_gradient_mean_squared_norm`. Dispersion is obtained by the exact variance
identity above; no new gradients or training steps are required. Prediction
cosines come from `output_vectors.pt` and are also recorded for the selected
events in [results.json](../results.json).

The run's calculations include all 5,000 examples and all parameters. Diagnostic
vectors use FP32, with norm/energy reductions in FP64. The variance identity
therefore holds to the numerical precision of the retained measurements.
These are descriptive findings from one seed and the selected control window,
not independent causal evidence that increasing coherence causes a loss spike.

See the [main observation](001-gradient-amplification.md) for the experiment,
verification, figures and mechanistic interpretation.
