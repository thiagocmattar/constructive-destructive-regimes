# Gradient coherence, prediction reversals, and intra-batch dispersion

Analysis062 / Run237. Explanation of the measured quantities, with LaTeX
equations for rendered Markdown. No new training experiment is involved.

The main distinction: absolute gradient dispersion increases at the peaks,
while dispersion relative to the mean gradient decreases. Both can happen
simultaneously.

## 1. Gradient coherence

For each training example $i$, define its loss and raw parameter gradient:

$$
\ell_i(\theta)=\frac12\left\|f_\theta(x_i)-\operatorname{onehot}(y_i)\right\|^2,
\qquad g_i=\nabla_\theta\ell_i(\theta).
$$

$$
\bar g=\frac1N\sum_{i=1}^N g_i,
\qquad E=\frac1N\sum_{i=1}^N\|g_i\|^2,
\qquad \boxed{C=\frac{\|\bar g\|^2}{E}}.
$$

Here $\|v\|$ denotes the Euclidean vector norm. All parameter coordinates are
included: the three weight matrices and their biases. $N=5000$, covering the
entire fixed training batch. Physical batches of 1,000 are only used to perform
the computation; the statistic is aggregated over all 5,000 examples.

These are raw gradients, before Adam's transformation. Each $g_i$ is the gradient
of one example's loss, without a $1/N$ factor. The mean gradient includes that
factor.

- $C=0$ means the gradients cancel exactly, provided $E$ is nonzero.
- $C=1$ means all gradients are identical, provided $E$ is nonzero.
- Increasing $C$ means a larger fraction of individual-gradient energy survives
  averaging.

This is not the average pairwise cosine, nor a cosine between consecutive
training steps. Coherence depends on gradient magnitudes as well as directions.
Even gradients with identical directions can have $C$ below one if their
magnitudes differ. A zero $E$ would make $C$ undefined; all measured states here
have positive $E$.

The denominator is measured using per-example backpropagation. At a linear
layer, let $a_i$ be the input features and $\delta_i$ the derivative of the example
loss with respect to the layer's preactivation:

$$
\nabla_W\ell_i=\delta_i a_i^\top,
\qquad \|\nabla_W\ell_i\|_F^2=\|\delta_i\|^2\|a_i\|^2,
\qquad \|\nabla_b\ell_i\|^2=\|\delta_i\|^2.
$$

The subscript $F$ denotes the Frobenius norm of a matrix.

Sum these contributions over all layers, then average over examples. This
avoids materializing a full parameter-gradient vector for every example while
still measuring its squared norm. It is not inferred from the batch gradient
alone. Analytically aggregated gradients were checked against native autograd.

Implementation: [Run237 diagnostics, observe()](../../../runs/237-2026-11-07-mlp-gradient-amplification/diagnostics.py).

## 2. Consecutive prediction changes

At state $t$, concatenate the 10 raw network outputs for every example, keeping
the same 5,000 examples in the same order:

$$
F_t=\operatorname{concat}_{i=1}^{5000} f_{\theta_t}(x_i)
\in\mathbb R^{50000},
\qquad \Delta F_t=F_{t+1}-F_t.
$$

The prediction-reversal statistic is

$$
\boxed{B_t=\cos(\Delta F_t,\Delta F_{t-1})}
=\frac{\langle\Delta F_t,\Delta F_{t-1}\rangle}
{\|\Delta F_t\|\,\|\Delta F_{t-1}\|}.
$$

A value near $-1$ means the latest output displacement nearly reverses the
preceding displacement. It compares changes in outputs, not predicted class
labels or the output vectors themselves. These are raw outputs of the final
linear layer, not softmax probabilities.

For the update entering peak 474:

$$
B_{473}=\cos(F_{474}-F_{473},F_{473}-F_{472})\approx-0.98505.
$$

For the update entering peak 522:

$$
B_{521}=\cos(F_{522}-F_{521},F_{521}-F_{520})\approx-0.98999.
$$

This is a global statistic over all examples and output coordinates. Large
changes contribute more strongly. It does not imply that every example or
coordinate reverses equally, and is not an average of per-example cosines.

The retained arrays contain residuals, $f_\theta(x_i)-\operatorname{onehot}(y_i)$. Since labels
are fixed, subtracting residuals at consecutive states gives exactly the same
prediction changes as subtracting outputs. The plotted value at state $s$
compares the incoming change $s-1\to s$ with the outgoing change $s\to s+1$.
All measured changes have nonzero norm.

Implementation: [Analysis062 report generator](../01_report.py), the `flat`,
`dr`, and `cosine` reductions.

## 3. Does intra-batch gradient dispersion increase?

Define absolute dispersion around the mean gradient:

$$
V=\frac1N\sum_{i=1}^N\|g_i-\bar g\|^2
=E-\|\bar g\|^2.
$$

$$
\text{Absolute RMS spread}=\sqrt V,
\qquad
\text{Relative RMS spread}=\frac{\sqrt V}{\|\bar g\|}.
$$

$V$ is the trace of the population covariance, using divisor $N$ rather than $N-1$.
It is the average squared distance of the individual gradients from their mean.
The following values are reduced from the retained full-example measurements:

| Quantity | State 465 | Peak 474 |
|---|---:|---:|
| Mean gradient norm $\lVert\bar g\rVert$ | 0.488 | 2.018 |
| Mean individual squared-gradient norm $E$ | 16.794 | 44.431 |
| Absolute dispersion $V$ | 16.555 | 40.360 |
| Absolute RMS spread $\sqrt V$ | 4.069 | 6.353 |
| Relative RMS spread | 8.335 | 3.149 |
| Coherence $C$ | 1.42% | 9.16% |

Absolute RMS spread increases about 56%, while the mean gradient norm increases
about 313%. The gradients spread farther apart in absolute parameter-gradient
space, but their collective mean grows much faster.

The connection between coherence and relative dispersion is exact:

$$
\boxed{\frac{V}{\|\bar g\|^2}=\frac1C-1},
\qquad
\frac{\sqrt V}{\|\bar g\|}=\sqrt{\frac1C-1}.
$$

These relative quantities require $\|\bar g\|>0$, as holds at all measured states.

The next large event shows the same distinction:

| Quantity | State 512 | Peak 522 |
|---|---:|---:|
| Mean gradient norm $\lVert\bar g\rVert$ | 0.410 | 2.184 |
| Absolute RMS spread $\sqrt V$ | 3.699 | 6.643 |
| Relative RMS spread | 9.032 | 3.042 |
| Coherence $C$ | 1.21% | 9.75% |

Thus the measured answer is:

- Absolute gradient dispersion increases.
- Dispersion relative to the mean gradient decreases.
- Angular dispersion among normalized nonzero gradients $g_i/\|g_i\|$ was not
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
