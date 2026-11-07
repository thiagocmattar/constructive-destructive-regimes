# Input distributions and intervention options

## TLDR

The input is already standardized separately for every pixel-and-channel coordinate. Across the actual 5000 training examples, column means range from -0.04434 to 0.02707 and population standard deviations from 0.98475 to 1.02076. All observed values lie between -2.20743 and 2.62508. Missing marginal normalization or a few enormous input values are not supported explanations here.

However, standardization does not decorrelate columns. The first input covariance mode contains 26.85% of variance, the first ten 64.50%, and the first 100 90.18%. The leading eigenvalue is about 825 times the average eigenvalue. This is a candidate mechanism for directional sensitivity of the first projection, not proof that this geometry causes the spikes or that the actual alternating updates occupy its leading eigenspaces.

We have located the earliest oscillating learned representation, not established a unique initiating cause. First-layer increments come from gradients propagated through the entire network and Adam's evolving state. A selective first-layer learning-rate intervention is the simplest next causal comparison. No intervention was launched in this audit.

## What is a column, and how is it normalized?

The matrix $X$ has 5000 rows (fixed training examples) and 3072 columns (3 RGB channels times 32 rows times 32 image columns). A column is one channel at one spatial position across all examples; it is not an entire image or one class.

For raw byte $p_{ij}$, the preparation code computes

$$
x_{ij}=\frac{p_{ij}/255-\mu_j}{s_j}.
$$

Each $\mu_j,s_j$ is fitted using **all 50000 training images**, with population standard deviation (denominator 50000). The experiment then selects the official first 5000 examples in unchanged order. The test set uses those same training statistics. Thus there are 3072 means and 3072 scales, not just three channel statistics. The saved statistics use HWC ordering; the network input uses CHW ordering, explicitly reconciled in this audit.

The original preprocessing uses FP64 centering and standardization, then stores FP32 inputs. These fixed values enter Flatten and the first linear layer directly. There is no additional sample-wise normalization, BatchNorm, LayerNorm, RMSNorm or whitening in this MLP. The slight mean/std deviations below reflect the 5000-example subset relative to the 50000-example fit; they are not evidence that normalization was skipped.

## Measured distributions

All moments below cover every example of every input coordinate. Population moments are computed in FP64 from the exact captured FP32 values. Entries in the table summarize the distribution of the **3072 column statistics**; they are not pooled input quantiles.

| Statistic per column | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Mean | -0.044343 | -0.003480 | 0.027069 |
| Population standard deviation | 0.984755 | 1.003110 | 1.020761 |
| Skewness | -0.103833 | 0.159136 | 0.588318 |
| Excess kurtosis | -1.358584 | -0.730612 | -0.344256 |

The pooled observed range is [-2.207429, 2.625075], with no values beyond absolute magnitude 3. Standardized does not mean Gaussian: these bounded empirical pixel distributions have coordinate-dependent shapes. The [per-column CSV](../input_column_distributions.csv) records the RGB channel and spatial position, normalization constants, mean, population std, minimum/maximum, 1st/5th/25th/50th/75th/95th/99th percentiles, skewness and excess kurtosis for all3072 coordinates. It summarizes distributions rather than retaining every empirical probability mass.

The original normalization scales, in raw [0,1] units, range from0.225166 to0.315486. There is no tiny normalization denominator in this preparation.

## Correlation survives marginal standardization

Center the actual 5000-example matrix for this diagnostic only:

$$
X_c=X-\mathbf{1}\bar{x}^{\mathsf T},\qquad
C_X=\frac{X_c^{\mathsf T}X_c}{5000}.
$$

$C_X$ is3072-by3072; its eigenvectors are combinations of input columns, not class directions or temporal patterns. This static input eigendecomposition is different from the earlier SVD of representation changes across optimizer steps.

| Input covariance quantity | Value |
|---|---:|
| First eigenmode variance fraction | 26.8549% |
| First ten eigenmodes | 64.4998% |
| First100 eigenmodes | 90.1795% |
| Largest eigenvalue / mean eigenvalue | 824.982 |
| Participation rank $(\sum\lambda)^2/\sum\lambda^2$ | 10.2694 |
| Median correlation of distinct input columns | 0.202741 |
| Median horizontally adjacent, same-channel correlation | 0.925913 |
| Range of horizontally adjacent, same-channel correlations | 0.879143--0.982850 |

Participation rank is an energy-concentration measure, not algebraic rank: all3072 computed covariance eigenvalues are positive. The centered covariance is not exactly the uncentered input metric used by the original model:

$$
\frac{X^{\mathsf T}X}{N}=C_X+\bar{x}\bar{x}^{\mathsf T}.
$$

For an arbitrary first-layer weight displacement $\Delta W$, ignoring bias for this identity,

$$
\frac{\|X\Delta W^{\mathsf T}\|_F^2}{N}
=\operatorname{tr}\!\left(\Delta W\frac{X^{\mathsf T}X}{N}\Delta W^{\mathsf T}\right).
$$

Consequently, equal-size weight increments can have different effects on the representation depending on their input-space direction. The strong covariance concentration motivates examining or intervening on that directional dependence. The factor825 is **not** a measured loss curvature, an Adam stability threshold, or a measured amplification factor for our actual increments. Neither the covariance spectrum nor normalization alone proves the source of the oscillation.

## Proposed interventions, ordered by interpretability

These are proposals, not executed results. Use an authenticated state before a selected rise (for example448), exactly match the full model, Adam moments and clocks, fixed5000 examples, ordering, loss, precision and update horizon, and change only the declared factor. The current controls provide candidate reuse evidence; qualify a replay at the selected fork before reusing them.

1. **Reduce only the first linear layer's LR**, including its bias, to0.5x and0.25x its existing0.001 rate; leave the other layers at0.001 and keep Adam state unchanged. Compare with continued Adam, global0.5x LR, and0.5x LR applied only to the later layers. This asks whether reducing the first layer's movement is especially effective. Different blocks have different parameter counts and increment norms; equal LR multipliers do not match total applied step length, so log that quantity and avoid attributing every difference solely to location.
2. **Suppress first-layer parameter updates temporarily** over the same prespecified window. Keep its gradients and Adam moment clocks updating while setting its applied LR tozero, so this tests movement suppression rather than a mixture of movement and frozen optimizer memory. Later-layer parameters and moments continue normally. If loss spikes remain, first-layer motion is not necessary in that branch. If they disappear, that supports a causal contribution but does not prove the original problem originates exclusively in that layer; suppressing a large block also slows optimization.
3. **Test input covariance through a function-preserving reparameterization**, or through selective damping of actual first-layer increments in the input covariance eigenbasis. These are more specific tests of the geometric hypothesis, but require an explicit optimizer-state contract and a matched amount of damping. Ordinary column standardization again would make only a small change here; it does not address the strong off-diagonal correlations. A regularized whitening arm is a candidate, not a proven remedy.

For the third option, simply replacing $X$ by whitened inputs at a trained state would change predictions immediately and confound subsequent comparisons. For an invertible column transform $A$, define

$$
X'=(X-\mathbf1\mu^{\mathsf T})A,\quad
W'_1=W_1A^{-\mathsf T},\quad
b'_1=b_1+W_1\mu.
$$

Then $X'W_1'^{\mathsf T}+\mathbf1b_1'^{\mathsf T}=XW_1^{\mathsf T}+\mathbf1b_1^{\mathsf T}$ in real arithmetic. An example is regularized whitening $A=(C_X+\epsilon I)^{-1/2}$ for a prespecified positive$\epsilon$. The transformed covariance is not exactly identity when regularized. Numerical prediction parity must be verified before advancing either branch.

This parameter remapping does **not** automatically preserve Adam dynamics. Its coordinatewise second moments do not supply all cross-coordinate history needed to rotate a full second-moment matrix. Centering also mixes weight/bias coordinates. A reparameterization experiment therefore needs explicitly specified state handling and appropriate reset controls, or a new matched training comparison from initialization. It is less direct than the first-layer LR test.

For every intervention, measure loss peak amplitude, mean and end-of-window loss, time to recovery, actual parameter increments per layer, first-linear and output change norms/alignments, and tanh transmission. Smoother curves alone do not demonstrate better optimization. Results remain conditional on this one initialization and selected window until replicated. In addition, project actual first-layer increments onto the input covariance eigenvectors before claiming that the high-variance input directions are the ones responsible for the observed reversal.

## Provenance and verification

This is a static local extension of Analysis062 using existing Run242 inputs and the pinned CIFAR cache; no new run, hypothesis, training, cloud execution or figure was introduced. [Script18](../18_input_distributions.py) generates the [full numerical results](../input_distribution_results.json) and per-column CSV. Source and generator SHA-256 hashes are included. It verifies the complete cached data-file identity, exact equality with Run242 inputs, and reconstructs all5000 normalized images bitwise from the official archived bytes and the stored normalization constants. Independent PyTorch reductions check every column mean/std; covariance eigenvalue trace and squared-Frobenius identities pass; CSV values are read back. Coverage is the fixed training subset, not an estimate of test-set behavior or a multi-seed result.
