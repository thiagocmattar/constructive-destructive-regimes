# Tracing the oscillation backward into the first hidden representation

## TLDR

**Yes: the first hidden representation already exhibits almost the same oscillatory structure as the second hidden representation and the outputs.** Both hidden layers have200 activation columns, evaluated on the same5000 sample rows. This similarity is measured across updates; it is not an assumption that neuron columns correspond between layers.

- The first-hidden change alignment is negative at **350 of353 states**. Its alignment time course correlates at **0.99590** with the second hidden layer and **0.99155** with the outputs.
- Its first ten fixed SVD modes capture **95.39%** of change energy, versus **97.40%** in the second hidden layer and **98.16%** in the outputs. It has a somewhat broader tail, but the dominant temporal patterns are very similar.
- The first and second hidden layers' top-ten temporal subspaces have overlap **0.99861**, where1 means identical subspaces. The first-layer temporal basis captures **97.374%** of second-layer change energy, nearly its own optimal top-ten coverage of97.399%.
- The **second tanh attenuates rather than amplifies displacement norm**: it retains86.84--89.30% of the preactivation displacement norm. Alignment before and after it correlates at0.99990. It reduces amplitude while largely preserving the back-and-forth pattern.
- In the additive accounting of second-hidden motion after tanh, changes arriving from the first hidden representation contribute **82.70%** by signed projection; changes in second-layer weights contribute **17.36%**. Bias and interaction are small. These are trajectory-accounting shares, not causal percentages or fractions of loss explained.
- First-hidden activation coordinate RMS stays between **0.89244 and0.89606**, even while its displacement amplitude rises near loss peaks. Bounded, nearly constant activation magnitude does not prevent oscillatory representation changes.

This locates the oscillation earlier in the network. It does not identify a unique initiating cause in the first layer's parameters or optimizer state.

Figures:

- [First hidden, second preactivation, second hidden and output SVD comparison](../figures/09-first-hidden-svd-comparison.pdf).
- [Transmission through the second linear layer and tanh](../figures/10-first-to-second-hidden-transmission.pdf).

## 1. Which matrices are being measured?

The network is Flatten(3072), Linear(3072,200), tanh, Linear(200,200), tanh, Linear(200,10). Write the representation pathway as

$$
H^{(1)}_t
\xrightarrow{\text{second linear layer}}
Z^{(2)}_t
\xrightarrow{\tanh}
H^{(2)}_t
\xrightarrow{\text{output linear layer}}
F_t.
$$

| Matrix at state t | Rows | Columns | What one entry means |
|---|---:|---:|---|
| $H^{(1)}_t$ | 5000 samples | 200 first-hidden activations | Activation after the first tanh for one sample and one first-hidden neuron |
| $Z^{(2)}_t$ | Same5000 samples | 200 second-layer preactivations | Value after the second linear layer but before its tanh |
| $H^{(2)}_t$ | Same5000 samples | 200 second-hidden activations | Activation after the second tanh |
| $F_t$ | Same5000 samples | 10 class outputs | Raw output for one sample and class |

Each operation acts on columns independently for each sample row. It does not mix samples. The two hidden matrices have equal widths, but their columns represent different neurons. We therefore do not calculate a direct cosine between flattened $\Delta H^{(1)}$ and $\Delta H^{(2)}$ as if their coordinates were interchangeable. Instead we compare each space's own temporal alignment, its SVD spectrum, and its temporal patterns in a shared update coordinate system.

$Z^{(2)}$ and $H^{(2)}$ do have corresponding coordinates, because tanh acts elementwise. Their relationship can be checked directly.

## 2. Coverage, alignment and SVD definitions

Use the unchanged Adam control and every state$t=448,\ldots,800$. State$t$ means after$t$ completed updates; the outgoing displacement is

$$
\Delta X_t=X_{t+1}-X_t,
\qquad
\rho_t^X=\cos\bigl(\operatorname{vec}(\Delta X_t),\operatorname{vec}(\Delta X_{t-1})\bigr).
$$

The same5000 examples, in the same order, are included at every state. Each cosine concatenates all samples and coordinates; it is neither a cosine of sample means nor an arithmetic mean of sample-wise cosines. All norms are nonzero in the measured window. A zero norm would give undefined/null, without epsilon regularization.

For each space, stack outgoing differences as rows and fit one uncentered SVD across the entire window. The matrix has353 time rows and1,000,000 sample/neuron columns for each of$H^{(1)},Z^{(2)},H^{(2)}$, and50,000 sample/class columns for outputs. No example averaging, temporal centering, row normalization, smoothing, subsampling or lag search is applied. Stored values are native FP32; differences and diagnostic reductions use FP64.

State447 supplies the incoming displacement for the cosine at448; state801 supplies the outgoing displacement at800. Neither becomes an additional fitted time row. Loss$L_t$ is before update$t\to t+1$. A coefficient at473 therefore describes the update entering the loss peak at474.

The basis is fixed within each space, with one deterministic sign per pattern. Plot coefficients are divided by$\sqrt{5000}$ for a per-example vector-RMS scale. This common scaling does not average examples or change reversals. Raw gradients remain undifferenced, as previously corrected by the user; this extension adds representation changes only.

## 3. Alignment and spectral evidence

| Alignment time courses | Pearson correlation |
|---|---:|
| First hidden / second preactivation | 0.996327 |
| First hidden / second hidden | 0.995896 |
| First hidden / outputs | 0.991551 |
| Second preactivation / second hidden | 0.999901 |
| Second hidden / outputs | 0.998471 |

First-hidden alignment is negative at350/353 states; the other three are negative at351/353. These are repeated output-space reversals already present in the first hidden representation, although not every sample or neuron must reverse individually.

| Representation changes | First-mode energy | First-ten energy | Modes for99% | Leading coefficient sign reversals |
|---|---:|---:|---:|---:|
| First hidden | 24.20% | 95.39% | 35 | 308/352 |
| Second preactivation | 25.41% | 97.32% | 17 | 306/352 |
| Second hidden | 25.58% | 97.40% | 16 | 306/352 |
| Outputs | 25.17% | 98.16% | 11 | 306/352 |

The first hidden layer has more small-energy temporal modes. Its dominant motion is nevertheless concentrated, and much of that same temporal structure survives downstream. This is a statement about the observed trajectory's temporal rank, not a count of active neurons or a general bound on representation capacity.

Selected states illustrate that the reversal is already strong earlier:

| State t | Loss | First-hidden cosine | Second-hidden cosine | Output cosine | First-hidden change norm |
|---|---:|---:|---:|---:|---:|
| 465 | 0.02819 | -0.74387 | -0.78493 | -0.81247 | 13.890 |
| 473 | 0.04770 | -0.96827 | -0.97900 | -0.98505 | 48.222 |
| 474 | 0.05367 | -0.96347 | -0.97634 | -0.98381 | 50.448 |
| 485 | 0.02772 | -0.14740 | -0.20197 | -0.21151 | 17.964 |
| 512 | 0.02306 | -0.77946 | -0.80981 | -0.82478 | 12.349 |
| 521 | 0.04891 | -0.97476 | -0.98484 | -0.98999 | 53.640 |
| 522 | 0.05506 | -0.96278 | -0.98158 | -0.99021 | 52.376 |
| 529 | 0.02268 | 0.09471 | 0.09503 | 0.11944 | 22.780 |

Table norms concatenate all1,000,000 hidden coordinates. Figure10 divides them by$\sqrt{5000}$; neither is the activation coordinate RMS.

## 4. Are these really similar temporal patterns, or just similar spectra?

Similar singular values alone would not show that the modes occur at the same times. We therefore compare the **left singular vectors**, which live in the same353-dimensional update space for every representation.

Let $U_{1,10}$ and $U_{2,10}$ contain the ten leading temporal vectors of the first and second hidden change matrices. Each has shape353-by10. Compute

$$
\mathrm{overlap}(1,2)=\frac{\|U_{1,10}^{\top}U_{2,10}\|_F^2}{10}.
$$

This is the mean squared cosine of the principal angles between the two temporal subspaces. It ranges from0 to1 and is invariant to reordering, sign flips, or orthogonal rotation of the basis within either subspace. It does not require identifying a first-hidden neuron with a second-hidden neuron.

| Temporal subspaces | Overlap |
|---|---:|
| First hidden / second preactivation | 0.998447 |
| First hidden / second hidden | 0.998609 |
| First hidden / output | 0.994599 |
| Second preactivation / second hidden | 0.999975 |

For another interpretation, project the second-hidden change matrix onto the first-hidden temporal basis:

$$
\frac{\|U_{1,10}^{\top}D_{H^{(2)}}\|_F^2}{\|D_{H^{(2)}}\|_F^2}=97.3741\%.
$$

Using the second-hidden layer's own optimal ten temporal modes yields97.3986%. Likewise, the first-hidden temporal basis captures98.0643% of output-change energy, compared with the output basis's98.1569%. This is stronger evidence of shared temporal structure than comparing spectra alone. It is still descriptive and fitted using the full window, not an online prediction or causal test.

## 5. How first-hidden changes pass through the next layer

With $W_{2,t}\in\mathbb R^{200\times200}$ and $b_{2,t}\in\mathbb R^{200}$,

$$
Z^{(2)}_t=H^{(1)}_tW_{2,t}^{\top}+\mathbf1b_{2,t}^{\top}.
$$

The second linear layer combines200 first-hidden columns into200 new columns, separately for each sample row. The output layer discussed previously combines200 columns into10. We must not automatically assign the second linear layer the earlier10/190 visible/null split: it is a different matrix with a different shape.

The exact preactivation increment is

$$
\Delta Z^{(2)}_t=
\underbrace{\Delta H^{(1)}_tW_{2,t}^{\top}}_{A_t:\ \text{first-hidden motion}}
+\underbrace{H^{(1)}_t\Delta W_{2,t}^{\top}}_{B_t:\ \text{second-layer weight motion}}
+\underbrace{\Delta H^{(1)}_t\Delta W_{2,t}^{\top}}_{C_t:\ \text{interaction}}
+\underbrace{\mathbf1\Delta b_{2,t}^{\top}}_{E_t:\ \text{bias motion}}.
$$

Every term has5000 rows and200 columns. Thus we can distinguish motion already present in the incoming representation from motion due to changing the next layer's parameters.

To handle tanh without replacing a finite step by a first-order approximation, define the **elementwise secant slope**

$$
S_{t,ij}=
\frac{\tanh(Z^{(2)}_{t+1,ij})-\tanh(Z^{(2)}_{t,ij})}
{Z^{(2)}_{t+1,ij}-Z^{(2)}_{t,ij}}.
$$

If the denominator is exactly zero, use its continuous derivative limit$1-\tanh^2(Z^{(2)}_{t,ij})$. For finite real inputs, $0\le S_{t,ij}\le1$. Then

$$
\Delta H^{(2)}_t=S_t\odot\Delta Z^{(2)}_t
=S_t\odot A_t+S_t\odot B_t+S_t\odot C_t+S_t\odot E_t.
$$

This identity uses the observed endpoints; $S_t$ depends on the whole observed displacement. It is not a counterfactual experiment in which each term changes alone. Stored FP32 forward values introduce small measured residuals, which are retained explicitly rather than forced to zero.

For a contribution$T_t$ to a measured change$Y_t$, report the signed projection share

$$
s_T=\frac{\sum_t\langle Y_t,T_t\rangle_F}{\sum_t\|Y_t\|_F^2}.
$$

These shares sum to1 including residuals, but are not nonnegative energy partitions. Terms can cancel. Results are:

| Contribution | Before second tanh: share of observed $\Delta Z^{(2)}$ | After second tanh: share of observed $\Delta H^{(2)}$ |
|---|---:|---:|
| First-hidden motion through existing second-layer weights | 82.8247% | 82.6974% |
| Second-layer weight changes at existing first-hidden features | 17.2348% | 17.3628% |
| Interaction | -0.06419% | -0.06610% |
| Second-layer bias changes | 0.00469% | 0.00585% |

Most second-hidden motion is therefore transmitted from the first-hidden representation, while second-layer weight changes make a meaningful smaller contribution. These percentages refer to motion in the second representation, not to final output motion or loss. They should not be multiplied by the earlier output-layer attribution percentages: those use a different projection and denominator.

## 6. What tanh does, and why the caps do not stop this oscillation

The measured norm ratio through the second tanh is

$$
\frac{\|\Delta H^{(2)}_t\|_F}{\|\Delta Z^{(2)}_t\|_F}
\in[0.86841,0.89303].
$$

Thus it removes about10.7--13.2% of the displacement norm, rather than amplifying it. Its alignment time course remains almost identical to the incoming preactivation alignment. Tanh bounds values and attenuates motion, but a bounded value can still move back and forth inside its permitted interval.

The existing second linear layer's gain along the actual first-hidden displacement is also below1:

$$
\frac{\|\Delta H^{(1)}_tW_{2,t}^{\top}\|_F}{\|\Delta H^{(1)}_t\|_F}
\in[0.81617,0.92572].
$$

This isolates transmission through the old weights. The *total* preactivation motion also includes the changing weights and other terms, so its norm can exceed this isolated transmitted contribution. The gain is directional; it does not claim every possible input direction is contracted or report the layer's spectral norm. The second weight matrix's maximum relative Frobenius drift from state448 is5.49%.

Meanwhile, coordinate RMS levels remain narrowly ranged:

| Representation | Minimum activation coordinate RMS | Maximum |
|---|---:|---:|
| First hidden | 0.89244 | 0.89606 |
| Second preactivation | 0.48367 | 0.49224 |
| Second hidden | 0.39577 | 0.40091 |

So the current evidence is **growing, reversing representation displacements with almost unchanged activation magnitude**, already visible after the first tanh. This analysis measures attenuation through the second tanh; it does not measure a pre-first-tanh gain or infer how many first-layer coordinates are saturated from RMS alone.

## 7. Captions, provenance and verification

**Figure09: earlier-layer SVD comparison.** Common full training loss and four columns for first hidden activations, second preactivations, second hidden activations, and network outputs. Each column shows incoming/outgoing change alignment, the leading fixed-mode coefficient, cumulative squared change-energy coverage and a complete10-by353 signed coefficient heatmap. All examples and consecutive updates are retained; no averaging, normalization of individual rows, or mode matching across spaces. Coefficients use the common$1/\sqrt{5000}$ scale, with a separate symmetric color range per space.

**Figure10: transmission of the first-hidden oscillation.** (a)Full training loss. (b)Aligned reversal time courses across representations. (c)Per-example vector RMS of representation changes. (d)Directional gain of first-hidden motion through the old second-layer weights, and total preactivation-to-hidden norm ratio through tanh. (e)Pointwise signed contributions to second-hidden motion after applying the observed elementwise secant. (f)Activation coordinate RMS, distinguishing nearly steady levels from changing displacement amplitudes. Contributions are accounting identities along the trajectory, not isolated interventions.

[Run241](../../../runs/241-2026-11-07-mlp-first-hidden-oscillation/README.md) preserves the same seed0, pinned5000 CIFAR-10 training examples, initialization, full mean half-squared-error objective, physical1000, Adam LR0.001/betas(0.9,0.999)/eps1e-8, no clipping/decay and TF32off. It captures355 first-hidden/preactivation arrays and second-layer states at447..801. All355 raw gradients and second-hidden arrays match Run240 exactly, all355 outputs match Run239, all801 native loss/gradient-norm pairs through800 match the original control, and full512/800/801 model/optimizer/RNG/current/previous-gradient checkpoints match. The observer preserves RNG and native gradients. No new training policy, seed, validation coverage or manuscript claim.

- [Reduction and finite-transmission script](../12_first_hidden_svd.py), reusing the already tested local SVD reduction.
- [Independent NumPy verification](../13_verify_first_hidden.py) and [verification record](../first_hidden_independent_verification.json).
- [Figure generator](../14_first_hidden_figures.py) and [figure audit](../first_hidden_figure_verification.json).
- [All353 numerical rows](../first_hidden_svd_values.csv) and [spectra, temporal subspaces and results](../first_hidden_svd_results.json).

Raw arrays are retained in Run241's ignored artifacts; previous second-hidden and output sources are reused by hash. New dense bases/Grams are retained under ignored `artifacts/first-hidden-svd/`. Independent checks cover7060 new coefficients,706 new cosines,18 selected full reconstructions, nine full-sample transmission checks and four temporal-subspace comparisons. Maximum independent coefficient discrepancy is$8.18\times10^{-14}$. Pre- and post-tanh accounting residuals are at most$1.34\times10^{-6}$ and$1.19\times10^{-6}$ per coordinate; direct tanh forward discrepancy is at most$1.09\times10^{-7}$. Both PDFs were rendered and visually reviewed.

Capture completed801 updates in18.354s, with270,178,304bytes peak GPU allocation. All47 input identities and12 downloaded artifact sizes/SHA256 passed. The worker exited;5,456,855,500bytes remain on the owned Spark including raw histories/archive/source records, within the6GB plan. No cloud cost or active worker remains. Reproduce reductions locally with scripts12,13,14 in order.

One seed/window, fixed examples, post hoc SVD bases fitted using later states, descriptive dependent-time correlations and no IID significance test. The measurement establishes that oscillation is already present at the first hidden representation and is largely transmitted downstream. It does not establish where the coupled optimizer/network dynamics first generate it.
