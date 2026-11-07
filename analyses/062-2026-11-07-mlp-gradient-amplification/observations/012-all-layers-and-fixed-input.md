# Complete layer trace: fixed inputs, oscillating learned representations

## TLDR

**The input itself does not oscillate. The oscillation is already present in the first learned representation, before either tanh, and persists through every later stage.**

- Every observed input tensor is exactly identical at all355 captured states447--801, including the order of all5000 samples. Flatten only reshapes it. Thus $\Delta X=0$: input-change energy and rank are zero, and the motion cosine and normalized SVD energy fractions are **undefined**, not zero.
- The first linear representation's change alignment is negative at350/353 plotted states and correlates at **0.99101** with the final output-change alignment. Its first ten SVD modes capture **95.57%** of change energy.
- Actual first-layer parameter increments also reverse: their consecutive cosine is negative at337/353 states. These are measured weight/bias increments, not raw gradients or gradient differences.
- The first tanh **strongly attenuates** motion, retaining only **32.16--36.82% of displacement norm**, while largely preserving reversal timing. The second tanh retains86.84--89.30%, as previously measured.
- Approximately **17.83--18.49% of sample-neuron coordinates** have a first-tanh finite-step slope above0.5; they carry **87.62--88.48% of the surviving first-hidden change energy**. These are coordinates for individual examples, not a fixed subset of18% of neurons.
- Changing first-layer weights accounts for almost all first-hidden motion by signed projection; first-layer bias changes contribute only about0.00331%. There is no changing-input term.

The raw inputs are fixed, but their geometry can still affect how parameter steps map into representation changes. These measurements locate the earliest changing forward representation; they do not establish the unique cause of the coupled optimizer/network oscillation.

Figures:

- [Complete forward-path SVD comparison, with fixed input explicitly identified](../figures/11-all-layer-svd-comparison.pdf).
- [Actual first-layer steps and transmission through the first tanh](../figures/12-fixed-input-first-layer-transmission.pdf).

## 1. Every stage, with rows and columns explicit

The network has no trainable input embedding or activation normalization before the first linear layer. Its stages are:

$$
\text{fixed images}\xrightarrow{\mathrm{Flatten}}X
\xrightarrow{W_1,b_1}Z^{(1)}
\xrightarrow{\tanh}H^{(1)}
\xrightarrow{W_2,b_2}Z^{(2)}
\xrightarrow{\tanh}H^{(2)}
\xrightarrow{W_3,b_3}F.
$$

| Stage at one state | Shape | Rows and columns |
|---|---|---|
| Fixed preprocessed image tensor | $5000\times3\times32\times32$ | Same5000 images, channels and spatial coordinates |
| Flattened input $X$ | $5000\times3072$ | Each row is a sample; each column is an input image coordinate |
| First linear output $Z^{(1)}$ | $5000\times200$ | Each row is the same sample; columns are first-layer preactivations |
| First tanh $H^{(1)}$ | $5000\times200$ | First hidden activation columns |
| Second linear output $Z^{(2)}$ | $5000\times200$ | Second-layer preactivation columns |
| Second tanh $H^{(2)}$ | $5000\times200$ | Second hidden activation columns |
| Final linear output $F$ | $5000\times10$ | Raw class-output columns |

The linear layers mix columns within each sample row. They never mix sample rows. Tanh acts on each entry separately. Flatten changes layout, not values. The input cache, labels and ordering are pinned, and this control uses the complete fixed training objective rather than changing mini-batches as its measurement unit.

“Input representation” can therefore mean two distinct objects here. The input image matrix$X$ is fixed. The first **learned** representation$Z^{(1)}=XW_1^\top+\mathbf1b_1^\top$ changes because$W_1,b_1$ change. It is the latter that already oscillates.

## 2. Input invariance was verified, not assumed

At every captured state447--801, a forward pre-hook on the first linear layer compares each actual input batch, after Flatten, with the same reference sample rows using exact tensor equality. All5000 rows are checked at each state. A capture test deliberately reverses sample order and verifies that this check rejects it. The saved input matrix is also independently verified against the pinned CIFAR-10 cache.

All355 checks pass, and the maximum input change is exactly zero. We retain one copy of$X$, its hash and the verification record; there is no need to store355 identical matrices.

Consequently,

$$
\Delta X_t=X_{t+1}-X_t=0.
$$

The hypothetical353-row input-change matrix is the zero matrix. All singular values are zero, its rank is zero, and no leading singular direction is identifiable. Expressions such as $0/\|0\|$ or $\sigma_1^2/\sum_k\sigma_k^2$ are undefined. Figure11 states this explicitly rather than drawing a zero-cosine curve or assigning meaningful SVD modes to the input. This is different from the trivially constant *input values*: their temporal similarity is not a measurement of input motion.

## 3. Same update convention and complete coverage

For every$t=448,\ldots,800$, analyze outgoing changes$\Delta A_t=A_{t+1}-A_t$ and

$$
\rho_t^A=\cos\left(\operatorname{vec}(\Delta A_t),\operatorname{vec}(\Delta A_{t-1})\right).
$$

The cosine compares incoming and outgoing movement around state$t$. State447 supplies the first incoming displacement; state801 supplies the final outgoing displacement. Loss$L_t$ is before the outgoing update. The displacement entering a loss peak at474 is labeled473.

Each learned-representation SVD uses353 time rows, all5000 examples and all coordinates:1,000,000 columns for each200-wide layer,50,000 for the output. There is no example averaging, centering, row normalization, smoothing, subsampling or lag selection. Native capture is FP32, while subtraction, Gram matrices, norms, cosines and SVD reductions use FP64. Signs are fixed once per basis by the largest-absolute coordinate. Different layers have independent bases; equal mode indices do not identify equal neuron-space directions.

We additionally analyze actual first-layer increments

$$
\delta\theta^{(1)}_t=
\operatorname{concat}\left(\operatorname{vec}(W_{1,t+1}-W_{1,t}),\ b_{1,t+1}-b_{1,t}\right),
$$

with614,600 coordinates:614,400 weights and200 biases. Its temporal matrix has353 rows. This is the motion actually applied by Adam in that parameter block, not$g_t$, not a moment, and not$g_{t+1}-g_t$. The user's correction to retain raw-gradient alignment remains in force.

## 4. The oscillation is already present before the first tanh

| Stream | Negative consecutive cosines | First-mode energy | First-ten energy | Modes for99% |
|---|---:|---:|---:|---:|
| Fixed input/Flatten changes | Undefined | Undefined | Undefined | No nonzero modes |
| Actual first-layer parameter increments | 337/353 | 19.08% | 88.85% | 68 |
| First linear changes $\Delta Z^{(1)}$ | 350/353 | 23.13% | 95.57% | 32 |
| First hidden changes $\Delta H^{(1)}$ | 350/353 | 24.20% | 95.39% | 35 |
| Second linear changes $\Delta Z^{(2)}$ | 351/353 | 25.41% | 97.32% | 17 |
| Second hidden changes $\Delta H^{(2)}$ | 351/353 | 25.58% | 97.40% | 16 |
| Output changes $\Delta F$ | 351/353 | 25.17% | 98.16% | 11 |

The leading first-linear coefficient reverses sign308/352 times; its first-hidden counterpart does too. The leading actual-parameter-step coefficient reverses310/352 times. These counts differ from full-vector negative-cosine counts because one fixed mode captures only part of a vector, and coefficient-pair counts exclude the extra447 row used for cosines.

| Alignment-series comparison | Pearson r |
|---|---:|
| Actual first-layer parameter increments / first linear changes | 0.837683 |
| Actual first-layer parameter increments / first hidden changes | 0.834878 |
| First linear / first hidden changes | 0.997591 |
| First linear / final output changes | 0.991012 |
| First hidden / final output changes | 0.991551 |

The top-ten temporal subspace overlap is0.998699 between first-linear and first-hidden changes, and0.995609 between first-linear and output changes. Here overlap is$\|U_A^\top U_B\|_F^2/10$ for their353-by10 temporal bases. The first-linear temporal basis captures98.0586% of output-change energy, nearly the output basis's own98.1569%. Thus similar spectra also correspond to closely matching timing; this is not merely a comparison of singular values.

Parameter increments have a broader spectrum and less exact temporal correspondence: their top-ten overlap with first-linear changes is0.894150. This reinforces the distinction between parameter-space motion and its effect on fixed examples.

## 5. Why fixed inputs can still change the apparent angle of parameter motion

The exact first-layer change is

$$
\Delta Z^{(1)}_t=X\Delta W_{1,t}^\top+\mathbf1\Delta b_{1,t}^\top.
$$

There is no$\Delta X$ term. The matrix$X$ is fixed, but a linear map need not preserve angles. For example, at state465 the actual first-layer parameter-step cosine is about **+0.017**, while the resulting first-linear displacement cosine is **-0.742**. At473 they are **-0.835** and **-0.960**. It would be incorrect to interpret this difference as the input itself moving.

For the exact geometry including biases, augment the inputs as$\widetilde X=[X,\mathbf1]\in\mathbb R^{5000\times3073}$ and the weights as$\widetilde W=[W_1,b_1]\in\mathbb R^{200\times3073}$. Then

$$
\langle\Delta Z^{(1)}_t,\Delta Z^{(1)}_{t-1}\rangle_F
=\operatorname{tr}\left(
\Delta\widetilde W_t\,\widetilde X^\top\widetilde X\,
\Delta\widetilde W_{t-1}^\top\right).
$$

Parameter-space inner products instead use the identity metric. The fixed input geometry weights parameter directions differently when their effects are measured on samples. Hence fixed inputs can influence the dynamics and their observed geometry without exhibiting temporal oscillation themselves. No additional input-spectrum or causal intervention is performed here.

## 6. First-layer weights versus biases

Separate$A_t=X\Delta W_{1,t}^\top$ and$B_t=\mathbf1\Delta b_{1,t}^\top$. Their sum reproduces the first-linear output change up to stored FP32 forward-roundoff. Propagate both through the observed elementwise first-tanh secant$S_t$ to account for$\Delta H^{(1)}_t$.

For an additive contribution$T_t$ to observed change$Y_t$, the aggregate signed projection share is

$$
s_T=\frac{\sum_t\langle Y_t,T_t\rangle_F}{\sum_t\|Y_t\|_F^2}.
$$

These shares sum to1 with the recorded roundoff residual. They are not independent nonnegative energy fractions or causal contributions to the loss.

| Contribution | First-linear motion | First-hidden motion after tanh |
|---|---:|---:|
| First-layer weight changes | 99.99960% | 99.99669% |
| First-layer bias changes | 0.000399% | 0.003310% |

This identifies weight motion as the direct source of the changing first learned representation in this fixed-input trajectory. It does not explain why Adam chooses those alternating increments.

Weight values do not abruptly explode at the selected peaks. From465 to474, the first-layer weight norm changes from14.3112 to14.3624 (about0.36%), while its outgoing increment norm grows from0.02808 to0.08084. From512 to522, the weight norm changes from14.5876 to14.6372 (about0.34%), while the increment norm grows from0.02580 to0.08516. Across the entire window the weight norm grows14.20 to15.73, so it is not globally constant. Distinguish the parameter level, its step size, and the representation effect of that step.

## 7. Tanh does attenuate motion; less-saturated coordinates carry what remains

For each sample-neuron coordinate, use the finite secant

$$
S_{t,ij}=\frac{\tanh(Z^{(1)}_{t+1,ij})-\tanh(Z^{(1)}_{t,ij})}
{Z^{(1)}_{t+1,ij}-Z^{(1)}_{t,ij}}.
$$

At exactly zero displacement, use the continuous derivative limit$1-\tanh^2(Z^{(1)}_{t,ij})$. This gives the real-arithmetic identity$\Delta H^{(1)}_t=S_t\odot\Delta Z^{(1)}_t$, with$0\le S\le1$. It is an endpoint-based decomposition of the observed step, not a first-order approximation or an isolated intervention.

The measured first-tanh norm ratio is

$$
\frac{\|\Delta H^{(1)}_t\|_F}{\|\Delta Z^{(1)}_t\|_F}
\in[0.32162,0.36818].
$$

So the first tanh removes roughly63--68% of displacement norm, much more than the second tanh. Nevertheless, first-linear and first-hidden reversal curves remain correlated at0.99759. Attenuating magnitude does not eliminate a back-and-forth direction pattern.

As a descriptive reduction of the captured endpoints, use thresholds$S>0.5$ and$S<0.01$:

| Quantity across states448--800 | Range |
|---|---:|
| Fraction of sample-neuron coordinates with$S>0.5$ | 17.83--18.49% |
| Fraction with$S<0.01$ | 45.45--46.79% |
| Fraction of surviving$\|\Delta H^{(1)}\|_F^2$ on coordinates with$S>0.5$ | 87.62--88.48% |

The threshold is a summary of finite-step transmission strength, not a binary neuron state. One neuron can be saturated for one example and responsive for another, and membership can change over time. The approximately18% therefore does not mean a fixed group of36 out of200 neurons. These summaries are descriptive post hoc reductions, not a tuned predictor of spike occurrence.

First-linear coordinate RMS stays between5.0606 and5.1444; first-hidden coordinate RMS stays0.89244--0.89606. For example, from465 to474 the first-linear *change norm* rises39.736 to148.867, while its activation coordinate RMS remains5.0662 versus5.0663. The issue is growing oscillatory displacement, not an abrupt explosion in the aggregate activation level.

## 8. Captions and interpretation limits

**Figure11: complete forward-path comparison.** The fixed-input banner reports exact invariance across355 captures and explicitly marks its motion cosine and normalized SVD fractions undefined. The five columns cover the first linear output, first tanh, second linear output, second tanh, and final linear output. Beneath common loss, rows show consecutive-change cosine, the leading fixed SVD coefficient, cumulative change-energy coverage and complete10-by353 signed coefficient heatmaps. Every sample is retained; coefficients are divided by$\sqrt{5000}$, with one symmetric scale per heatmap and no per-mode normalization. Independent bases are not assumed to match spatial directions.

**Figure12: fixed input and changing first-layer projection.** (a)Full training loss. (b)Actual first-layer parameter-step and representation-change alignments. (c)Per-example vector RMS of pre/post first-tanh displacements. (d)Finite-displacement gains of both tanh layers. (e)Pointwise signed weight/bias contributions to first-hidden motion, using the observed secant. (f)The fraction of sample-neuron coordinates with secant above0.5 and the fraction of surviving change energy they carry. Roundoff residuals are retained numerically. These are measured-trajectory relationships, not isolated causal interventions.

All correlations use dependent time points from one seed and one selected window, with no IID confidence claims. SVD bases are fitted post hoc using the full window. Cosines are global over fixed examples; not every sample, neuron or mode must reverse individually. Static input does not imply data geometry is irrelevant to training. This study completes the forward-layer trace but does not measure gradients with respect to input pixels or establish the unique initiating optimizer mechanism.

## 9. Sources, verification and reproduction

[Run242](../../../runs/242-2026-11-07-mlp-all-layer-oscillation/README.md) captures$Z^{(1)}$, first-layer weights/biases and the fixed input. Its observer verifies all355 raw gradients and second-hidden arrays against Run240, first-hidden/second-preactivation arrays against Run241, and outputs against Run239. Native reference boundaries through800 and full512/800/801 model/optimizer/RNG/current/previous-gradient states match. RNG and native gradients are preserved. Same seed0, pinned5000 training samples/order, full mean half-squared-error objective, physical1000, tanh MLP, Adam LR0.001/betas(0.9,0.999)/eps1e-8, no clipping/decay, FP32 native/FP64 diagnostic arithmetic and TF32off. No new training policy or manuscript change.

- [Reduction script](../15_all_layer_svd.py), [independent NumPy verification](../16_verify_all_layers.py), [figure generator](../17_all_layer_figures.py).
- [All353 numerical rows](../all_layer_svd_values.csv), [results/spectra](../all_layer_svd_results.json), [independent audit](../all_layer_independent_verification.json), [figure audit](../all_layer_figure_verification.json).
- Raw evidence: Run242 ignored artifacts `inputs.npy`, `preactivation1.npy`, `first_weights.npy`, `first_biases.npy`, `input_verification.json`; previously qualified downstream histories are reused by hash. New bases and Grams remain under ignored `artifacts/all-layer-svd/`.

Independent checks verify the saved input against the pinned cache and its tensor hash, the355 input-equality record,7060 new SVD coefficients,706 new cosines,18 selected full reconstructions, nine full-sample first-layer transmission checks, three temporal-subspace comparisons and all353 actual-increment norms. Maximum independent coefficient discrepancy is$2.28\times10^{-13}$. Maximum coordinate accounting residual is$1.89\times10^{-5}$ before the first tanh and$1.10\times10^{-5}$ after it, reflecting native FP32 forward arithmetic. All source/download sizes and hashes pass. Both PDFs were rendered and visually reviewed.

Capture completed801 updates in17.615s,344,022,528bytes peak GPU allocation.49 input identities and14 downloaded artifact identities verified. Worker exited;4,592,317,389bytes retained on owned Spark including raw histories/archive/source records, within the5GB plan. No cloud charge or active worker remains. Reproduce with scripts15,16,17 in order.
