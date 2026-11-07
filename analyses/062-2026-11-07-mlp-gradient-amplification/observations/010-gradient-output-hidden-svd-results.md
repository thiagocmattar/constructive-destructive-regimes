# Raw-gradient, output-change and hidden-change SVDs

## TLDR: the oscillation reaches the hidden representation

**The final hidden features oscillate strongly, with motion preferentially concentrated in the output-visible subspace. The remaining hidden component also oscillates; it is not stationary.** The two possibilities proposed earlier are therefore not mutually exclusive in these data.

- Consecutive hidden-change alignment closely tracks output-change alignment: Pearson **r=0.99847** over all353 states. Both the visible and null components reverse repeatedly.
- The output head's fixed10-dimensional row space occupies5% of the200-dimensional hidden space per example. It contains only **4.45--4.92% of activation energy**, but carries **68.14% of total hidden-change energy** across the window. The other190 dimensions carry31.86% of change energy.
- Hidden activation coordinate RMS stays within **0.39577--0.40091**. This is stable activation magnitude, not stationary representations: hidden displacement norms grow substantially near loss peaks.
- The first ten temporal SVD modes capture **99.04% of raw-gradient energy**, **98.16% of output-change energy**, and **97.40% of hidden-change energy**. The leading mode alone captures roughly one quarter in each space. These are independently fitted directions, not a claim that mode1 is geometrically the same across spaces.
- Hidden changes through the existing output head supply **98.41% of the signed projection onto the observed output motion** in an exact additive decomposition. Output-weight changes supply1.685%, the interaction contributes-0.0958%, and biases0.00129%. These are signed trajectory-accounting shares, not fractions of loss variation or causal attribution.
- As corrected by the user, the gradient SVD uses **raw $g_t$**, and its alignment is $\cos(g_t,g_{t-1})$. We do **not** difference gradients. Output and hidden SVDs use consecutive differences.

Two new figures:

- [Three-space comparison: alignments, coefficients, spectra and heatmaps](../figures/07-three-space-svd-comparison.pdf).
- [Visible/null hidden motion and output attribution](../figures/08-hidden-visible-nullspace.pdf).

## 1. What was replayed and what was matched

[Run240](../../../runs/240-2026-11-07-mlp-hidden-change-svd/README.md) replays the unchanged Adam control through state801. Native loss/gradient-norm pairs and gradient alignment statistics through800 match Run223; full model, optimizer, RNG, current and previous gradient states match at512/800, and the full801 state matches Run239. All355 captured output arrays447..801 match Run239 exactly. The observer preserves native gradients and RNG.

Same seed0, fixed5000 training examples in identical order, physical batch1000, raw half-squared-error objective against one-hot labels,3072-200-200-10 tanh MLP, Adam LR0.001, betas(0.9,0.999), eps1e-8, no clipping/decay. Native values are FP32; differences, Gram matrices, SVD reductions, projections and reported norm/cosine calculations use FP64. TF32 is disabled during replay. No optimizer intervention, extra seed, validation/test claim or manuscript change.

The user corrected "gradient changes" to "gradient alignment" after capture launch. The worker was already capturing raw gradients, so no data acquisition changed. The original executed launch configuration is preserved. The [corrected contract](009-three-space-svd-contract.md) controls this reduction and supersedes that one part of the original plan.

## 2. Matrix dimensions and time indexing

State$t$ means after$t$ completed optimizer updates. Let $\mathcal F_t$ be the5000-by10 output matrix and $H_t$ the5000-by200 final hidden activation matrix, after the second tanh and before the output linear layer. Flattening preserves every example and coordinate.

Each SVD has353 rows, for$t=448,\ldots,800$:

| Stream | Fitted row | Matrix shape | Meaning of one column |
|---|---|---|---|
| Gradient | $g_t^\top$ | $353\times656810$ | One parameter's raw full-mean loss gradient, including biases |
| Output | $\operatorname{vec}(\mathcal F_{t+1}-\mathcal F_t)^\top$ | $353\times50000$ | One fixed example/output coordinate |
| Hidden | $\operatorname{vec}(H_{t+1}-H_t)^\top$ | $353\times1000000$ | One fixed example/hidden-unit coordinate |

The gradient is already a derivative with respect to parameters. Its SVD describes the directions of the observed gradients themselves, not the changes of those gradients. It is also not a SVD of momentum or actual Adam parameter increments.

For alignment, use $\cos(g_t,g_{t-1})$ in gradient space and the incoming/outgoing displacement cosine around$t$ in the other spaces, for example

$$
\rho_t^H=\cos\left(\operatorname{vec}(H_{t+1}-H_t),\operatorname{vec}(H_t-H_{t-1})\right).
$$

An extra row at447 supplies the preceding vector for alignment at448; it is excluded from the SVD fit. Outputs and hidden activations at801 supply the outgoing displacement at800. Loss$L_t$ is measured before the outgoing update. In particular, the update entering the state474 peak is labeled473.

No examples are averaged before SVD. No temporal centering, per-vector normalization, coordinate standardization, state subsampling, smoothing or lag optimization is applied. A zero denominator would produce undefined/null; all actual compared vectors have nonzero norms.

## 3. Fixed SVDs and how concentrated they are

For each space independently, compute $X=U\Sigma V^\top$. Its coefficients are $a_{t,k}=\langle X_{t,:},v_k\rangle=U_{t,k}\sigma_k$. Each pattern is fixed across the whole fitted window, with its largest-absolute coordinate oriented positively once. Consequently, coefficient sign changes cannot result from refitting signs at individual steps.

| Space | First mode | First3 | First10 | Modes for95% | Modes for99% | Leading coefficient sign reversals |
|---|---:|---:|---:|---:|---:|---:|
| Raw gradients | 26.59% | 67.01% | 99.04% | 7 | 10 | 305/352 |
| Output changes | 25.17% | 63.44% | 98.16% | 8 | 11 | 306/352 |
| Hidden changes | 25.58% | 61.96% | 97.40% | 9 | 16 | 306/352 |
| Visible hidden changes | 25.56% | 62.62% | 98.18% | 8 | 11 | 308/352 |
| Null hidden changes | 26.02% | 60.69% | 95.76% | 10 | 34 | 304/352 |

Fractions are $\sigma_k^2/\sum_j\sigma_j^2$ within each stream. Gradient energy and hidden/output displacement energy have different units. A gradient coefficient sign flip means the gradient component reverses; output/hidden coefficient flips mean projected displacement reverses. Only the latter are increments of projected network outputs or representations.

The leading mode is not sufficient on its own: about three quarters of energy lies in other directions. Nevertheless, a small number of fixed temporal patterns describes most observed motion. The null component has a broader small-energy tail, needing34 modes for99% versus11 for the visible component.

The output SVD reproduces the previously saved coefficients and spectrum. Figure07 uses raw gradient coefficients and output/hidden coefficients divided by $\sqrt{5000}$, with one symmetric scale within each heatmap. There is no per-mode normalization. Different spaces have different units and color ranges; visual resemblance does not prove their modes correspond.

## 4. Does hidden motion really oscillate?

Yes. These correlations compare the entire353-point alignment curves, with the fixed indexing above:

| Pair of alignment series | Pearson r |
|---|---:|
| Raw gradient / output change | 0.776295 |
| Raw gradient / hidden change | 0.774865 |
| Output change / all hidden change | 0.998471 |
| Output change / visible hidden change | 0.999178 |
| Output change / null hidden change | 0.991516 |

Hidden-change cosine is negative at351/353 states; the visible component is negative at351/353 and the null component at350/353. This is evidence of reversal in both components, not just a correlation caused by their being flat or zero. A global cosine need not describe every example or every neuron separately.

The near-perfect hidden/output alignment relationship says that the final hidden representation's outgoing/incoming motion reflects almost the same reversal geometry as the output. The moderate raw-gradient comparison remains a separate relationship: Adam's transformation and the network Jacobian intervene between gradients and functional displacement.

## 5. The visible/null split: concentrated motion, with a moving remainder

Let $W_*=W_{448}\in\mathbb R^{10\times200}$. Its rank is10. An orthonormal row basis $Q\in\mathbb R^{10\times200}$ defines

$$
P=Q^\top Q,\qquad
\Delta H_t^{\rm visible}=\Delta H_tP,\qquad
\Delta H_t^{\rm null}=\Delta H_t(I-P).
$$

These components are orthogonal, so squared Frobenius norms and temporal Gram matrices add exactly in real arithmetic. We use compressed visible coordinates $\Delta H_tQ^\top$ of shape5000-by10: they preserve all norms and inner products of $\Delta H_tP$. Its temporal SVD matrix is353-by50000. The null SVD uses353-by1000000 coordinates in the original hidden basis.

Across the entire window,

$$
\frac{\sum_t\|\Delta H_tP\|_F^2}{\sum_t\|\Delta H_t\|_F^2}=0.681419.
$$

Pointwise visible change-energy shares range63.94--70.77%. In contrast, $\|H_tP\|_F^2/\|H_t\|_F^2$ ranges4.45--4.92%. The motion is therefore strongly biased toward the head-visible subspace, although most activation energy lies elsewhere. These ten dimensions are mixed directions in the200-dimensional hidden space, not ten selected neurons. Across5000 examples, the visible space can still contain50,000 coordinates; it is not a claim that all hidden motion is ten-dimensional globally.

The remaining31.86% of motion is substantial and also oscillatory. Thus the earlier phrase "while much of the representation remains steady" must be qualified: **the overall activation magnitude remains steady, but the orthogonal component is not motionless**. Both visible and null displacement norms grow near spikes.

| State t | Loss | Full hidden-change norm | Visible-change norm | Null-change norm | Visible energy share | Hidden coordinate RMS |
|---|---:|---:|---:|---:|---:|---:|
| 465 | 0.02819 | 12.627 | 10.360 | 7.218 | 67.32% | 0.39626 |
| 473 | 0.04770 | 44.836 | 36.575 | 25.934 | 66.54% | 0.39595 |
| 474 | 0.05367 | 46.840 | 38.117 | 27.223 | 66.22% | 0.39617 |
| 485 | 0.02772 | 16.162 | 13.135 | 9.417 | 66.05% | 0.39721 |
| 512 | 0.02306 | 11.268 | 9.152 | 6.572 | 65.98% | 0.39640 |
| 521 | 0.04891 | 49.993 | 41.932 | 27.221 | 70.35% | 0.39653 |
| 522 | 0.05506 | 48.677 | 40.905 | 26.386 | 70.62% | 0.39595 |
| 529 | 0.02268 | 20.121 | 16.425 | 11.622 | 66.64% | 0.39673 |

Change norms in the table concatenate all examples; Figure08 divides them by $\sqrt{5000}$ to give per-example vector RMS. Activation coordinate RMS divides $\|H_t\|_F$ by $\sqrt{5000\times200}$. These are explicitly different normalizations.

Concentration is present throughout the window, rather than appearing suddenly at each spike. For example, the visible share is slightly lower at474 than465, even though both change components grow. A spike is therefore not simply the moment when an otherwise invisible displacement first enters the head's row space. Here a persistent directional bias combines with growing, reversing displacement amplitude.

### Does the head move enough to invalidate the fixed split?

The head's maximum relative Frobenius drift from$W_{448}$ is7.69%. Its component outside the reference row space has at most4.31% of its current Frobenius norm. These are weight-space diagnostics, not bounds on every output effect.

As a sensitivity check, using each instantaneous head's row space gives pointwise visible shares64.19--71.09%, close to the fixed-head result. More directly, passing the fixed-null hidden motion through the *current* head yields an aggregate RMS only0.526% of the observed output-change RMS. Thus it is approximately invisible along this measured trajectory, although it is exactly null only for the reference head.

## 6. Hidden motion versus output-head motion: exact decomposition

For each outgoing update, with$\mathcal F_t=H_tW_t^\top+\mathbf1b_t^\top$,

$$
\Delta\mathcal F_t=
\underbrace{\Delta H_tW_t^\top}_{A_t}
+\underbrace{H_t\Delta W_t^\top}_{B_t}
+\underbrace{\Delta H_t\Delta W_t^\top}_{C_t}
+\underbrace{\mathbf1\Delta b_t^\top}_{E_t}.
$$

All four terms have shape5000-by10. This identity is exact for real-arithmetic linear heads. With retained FP32 forward outputs, the largest coordinate discrepancy is$7.77\times10^{-7}$, and the leading-mode coefficient discrepancy is$2.06\times10^{-7}$ or less; the measured roundoff residual is retained explicitly.

To summarize additive contributions without incorrectly adding squared norms of nonorthogonal terms, define

$$
s_A=\frac{\sum_t\langle\Delta\mathcal F_t,A_t\rangle_F}
{\sum_t\|\Delta\mathcal F_t\|_F^2},
$$

and similarly for the other terms. These **signed projection shares** sum to1, including roundoff. They can be negative or exceed1 if terms cancel; they are not independent nonnegative energy fractions.

| Contribution | Full-output signed projection share | Leading-output-coefficient signed projection share |
|---|---:|---:|
| Hidden change through existing head | 98.4093% | 98.4455% |
| Head-weight change at existing features | 1.6852% | 1.6748% |
| Hidden/head interaction | -0.0958% | -0.1217% |
| Bias change | 0.00129% | 0.00135% |

The leading-coefficient shares use the same formula after projecting each term onto the existing fixed output$v_1$. In Figure08(e), the hidden contribution nearly overlaps the observed alternating coefficient. This supports the interpretation that the output oscillation is carried mainly by changing hidden features through the head, rather than by a rapidly changing final linear layer or its bias. It does not identify the upstream initiating mechanism or isolate independent causal interventions on layers.

## 7. Figure captions

**Figure07: fixed SVDs across three network spaces.** Common full training loss, followed by columns for raw parameter gradients, output displacements and final hidden displacements. Rows show consecutive alignment, the leading fixed-mode coefficient, cumulative squared-energy coverage, and signed coefficients for the ten leading patterns. All353 states448--800 are retained. Gradient rows are$g_t$, not gradient differences. Output and hidden coefficients are divided by$\sqrt{5000}$; gradients are unscaled. Each heatmap has one symmetric scale and embeds all10-by353 cells. Independent bases do not establish mode correspondence across spaces.

**Figure08: concentration and reversal of hidden motion.** (a)Full training loss. (b)Incoming/outgoing alignment for outputs and full, fixed-visible and fixed-null hidden changes. (c)Per-example RMS hidden displacement norm for the three hidden components. (d)The head-visible share of hidden change energy, compared with its share of activation energy and an instantaneous-head sensitivity check. (e)The leading output coefficient and contributions from hidden motion, head-weight motion and their interaction. Bias and floating-point residuals remain in the saved numerical decomposition. The fixed reference head is from448; its rank10 row space is defined per example.

## 8. Sources, verification, reproduction and limits

- Raw evidence: Run240 `artifacts/spark/payload/worker/{gradients.npy,hidden.npy,outputs.npy,heads.pt,boundaries.jsonl}`; metadata names parameters, states and hidden placement. Raw histories remain ignored and retained locally and on Spark.
- [Reduction script](../09_three_space_svd.py), [independent verification](../10_verify_three_space_svd.py), and [figure generator](../11_three_space_figures.py).
- [All353 tabulated states](../three_space_svd_values.csv), [spectra and results](../three_space_svd_results.json), [independent audit](../three_space_svd_independent_verification.json), and [figure audit](../three_space_figure_verification.json).
- Dense bases and Gram matrices are retained under ignored `artifacts/three-space-svd/`. The output mode is verified against the previously retained basis, and prior output/gradient alignment series are reproduced to numerical precision.

Independent NumPy arithmetic verifies17,650 coefficients,1,765 cosines, five orthonormal bases, mode energies, and45 selected full-vector reconstruction residuals. The null Gram is checked against actual null vectors, not only Gram subtraction. Projector idempotence/null action, all353 native gradient norms, the energy split, and output attribution are checked. Maximum independent coefficient discrepancy is$7.11\times10^{-14}$. Source and download sizes/SHA256 are verified; both PDFs were rendered and visually reviewed.

Reproduce locally with scripts09,10,11 in that order. Capture completed801 updates in13.766s, with245,492,224bytes peak GPU allocation. All45 input files and13 downloaded artifacts were verified. Worker exited; approximately4.718GB remains on the owned Spark including raw files and archive. No cloud charges or active worker remain.

One seed, one selected trajectory/window, fixed examples, and post hoc bases fitted using future states within that window. Correlations are descriptive, without IID significance claims. Stable hidden norms do not imply stationary representations; low temporal rank does not imply that only a few neurons participate; visible/null claims refer to a specified head. These diagnostics locate the oscillation in the final hidden representation and quantify its transmission to outputs, but do not yet identify its unique initiating cause in earlier layers or optimizer state.
