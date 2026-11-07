# Understanding output oscillations, SVD, and hidden representations

## TLDR: what we can conclude

1. **The predictions move back and forth during the studied spikes.** Consecutive output-change vectors have strongly negative cosine similarity. This measures reversal of motion in output space, not similarity between the output values themselves.
2. **Gradient and output oscillation occur together, but are not identical measurements.** Their plotted alignment series have a descriptive Pearson correlation of **0.7763** over states 448--800. This is not a causal proof or a mathematical identity between the two quantities. The time indexing of the two series also differs slightly, as explained below.
3. **The current SVD decomposes output differences across time.** It uses a **353-by-50,000** matrix: one row per outgoing update, one column per fixed example/output coordinate. All 5,000 examples and all 10 output coordinates are retained. We do not average examples.
4. **The SVD patterns are fixed across the entire window.** The leading pattern $v_1$ is a singular **vector**, not a singular value. The coefficient $\alpha_t$ measures signed movement along this fixed pattern. Its sign reverses in **306 of 352 adjacent pairs (86.93%)**, and at every step over the coefficient intervals 465--474 and 512--522. These are real projected reversals, not arbitrary SVD sign changes between steps.
5. **Several fixed directions matter.** The first mode captures **25.17%** of squared output-change energy; the first ten capture **98.16%**. This differs from the earlier approximately 93% result, which used a different matrix and a single selected event.
6. **We can trace the motion into hidden representations.** The useful next measurement separates output changes caused by hidden activation changes, output-weight changes, bias changes, and their interaction. It should distinguish motion throughout the hidden representation from motion concentrated in directions visible to the output head. This decomposition is described below as a proposed diagnostic; it has not been measured by this note.

This document explains existing measurements and a next diagnostic. It adds no training run, intervention, new numerical measurement, or causal claim. The results come from the unchanged CIFAR-10 MLP Adam control, one seed and the selected time window.

## 1. What is one network output, and what is the full output matrix?

The network is

$$
3072\ \longrightarrow\ 200\ \xrightarrow{\tanh}\ 200\ \xrightarrow{\tanh}\ 10,
$$

with a linear map at each arrow to a new width, biases in all three linear layers, and no activation normalization. More explicitly: flatten the CIFAR-10 image, apply Linear(3072,200), tanh, Linear(200,200), tanh, and Linear(200,10).

At state $t$, meaning **after $t$ completed optimizer updates**, evaluate the same $N=5000$ examples in the same order. For example $i$, the network returns ten real numbers:

$$
f_t(x_i)=(f_{t,i,1},\ldots,f_{t,i,10}).
$$

There is one output coordinate per class. These are raw output values, not softmax probabilities or predicted class labels. This experiment uses half squared error against one-hot labels:

$$
L_t=\frac{1}{N}\sum_{i=1}^{N}\frac12\sum_{c=1}^{10}
\bigl(f_{t,i,c}-y_{i,c}\bigr)^2.
$$

To keep matrices and flattened vectors distinct in this explanation, write

$$
\mathcal F_t\in\mathbb R^{5000\times10},
\qquad F_t=\operatorname{vec}(\mathcal F_t)\in\mathbb R^{50000}.
$$

| Object | Rows | Columns | Meaning of one entry |
|---|---|---|---|
| $\mathcal F_t$ at one state | 5,000 fixed examples | 10 class coordinates | Raw output for one example and one class |
| $F_t$, flattened | 50,000 coordinates | One vector | The same information, with example and class indices combined |
| Stored output tensor | First axis: 355 states, 447--801 | Other axes: 5,000 examples and 10 classes | Raw output at one state/example/class |

Flattening simply lays out the entries in a fixed order, for example

$$
F_t=(f_{t,1,1},\ldots,f_{t,1,10},f_{t,2,1},\ldots,f_{t,5000,10})^\top.
$$

It does not average, discard, or mix examples. The same coordinate always refers to the same example and class throughout training. A state axis is added only when we collect multiple evaluations over time.

## 2. Output values versus output changes: two different questions

The cosine between $F_t$ and $F_{t-1}$ asks whether the **output values** look similar. Our diagnostic instead compares two consecutive **movements** of those values.

Using the incoming-update convention from the discussion, define

$$
\Delta^-F_t=F_t-F_{t-1},
\qquad \rho_t^f=\cos(\Delta^-F_t,\Delta^-F_{t-1}).
$$

This compares the motions $t-2\to t-1$ and $t-1\to t$.

- Near $+1$: the two changes point in almost the same direction.
- Near $0$: they are approximately orthogonal.
- Near $-1$: they point in almost opposite directions.

For example, if the latest change is $-0.8$ times the previous change, the cosine is $-1$, although the second movement only partly retraces the first. A cosine measures direction, not distance. Exact reversal of direction does not require returning to the identical earlier outputs.

The measured cosine is global:

$$
\cos(A,B)=
\frac{\sum_{i=1}^{5000}\sum_{c=1}^{10}A_{ic}B_{ic}}
{\sqrt{\sum_{i,c}A_{ic}^2}\sqrt{\sum_{i,c}B_{ic}^2}}.
$$

It equals the cosine of the flattened vectors, or equivalently the normalized Frobenius inner product of the matrices. It is **neither a cosine of class means nor an arithmetic mean of 5,000 per-example cosines**. Examples with larger displacement norms can contribute more strongly. A negative global cosine does not require every example's individual cosine to be negative. If either entire displacement were zero, its cosine would be undefined; no undefined pairs occur in this window.

### The indexing used in the existing figures

The figures label an **outgoing** change by its starting state:

$$
\Delta^+F_t=F_{t+1}-F_t.
$$

Their output-alignment curve at plotted state $t$ is

$$
B_t=\cos(\Delta^+F_t,\Delta^+F_{t-1})
=\cos(F_{t+1}-F_t,F_t-F_{t-1})
=\rho^f_{t+1}.
$$

Both conventions describe the same geometric measurement, with a one-step labeling difference. We use the outgoing convention for the rest of this note to match the saved coefficients and figures.

For example, at plotted $t=474$:

| Quantity | What it uses |
|---|---|
| $L_{474}$ | Outputs at state 474 |
| Gradient alignment $\rho_{474}$ | Raw gradients at states 473 and 474 |
| Output alignment $B_{474}$ | Motions 473 to 474 and 474 to 475 |
| Coefficient $\alpha_{474}$ | Motion 474 to 475 |

The reported correlation is specifically

$$
\operatorname{corr}_{t=448}^{800}(\rho_t,B_t)=0.7762948.
$$

There was no lag search, smoothing, or subsampling. The correlation with the incoming-labeled $\rho_t^f$ at the same $t$ would be a different calculation; the reported number should not be silently transferred to that pairing.

## 3. Why gradient and output oscillation are related but not equivalent

The raw gradient $g_t$ lives in parameter space: it contains derivatives with respect to all weights and biases. The vector $F_t$ lives in output space: it contains predictions on fixed examples. These are different spaces with different dimensions.

Let $\delta\theta_t=\theta_{t+1}-\theta_t$ be the actual parameter update. Locally,

$$
\Delta^+F_t\approx J_t\delta\theta_t,
$$

where $J_t$ is the Jacobian of all 50,000 outputs with respect to the parameters. This is a first-order approximation, not the exact finite-step identity. Adam constructs $\delta\theta_t$ using momentum and adaptive scaling; it is not simply $-\eta g_t$. Furthermore, multiplication by $J_t$ need not preserve angles, and $J_t$ changes with the network state.

Therefore, the observations support **co-occurring gradient and functional oscillation**, not a universal implication $\rho_t<0\Rightarrow B_t<0$. Negative raw-gradient alignment also does not by itself prove that the actual Adam step increases loss. The observed correlation is descriptive: these are dependent time points from one trajectory, not independent experimental replications.

## 4. The current SVD: one matrix containing many output changes

For each outgoing update $t=448,\ldots,800$, construct the 50,000-coordinate difference $\Delta^+F_t$. Stack those vectors as rows:

$$
D=
\begin{bmatrix}
(\Delta^+F_{448})^\top\\
(\Delta^+F_{449})^\top\\
\vdots\\
(\Delta^+F_{800})^\top
\end{bmatrix}
\in\mathbb R^{353\times50000}.
$$

**A row now means a time step, not an example.** A column means a particular example/class coordinate, retained across all steps. For instance, the column for example 17, class 4 records how that output coordinate changes at every update in the window.

There are $800-448+1=353$ outgoing updates. They require outputs at states 448--801. The additional stored state 447 supplies the incoming displacement for the alignment at 448; it is not a row of the SVD matrix.

The decomposition is an uncentered SVD:

$$
D=U\Sigma V^\top.
$$

For the economy-size decomposition, the shapes are:

| Matrix | Shape | Role |
|---|---|---|
| $D$ | $353\times50000$ | Observed changes across time |
| $U$ | $353\times353$ | Temporal factors |
| $\Sigma$ | $353\times353$ | Diagonal matrix of nonnegative singular values |
| $V$ | $50000\times353$ | Orthogonal patterns in example/output space |
| First ten columns of $V$ | $50000\times10$ | Patterns retained in the heatmap |

There is no averaging over examples, no subtraction of temporal or coordinate means, and no coordinate standardization. Thus squared singular values partition **squared change energy**, not centered variance or loss variation.

If instead we stacked vectors as columns, using the user's notation $[\Delta F_1,\ldots,\Delta F_T]$, that matrix would be $D^\top$. It contains the same information and has the same singular values; the output-space patterns would then be its **left** singular vectors. The geometry is unchanged by this bookkeeping choice.

## 5. Singular vectors, singular values, and coefficients

Let $v_k$ be column $k$ of $V$. Each $v_k$ is a unit vector with 50,000 entries and can be reshaped into a 5,000-by-10 pattern. The expansion is

$$
\Delta^+F_t=\sum_{k=1}^{353}\alpha_{t,k}v_k,
\qquad
\alpha_{t,k}=\langle\Delta^+F_t,v_k\rangle
=U_{j(t),k}\sigma_k,
$$

where $j(t)=t-448+1$ is the one-based row index of update $t$. Zero singular values, if present, contribute nothing.

| Object | Does it vary with step $t$? | Interpretation |
|---|---|---|
| $v_k$ | No, in this fixed-window fit | One direction of coordinated example/output motion |
| $\sigma_k$ | No | Aggregate strength of that mode across the fitted window |
| $\alpha_{t,k}$ | Yes | Signed displacement along that direction at this step |

For a hypothetical two-coordinate pattern $v=(1,-1)/\sqrt2$, a positive coefficient raises the first coordinate and lowers the second. A negative coefficient reverses both effects. In our actual pattern, the same principle applies to 50,000 coordinates, including many examples and classes. One mode is not necessarily one class, one neuron, or one coordinate that dominates separately in each example.

The singular values are ordered $\sigma_1\ge\sigma_2\ge\cdots\ge0$. Their relation to the coefficient time series is

$$
\sigma_k^2=\sum_t\alpha_{t,k}^2,
\qquad
\text{energy share of mode }k
=\frac{\sigma_k^2}{\|D\|_F^2}.
$$

The first mode is the unit direction capturing the most total squared projection over the full window. It need not dominate at every individual step. Here its share is 25.17%; the first ten together capture 98.16%. A ten-mode reconstruction therefore leaves 1.84% of squared change energy unexplained. These percentages are not percentages of classification accuracy, loss variation, or causal explanation.

## 6. What the leading coefficient proves about oscillation

For the leading mode, abbreviate $\alpha_t=\alpha_{t,1}$. Define the projected output position relative to state 448:

$$
z_t=\langle F_t-F_{448},v_1\rangle.
$$

Because $v_1$ is fixed,

$$
\alpha_t=z_{t+1}-z_t.
$$

Thus $z_t$ tells us **where the outputs are along this axis**, while $\alpha_t$ tells us **how they move along it**. If $\alpha_{t-1}\alpha_t<0$, the projected position reverses direction between two adjacent updates. This is more specific than observing that two full vectors have a negative cosine: it tracks repeated reversals along one identified axis.

The observed coefficients reverse sign in 306/352 adjacent pairs. For example, coefficients 470 through 475 are approximately

$$
8.87,\ -12.27,\ 15.79,\ -18.35,\ 18.32,\ -14.03.
$$

The state474 loss peak is entered by coefficient473; coefficient474 describes the following update. The sequence shows alternating projected motion whose magnitude grows approaching the peak. It does not alone establish that this mode causes the peak or that the full output vector retraces the same path.

The sign of a singular vector is arbitrary: replacing $v_1$ by $-v_1$ flips every coefficient sign but leaves every reversal unchanged. Our implementation orients each pattern once, making its largest-absolute coordinate positive. It does not refit or choose a new sign at each step. A separate SVD at every step would require addressing sign ambiguity and changing directions before making this interpretation.

The heatmap displays $\alpha_{t,k}/\sqrt{5000}$, while the leading-coefficient line plot displays raw $\alpha_t$. This common positive scaling changes neither sign reversals nor relative mode energy shares. It expresses a mode's magnitude on a per-example vector-RMS scale; it does not average the examples before fitting.

### How SVD relates back to the full output cosine

With a complete orthonormal basis for the observed changes,

$$
B_t=
\frac{\sum_k\alpha_{t,k}\alpha_{t-1,k}}
{\sqrt{\sum_k\alpha_{t,k}^2}\sqrt{\sum_k\alpha_{t-1,k}^2}}.
$$

The leading-mode sign reversal makes the $k=1$ term in the numerator negative. Other modes also contribute. This explains why tracking $\alpha_t$ is informative but is not equivalent to measuring the complete output cosine. The ten plotted modes give an approximation to this expression; the full expression includes all modes needed to represent the observed changes.

## 7. Why the earlier approximately 93% result is different

Two SVD analyses have appeared in this investigation:

| Question | Matrix | Row meaning | Column meaning | Leading direction |
|---|---|---|---|---|
| One selected before-to-peak event | $\mathcal F_b-\mathcal F_a$, shape $5000\times10$ | Example | Class coordinate | A 10-dimensional class-space vector |
| Current trajectory of individual updates | $D$, shape $353\times50000$ | Outgoing update | Example/class coordinate | A 50,000-dimensional functional pattern |

For the event matrix, a near-rank-one approximation means

$$
(\mathcal F_b-\mathcal F_a)_{i,:}\approx s_i q_1^\top,
\qquad q_1\in\mathbb R^{10}.
$$

Examples share a mixed class-space direction $q_1$, with different signed amplitudes $s_i$. It does **not** mean that each example changes only one class coordinate, with a different class selected for every example. Positive and negative amplitudes can cancel in a per-class mean even when the rank-one pattern is strong.

That selected-event class-space concentration was approximately 93%. The current 25.17% measures the strength of one fixed example-and-class pattern across many updates. There is no contradiction: a locally strong class-space direction can change between events, as can the amplitudes across examples.

## 8. Moving inward: measure hidden representations directly

Let $H_t\in\mathbb R^{5000\times200}$ contain the last hidden activations, **after the second tanh and before the final linear layer**. Each row is one example; each column is one hidden unit. Let $W_t\in\mathbb R^{10\times200}$ be the output-layer weights and $b_t\in\mathbb R^{10}$ its biases. Then

$$
\mathcal F_t=H_tW_t^\top+\mathbf1b_t^\top,
$$

where $\mathbf1\in\mathbb R^{5000}$ broadcasts the same bias to every example. The matrix product has shape $(5000\times200)(200\times10)=5000\times10$.

We cannot uniquely recover $H_t$ from $\mathcal F_t$: the head maps 200 hidden coordinates to at most 10 independent output coordinates. Multiple hidden representations can produce the same outputs. It is preferable to evaluate hidden activations from the network states. Existing selected-state representation probes support this approach; complete per-update hidden coverage for a new time-series analysis must be checked or collected rather than assumed from the retained output tensor.

### Exact decomposition of an outgoing output change

Define $\Delta H_t=H_{t+1}-H_t$, $\Delta W_t=W_{t+1}-W_t$, and $\Delta b_t=b_{t+1}-b_t$. Expanding the linear head gives

$$
\mathcal F_{t+1}-\mathcal F_t
=
\underbrace{\Delta H_tW_t^\top}_{A_t:\ \text{hidden change through old head}}
+\underbrace{H_t\Delta W_t^\top}_{B_t^{\rm head}:\ \text{head-weight change at old features}}
+\underbrace{\Delta H_t\Delta W_t^\top}_{C_t:\ \text{interaction}}
+\underbrace{\mathbf1\Delta b_t^\top}_{E_t:\ \text{bias change}}.
$$

Every term has shape 5,000 by 10. This identity is exact, not a first-order approximation. The symbols for these four contributions are local to this decomposition; $B_t^{\rm head}$ is not the output-alignment cosine $B_t$.

Projecting the four terms onto the existing fixed $v_1$ yields

$$
\alpha_t=
\langle\operatorname{vec}(A_t),v_1\rangle+
\langle\operatorname{vec}(B_t^{\rm head}),v_1\rangle+
\langle\operatorname{vec}(C_t),v_1\rangle+
\langle\operatorname{vec}(E_t),v_1\rangle.
$$

These signed contributions would reveal which measured source of output motion supplies the alternating coefficient. They may reinforce or cancel each other; their norms do not generally add. This is an accounting identity along the observed trajectory, not an intervention establishing independent causal effects. The decomposition uses the old state as reference and displays the interaction separately.

### Is the whole hidden representation oscillating, or just its output-visible part?

First measure

$$
\rho_t^H=\cos\bigl(\operatorname{vec}(\Delta H_t),
\operatorname{vec}(\Delta H_{t-1})\bigr).
$$

This cosine uses $5000\times200=1,000,000$ hidden coordinates. As with outputs, the same examples and hidden-unit ordering must be preserved.

Then separate directions visible to a chosen fixed reference head $W_*$ from its nullspace. The orthogonal projector onto its row space is

$$
P_*=W_*^\top(W_*W_*^\top)^+W_*
\in\mathbb R^{200\times200},
$$

where $+$ denotes the Moore--Penrose pseudoinverse. For any hidden change,

$$
\Delta H_t^{\rm visible}=\Delta H_tP_*,
\qquad
\Delta H_t^{\rm null}=\Delta H_t(I-P_*).
$$

The null component satisfies $\Delta H_t^{\rm null}W_*^\top=0$: it produces no output change through that fixed head. The visible subspace has dimension at most ten; visibility does not mean equal amplification in all its directions. A fixed reference makes temporal comparisons interpretable. Using the instantaneous head instead would require explicitly accounting for the changing projector.

Comparing norms and consecutive cosines in these components can distinguish broad hidden oscillation from oscillation concentrated in a small output-sensitive component. Earlier selected-state probes support investigating the latter, but do not replace this proposed per-update measurement. Tanh bounds individual hidden activations; it does not prevent bounded activations from moving back and forth in directions to which the output head is sensitive.

## Sources and scope

- [Gradient/output alignment definitions, pairing, and measured correlation](004-gradient-output-alignment-comparison.md).
- [Fixed output-change SVD, energy shares, source data, and verification](006-svd-output-change-heatmap.md).
- [Leading-coefficient reversals and numerical verification](007-leading-svd-coefficient.md).
- [Earlier internal-representation probes and selected-event evidence](../../061-2026-11-07-mlp-adam-spike-internals/observations/001-spike-internals.md).
- [Original gradient-amplification analysis and interpretation limits](001-gradient-amplification.md).

The SVD basis is fitted post hoc to the entire window, including later states. It is descriptive, not an online predictor. No cross-seed uncertainty estimate is available. This explanation keeps observed results separate from the proposed hidden-space diagnostic and does not claim a unique initiating cause of the spikes.
