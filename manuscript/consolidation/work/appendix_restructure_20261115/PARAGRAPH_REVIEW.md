# Appendix restructuring: sequential decisions

15 November 2026. Original, diagnosis, proposed solution, and replacement are recorded before each mutation. Structural transfers are identified explicitly.

## A title

Source: `sections/appendix.tex`

**Read**

```tex
\section{Mathematical Scope of the Diagnostics}
```

**Diagnosis:** The title does not tell readers which mathematical distinctions the appendix establishes.

**Proposed solution:** Name exactness and approximation limits explicitly.

**Rewrite:**

```tex
\section{Exact Identities and Approximation Limits}
```

## A1 exact curvature conditions

Source: `sections/appendix.tex`

**Read**

```tex
The distinction between an identity and a local approximation is
essential to interpreting the diagnostics. Let $f_t=\nabla\mathcal L(\theta_t)$
and $u_{t-1}=\theta_t-\theta_{t-1}\ne0$ be the actual applied increment.
For a twice continuously differentiable objective along this segment,

```

**Diagnosis:** The opening emphasizes interpretation without first stating the full-gradient and fixed-objective conditions.

**Proposed solution:** State the conditions and the recovered quantity before the integral.

**Rewrite:**

```tex
Full gradients of a fixed objective recover directional curvature
averaged along an update. Let $f_t=\nabla\mathcal L(\theta_t)$ and
$u_{t-1}=\theta_t-\theta_{t-1}\ne0$ be the actual applied increment.
For a twice continuously differentiable objective along this segment,

```

## A1 starting-point comparison

Source: `sections/appendix.tex`

**Read**

```tex
Taking its inner product with $u_{t-1}/\|u_{t-1}\|^2$ proves
Eq.~\eqref{eq:kappa-step}. This step-average identity holds for the
actual increment of any optimizer. If the Hessian is $L_H$-Lipschitz
in operator norm on the segment, its difference from starting-point
directional curvature satisfies

```

**Diagnosis:** The estimator connection silently assumes full gradients and the referent of its difference is unclear.

**Proposed solution:** Make the full-gradient substitution explicit and introduce the bound as a comparison of two quantities.

**Rewrite:**

```tex
Taking the inner product with $u_{t-1}/\|u_{t-1}\|^2$ gives
Eq.~\eqref{eq:kappa-step} with $g_t=f_t$, regardless of how the optimizer
selected the update. This step average need not equal curvature at the
starting state. If the Hessian is $L_H$-Lipschitz in operator norm along
the segment, their difference satisfies

```

## A1 bound and applicability

Source: `sections/appendix.tex`

**Read**

```tex
Indeed, the integral of the Hessian difference is bounded by
$\int_0^1L_Ha\|u_{t-1}\|\,da$. For sampled gradients $g_t=f_t+\xi_t$,
the measured estimate additionally contains
$u_{t-1}^{\top}(\xi_t-\xi_{t-1})/\|u_{t-1}\|^2$.
Under plain SGD, $u_{t-1}=-\eta g_{t-1}$, this term is
$-\varepsilon_t/\eta$. For Adam, the raw-gradient projection in
Eq.~\eqref{eq:adam-projected-response} generally differs from projection
onto the applied increment. Across nonsmooth activation boundaries,
the smooth derivation need not apply, although the diagnostic remains
computable. A zero increment leaves step-direction curvature undefined.


```

**Diagnosis:** One paragraph interleaves a bound, sampling correction, optimizer distinction, and applicability limits.

**Proposed solution:** Keep the short bound justification and applicability limits here; move the unchanged sampling correction beside the sampling assumptions.

**Rewrite:**

```tex
The bound follows by integrating $L_Ha\|u_{t-1}\|$ over $a\in[0,1]$.
For Adam, projection onto the applied increment must be distinguished
from the raw-gradient projection in Eq.~\eqref{eq:adam-projected-response}.
Across nonsmooth activation boundaries, the smooth derivation need not
apply even though the diagnostic remains computable. A zero increment
leaves step-direction curvature undefined.


```

## A2 loss remainder purpose

Source: `sections/appendix.tex`

**Read**

```tex
The loss-rise condition also has a finite-step remainder. Set
$x=\theta_{t-1}$ and $u=-\eta g_{t-1}$. Integration by parts gives

```

**Diagnosis:** The remainder is introduced as another formula rather than a limit on an existing interpretation.

**Proposed solution:** State the question it answers before defining the unchanged remainder.

**Rewrite:**

```tex
The quadratic loss-sign criterion can fail on a finite nonlinear step.
To quantify the discrepancy, set $x=\theta_{t-1}$ and $u=-\eta g_{t-1}$.
Integration by parts gives the remainder

```

## A2 exact quadratic and nonlinear cases

Source: `sections/appendix.tex`

**Read**

```tex
Substituting $f_t=g_t-\xi_t$ gives the loss identity in the main text.
For a constant Hessian the integral vanishes, and full-gradient plain
GD obeys $\Delta L_t=\eta\|g_{t-1}\|^2(\eta\widehat\kappa_t-2)/2$
exactly. Under the Lipschitz assumption, subtracting the midpoint Hessian
inside the integral gives $|\mathcal E_t|\leq L_H\|u\|^3/12$.
Thus a nonlinear finite step can disagree with the quadratic loss sign,
particularly when the predicted change is near zero.


```

**Diagnosis:** The mathematical consequence can be stated more directly, and the sampling correction needs to precede its assumptions.

**Proposed solution:** Preserve the formulas, explain the sign limitation, and relocate the original noise term here.

**Rewrite:**

```tex
Substituting $f_t=g_t-\xi_t$ gives the loss identity in the main text.
For a constant Hessian the remainder vanishes, and full-gradient plain
GD obeys $\Delta L_t=\eta\|g_{t-1}\|^2(\eta\widehat\kappa_t-2)/2$
exactly. Under the Lipschitz assumption, subtracting the midpoint Hessian
inside the integral gives $|\mathcal E_t|\leq L_H\|u\|^3/12$.
The remainder can therefore change the predicted loss sign on a nonlinear
step, particularly when the quadratic prediction is near zero.

For sampled gradients $g_t=f_t+\xi_t$, the curvature measurement contains
the additional term $u_{t-1}^{\top}(\xi_t-\xi_{t-1})/\|u_{t-1}\|^2$.
Under plain SGD, $u_{t-1}=-\eta g_{t-1}$, this term is
$-\varepsilon_t/\eta$.


```

## A3 trajectory versus fixed-state sampling

Source: `sections/appendix.tex`

**Read**

```tex
Sampling identities use conditional unbiasedness, without assuming
that the new state is independent of the preceding noise. Fresh sampling
gives $\E[g_{t-1}^{\top}\xi_t]=0$, whereas
$\E[g_{t-1}^{\top}\xi_{t-1}]=\E\|\xi_{t-1}\|^2$.
For uniform subsets of size $b$ from $N$ examples, sampled without
replacement, $\operatorname{Cov}(\xi\mid\theta)=(N-b)\Sigma/[b(N-1)]$
when $\Sigma$ uses denominator $N$. The expected-cosine formula
additionally replaces a random product of norms by its moment-based
approximation; it is not an exact ratio-of-expectations identity.


```

**Diagnosis:** The paragraph moves from dependent trajectory states to independent fixed-state draws without marking the change.

**Proposed solution:** Separate the two experiments and retain the covariance and denominator-approximation qualifications.

**Rewrite:**

```tex
Along the SGD trajectory, conditional unbiasedness does not require
the new state to be independent of the preceding noise. Fresh sampling
gives $\E[g_{t-1}^{\top}\xi_t]=0$, whereas
$\E[g_{t-1}^{\top}\xi_{t-1}]=\E\|\xi_{t-1}\|^2$.

Separately, at a fixed state $\theta$, uniform subsets of size $b$ from
$N$ examples, sampled without replacement, have
$\operatorname{Cov}(\xi\mid\theta)=(N-b)\Sigma/[b(N-1)]$
when $\Sigma$ uses denominator $N$. For two independent draws at that
state, the expected-cosine formula additionally approximates their random
product of norms by a moment-based value. It is not an exact identity
obtained by taking a ratio of expectations.


```

## B structure

Source: `sections/appendix.tex`

**Read**

```tex
\section{Reproduction Details and Independent Checks}
\label{app:validation}

\subsection{Objectives, sampling, and measurement timing}
```

**Diagnosis:** The existing headings combine methods and checks without identifying the objects being verified.

**Proposed solution:** Separate experimental definitions, gradient checks, and representation checks.

**Rewrite:**

```tex
\section{Experimental Definitions and Diagnostic Checks}
\label{app:validation}

\subsection{Objectives and experimental design}
```

## B-A1 objective-defining choices

Source: `sections/appendix.tex`

**Read**

```tex
The MLP is a 3072--200--200--10 tanh network with 656,810 parameters;
its half-squared error sums over ten outputs and averages over the
first 5,000 CIFAR-10 images. ResNet-32 uses all 50,000 training images
and 50 fixed 1,000-image training-mode BatchNorm groups. MLP input
standardization is coordinate-wise; ResNet standardization is
channel-wise, using training-split statistics. There is no augmentation
or dropout. The two-layer, two-head Transformer has embedding and
feed-forward width 200. Twenty streams form sequences of at most 35
tokens, giving 59,680 sequences and 2,088,600 supervised targets.
Its 33,278-word dictionary follows the pinned PyTorch example
revision~\cite{pytorchWordLanguageModelPinned}, including words from
training, validation, and test text. This is a fixed-objective dynamics
experiment rather than a train-only-vocabulary benchmark.


```

**Diagnosis:** The paragraph repeats Table I and buries objective-defining preprocessing among architecture counts.

**Proposed solution:** Refer to Table I, retain normalization and vocabulary choices, and split image and language preprocessing. Full architecture counts remain in Supplement S2.

**Rewrite:**

```tex
Table~\ref{tab:setup} gives architectures, objective reductions, and
data coverage. MLP inputs use coordinate-wise standardization; ResNet
uses channel-wise standardization, both from training-split statistics.
ResNet retains fixed training-mode BatchNorm groups throughout training
and measurement. There is no augmentation or dropout.

The Transformer forms sequences of at most 35 tokens from twenty streams.
Its 33,278-word dictionary follows the pinned PyTorch example
revision~\cite{pytorchWordLanguageModelPinned}, including words from
training, validation, and test text. This is a fixed-objective dynamics
experiment rather than a train-only-vocabulary benchmark.


```

## B-A2 numerical conventions

Source: `sections/appendix.tex`

**Read**

```tex
Parameters, gradients, and Adam moments are FP32; diagnostic products
and reductions use FP64, with TF32 disabled. Full training objectives
supply the updates. Spectral measurement coverage and interpolation are
specified in Fig.~\ref{fig:gd-loss}. The finite-step audit finds positive
all-pair interactions in all nine GD trajectories and negative adjacent
interactions in seven. The highest-rate Transformer has 1,325 sign
disagreements among 2,999 paired transitions, illustrating the limits of
the local quadratic surrogate even without sampling noise.


```

**Diagnosis:** A blanket full-objective statement is followed by mini-batch methods, and empirical findings interrupt numerical settings.

**Proposed solution:** Scope the full-objective claim and move the interaction/sign findings into B-B with their evidence.

**Rewrite:**

```tex
Parameters, gradients, and Adam moments are FP32; diagnostic products
and reductions use FP64, with TF32 disabled. Full-batch GD and Adam
updates use the complete training objective. Figure~\ref{fig:gd-loss}
specifies the separate spectral measurement coverage and interpolation.


```

## B-A3 sampling design

Source: `sections/appendix.tex`

**Read**

```tex
MLP mini-batches contain 32, 128, 512, 2048, or 5000 images.
ResNet uses 1, 5, 20, or 50 complete normalization groups;
Transformer batches contain 8, 32, 128, 512, or 59680 sequences.
Batches are independently redrawn uniform subsets; different sizes use
prefixes of a common permutation within each repetition. For
sequence-summed losses $\ell_i$, the Transformer estimator is
$N(bM)^{-1}\sum_{i\in B}\ell_i$, where $N=59680$ and $M=2088600$.
Its fixed denominator preserves the token-weighted objective.
Three initializations provide matched states at steps 0 and 2000,
with 200-update continuations and eight stochastic repetitions per
cell. Equal update counts do not imply equal data exposure. The two
smallest ResNet batches in initialization 2 use a different runtime;
a full-GD bridge agrees to parameter relative error $2.36\times10^{-7}$,
but runtime and batch size remain confounded in that block.


```

**Diagnosis:** Sampling independence and across-size coupling are compressed together; the runtime bridge number distracts from its unresolved confound.

**Proposed solution:** Explain permutation coupling directly and retain the confound while pointing to the existing supplementary bridge result.

**Rewrite:**

```tex
MLP mini-batches contain 32, 128, 512, 2048, or 5000 images;
ResNet uses 1, 5, 20, or 50 complete normalization groups; Transformer
batches contain 8, 32, 128, 512, or 59680 sequences. At each draw within
a repetition, sizes use nested prefixes of a common random permutation;
a fresh permutation is drawn for each sampling event. For sequence-summed
losses $\ell_i$, the Transformer estimator is
$N(bM)^{-1}\sum_{i\in B}\ell_i$, where $N=59680$ and $M=2088600$.
The fixed denominator preserves the token-weighted objective.

Three initializations provide matched states at steps 0 and 2000,
with 200-update continuations and eight stochastic repetitions per cell.
Equal update counts do not imply equal data exposure. The two smallest
ResNet batches in initialization 2 use a different runtime; batch size
and runtime remain confounded in that block. Supplementary Section S2
retains the numerical bridge check and complete sampling protocol.


```

## B-A4 Adam design and clock

Source: `sections/appendix.tex`

**Read**

```tex
The Adam grid uses MLP rates $.0001/.0003/.001$ and ResNet/Transformer
rates $.0003/.001/.003$, for 1,024 updates with one initialization per
model. Initial parameters match across rates; moments start at zero,
$(\beta_1,\beta_2)=(.9,.999)$, $\epsilon=10^{-8}$, with no decay,
clipping, or schedule. The overview displays the highest rates.
An excursion starts at a strict loss rise and ends on return to its
pre-rise baseline; an unrecovered event is right-censored. These events
are distinct from the next-step targets in Table~\ref{tab:severity}.
At state $t$, gradients are available before update $t\to t+1$;
representation changes ending at $t$ belong to the preceding update.


```

**Diagnosis:** Coefficients repeat the main text and censoring is named without a definition.

**Proposed solution:** Retain design and clock, reference the existing coefficients, and explain censoring.

**Rewrite:**

```tex
The Adam grid uses MLP rates $.0001/.0003/.001$ and ResNet/Transformer
rates $.0003/.001/.003$, for 1,024 updates with one initialization per
model. Initial parameters match across rates; moments start at zero
with the coefficients specified in Section~\ref{sec:adam}. These runs
use no weight decay, clipping, or schedule; the overview displays the
highest rates. An excursion starts at a strict loss rise and ends on
return to its pre-rise baseline. Recovery not observed before the run
ends is treated as right-censored. These events differ from the next-step
targets in Table~\ref{tab:severity}. At state $t$, gradients precede
update $t\to t+1$; representation changes ending at $t$ describe the
preceding update.


```

## B-B1 accumulated-interaction evidence

Source: `sections/appendix.tex`

**Read**

```tex
\subsection{Curvature and representation checks}
```

**Diagnosis:** The central accumulated-reinforcement finding is repeated in prose without its compact numerical support in the main appendix.

**Proposed solution:** Expose the existing nine-row evidence, projected from the supplementary audit, without introducing new measurements.

**Rewrite:**

```tex
\subsection{Gradient and curvature checks}
The accumulated-interaction check tests whether opposition between
consecutive updates dominates the complete trajectory.
Table~\ref{tab:app-gd-audit} reports positive total interactions in all
nine GD trajectories, including seven with negative adjacent sums.
Its loss-sign comparison also exposes finite-step limits: the highest-rate
Transformer has 1,325 disagreements between the incoming quadratic
prediction and the observed loss change among 2,999 paired transitions.

\input{tables/appendix-gd-audit}

```

## B-B2 independent curvature comparison

Source: `sections/appendix.tex`

**Read**

```tex
Independent directional Hessian--vector products test the cheap
curvature estimate against starting-point curvature. The moving MLP
study supplies 1,188 paired observations from 297 continuations over
three starts and five batch sizes. Full gradients at both states
supply the realized noise correction. Table~\ref{tab:app-hvp}
retains the raw discrepancy and corrected discrepancy; the correction
is an offline check, not an additional online estimator. Measurements
within a trajectory are dependent.


```

**Diagnosis:** Cheap estimate versus curvature obscures step-average versus starting-point quantities.

**Proposed solution:** Identify both quantities and explain the scope and residual discrepancy of the offline check.

**Rewrite:**

```tex
Independent Hessian--vector products measure directional curvature
at the starting state. Comparing them with the gradient-difference
measurement checks its finite-step approximation and sampling contribution.
The moving MLP study supplies 1,188 pairs from 297 continuations over
three starts and five batch sizes. Full gradients at both states supply
the realized noise correction. Table~\ref{tab:app-hvp} retains raw and
corrected discrepancies; finite-step differences can remain after
correction. This is an offline check, not an additional online estimator,
and measurements within a trajectory are dependent.


```

## B-B3 fixed-state approximation check

Source: `sections/appendix.tex`

**Read**

```tex
Frozen-state sampling supplies a separate check on the moment
approximation: at nine MLP states, each batch-size cell has 32
independent sequences of 128 draws. Maximum relative discrepancies
in the two moment identities are .50\% and .74\%, while maximum
absolute expected-cosine error is .00745. The approximation lies
outside 21 of 36 stochastic conditional 95\% bootstrap intervals;
agreement of scale should therefore not be read as exactness.


```

**Diagnosis:** Protocol counts bury the distinction between small absolute error and statistical agreement.

**Proposed solution:** Keep every accuracy/coverage result, refer to the existing full sampling protocol, and separate representation checks structurally.

**Rewrite:**

```tex
Independent resampling at nine fixed MLP states checks the moment
identities and expected-cosine approximation (Supplementary Section S2).
Maximum relative discrepancies in the two moment identities are .50\%
and .74\%; maximum absolute expected-cosine error is .00745. The
approximation lies outside 21 of 36 stochastic conditional 95\% bootstrap
intervals. It captures the scale of the expected cosine but is not exact
at the resolution of these checks.

\subsection{Representation measurements and spectral controls}

```

## B-C1 measurement validity

Source: `sections/appendix.tex`

**Read**

```tex
Representation measurements and spectral controls}
Representation measurements preserve input order, masks, normalization
groups, model mode, RNG, buffers, and gradients. Identical-state and
plain-forward checks establish observer invariance. ResNet's selected
trace reproduces all 187 native-continuation loss states from 264 to
450, with gradient and parameter checks; it is not asserted to recover
a different runtime's trajectory bitwise. MLP outputs are raw values;
ResNet/Transformer outputs are log probabilities over all classes or
words. The latter remove common-logit-shift invariance but retain equal
coordinate weighting, rather than weighting by loss sensitivity.


```

**Diagnosis:** Observer invariance and a detailed replay inventory interrupt the meaning of the measurement.

**Proposed solution:** Explain non-perturbation directly, retain native-continuation scope and output geometry, and leave exact replay counts in Supplement S4.

**Rewrite:**

```tex
Representation measurements preserve input correspondence, masks,
normalization groups, model mode, random-number-generator state, buffers,
and gradients. Repeated-state and plain-forward checks verify that
measurement leaves the relevant training state unchanged. The selected
ResNet trace is checked against its own native continuation, not claimed
as bitwise recovery of another runtime's trajectory (Supplementary
Section S4). MLP outputs are raw values; ResNet/Transformer outputs are
log probabilities over all classes or words. These remove sensitivity
to a common logit shift but weight coordinates equally, rather than by
loss sensitivity.


```

## B-C2 temporal reduction

Source: `sections/appendix.tex`

**Read**

```tex
Temporal Grams use all coordinates in FP64, without centering,
projection, or primary row normalization. The extra change needed for
the first cosine is excluded from the fitted 51/61/49-row matrices.
Symmetry, positive-semidefiniteness, trace, eigenvalues, and coefficient
reconstruction are checked. Retention uses
$\sqrt{\mathbf1^\top K\mathbf1}/\sum_i\sqrt{K_{ii}}$ without truncating
modes; direct and full-eigendecomposition calculations agree.


```

**Diagnosis:** An implementation-check inventory distracts from the reduction and full-mode retention contract.

**Proposed solution:** Retain the measurement-defining operations and summarize numerical checks with a supplementary pointer.

**Rewrite:**

```tex
Temporal Gram matrices use all coordinates in FP64, without centering,
projection, or primary row normalization. The extra change needed for
the first cosine is excluded from the fitted 51/61/49-row matrices.
Matrix and reconstruction checks verify the reduction (Supplementary
Section S4). Retention uses
$\sqrt{\mathbf1^\top K\mathbf1}/\sum_i\sqrt{K_{ii}}$ without truncating
modes; direct and full-eigendecomposition calculations agree.


```

## B-C3 alternative explanations for concentration

Source: `sections/appendix.tex`

**Read**

```tex
Table~\ref{tab:app-spectral} checks whether concentration merely reflects
a few large steps. The orthogonal control zeros the off-diagonal
entries of $K$, retaining step norms while
removing directional recurrence. Unit-row normalization instead uses
$B^{-1/2}KB^{-1/2}$, with $B=\operatorname{diag}(\operatorname{diag}K)$,
excluding zero rows. Both amplitude and repeated directions contribute.
Neither control is a randomized significance test; signed coefficients
and consecutive cosines supply the evidence of reversal.


```

**Diagnosis:** The controls are defined as matrix operations before explaining the alternative explanations they test.

**Proposed solution:** Introduce each control through its scientific question and interpret the existing architecture-dependent table values.

**Rewrite:**

```tex
Could spectral concentration arise mainly from a few large steps?
The orthogonal control in Table~\ref{tab:app-spectral} zeros off-diagonal
entries of $K$, preserving step norms while removing directional
recurrence. Does concentration persist when all changes have equal
length? Unit-row normalization uses $B^{-1/2}KB^{-1/2}$, with
$B=\operatorname{diag}(\operatorname{diag}K)$, excluding zero rows.
Normalization retains strong concentration in the MLP but increases
the modes needed for 95\% in ResNet and Transformer. These descriptive
controls are not randomized significance tests. Signed coefficients
and consecutive cosines, rather than concentration alone, establish
reversal.


```

## C structure

Source: `sections/appendix.tex`

**Read**

```tex
\section{Intervention Integrity, Endpoint Sensitivity, and Cost}
\label{app:interventions}
```

**Diagnosis:** The section mixes implementation validity and outcome robustness.

**Proposed solution:** Separate the matched-comparison contract from endpoint and cost checks.

**Rewrite:**

```tex
\section{Intervention Verification, Endpoint Sensitivity, and Cost}
\label{app:interventions}

\subsection{Matched conditions and the local displacement bound}
```

## C-A1 matched-state verification

Source: `sections/appendix.tex`

**Read**

```tex
All 12 conditions within a model share the complete pre-intervention
parameter, optimizer, random, and data-order state. The 36 continuations
contain 11,904 updates, including 10,912 treated updates. Eight preceding
untreated updates supply cap calibration; the budgets are frozen before
the common start. Qualification includes shared-prefix fingerprints,
108 exact gradient/reference-evaluation/native-update audits, calibration
reconstruction, observer invariance, and bounds checked on stored
increments. These checks are not statistical replications.


```

**Diagnosis:** Counts and shorthand audit names obscure what was matched and verified.

**Proposed solution:** State the actual comparison guarantees and move the inventory to its existing Supplement S5 record.

**Rewrite:**

```tex
All 12 conditions within a model share the complete pre-intervention
parameter, optimizer, random, and data-order state. Eight preceding
untreated updates supply calibration; budgets are frozen before the
common start. Checks verify common starting states, unchanged-Adam
gradient and update calculations, reference evaluations, calibration
values, and bounds on stored increments. Supplementary Section S5
retains the audit counts and full coverage. These checks verify execution;
they are not statistical replications.


```

## C-A2 intervention semantics

Source: `sections/appendix.tex`

**Read**

```tex
Late $\beta_2$ reductions retain moments and step counters, using the new
coefficient in the recurrence and native bias correction. Gradient
clipping acts on the complete accumulated raw gradient before Adam.
The cap targets the two hidden MLP linear maps, 31 ResNet convolutions,
and two QKV/four feed-forward Transformer maps; other increments remain
unchanged. Adam's moments advance normally. For simultaneously changing
upstream inputs $H$ and local weights, the preactivation change is

```

**Diagnosis:** Implementation rules repeat Section VI and QKV remains shorthand.

**Proposed solution:** Retain site counts and state semantics concisely, and introduce the unchanged propagated-change equation.

**Rewrite:**

```tex
The interventions follow Section~\ref{sec:vc-interventions}: late
$\beta_2$ changes retain optimizer state, clipping precedes Adam, and
the cap scales proposed increments without recomputing moments. The cap
targets two hidden MLP linear maps, 31 ResNet convolutions, and two
query/key/value and four feed-forward Transformer maps; other increments
remain unchanged. When upstream inputs $H$ and local weights change
together, the preactivation change is

```

## C-A2 propagated-change reference

Source: `sections/appendix.tex`

**Read**

```tex
\begin{equation}
 \begin{aligned}
 \Delta Z={}&H(\Delta W^{\rm cap})^\top+
             \mathbf1(\Delta b^{\rm cap})^\top\\
            &+\Delta H W^\top+\Delta H(\Delta W^{\rm cap})^\top.
 \end{aligned}
\end{equation}
```

**Diagnosis:** The local-bound equation lacks a reference label.

**Proposed solution:** Add only a label; preserve its mathematics byte for byte.

**Rewrite:**

```tex
\begin{equation}
 \begin{aligned}
 \Delta Z={}&H(\Delta W^{\rm cap})^\top+
             \mathbf1(\Delta b^{\rm cap})^\top\\
            &+\Delta H W^\top+\Delta H(\Delta W^{\rm cap})^\top.
 \end{aligned}
\label{eq:app-propagated-change}
\end{equation}
```

## C-A2 meaning of the bounded terms

Source: `sections/appendix.tex`

**Read**

```tex
Only the first line is locally bounded. Thus the cap does not guarantee
a bound on propagated output or loss. A zero score uses factor one;
a zero calibrated budget disables that site's cap.


```

**Diagnosis:** The equation terms are not interpreted and edge conventions repeat Section VI.

**Proposed solution:** Explain direct, upstream, and interaction terms; retain the scope limit and remove duplicate zero conventions.

**Rewrite:**

```tex
The first line is the direct local change and is bounded by the cap.
The second contains the change arriving through upstream inputs and
its interaction with the local weight update. Neither is bounded by
the local rule, so Eq.~\eqref{eq:app-propagated-change} gives no guarantee
on total propagated output or loss.

\subsection{Endpoint sensitivity and computational cost}

```

## C-B1 endpoint sensitivity

Source: `sections/appendix.tex`

**Read**

```tex
The final loss in Fig.~\ref{fig:vc-tradeoff} is sensitive to the phase of
any remaining oscillation. Table~\ref{tab:app-late} averages the last 32
states of every arm, without selecting a favorable checkpoint. The
endpoint advantage of q25 persists in this check for all three models.
ResNet q50 remains worse than unchanged Adam, whereas q25 and q75
improve the late mean. Neither a fixed endpoint nor a short late average
establishes asymptotic convergence; all comparisons use one initialization
per model and unequal effective intervention strengths.


```

**Diagnosis:** The averaging interval is implicit and arms is unnecessary jargon.

**Proposed solution:** Specify the exact states while retaining the complete result and limitations.

**Rewrite:**

```tex
The final loss in Fig.~\ref{fig:vc-tradeoff} depends on the phase of
remaining oscillation. Table~\ref{tab:app-late} averages states
$t_f-31,\ldots,t_f$ for every setting, without selecting a favorable
checkpoint. The endpoint advantage of q25 persists for all three models.
ResNet q50 remains worse than unchanged Adam, whereas q25 and q75
improve the late mean. Neither a fixed endpoint nor a short late average
establishes asymptotic convergence; comparisons use one initialization
per model and unequal effective intervention strengths.


```

## C-B2 measured cost

Source: `sections/appendix.tex`

**Read**

```tex
The recorded q50 component times in Table~\ref{tab:app-cost} retain the
cost caveat. Gradient and reference means include the final boundary;
controller means count applied updates. The reference evaluation collects
the current inputs used by the cap; its work is additional to local
products and reductions. Other diagnostics and I/O contribute further
wall time. Hosts and instrumentation differ, so these measurements
cannot establish a time-to-loss advantage. In particular, the MLP cap
cost exceeds the full-gradient calculation.


```

**Diagnosis:** Retain the cost caveat is document commentary, and final boundary is undefined.

**Proposed solution:** State the timing units, what work they include, and the supported limitation directly.

**Rewrite:**

```tex
Table~\ref{tab:app-cost} reports recorded q50 component times. Gradient
and reference means cover evaluated parameter states, including the
final state; controller means cover applied updates. The reference pass
collects current cap inputs in addition to the local products and
reductions. Other diagnostics and I/O add wall time. Hosts and
instrumentation differ, so these measurements do not establish a
time-to-loss advantage. In the MLP, cap computation exceeds the
full-gradient calculation.


```

## D transfer out of main appendix

Source: `sections/appendix.tex`

**Read**

```tex
\section{Applied Updates and Selected Functional Trajectories}
\label{app:plots}

Figure~\ref{fig:app-updates} retains every update and every intervention
setting. It distinguishes the applied parameter increment from the raw
gradient and shows why nominal coefficients do not match effective
intervention strength. Figure~\ref{fig:app-activity} shows how often
clipping or the cap changes the candidate update.
Figure~\ref{fig:app-selected} follows three selected conventional
settings, with unchanged Adam retained only as a loss reference.
They illustrate contrasting outcomes rather than a selection of each
family's best setting; Table~\ref{tab:vc-outcomes} includes all settings.

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/a01-matched-applied-updates.pdf}
\caption{Actual applied update norms for all matched interventions. Rows correspond to immediate learning-rate reduction, late $\beta_2$ reduction, global gradient clipping, and distributed displacement caps. Gray is the same unchanged-Adam control in every row. All post-start updates are retained without smoothing; the log scale is shared across families within each model. Update $u_{t-1}$ is placed at its completed-state endpoint $t$.}
\label{fig:app-updates}
\end{figure*}

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/a02-matched-controller-activity.pdf}
\caption{Controller activity over the same complete windows. Top: multiplicative factor retained by global gradient clipping before Adam; one means no clipping. Bottom: fraction of targeted units whose proposed increments are scaled by the displacement cap, pooled over all targeted sites; zero means no units capped. A clipping factor is not an Adam update-retention factor. All values are aligned with completed updates and unsmoothed.}
\label{fig:app-activity}
\end{figure*}

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/a03-selected-intervention-motion.pdf}
\caption{Selected conventional interventions across the three models: $\beta_2=0.5$ (top block), LR $\times0.1$ (middle), and clip threshold 1.0 (bottom). Each block shows full training loss, consecutive representation-change cosine, and relative RMS displacement. Gray loss curves are unchanged Adam; colored curves and all heatmaps belong to the selected intervention. No Adam heatmaps are repeated. Log-loss limits are shared across the three blocks within each model. RMS uses each site's unchanged-Adam maximum as a common denominator, with one color scale across all selected settings and models; values above one exceed that baseline maximum. Every post-start change is retained, without smoothing, layer averaging, or peak markers. Sites match Fig.~\ref{fig:trace-motion}.}
\label{fig:app-selected}
\end{figure*}

```

**Diagnosis:** The complete trajectory gallery consumes main-paper space without a new central inference.

**Proposed solution:** Transfer all three existing assets to Supplement S7, preserving data and measurement captions.

**Rewrite:**

```tex

```

## Remove obsolete float-page reset

Source: `sections/appendix.tex`

**Read**

```tex
\clearpage
% Reset column height after the full-width float pages before the bibliography.
\onecolumn\twocolumn

```

**Diagnosis:** The column reset was specific to the removed full-width float pages.

**Proposed solution:** Remove it provisionally and inspect the rebuilt appendix-to-bibliography transition.

**Rewrite:**

```tex

```

## Compact existing GD audit

Source: `tables/appendix-gd-audit.tex`

**Read**

```tex

```

**Diagnosis:** The full supplementary audit includes loss and retention columns already covered in the main paper.

**Proposed solution:** Copy all nine rows exactly, selecting model, rate, interaction ratios, and sign disagreements.

**Rewrite:**

```tex
\begin{table}[!t]
\caption{Complete GD interaction and loss-sign audit. $E=\sum_t\|g_t\|^2$, $I_{\rm all}=2\sum_{i<j}g_i^\top g_j$, and $I_1=2\sum_tg_{t-1}^\top g_t$ use all applied gradients. Ratios remove the common $\eta^2$ factor. Sign disagreements compare the incoming quadratic prediction with 2,999 observed loss changes; the final unpaired update is excluded.}
\label{tab:app-gd-audit}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}
\toprule
Model & $\eta$ & $I_{\rm all}/E$ & $I_1/E$ & Sign disagreements \\
\midrule
MLP & 0.004 & 847.349 & 1.905 & 0/2999 \\
MLP & 0.02 & 2.988 & -1.970 & 145/2999 \\
MLP & 0.04 & 0.659 & -1.945 & 281/2999 \\
ResNet-32 & 0.025 & 37.798 & -1.884 & 595/2999 \\
ResNet-32 & 0.05 & 7.375 & -1.900 & 694/2999 \\
ResNet-32 & 0.1 & 2.427 & -1.900 & 814/2999 \\
Transformer & 0.05 & 620.733 & 1.977 & 0/2999 \\
Transformer & 0.1 & 50.112 & -1.631 & 542/2999 \\
Transformer & 0.2 & 7.671 & -1.927 & 1325/2999 \\
\bottomrule
\end{tabular}
\end{table}

```

## S7 transferred trajectory gallery

Source: `sections/s07-intervention-trajectories.tex`

**Read**

```tex

```

**Diagnosis:** Transferred figures need a standalone supplementary introduction and valid cross-document wording.

**Proposed solution:** Reuse the original assets and captions, adapting only float sizing and the main-paper figure reference.

**Rewrite:**

```tex
\clearpage
\section{Extended Intervention Trajectories}
\label{supp:extended-interventions}
Figures~\ref{fig:app-updates}--\ref{fig:app-selected} retain complete
post-intervention update histories, intervention activity, and selected
representation trajectories. The applied increments differ from raw
gradients; nominal coefficients do not match effective intervention
strength. The selected conventional settings illustrate contrasting
behaviors, rather than rank each family's best setting. Section S5
retains the complete matched outcomes.

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth,height=.78\textheight,keepaspectratio]{figures/a01-matched-applied-updates.pdf}
\caption{Actual applied update norms for all matched interventions. Rows correspond to immediate learning-rate reduction, late $\beta_2$ reduction, global gradient clipping, and distributed displacement caps. Gray is the same unchanged-Adam control in every row. All post-start updates are retained without smoothing; the log scale is shared across families within each model. Update $u_{t-1}$ is placed at its completed-state endpoint $t$.}
\label{fig:app-updates}
\end{figure}

\clearpage
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth,height=.78\textheight,keepaspectratio]{figures/a02-matched-controller-activity.pdf}
\caption{Controller activity over the same complete windows. Top: multiplicative factor retained by global gradient clipping before Adam; one means no clipping. Bottom: fraction of targeted units whose proposed increments are scaled by the displacement cap, pooled over all targeted sites; zero means no units capped. A clipping factor is not an Adam update-retention factor. All values are aligned with completed updates and unsmoothed.}
\label{fig:app-activity}
\end{figure}

\clearpage
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth,height=.78\textheight,keepaspectratio]{figures/a03-selected-intervention-motion.pdf}
\caption{Selected conventional interventions across the three models: $\beta_2=0.5$ (top block), LR $\times0.1$ (middle), and clip threshold 1.0 (bottom). Each block shows full training loss, consecutive representation-change cosine, and relative RMS displacement. Gray loss curves are unchanged Adam; colored curves and all heatmaps belong to the selected intervention. No Adam heatmaps are repeated. Log-loss limits are shared across the three blocks within each model. RMS uses each site's unchanged-Adam maximum as a common denominator, with one color scale across all selected settings and models; values above one exceed that baseline maximum. Every post-start change is retained, without smoothing, layer averaging, or peak markers. Sites match the main paper's representation-motion figure.}
\label{fig:app-selected}
\end{figure}
\clearpage

```

## Include S7

Source: `supplementary.tex`

**Read**

```tex
\input{sections/s06-reproducibility}
```

**Diagnosis:** The supplementary wrapper has no destination for the transferred gallery.

**Proposed solution:** Append S7 after S6, retaining the existing section numbering.

**Rewrite:**

```tex
\input{sections/s06-reproducibility}
\input{sections/s07-intervention-trajectories}
```

## Supplement roadmap

Source: `supplementary.tex`

**Read**

```tex
S5 supplies intervention methods and outcomes. Historical MLP replication
```

**Diagnosis:** The roadmap does not mention the new gallery or existing provenance section.

**Proposed solution:** Extend the roadmap without changing prior content.

**Rewrite:**

```tex
S5 supplies intervention methods and outcomes; S6 records reproducibility and scope; S7 retains extended intervention trajectories. Historical MLP replication
```

## VI full-history pointer

Source: `sections/05c-interventions.tex`

**Read**

```tex
Appendix~\ref{app:plots} retains all applied-update histories and selected functional trajectories.
```

**Diagnosis:** The referenced main appendix has moved.

**Proposed solution:** Point to its explicit supplementary destination.

**Rewrite:**

```tex
Supplementary Section S7 provides complete applied-update histories
and selected representation trajectories.
```

## VI motion pointer

Source: `sections/05c-interventions.tex`

**Read**

```tex
Appendix~\ref{app:plots} compares this functional motion with selected
conventional interventions.
```

**Diagnosis:** The referenced main appendix has moved.

**Proposed solution:** Retain the supporting pointer without implying a new comparison or ranking.

**Rewrite:**

```tex
Supplementary Section S7 shows representation trajectories for selected
conventional interventions.
```

## IV accumulated-interaction evidence pointer

Source: `sections/04-gd-sgd.tex`

**Read**

```tex
the adjacent cancellations (Appendix~\ref{app:validation}).
```

**Diagnosis:** A broad appendix pointer obscures the specific evidence for the accumulated-interaction finding.

**Proposed solution:** Reference the newly exposed existing audit table directly.

**Rewrite:**

```tex
the adjacent cancellations (Table~\ref{tab:app-gd-audit}).
```

## Restore B-C heading boundary

**Read**

```tex
\subsection{Representation measurements preserve input correspondence, masks,
```

**Diagnosis:** The first build found that the replacement matched the newly inserted subsection title as well as the paragraph start, leaving the heading unclosed.

**Proposed solution:** Restore the complete planned heading and separate the prose paragraph; retain all scientific text.

**Rewrite:**

```tex
\subsection{Representation measurements and spectral controls}
Representation measurements preserve input correspondence, masks,
```

## Final source whitespace cleanup

**Read**

```tex
full-gradient calculation.


```

**Diagnosis:** The staged whitespace check found a blank line at the end of the edited appendix.

**Proposed solution:** Remove only the redundant final blank line; no prose or TeX content changes.

**Rewrite:**

```tex
full-gradient calculation.

```
