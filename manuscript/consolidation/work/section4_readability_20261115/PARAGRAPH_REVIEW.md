# Conservative Section IV readability edits

15 November 2026. Each diagnosis and proposed solution was recorded before its source edit. Unlisted prose was retained. Equations, notation, labels, and subsection order are protected.

## IV opening

**Read**

```tex
We test how constructive and destructive regimes emerge, how their
curvature and gradient signatures relate, and how repeated reversal
affects accumulated parameter change. Table~\ref{tab:setup} summarizes
the three model--dataset cases and learning rates. Within each model,
all GD rates share one initialization and the complete training objective
for 3,000 updates. Full-batch GD isolates these dynamics from sampling
noise; subsequent mini-batch continuations test the effect of sampling.
```

**Diagnosis:** Curvature and gradient signatures leaves the measured quantities implicit. The opening should identify the comparison without changing experimental scope.

**Proposed solution:** Name curvature, gradient size and alignment, retain the common initialization, full-objective coverage and update budget.

**Rewrite:**

```tex
We test how constructive and destructive regimes emerge, how curvature
relates to gradient size and alignment, and how repeated reversal
coexists with accumulated parameter change. Table~\ref{tab:setup}
summarizes the three model--dataset cases and learning rates. Within
each model, all GD rates share one initialization and the complete
training objective for 3,000 updates. Full-batch GD isolates these
dynamics from sampling noise; subsequent mini-batch continuations
test the effect of sampling.
```

## IV observed loss trajectories

**Read**

```tex
Figure~\ref{fig:gd-loss} shows a clear separation between smooth and
oscillatory convergence. The lowest MLP and Transformer rates produce
smooth loss curves and $\eta\widehat\kappa_t$ near zero. At higher rates,
the loss develops fluctuations while its longer-term trend continues
downward; the estimated scaled curvature approaches two. ResNet enters
this regime at every tested rate, with earlier transitions at larger
learning rates.
```

**Diagnosis:** Oscillatory convergence claims more than the finite training curves establish; clear separation adds emphasis without information.

**Proposed solution:** Describe the observed smooth and oscillatory loss reduction, retaining the model-specific observations.

**Rewrite:**

```tex
Figure~\ref{fig:gd-loss} shows both smooth and oscillatory loss reduction.
The lowest MLP and Transformer rates produce smooth loss curves and
$\eta\widehat\kappa_t$ near zero. At higher rates, the loss fluctuates
while its longer-term trend continues downward; the estimated scaled
curvature approaches two. ResNet enters this regime at every tested
rate, with earlier transitions at larger learning rates.
```

## IV smoothed curvature interpretation

**Read**

```tex
Near the oscillatory regime, the smoothed $\eta\widehat\kappa_t$ stays
close to two. This identifies the directional loss-balance boundary but
hides how the oscillation changes in size. Figure~\ref{fig:gd-alignment-displacement}
provides the missing information by separating reversal from gradient
growth and accumulated displacement.
```

**Diagnosis:** A smoothed quantity near two is treated as identifying a stepwise loss boundary despite averaging and the nonlinear remainder.

**Proposed solution:** Call two the quadratic reference threshold, state the limits of the smoothed nonlinear observation, then introduce the complementary measurements.

**Rewrite:**

```tex
During oscillatory training, the smoothed $\eta\widehat\kappa_t$ stays
close to two, the quadratic reference threshold for a loss increase.
This average does not classify individual loss changes, and the nonlinear
criterion has a finite-step remainder. Figure~\ref{fig:gd-alignment-displacement}
separates gradient reversal, gradient growth, and accumulated displacement
to show features of the motion that the smoothed curvature alone obscures.
```

## IV-A displacement and final loss

**Read**

```tex
Normalized displacement then declines strongly. The larger steps
accumulate path length faster than net parameter change, so $R_t$
falls even as training continues to reduce the loss. At the highest
rates, the retained fraction after 3,000 updates is approximately
2.6\% for the MLP, 3.7\% for ResNet, and 5.4\% for the Transformer.
Nevertheless, each highest-rate condition reaches its model's lowest
final training loss. These destructive regimes therefore coexist with continued loss reduction
despite extensive local reversal. The risk of an upward loss excursion
depends on how strongly gradient growth amplifies that reversal.
```

**Diagnosis:** Lowest final loss lacks its comparison set. The final risk statement turns the descriptive result into an unqualified mechanism for nonlinear loss excursions.

**Proposed solution:** Limit the comparison to the tested rates and end with the observed coexistence of reversal and loss reduction. Keep every reported displacement value.

**Rewrite:**

```tex
Normalized displacement then declines strongly. The larger steps
accumulate path length faster than net parameter change, so $R_t$
falls even as training continues to reduce the loss. At the highest
rates, the retained fraction after 3,000 updates is approximately
2.6\% for the MLP, 3.7\% for ResNet, and 5.4\% for the Transformer.
Nevertheless, each highest-rate condition reaches its model's lowest
final training loss among the tested learning rates. These destructive
regimes therefore coexist with continued loss reduction despite
extensive local reversal.
```

## IV-A accumulated-interaction result

**Read**

```tex
The complete displacement sums confirm that local reversal need not
dominate accumulated motion. Across all nine trajectories, the sum of
all cross-step interactions is positive, including the seven whose
adjacent-pair sum is negative. Interactions at longer lags more than offset
the adjacent cancellations (Appendix~\ref{app:validation}).
```

**Diagnosis:** The positive accumulated sum can appear inconsistent with the preceding low retention values because their reference quantities are not explained.

**Proposed solution:** Retain the numerical finding and explain the two different comparisons using the existing displacement identity and R_t definition.

**Rewrite:**

```tex
The complete displacement sums confirm that local reversal need not
dominate accumulated motion. Across all nine trajectories, the sum of
all cross-step interactions is positive, including the seven whose
adjacent-pair sum is negative. Interactions at longer lags more than offset
the adjacent cancellations (Appendix~\ref{app:validation}). Positive total
interaction means that squared displacement exceeds the sum of squared
step lengths. This is compatible with low $R_t$, which compares
displacement with the total distance traveled.
```

## IV-A ordering of evidence and interpretation

**Read**

```tex
The transition from constructive to destructive alignment changes how
parameter updates accumulate: nearly opposite gradients undo much of
the preceding step. In a quadratic picture this is overshooting
and returning across a directional minimum. In a neural network, declining
loss together with growing gradient norms also permits descent along a
changing valley while oscillating across its steeper directions. Neither
a move between distinct basins nor overwriting a learned representation
can be inferred from these parameter-space measurements alone. The
functional analysis below asks directly whether representations also
reverse. This separates an interpretable geometric possibility from an
observed change in the learned function.

The complete displacement sums confirm that local reversal need not
dominate accumulated motion. Across all nine trajectories, the sum of
all cross-step interactions is positive, including the seven whose
adjacent-pair sum is negative. Interactions at longer lags more than offset
the adjacent cancellations (Appendix~\ref{app:validation}). Positive total
interaction means that squared displacement exceeds the sum of squared
step lengths. This is compatible with low $R_t$, which compares
displacement with the total distance traveled.
```

**Diagnosis:** The central accumulated-interaction result follows a speculative geometric interpretation, weakening the empirical progression.

**Proposed solution:** Move the revised accumulated-interaction paragraph directly after displacement retention. Retain the geometric interpretation unchanged after it.

**Rewrite:**

```tex
The complete displacement sums confirm that local reversal need not
dominate accumulated motion. Across all nine trajectories, the sum of
all cross-step interactions is positive, including the seven whose
adjacent-pair sum is negative. Interactions at longer lags more than offset
the adjacent cancellations (Appendix~\ref{app:validation}). Positive total
interaction means that squared displacement exceeds the sum of squared
step lengths. This is compatible with low $R_t$, which compares
displacement with the total distance traveled.

The transition from constructive to destructive alignment changes how
parameter updates accumulate: nearly opposite gradients undo much of
the preceding step. In a quadratic picture this is overshooting
and returning across a directional minimum. In a neural network, declining
loss together with growing gradient norms also permits descent along a
changing valley while oscillating across its steeper directions. Neither
a move between distinct basins nor overwriting a learned representation
can be inferred from these parameter-space measurements alone. The
functional analysis below asks directly whether representations also
reverse. This separates an interpretable geometric possibility from an
observed change in the learned function.
```

## IV-B backward-lag interpretation

**Read**

```tex
The early oscillation envelope contracts, rapidly for the MLP and ResNet
and more slowly for the Transformer. Across the full window, alignment
can strengthen again, so the envelope is not monotone. These recurrences
illustrate how local destructive interference coexists with reinforcement
over longer lags. This persistent oscillation across multiple steps, including renewed
alignment at longer lags, reveals temporal structure beyond immediate
neighbors. Because plain GD has no momentum state, this apparent memory
comes from the evolving objective gradients, rather than an explicit
optimizer memory. The complete displacement sums quantify its accumulated effect.
```

**Diagnosis:** Early can be read as early training rather than small backward lag. Envelope and apparent memory add abstraction, and the temporal-structure claim repeats.

**Proposed solution:** Describe peaks as backward separation increases, retain nonmonotone recurrence and the absence of momentum, and remove repeated interpretation.

**Rewrite:**

```tex
Across the first several backward lags, the positive and negative
alignment peaks shrink rapidly for the MLP and ResNet and more slowly
for the Transformer. This describes increasing separation from the fixed
endpoint. At longer lags, alignment can strengthen again, so its
magnitude does not decrease monotonically with separation. These
recurrences show how opposition to the immediate predecessor can coexist
with alignment to earlier gradients. Such dependence across multiple
lags occurs under plain GD, which has no momentum state.
```

## IV-C evolving-trajectory experiment

**Read**

```tex
To examine the directional dilution in Eq.~\eqref{eq:noise-cosine-approx}
during training, we vary batch size at initialization and after 2,000 GD
updates. Learning rates are $0.02$, $0.1$, and $0.2$ for MLP, ResNet, and
Transformer. Each starting state receives 200 updates, with three
initializations, eight sampling repetitions per stochastic condition,
and one full-GD control per initialization. Every condition therefore
has the same number of optimizer steps, but a larger batch exposes the
model to more examples within those steps.
```

**Diagnosis:** The opening sounds like a direct test of the fixed-state cosine approximation, although parameters evolve over the continuations.

**Proposed solution:** Identify the changing gradients and trajectory before retaining the complete experimental design and unequal-exposure qualification.

**Rewrite:**

```tex
We next vary batch size during training, where sampling changes both
the measured gradients and the parameter trajectory. This differs from
the fixed-state calculation in Eq.~\eqref{eq:noise-cosine-approx}.
Continuations start at initialization or after 2,000 GD updates.
Learning rates are $0.02$, $0.1$, and $0.2$ for MLP, ResNet, and
Transformer. Each starting state receives 200 updates, with three
initializations, eight sampling repetitions per stochastic condition,
and one full-GD control per initialization. Every condition therefore
has the same number of optimizer steps, but a larger batch exposes the
model to more examples within those steps.
```

## IV-C observed sampling effects and limits

**Read**

```tex
Figure~\ref{fig:sgd-moving-geometry} shows that the smallest batches move
the strongly negative trained-state alignment toward zero in all three
cases. Sampling weakens coherent reversal, consistent with attenuation
of the destructive regime. However, it also changes gradient norms and
the trajectory, so weaker opposition alone is not a stability guarantee.
At initialization, smaller MLP batches can induce negative mean alignment
despite a positive full-GD mean; in the Transformer, the same small batch
has similar negative cosine from both starting states but decreasing loss
from initialization and increasing loss from the trained state. Batch
size therefore influences both the coherence of updates and which region
of the objective they reach. The joint angle/norm condition explains why
these effects must be considered together.
```

**Diagnosis:** The final sentence invokes the joint angle/norm criterion as a full explanation of SGD loss behavior, omitting the noise term and nonlinear remainder. Multiple observations and their interpretation are packed into one paragraph.

**Proposed solution:** Separate trained-state attenuation from contrasting starting-state outcomes. State the supported conclusion that cosine alone is insufficient, and identify the other terms needed to interpret SGD loss changes.

**Rewrite:**

```tex
Figure~\ref{fig:sgd-moving-geometry} shows that the smallest batches move
the strongly negative mean alignment from the trained starting states
toward zero in all three cases. This indicates weaker reversal in these
continuations. Sampling also changes gradient norms and the trajectory,
so weaker opposition alone does not establish stability.

At initialization, smaller MLP batches can induce negative mean alignment
despite a positive full-GD mean. In the Transformer, the same small batch
has similar negative cosine from both starting states but decreasing loss
from initialization and increasing loss from the trained state. Cosine
alone therefore does not distinguish these loss behaviors. Interpreting
the loss changes requires considering gradient magnitudes, the sampling
contribution in Eq.~\eqref{eq:loss-curvature}, and the finite-step
remainder for nonlinear objectives.
```

## IV-C closing sentence (layout review)

**Read**

```tex
Interpreting
the loss changes requires considering gradient magnitudes, the sampling
contribution in Eq.~\eqref{eq:loss-curvature}, and the finite-step
remainder for nonlinear objectives.
```

**Diagnosis:** The longer closing sentence leaves a three-word tail at the top of the following text page before Section V.

**Proposed solution:** State the same terms more concisely to keep the section ending together, without altering the mathematical scope.

**Rewrite:**

```tex
The loss relation also includes gradient magnitudes, the sampling
term in Eq.~\eqref{eq:loss-curvature}, and the nonlinear finite-step
remainder.
```
