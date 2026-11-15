# Conservative Section VI readability edits

15 November 2026. Each diagnosis and proposed solution was recorded before its source edit. Unlisted prose was retained. Equations, notation, labels, and subsection order are protected.

## VI intervention question and distributed scope

**Read**

```tex
The preceding analysis shows substantial functional movement repeatedly
reversing with little net displacement. Our working assumption is that
limiting unusually large local changes can reduce the disruptive
component of this oscillation while preserving subsequent progress.
We test a distributed displacement cap that selectively scales Adam's
proposed unit updates according to the representation change they would
produce. Learning-rate reductions, second-moment changes, and gradient
clipping provide complementary controls at other points in the update.
```

**Diagnosis:** The opening introduces a distributed cap without explaining its scope, and the intervention question can become secondary to the method.

**Proposed solution:** Lead with the empirical question, explain distributed as applying the cap to individual units across selected sites, and retain the conventional controls.

**Rewrite:**

```tex
The preceding analysis shows substantial functional movement repeatedly
reversing with little net displacement. We test whether limiting unusually
large local changes can improve training outcomes while reversals persist.
A distributed displacement cap applies the rule to individual units
across selected sites, scaling Adam's proposed increments according to
the direct representation changes they would produce. Learning-rate
reductions, second-moment changes, and gradient clipping provide
complementary controls at other points in the update.
```

## VI-A meaning of the budget

**Read**

```tex
For a positive movement budget $\tau_\ell$ in site $\ell$, scale the
proposed weight-row increment and bias increment by the same factor
```

**Diagnosis:** Movement budget is used before stating precisely which measured quantity it limits.

**Proposed solution:** Define the budget as the limit on the unit's direct RMS change, without modifying the cap equation.

**Rewrite:**

```tex
Let the positive budget $\tau_\ell$ limit each unit's direct RMS change
at site $\ell$. Scale the proposed weight-row increment and bias
increment by the same factor
```

## VI-A scope of the cap guarantee

**Read**

```tex
For example, an update that would move the unit by twice its budget
is halved. A smaller update is left unchanged. This preserves the
direction of the unit's proposed weight-and-bias change while limiting
its direct effect on the current inputs. A zero score uses factor one;
a zero calibrated budget disables that site's cap.
```

**Diagnosis:** The local bound is stated, but the consequence of holding the inputs fixed is not explicit beside the rule. Readers could infer a bound on the full representation change.

**Proposed solution:** Keep the example and edge cases, and distinguish the bounded local contribution from changes propagated from earlier layers.

**Rewrite:**

```tex
An update whose direct RMS change is twice its budget is halved;
a smaller update is unchanged. This preserves the direction of the
unit's proposed weight-and-bias change. The bound covers only its
direct effect at fixed inputs: changes arriving from earlier layers
can still produce larger total representation changes. A zero score
uses factor one; a zero calibrated budget disables that site's cap.
```

## VI-A quantile interpretation

**Read**

```tex
The budget remains fixed during the treated continuation, while the
scores use the site's current inputs at every step. The matched experiment prespecified q50 and also tested q25 and q75;
the illustration below uses q25, with all three retained in the comparison.
Quantiles use linear interpolation.
```

**Diagnosis:** The labels do not explain what their numbers mean and could be mistaken for retained update fractions or fractions of units capped.

**Proposed solution:** Define the percentile labels, connect smaller quantiles to smaller fixed budgets, and retain prespecification, interpolation, and the complete comparison.

**Rewrite:**

```tex
The labels q25/q50/q75 denote the 25th/50th/75th percentiles of this
pooled score distribution, using linear interpolation. Lower quantiles
give smaller budgets, not fixed fractions of an update or preset numbers
of capped units. Budgets remain fixed during treatment, while scores
use each site's current inputs. The experiment prespecified q50 and
also tested q25 and q75; the illustration uses q25, with all three
retained in the comparison.
```

## VI-A prior methods and contribution framing

**Read**

```tex
The closest representation-based comparator is the relative-change
correction of Kosson et al.~\cite{kosson2024warmup}, developed to regulate
updates during warmup. Here the target is the \emph{absolute} per-unit
displacement during an established oscillatory trajectory, using a budget
calibrated from its preceding updates. This avoids dividing by a possibly
small current activation scale and leaves smaller unit updates intact.
AGC instead compares raw-gradient and weight norms~\cite{brock2021normalization};
FOOF constructs a direction for descent in neuron space~\cite{benzing2022neurons}.
Our contribution is to control a different quantity while retaining
Adam's proposed direction within each unit and its evolving moments.
The experiment below tests this choice against conventional interventions;
it does not rank these other representation-based optimizers.
```

**Diagnosis:** The contribution is framed as controlling a new quantity rather than testing an intervention target. Acronyms obscure the roles of the related methods.

**Proposed solution:** Retain the distinctions and references, use descriptive method names, and frame the cap as an experimental test while preserving the scope of the comparisons.

**Rewrite:**

```tex
Kosson et al.~\cite{kosson2024warmup} regulate relative representation
changes during warmup. Here we test \emph{absolute} per-unit displacement
during an established oscillatory trajectory, with budgets calibrated
from preceding updates. This avoids dividing by a possibly small
activation scale and leaves smaller unit updates intact. Adaptive
gradient clipping compares raw-gradient and weight norms~\cite{brock2021normalization};
Benzing derives a descent direction in the space of neuron
outputs~\cite{benzing2022neurons}. Our cap tests local displacement as an
intervention target while retaining Adam's direction within each unit
and its evolving moments. The experiment compares conventional
interventions, without ranking these other representation-based methods.
```

## VI-B intervention families and execution

**Read**

```tex
Learning-rate schedules motivate changing the step scale~\cite{smith2017cyclical,loshchilov2016sgdr};
we test immediate multipliers $.75/.5/.1$. Following the second-moment
mechanism studied for Adam spikes~\cite{bai2026spikes}, we reduce
$\beta_2$ to $.95/.9/.5$. Global gradient clipping~\cite{pascanu2013difficulty}
uses norm thresholds $.5/1$;
and distributed caps q25/q50/q75. The $\beta_2$ changes retain the
moments and step counter, using the new coefficient in both the
recurrence and the native bias correction. Clipping acts on the full
accumulated gradient before Adam. The cap acts on the resulting
proposed update, targeting both hidden MLP linear maps, all ResNet
convolutions, and the Transformer's QKV and feed-forward linear maps.
Untargeted parameters keep their proposed increments. No intervention
waits for a detected loss spike.
```

**Diagnosis:** One paragraph combines intervention settings with optimizer-state handling and parameter targeting. The dangling and distributed caps clause is hard to parse, and QKV is unexplained.

**Proposed solution:** List the four families in complete sentences, then separate their execution details. Expand QKV and retain every setting, state choice, target, and start-time rule.

**Rewrite:**

```tex
We compare four intervention families. Learning-rate schedules motivate
changing the step scale~\cite{smith2017cyclical,loshchilov2016sgdr};
we test immediate multipliers $.75/.5/.1$. Following the second-moment
mechanism studied for Adam spikes~\cite{bai2026spikes}, we reduce
$\beta_2$ to $.95/.9/.5$. Global gradient clipping~\cite{pascanu2013difficulty}
uses norm thresholds $.5/1$. The displacement caps use q25/q50/q75.

The $\beta_2$ changes retain the moments and step counter, using the
new coefficient in both the recurrence and native bias correction.
Clipping acts on the full accumulated gradient before Adam. The cap
acts on Adam's proposed update, targeting both hidden MLP linear maps,
all ResNet convolutions, and the Transformer's query, key, and value
(QKV) and feed-forward linear maps. Untargeted parameters keep their
proposed increments. No intervention waits for a detected loss spike.
```

## VI-B scope of outcome metrics

**Read**

```tex
To distinguish smoother loss from faster training progress, define
```

**Diagnosis:** Faster training progress may be read as a wall-clock efficiency claim, while the experiment matches update counts.

**Proposed solution:** State that the metrics distinguish smoother loss from better outcomes within the same update budget.

**Rewrite:**

```tex
To distinguish smoother loss from better loss outcomes over the same
update budget, define
```

## VI-B motion interpretation

**Read**

```tex
The top row of Fig.~\ref{fig:vc-motion} shows the resulting loss
trajectories, and the paired maps show what changes inside the network.
Large output jumps become smaller, while the cosine maps remain strongly
oscillatory. The q25 cap can therefore improve training while retaining
back-and-forth movement. Its effect is on the size and distribution of
that movement, rather than eliminating every reversal. The complete
window also matters: attenuating the traced ResNet peak does not remove
all later oscillations. Appendix~\ref{app:plots} compares their functional motion with selected conventional interventions.
```

**Diagnosis:** Improve training is broader than the reported outcomes, and distribution of movement is unspecified, potentially suggesting an unreported SVD result.

**Proposed solution:** Tie improvement to the training-loss outcomes and describe changes in movement size across sites and time, retaining the complete-window qualification.

**Rewrite:**

```tex
The top row of Fig.~\ref{fig:vc-motion} shows the loss trajectories;
the paired maps show movement size and reversal at each site over time.
Large output jumps become smaller, while repeated reversals remain.
Thus q25 improves the reported training-loss outcomes while preserving
back-and-forth movement. The complete window also matters: attenuating
the traced ResNet peak does not remove all later oscillations.
Appendix~\ref{app:plots} compares this functional motion with selected
conventional interventions.
```

## VI-B role of the exact output identity

**Read**

```tex
For the squared-loss MLP, the connection to output displacement is exact.
Let $F_t$ be the output matrix on all $N=5000$ training examples,
$Y$ their targets, and $\Delta F_t=F_{t+1}-F_t$. Then
```

**Diagnosis:** The connection is exact can suggest an exact causal account of the intervention, rather than an identity for the observed output and loss changes.

**Proposed solution:** Name the identity as an exact accounting of observed loss changes, retaining all definitions and the equation.

**Rewrite:**

```tex
For the squared-loss MLP, output changes give an exact accounting of
observed loss changes. Let $F_t$ be the output matrix on all $N=5000$
training examples, $Y$ their targets, and $\Delta F_t=F_{t+1}-F_t$. Then
```

## VI-B direction-sensitive linear and squared terms

**Read**

```tex
This identity separates direction from size in function space.
The first term is negative when the output change points toward reducing
the current error. The second is the cost of the movement's size. For all 161 baseline rises and 139 of 140 q25 rises above $10^{-8}$, the first term is negative but the second is larger. A directionally
corrective output change can therefore still be too large. This is an
output-space identity; its first term need not equal the parameter-space
directional derivative, and cross-entropy requires its own loss readout.
```

**Diagnosis:** Separates direction from size is inaccurate because the first term also depends on the size of the change. The identity describes observed changes but does not identify the cap's causal mechanism.

**Proposed solution:** Explain the linear term's dependence on direction and magnitude and the nonnegative quadratic term. Preserve all rise counts, thresholds, and parameter-space/cross-entropy qualifications.

**Rewrite:**

```tex
The first term depends on both the direction and magnitude of the
output change; it is negative when that change points toward reducing
the current error. The second is nonnegative and grows with the squared
size of the change. For all 161 baseline rises and 139 of 140 q25 rises
above $10^{-8}$, the first term is negative but the second is larger.
An output change directed toward reducing error can therefore still be
too large. This accounts for the observed loss changes without
establishing why the cap improves training. The first term need not
equal the parameter-space directional derivative, and cross-entropy
requires its own loss readout.
```

## VI concluding contribution and cost scope

**Read**

```tex
These comparisons support selective control of local movement in the
tested training trajectories. The implementation adds measurable cost;
the recorded MLP cap computation exceeds its full-gradient calculation
(Appendix~\ref{app:interventions}). One initialization per model and unequal
intervention strengths limit conclusions about a universal optimizer
ranking. The consistent reductions in mean loss and upward variation
nevertheless show that persistent oscillation can be made less disruptive
without requiring consecutive changes to become constructive.
```

**Diagnosis:** The conclusion can better align with the intervention-target contribution and explicitly distinguish fixed-update loss outcomes from elapsed-time efficiency.

**Proposed solution:** Retain the cost and comparison limitations, frame the result as support for the intervention target, and state that faster wall-clock training is not established.

**Rewrite:**

```tex
These comparisons support local representation displacement as an
intervention target in the tested trajectories. The implementation adds
measurable cost: the recorded MLP cap computation exceeds its full-gradient
calculation (Appendix~\ref{app:interventions}). The results do not establish
faster wall-clock training. One initialization per model and unequal
intervention strengths limit conclusions about a universal optimizer
ranking. The consistent reductions in mean loss and upward variation
show that persistent oscillation can be made less disruptive without
requiring consecutive changes to become constructive.
```
