# Conservative Section V readability edits

15 November 2026. Each diagnosis and proposed solution was recorded before its source edit. Unlisted prose was retained. Equations, notation, labels, and subsection order are protected.

## V gradient-response interpretation

**Read**

```tex
We first characterize the gradient response along Adam's trajectory,
then trace the corresponding functional changes through the network.
The following section tests interventions from matched training states.
The complete experimental grid is specified in Appendix~\ref{app:validation}.
```

**Diagnosis:** The reused curvature notation is not immediately distinguished from the actual-step directional curvature. Readers must infer the difference from the mixed projection in the integral.

**Proposed solution:** Explain what the Adam quantity measures immediately after the equation, explicitly distinguishing its projection from the applied-update projection without changing either formula.

**Rewrite:**

```tex
Under Adam, $\eta\widehat\kappa_t$ describes gradient reversal and growth
by projecting the gradient response onto the preceding raw gradient.
That direction generally differs from the applied update. Consequently,
this quantity is not generally the step-direction curvature
$\widehat\kappa_t^{\mathrm{step}}$ in Eq.~\eqref{eq:kappa-step},
despite the retained notation.

We first characterize the gradient response along Adam's trajectory,
then trace the corresponding functional changes through the network.
The following section tests interventions from matched training states.
The complete experimental grid is specified in Appendix~\ref{app:validation}.
```

## V-A observations before mechanism

**Read**

```tex
Unlike the sustained GD oscillations, the Adam trajectories exhibit
distinct excursions and recoveries. This is consistent with the adaptive
preconditioner mechanism analyzed by Bai et al.~\cite{bai2026spikes}:
changing second moments can alter the effective step size independently
of changes in the underlying Hessian. Our measurements describe how the
resulting motion appears in consecutive gradients. Early constructive
intervals give way to persistent reversal.
Prominent excursions occur with negative cosine and growing gradient
norms, producing large peaks in $\eta\widehat\kappa_t$. In ResNet and
Transformer, recovery often includes renewed constructive alignment and
falling norms.
```

**Diagnosis:** The paragraph introduces a mechanism before reporting the observed gradients, and resulting motion can suggest that this experiment established that mechanism.

**Proposed solution:** Report excursions, reversal, norm growth, and recovery first. Then identify the cited Adam mechanism as a compatible explanation, not something isolated by these measurements.

**Rewrite:**

```tex
The Adam trajectories exhibit distinct loss excursions and recoveries.
Early constructive intervals give way to persistent reversal.
Prominent excursions occur with negative cosine and growing gradient
norms, producing large peaks in $\eta\widehat\kappa_t$. In ResNet and
Transformer, recovery often includes renewed constructive alignment and
falling norms. These observations are compatible with the mechanism
analyzed by Bai et al.~\cite{bai2026spikes}, in which Adam's averaging
of squared gradients changes the effective step size independently of
the underlying Hessian. Our gradient measurements do not isolate that
mechanism.
```

## V-A post-excursion interpretation

**Read**

```tex
The loss can reach lower values after an excursion. Such an excursion
could accompany movement into another basin, while momentum can delay
the update's response to a newly changing gradient direction. Whether
that movement helps depends on the current landscape, architecture,
and optimizer state. Related work on SGD catapults shows that spikes can
accompany improved feature learning~\cite{zhu2024catapults}; their role
under Adam must likewise be evaluated together with subsequent progress.
```

**Diagnosis:** A lower loss after an excursion is followed by unmeasured basin-transition and momentum explanations, distracting from the supported observation and practical question.

**Proposed solution:** Retain the observation and the cited SGD precedent, and remove speculative mechanisms that are not tested here.

**Rewrite:**

```tex
The loss can reach lower values after an excursion, although this does
not establish that the excursion caused the improvement. Related work on
SGD catapults shows that spikes can accompany improved feature
learning~\cite{zhu2024catapults}. Under Adam, their role likewise needs
to be evaluated through subsequent progress.
```

## V-A prediction target and clock

**Read**

```tex
After computing the raw gradient, but before applying the next update,
we may want to classify whether that update will increase the loss.
Let $L_t=\mathcal L(\theta_t)$ and define an increase as
$(L_{t+1}-L_t)/L_t\geq0.001$. At decision step $t$, both $g_t$ and $L_t$
are available. However, the pair $(g_{t-1},g_t)$ describes the response
to the preceding update, from $\theta_{t-1}$ to $\theta_t$. Using it to
classify the outgoing update tests whether that response carries forward.
For quadratic full-batch plain GD, the strict condition
$\eta\widehat\kappa_t>2$ exactly characterizes a positive loss change
on the preceding step. This exact retrospective equivalence does not
extend to a finite nonlinear step or to Adam, whose applied direction
also depends on its moments.
```

**Diagnosis:** The next-step target is defined, but its difference from the plotted multi-step excursions is left implicit. The long paragraph combines the empirical task and the theoretical timing qualification.

**Proposed solution:** State the 0.1% single-step target beside the figure's excursions exceeding 10%, then preserve the available-information clock and exact-versus-predictive distinction in a separate paragraph.

**Rewrite:**

```tex
After computing the raw gradient, but before applying the next update,
we classify whether that update will increase the loss by at least
0.1\%. This single-step target differs from the excursions exceeding
10\% marked in Fig.~\ref{fig:adam-overview}. Let
$L_t=\mathcal L(\theta_t)$ and define the target as
$(L_{t+1}-L_t)/L_t\geq0.001$.

At decision step $t$, both $g_t$ and $L_t$ are available. The pair
$(g_{t-1},g_t)$ describes the response to the preceding update, from
$\theta_{t-1}$ to $\theta_t$. Using it to classify the outgoing update
tests whether that response carries forward. For quadratic full-batch
plain GD, the strict condition $\eta\widehat\kappa_t>2$ exactly
characterizes a positive loss change on the preceding step. This exact
retrospective equivalence does not extend to a finite nonlinear step or
to Adam, whose applied direction also depends on its moments.
```

## V-A named rule scope

**Read**

```tex
The \emph{destructive loss-spike condition} combines reversal with
sufficient gradient growth, $\rho_t<0$ and
```

**Diagnosis:** The established name can imply that the table predicts the marked large excursions rather than the specified single-step target.

**Proposed solution:** Retain the name and formula while explicitly tying its use to this next-step prediction task.

**Rewrite:**

```tex
For this next-step prediction task, the \emph{destructive loss-spike
condition} combines reversal with sufficient gradient growth, $\rho_t<0$ and
```

## V-B measured output coordinates

**Read**

```tex
A measured \emph{site} is a named output in the computation graph.
It can be a single layer, an attention module, or the end of a residual
block group; we record that output rather than averaging internal layers.
Write its activation matrix as $A_t^\ell\in\mathbb R^{N_\ell\times m_\ell}$.
The $N_\ell$ rows identify reference examples and, where relevant, spatial or token
positions; columns identify the $m_\ell$ features at each position.
The preceding update changes this representation by
```

**Diagnosis:** The output-space choice is material to Euclidean movement and SVD but is only stated in the appendix.

**Proposed solution:** Bring the existing raw-output versus log-probability convention into the site definition, without changing coordinates or measurement scope.

**Rewrite:**

```tex
A measured \emph{site} is a named output in the computation graph.
It can be a single layer, an attention module, or the end of a residual
block group; we record that output rather than averaging internal layers.
At the final site, we use raw outputs for the MLP and log probabilities
over all classes or words for ResNet and Transformer.
Write its activation matrix as $A_t^\ell\in\mathbb R^{N_\ell\times m_\ell}$.
The $N_\ell$ rows identify reference examples and, where relevant, spatial
or token positions; columns identify the $m_\ell$ features at each position.
The preceding update changes this representation by
```

## V-B aggregate cosine interpretation

**Read**

```tex
A negative $c_t^\ell$ means the new functional change opposes the
preceding one; positive values indicate reinforcement. As $\rho_t$ measures alignment between consecutive raw gradients,
$c_t^\ell$ measures alignment between consecutive functional changes at
site $\ell$, giving a representation counterpart of constructive and
destructive regimes. The root-mean-square (RMS) displacement $a_t^\ell$
measures the change per coordinate. A zero change has undefined cosine.
```

**Diagnosis:** The flattened cosine is interpreted without explicitly distinguishing site-level aggregate opposition from reversal of every feature or example.

**Proposed solution:** State that the cosine concerns the combined coordinates, retain its connection to gradient alignment, and keep the RMS and undefined-case definitions.

**Rewrite:**

```tex
A negative $c_t^\ell$ indicates opposition between the two flattened
changes at site $\ell$; positive values indicate reinforcement. This
aggregate does not require every feature or reference input to reverse.
Thus $c_t^\ell$ provides a representation counterpart to the consecutive
gradient alignment measured by $\rho_t$. The root-mean-square (RMS)
displacement $a_t^\ell$ measures the RMS size of coordinate changes.
A zero change has undefined cosine.
```

## V-B MLP reversal and RMS interpretation

**Read**

```tex
Figure~\ref{fig:trace-motion} shows where functional oscillation appears
and how its size changes. The top row locates the loss excursion.
Reading the cosine maps backwards from output to input reveals how
widely reversal is shared. In the MLP, every displayed change has
negative cosine at every measured site, from the first linear input
map to the final output. Oscillation is therefore homogeneous across
the measured depth, even while loss rises and falls. The RMS maps add
its changing amplitude. Comparing pre- and post-nonlinearity measurements
shows that tanh attenuates displacement; each site's separate
normalization in the figure highlights timing rather than absolute
differences in magnitude across sites.
```

**Diagnosis:** Homogeneous oscillation overgeneralizes shared aggregate cosine signs. The attenuation statement appears to follow from heatmaps that normalize each site independently and therefore hide absolute cross-site scale.

**Proposed solution:** Restrict the depth observation to negative aggregate cosine at every measured site. Explain normalization before amplitude interpretation and identify tanh's mathematical bound, without claiming that the colors quantify attenuation.

**Rewrite:**

```tex
Figure~\ref{fig:trace-motion} shows where functional oscillation appears.
The top row locates the loss excursion. Reading the cosine maps from
output toward input shows that, in the MLP, every displayed change has
negative aggregate cosine at every measured site, even while loss rises
and falls. This shared sign does not imply equal amplitudes or reversal
of every coordinate.

Each RMS row is divided by that site's maximum within the window.
Its colors therefore show within-site variation, not absolute magnitude
differences between sites. For matched pre- and post-tanh coordinates,
tanh cannot enlarge displacement because its derivative has magnitude
at most one. Quantifying that attenuation requires unnormalized
displacements; it cannot be read from these independently scaled rows.
```

## V-B depth observations and causal scope

**Read**

```tex
ResNet and Transformer exhibit a less homogeneous pattern. Their
output-side representations reverse while earlier representations can
continue to reinforce their preceding change. The first displayed output
reversal precedes the 10\% loss marker by 15 steps in ResNet and five
in Transformer. In ResNet, the first convolution reaches its largest
displacement at the loss peak, followed one step later by the remaining
displayed sites. In Transformer, the embedding becomes more oscillatory
near the large excursion and returns to constructive movement sooner
than the later sites. Larger backpropagated errors could recruit earlier
representations into stronger changes, while constructive intervals
could accumulate smaller changes before that reorganization. This
architecture-dependent interpretation motivates examining both the
direction and magnitude throughout the network.
```

**Diagnosis:** The paragraph moves directly from measured timing to an untested backpropagated-error and reorganization mechanism. Homogeneous is also broader than the measured cosine pattern.

**Proposed solution:** Retain every timing and depth observation, describe the contrast in terms of reversal, and close with the descriptive scope rather than an untested mechanism.

**Rewrite:**

```tex
In ResNet and Transformer, output-side representations reverse while
earlier representations can continue to reinforce their preceding change.
The first displayed output reversal precedes the 10\% loss marker by
15 steps in ResNet and five in Transformer. In ResNet, the first
convolution reaches its largest displacement at the loss peak, followed
one step later by the remaining displayed sites. In Transformer, the
embedding becomes more oscillatory near the large excursion and returns
to constructive movement sooner than the later sites. These differences
motivate measuring direction and magnitude throughout the network.
The measurements locate the observed changes without establishing
their causal origin.
```
