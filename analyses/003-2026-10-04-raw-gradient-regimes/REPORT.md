# Conclusion and manuscript direction

**Narrowing the central claim is the right direction. The three-seed evidence does not support
the broad explanation that larger models perform better because their raw-gradient
regime becomes less destructive. A more specific observation is supported: local
negative alignment depends on size and training phase, and coexists with lower
loss and positive interaction over the complete gradient sequence.**

The [observation](observations/001-raw-gradient-regimes.md) owns definitions,
source/config hashes, coverage, results and caveats. The
[completed replication audit](observations/003-replicated-conclusions.md)
confirms the restricted conclusion in all three seeds. The
[requested panel](figures/loss_and_raw_cosine.pdf) compares training loss and raw
cosine; [the second panel](figures/negative_alignment.pdf) separates negative
frequency and unconditional intensity. This uses completed local evidence only.
No running replication, worker, active log or scientific configuration was changed.

## Size does not have one meaning of "less destructive"

From 14M to 31M to 70M in matched Run 002/003/004 seed 1234 trajectories, last 31-update
token-weighted training loss falls 5.2311 -> 4.5869 -> 3.9846 and complete-validation
loss falls 5.1824 -> 4.6118 -> 4.0632. Negative adjacent raw-gradient pairs increase
49.23% -> 56.96% -> 67.65%; mean max(-cosine, 0) increases 0.1606 -> 0.1753 -> 0.2002.

One narrower reading of the conjecture is supported descriptively: **given a
negative pair, its mean negative magnitude decreases** 0.3262 -> 0.3078 -> 0.2959.
Pairs are slightly shallower by this conditional statistic, but frequent enough
that unconditional negativity rises. These are different hypotheses. The deepest
smoothed trough also changes its ordering with smoothing width.

The strongest difference is late in training: final-quarter negative fractions
are 2.81%, 17.98%, 48.31%. The larger model retains more negative alignment while
its loss remains lower. Some middle-phase orderings reverse. These are continuous
temporal patterns, not evidence for exactly two discrete regimes or a phase transition.

The pilot alone could not supply a reliable scaling law.
Run 001 is a same-seed 14M replay, not another seed; its intensity 0.1764 slightly
exceeds 31M's 0.1753, breaking strict intensity ordering when that replay is selected.
The completed replication retains increasing negative frequency and unconditional
intensity, decreasing conditional magnitude, and lower validation loss in 3/3
seed blocks. This strengthens repeatability within the recipe while preserving
the replay sensitivity. Three seeds still do not establish a scaling law or
that alignment causes performance.

## Dimensionality is an open mechanism

For independent uniform unit vectors u, v in R^d, symmetry gives E[u dot v]=0,
E[(u dot v)^2]=1/d and P(u dot v<0)=1/2 for every d>=2. Increasing dimension in
this toy model narrows cosine magnitudes without reducing negative-sign frequency.
This is an illustrative mathematical baseline, not a model of trained gradients.

A concentration argument can motivate smaller magnitudes under explicit isotropy/
independence assumptions. It cannot alone explain fewer negative pairs or better
loss. Training gradients are structured and coupled across changing parameters.
Width/head configuration, initialization scales and optimizer/curvature effects
co-vary with count here. Effective dimension, isotropy and independence were not
measured. The conditional-magnitude decrease does not identify its mechanism.

Negative alignment need not imply poor convergence even in an exact elementary
example: for L(theta)=theta^2/2 and gradient descent with eta=1.5,
theta_(t+1)=-0.5 theta_t. Consecutive gradients have cosine -1 while loss shrinks
by a factor 0.25 every step. This verified counterexample separates cancellation
from harmful learning. It is not an explanation fitted to Pythia.

## A claim that specifies vector and horizon

I read the working introduction, methods, results, conclusion, related work,
two-step-loss appendix, supplied editor feedback and repository review map.
The draft currently links negative rho to overwriting learned structure, slower
learning and instability, and treats lag-one interaction as a proxy for the full
cross-step term. These links need assumptions and evidence beyond these metrics.

Here net all-pair raw-gradient interaction J is **positive at every size**:
+11435.22, +10081.85, +8710.80. Lag-one truncation misses 98.54%, 98.48%, 98.06% of
the squared cumulative raw-gradient norm. Norm-weighted lag-one interaction is
+3.34, -4.82, +1.68, rather than following cosine medians. Local cancellation,
adjacent norm-weighted interaction and whole-trajectory interaction are distinct
observables. Low norm-of-sum/sum-of-norms also does not prove net negative interaction.

A candidate central claim is:

> Adjacent raw-gradient cancellation characterizes local trajectory geometry,
> but its sign does not determine net cumulative interaction or optimization
> quality. Across three paired seeds in matched Pythia/MiniPile trajectories, larger models reach
> lower loss while exhibiting more frequent local cancellation and positive
> net interaction over the complete gradient sequence.

The first sentence is a diagnostic distinction; the second is explicitly bounded
empirical evidence. A possible title is *Temporal Gradient Alignment in Language-Model
Training: Local Cancellation and Cumulative Reinforcement*. Constructive/destructive
terminology remains usable when attached to a declared vector and horizon.
This is a sharper direction than a general claim about undefined stability,
provided robustness and novelty are demonstrated.

For the rewrite, retain exact SGD identities under stated assumptions; measure
the lag-one residual instead of substituting lag-one for all-pair interaction.
Define local cancellation without equating it to forgetting. Distinguish AdamW
raw gradients, adaptive directions and actual updates. Tie abstract/introduction/
conclusion language to recorded observables. The two-step Taylor expansion also
has noise and curvature terms; its cosine term alone cannot establish the sign
of total loss change. No full equation audit or manuscript replacement was performed.

This is not a novelty assessment. [Xing et al., *A Walk with SGD*](https://arxiv.org/abs/1802.08770)
studies successive-iteration geometry and bouncing through valley-like regions.
[Cohen et al., *Gradient Descent on Neural Networks Typically Occurs at the Edge
of Stability*](https://arxiv.org/abs/2103.00065) documents short-horizon nonmonotone
loss alongside longer-horizon decrease. These primary sources provide context,
not evidence for our Pythia mechanism or an edge-of-stability diagnosis here.
A focused comparison with prior work is needed before positioning this contribution.

The supplied archive, PDFs, reviews and draft were preserved. Proposed wording
is recorded here for scientific review. The completed seeds reinforce this
horizon-specific claim. They do not identify dimension-mediated performance
as its mechanism.

## Follow-up: could the LR schedule explain late recovery?

Yes, as a plausible contributor. [Observation 002](observations/002-schedule-and-recovery.md)
checks the exact schedule and separates common phases. The decline coincides
with recovery; a simple quadratic demonstrates how reducing step size can remove
gradient alternation even while both the higher and lower LR produce decreasing
loss. But raw cosine is invariant to positive scalar rescaling, so any LR effect
is indirect through the evolving parameters and optimizer state.

All three sizes have the identical schedule, with LR fixed at 0.0001 from update 575.
Among pairs 576-711, medians are +0.3602,+0.1560,-0.0034 and negative fractions
2.94%,19.12%,52.21%. The 70M mean cosine changes from +0.0389 to -0.1105 across
two subintervals of that same floor-LR segment. A nominal LR threshold is therefore
insufficient. LR history, model-dependent effective steps, optimization progress
and changing batches remain possible contributors.

The user's earlier-recovery observation is well motivated; 14M's numeric full-pass
cosine median is higher, consistent with lower negativity. The timing of recovery
helps shape the global median, while middle-phase comparisons have different size
orderings. Exact recovery markers depend on smoothing. Existing trajectories
cannot isolate the LR effect from training time; a matched schedule intervention
would be needed. No such experiment was launched. Use "recovery of local alignment"
as the observable, leaving convergence to a separately defined criterion.

## Replication and recommended schedule intervention

[Observation003](observations/003-replicated-conclusions.md) extends the pilot
reductions to all nine completed trajectories without changing Analysis002's
evidence. All three seed blocks retain the F/D increase, M decrease, lower raw
median cosine and B/A, and lower validation loss with size. Net all-pair raw
interaction is positive in 9/9 runs. At the common floor LR, 14M/31M median
alignment is positive and 70M slightly negative in every seed. The restricted
conclusions are retained; causal schedule and dimensionality explanations remain
open.

The [recommended experiment](SCHEDULE_EXPERIMENT.md) compares original decay,
held midpoint LR, and twice-as-fast decay from each shared update356 checkpoint
at 14M/70M, with three seeds. It supplies a direct contrast for the contribution
of late decay, with loss measured independently of alignment. Adaptive schedules
would introduce feedback and are better considered after this comparison.
This is a design recommendation, with no new execution or spending authorization.
