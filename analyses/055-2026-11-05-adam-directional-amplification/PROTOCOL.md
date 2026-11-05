# Step 1: complete-trajectory descriptive protocol

Written before executing this analysis on 2026-11-05. The trajectories and some
historical forecasting results have already been inspected, so this is an
exploratory development analysis, not a preregistered independent confirmation.
This protocol fixes reductions, comparisons and display coverage before their
new results are calculated. Do not tune them to obtain a preferred finding.

## Question and design

Does direction distinguish gradient amplification associated with subsequent
loss excursions from amplification during ordinary training? Vary observed
raw-gradient direction within a trajectory; retain model, LR, seed, objective,
optimizer and measurement contract. Compare similar magnitude/growth/time/loss
contexts descriptively. Direction is not randomized and no causal claim follows.

Use all nine original Run215 arms: MLP LR 0.0001/0.0003/0.001, ResNet-32 and
Transformer LR 0.0003/0.001/0.003; seed0, 1,024 updates, fresh zero Adam moments,
betas 0.9/0.999, epsilon 1e-8, no clipping, decay or schedule. Initial tensors
are identical across rates within each model. Runtime/device differences are
retained in source identities; they are not isolated experimental factors.

The complete objectives are MLP mean half squared error on the prepared 5,000
CIFAR-10 images; ResNet mean cross entropy on all 50,000 training images;
Transformer token-weighted NLL on all 2,088,600 prepared targets. All raw
whole-model gradients are accumulated in FP32 with FP64 geometric reductions.
No validation subset is introduced; held-out prediction/generalization is not
an outcome of this analysis. Original data/cache/model/tokenizer identities
remain pinned by Run215 config and input manifests, not re-created here.

## Clock, measurements and availability

At parameter state s, loss L_s and gradient g_(s+1) are evaluated at theta_s.
Run215 metrics[s] consumes that gradient for update s+1. Define

```
rho_s = cos(g_s, g_(s+1))
r_s   = ||g_(s+1)|| / ||g_s||
Q_s   = 1 - rho_s * r_s = recorded eta*kappa_hat
```

Here rho_s/Q_s are analysis state-indexed names; manuscript gradient indices
are s+1. These measure the incoming response to the preceding displacement
and are available before the outgoing update. Loss from the current forward
pass is already available too. No outgoing actual-step product is a predictor.
Undefined first measurements and zero-denominator cases remain missing, with
no epsilon or zero filling. All recorded finite values remain visible.

Full plots cover all 1,025 losses (0..1024) and all 1,024 prospective raw-norm
records (0..1023); adjacent statistics cover 1..1023. A separately measured
terminal raw gradient exists, but there is no outgoing update at 1024; exclude
that terminal vector consistently from these prospective diagnostic displays.
Norms in time plots are divided by their state-0 norm; r_s in the geometry
plot is the consecutive ratio and is a different normalization.

## Outcomes and eligible decisions: inherited, not optimized

The fixed-reference excursions in Analysis036 remain the event definition.
Reconstruct and audit all 812 candidates, including 806 recovered and six
right-censored. Use >10% recovered events only to shade rise intervals from
pre-rise a to peak p in full plots; censored >10% rises receive an outline
and remain flagged rather than asserted to have recovered.

For a descriptive future-outcome marker, reuse the archived horizon H=8 and
increase delta=0.10, without fitting or reusing any predicted probability:

```
E_s = max(L_(s+1), ..., L_(s+8)) / L_s - 1
Y_s = 1[E_s >= 0.10]
quiet_s = (s >= 64) and (L_s < 1.05 * min(L_(s-16), ..., L_s))
```

The terminal eight states have incomplete future windows, not negative labels.
Quiet eligibility uses past/current loss only. Inspect every state in the full
plots; geometry shows all states with defined rho/r, with quiet complete-window
decisions distinguished by Y. Ordinary states, nonquiet states and zero-positive
arms remain visible. Compare directly to every stored forecast row (4,534) to
audit inherited labels, not to treat old test states as untouched evidence.

Event membership, a future rise relative to current loss, and the historical
trailing-minimum episode definition have different denominators. Do not merge
their counts. Y does not require recovery and is not an optimizer-stability label.

## Descriptive norm/context comparison

Primary comparison: among quiet eligible decisions with r_s>1, compare rho<0
with rho>=0 within each task/LR arm. Secondary comparison, declared now to
address persistent reversal: among the same decisions with rho<0, compare
rho<-0.8 with -0.8<=rho<0. The secondary split is illustrative, not a learned
threshold or a proposed deployment rule. Report both contrasts in every arm.

Pair states one-to-one without replacement within an arm. Require every pair
to satisfy all six fixed calipers:

| Context variable | Maximum absolute difference |
| --- | --- |
| log current gradient norm | log(1.5) |
| log consecutive norm ratio | log(1.10) |
| completed-update state | 128 |
| log current loss | log(1.25) |
| log four-update loss ratio | 0.02 |
| log draw-up above trailing 16-state minimum | 0.02 |

Minimize the sum of squared caliper-normalized differences after maximizing
valid pair count (rectangular Hungarian assignment with an invalid-pair cost
larger than every possible sum of valid costs). Fixed chronological row order
and deterministic tie-breaking; neither Y nor E enters matching. Report
eligible group sizes, paired support fractions, all unmatched counts, pairwise
covariate differences and balance before/after. Do not relax calipers if few
pairs exist. No pair can cross arms; temporal adjacency and overlapping future
windows remain dependent. No independent-event confidence intervals or p-values.

Summarize the two groups' observed Y frequencies, mean E and within-pair
differences. Retain zero-positive groups. Mark fewer than 20 pairs as sparse
and zero pairs as unavailable. These are local descriptive comparisons, not
calibrated probabilities, classifier evaluation or an estimate of causal effect.

## Outputs and interpretation

Five vector PDFs: complete low/middle/high LR-tier trajectories (four rows,
three model columns each); a nine-arm rho/r geometry plot; and a two-contrast
matched-context summary. Time curves are raw green, unsmoothed, unthinned and
unclipped. Loss/norm use logarithmic axes; cosine and Q use linear axes.
Gray bands have a single stated loss-only meaning. The joint plot includes
the GD quadratic reference Q=2 as a geometric reference, never an Adam boundary.
Data-driven axis bounds include every finite measurement; no quantile clipping.

Every figure gets a local observation with source, reduction, caption, counts,
uncertainty and caveats. Manifest pins this protocol before reduction and each
generator; independent verification re-reads raw logs and reconstructs labels,
events, pair constraints, summary statistics and figure arrays. Render every
PDF with Poppler and inspect text/ticks/legends before handoff.

No new classifier, threshold search, validation split, action policy, HVP,
model training, cloud resource or manuscript change. Backlog records tests2-5
and the beyond-200M demonstration separately.
