# Analysis055: what the complete trajectories add to the Adam argument

The first descriptive step is complete. Direction has a useful-looking local
association with later loss amplification after restricting norm/context
differences, but a universal negative-cosine/growing-norm alarm is contradicted
by ordinary periods and low-rate controls. This is evidence for studying a
conditional warning diagnostic, not a validated classifier or an Adam stability
criterion. The manuscript and Analysis036 remain unchanged.

## Reading route

1. Inspect [high-rate trajectories](figures/adam_full_trajectories_high.pdf),
   then [low-rate controls](figures/adam_full_trajectories_low.pdf).
   The [middle-rate trajectories](figures/adam_full_trajectories_middle.pdf)
   complete the nine-arm cohort. These are complete-history companions to the
   author's preferred Analysis036 multiple-spike figure.
2. Read the [direction/amplification map](figures/adam_direction_amplification.pdf).
   This asks where future loss increases occur in the same geometry occupied
   by ordinary observations.
3. Read the [matched-context comparison](figures/adam_matched_context.pdf).
   This asks if the contrast remains when magnitude/growth/time/loss context
   are restricted to similar ranges, and shows when support is sparse.

## What we measured, in plain language

Each point represents a training state before its next Adam update. We know
the current loss and gradient, compare this gradient with the preceding one,
then look forward in the saved history. The descriptive marker is:

**Does loss reach at least1.10 times its current value during the next8 updates?**

We inherited that marker from the old analysis rather than choosing a horizon
or threshold to improve this result. For the colored comparison we retain
states at least64 updates into training whose loss is still less than 5% above
its trailing16-state minimum. This asks about developing excursions while loss
is comparatively quiet. All other states remain in the full trajectories and
are gray in the joint map. The marker does not require eventual recovery.

There are 7,204 eligible decision states, of which645 have the future-rise marker.
These include the earlier time periods omitted from the old forward-test table.
All 4,534 archived forecast labels agree exactly with our reconstructed outcomes.
The new7,204-state frequencies are descriptive full-history summaries, not new
test-set prediction performance. Consecutive decisions often concern the same
future excursion, so645 is not645 independent spikes.

Full plots include9,225 loss observations,9,216 norm records and 9,207 defined
adjacent gradient pairs. At s=0 the adjacent pair is undefined; the final loss
at 1024 is shown. Prospective vector histories end at 1023. Loss states 1017..1024
lack an eight-update future; seven of these have prospective vector rows here.
They remain visible and are not labeled negative. No smoothing or thinning.

## Finding1: the original pattern repeats, but its background matters

The high-rate view shows repeated MLP and Transformer norm bursts/reversals.
The high-rate ResNet instead has a prolonged later period with positive cosine,
small projected change and steadily declining loss/norm after its large events.
Persistent raw reversal therefore is not a requirement for every stage of Adam
training. Its meaning depends on the accompanying magnitude and trajectory.

The low-rate views are especially informative. ResNet and Transformer show
strong reversal and repeated or sustained gradient amplification without any
eligible >=10% future rise within eight updates. The low-rate ResNet norm grows
by tens of times its initial value even as loss keeps falling. A large norm,
negative cosine, or their informal conjunction does not by itself identify
a pronounced future excursion.

| Lowest-rate control | Quiet eligible states | Negative cosine | Q>2 | Qualifying future rises |
| --- | ---: | ---: | ---: | ---: |
| ResNet,0.0003 | 953 | 726 | 186 | 0 |
| Transformer,0.0003 | 953 | 764 | 215 | 0 |

These are full-eligible-history counts, not the archived later-block alarm
counts. An absent rise within eight updates says nothing about a different
unmeasured horizon. Q=2 remains a GD quadratic reference, not an Adam threshold.

## Finding2: future-rise states occupy a shared, not exclusive, region

In the joint map, horizontal position is consecutive cosine rho and vertical
position is consecutive norm ratio r (log scale). Left means reversal; above
one means norm growth. Orange triangles indicate a future rise, green dots an
eligible state without it, gray dots an ineligible state. This is the same
raw-gradient geometry for every point, not a plot restricted to spike events.

Many orange observations lie toward reversal and amplification, particularly
for Transformer. They overlap green observations; the two lowest-rate controls
populate this geometry with no orange observations. In the MLP, many future-rise
states already have a temporarily declining norm. The present direction and
growth can change before the future event unfolds.

Among645 eligible positive decisions,38 have nonnegative current cosine and 262
have r<=1 (groups can overlap). Thus neither current reversal nor current norm
growth is necessary for this eight-update future marker. This does not conflict
with the noiseless quadratic-GD completed-step relation: that relation is about
a different update/outcome, and Adam follows an adaptive direction.

The dashed curve is rho*r=-1, equivalently Q=2. Its role is to explain the
direction/magnitude product geometrically. Orange and green points on both
sides prevent presenting this curve as a learned risk boundary.

## Finding3: direction retains a local association in comparable contexts

The main comparison restricts attention to quiet decisions with growing norms.
We pair a negative-cosine state with a nonnegative-cosine state in the same
model/LR trajectory, with bounded differences in six quantities: current norm,
norm ratio, training age, current loss, recent loss trend and recent draw-up.
Pairs are chosen without future losses entering the matching objective.

For example, current norms may differ by at most a factor1.5, norm ratios by
a factor1.10, and ages by 128 updates. The [protocol](PROTOCOL.md) gives every
bound and the fixed maximum-count/minimum-distance assignment. These are
similar contexts within explicit tolerances, not identical optimizer states.

| Model | LR | Pairs | Negative rho: later rise | Nonnegative rho: later rise |
| --- | ---: | ---: | --- | --- |
| MLP | 0.0001 | 17* | 2/17 (11.8%) | 0/17 (0%) |
| MLP | 0.0003 | 12* | 5/12 (41.7%) | 2/12 (16.7%) |
| MLP | 0.001 | 7* | 3/7 (42.9%) | 2/7 (28.6%) |
| ResNet-32 | 0.0003 | 50 | 0/50 (0%) | 0/50 (0%) |
| ResNet-32 | 0.001 | 41 | 1/41 (2.4%) | 0/41 (0%) |
| ResNet-32 | 0.003 | 14* | 3/14 (21.4%) | 0/14 (0%) |
| Transformer | 0.0003 | 37 | 0/37 (0%) | 0/37 (0%) |
| Transformer | 0.001 | 69 | 9/69 (13.0%) | 0/69 (0%) |
| Transformer | 0.003 | 41 | 13/41 (31.7%) | 0/41 (0%) |

Star means fewer than 20 pairs, a descriptive sparsity flag with no inferential
meaning. Six arms have higher observed frequency for reversal, three tie at
zero, none favor nonnegative cosine in this selected common-support comparison.
Do not summarize this as nine independent replications: rates share one model
initialization, pairs/states are dependent and several outcomes concern the
same excursions. A rate here means observations followed by an excursion,
not the fraction of independent spikes predicted by a classifier.

Primary matched coverage is limited: only 4.6%-19.3% of eligible growing,
negative-cosine states find a nonnegative counterpart within the fixed bounds.
The unmatched periods remain in the other figures and exported tables. The
comparison supports a local association within overlapping contexts, not a
conclusion about every reversing state.

The bottom row addresses persistent reversal: match rho<-0.8 against
-0.8<=rho<0 with the same rules. This predeclared secondary contrast yields 476
pairs across nine arms. It is more favorable for strongly negative MLP states
(at the highest rate,15/20 versus 5/20 future-rise observations), but differs
little in the highest-rate Transformer (9/34 versus 8/34) and ties in middle-rate
ResNet (2/52 versus 2/52). More negative cosine is not a universal risk ordering.
The0.8 split is illustrative, not a selected alarm threshold.

Residual context imbalance also matters. Calipers bound each pair but do not
make group means identical. The largest remaining absolute standardized mean
difference is 0.611 for log gradient norm in the sparse high-rate ResNet secondary
comparison. The primary Transformer middle-rate comparison retains a0.445
standardized difference in the four-step loss trend (absolute log difference
0.0042). See [balance.csv](derived/balance.csv). Thus this analysis does not
establish information beyond a comparably expressive recent-loss/norm predictor.

## What goes into the argument now

Supported: repeated excursions have interpretable raw-gradient dynamics;
complete-trajectory controls reject a universal alarm; local context comparisons
make a direction-conditioned warning hypothesis worth testing. The observable
is available before the outgoing update and needs no additional HVP.

Still open: calibrated probability, incremental out-of-sample usefulness,
beneficial action, an Adam stability boundary, noise-induced Adam stabilization,
and a measured >200M scalability result. The [backlog](BACKLOG.md) preserves
these as separate work rather than conclusions from the figures.

Suggested main-text sequence: explain the observable; use the winning micro
figure; show complete-trajectory controls; use the context comparison if its
limited support is stated; finish with the precise warning hypothesis.
The joint map is explanatory and can be supplementary if space is tight.

## Verification and provenance

[manifest.json](manifest.json) pins protocol, generator, source histories and
derived tables. [plot_manifest.json](plot_manifest.json) pins five vector PDFs
and the numeric hashes of 99 plotted records /46,098 points.
[verification.json](verification.json) independently audits all 9,216 state rows,
812 event boundaries,4,534 historical labels,18 maximum-cardinality matching
cases,764 pair records across both contrasts and 4,584 pair-caliper checks.
Pairs can share states across the two separate contrasts;764 is not a unique
independent sample size. All five figures passed text/font/page and visual QA.

There is one seed per model and no independent-seed uncertainty interval.
All source trajectories, old figures, manuscript files and unrelated user work
remain preserved. Local scalar reanalysis only; zero cloud cost or retained
cloud storage attributable to this task.
