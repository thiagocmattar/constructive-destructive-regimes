# Observation002: joint geometry and comparable-context frequencies

Outputs: [direction/amplification map](../figures/adam_direction_amplification.pdf)
and [matched-context comparison](../figures/adam_matched_context.pdf).
Generator: [02_plot.py](../02_plot.py), from the fixed reductions of
[01_analyze.py](../01_analyze.py). Sources, objective/precision, matching rules
and outcome timing are in the [protocol](../PROTOCOL.md) and manifest.

## Geometry: how to read it

Each panel is one model/LR, with low/middle/high rates in successive rows and
models in columns. Each of 9,207 defined adjacent pairs appears once. Horizontal
position is consecutive raw-gradient cosine; vertical position is consecutive
norm ratio on log axes. Zero cosine and unit ratio divide sign/growth regimes.
Current diagnostics are available before the next update. No momentum or actual
update geometry is substituted for the raw-gradient measurements.

Green circles mark6,559 quiet eligible decisions without a >=10% rise relative
to current loss in the next8 updates; orange triangles mark645 with it. The
remaining2,003 points are gray: startup, nonquiet or incomplete future window.
Counts printed in panels use positive/eligible decisions, not independent
events. The same eligible mask applies in every comparison. No point was
removed because it is a false alarm or an unhelpful example.

The dashed curve rho*r=-1 (Q=2) is the quadratic-GD geometric reference, not a
fit or an Adam boundary. Low-rate ResNet and Transformer occupy its vicinity
with no qualifying future-rise markers. Orange/green overlap shows imperfect
separation. A future eight-update rise can follow a currently nonnegative
cosine or declining norm, since the future path can change regime.

Caption: Raw-gradient direction and amplification across all nine full-objective
Adam arms. Every defined adjacent pair is shown. Green/orange distinguish quiet
states without/with a >=10% future rise within eight updates; gray shows other
observations. Quiet eligibility uses only past/current loss. Dashed curves show
the noiseless quadratic-GD reference rho*r=-1, not a calibrated Adam risk boundary.
Fractions count decision states, not independent excursions. One initialization
per model; no fitted classifier or uncertainty interval.

## Context comparison: how to read it

Each horizontal connector compares two groups after one-to-one matching within
the same model/LR arm. All matched decisions are quiet and have r>1. Top row:
negative versus nonnegative cosine. Bottom row: strong (<-0.8) versus weaker
negative cosine. Six fixed context calipers bound norm, norm ratio, age, current
loss, four-update loss trend and trailing16-state draw-up differences. Outcomes
are not used to choose pairs. Read [matched_summary.csv](../derived/matched_summary.csv)
and [matched_pairs.csv](../derived/matched_pairs.csv) for all denominators.

The horizontal axis is the observed fraction followed by the specified future
rise. A green diamond farther right means greater frequency in the more-reversing
group; the open circle is the comparator. n is the number of pairs, with n
observations per group; a star marks n<20. No error bars are shown because
dependent states from one initialization do not supply an independent-sample CI.

The primary comparison has 288 pairs: six arms favor the reversing group and
three tie at zero. Its local support covers4.6%-19.3% of otherwise eligible
growing/reversing states across arms. Secondary476 pairs show that increasing
reversal strength does not yield a uniform risk ordering. Both contrasts are
retained; states may recur across contrasts. Balance remains imperfect: all
pair calipers pass, but group means can differ. The maximum post-match absolute
standardized difference is 0.611 in sparse high-rate ResNet secondary log norm;
the primary Transformer middle-rate loss-trend difference is 0.445. Reported
frequencies do not establish incremental predictive value beyond loss history.

Caption: Subsequent loss-rise frequency in comparable growing-gradient contexts.
Top panels compare negative/nonnegative raw-gradient cosine; bottom panels
compare stronger/weaker negative cosine. One-to-one pairs share a model/LR arm
and satisfy fixed norm, growth, time and recent-loss calipers, without using future
outcomes in matching. Diamonds/open circles denote more/less reversal. n gives
pairs per contrast; stars mark fewer than 20. Outcomes use a >=10% rise within
eight future updates at quiet decision states. Frequencies are descriptive,
not calibrated probabilities, causal effects or independent-seed estimates.

## Verification and uncertainty

All map counts and 9,207 point coordinates match the derived/raw state records.
All 36 frequency coordinates match independent paired summaries. Maximum matching
cardinality is independently checked in 18 graph cases; all 4,584 pair-caliper
constraints pass. Numeric curve hashes and PDF/font/text checks are in
plot_manifest.json and verification.json. All five publication PDFs were
rendered and reviewed. Threshold/horizon, calipers and secondary split were
fixed before these calculations but informed by previously inspected data.
No p-values, causal mechanism, stability guarantee or untouched validation claim.
