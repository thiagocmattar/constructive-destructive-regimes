# Consolidated findings

[Analysis014's Gaussian gradient-noise sweep](../../analyses/014-2026-10-10-rosenbrock-gradient-noise/observations/006-sigma-comparison.md)
keeps Rosenbrock, AdamW decay0.1, start(0,1), three LRs and 2,000 updates,
with sigma0/0.01/0.1/1/10 and one shared seed1234. In this realization, final
loss increases and raw full-prefix B/A decreases with sigma at each LR; the
lowest-loss LR changes with noise level. At sigma10 late raw cosine averages
are near zero while momentum remains0.81-0.82, with much higher objective.
Squared-sum identities still hold pathwise, including realized signal/noise
cross terms; noise-only expected energy/squared sum is not a pathwise equality.
All paths fail joint stationarity. One paired realization does not establish
an ensemble noise law, sampling uncertainty, causal alignment benefit or LLM transfer.

[Analysis013's new-start comparison](../../analyses/013-2026-10-10-rosenbrock-adamw-new-start/observations/001-initial-point-comparison.md)
changes only the Rosenbrock AdamW start to (0,1), with the same three LRs,
decay 0.1 and 2,000 updates. Smallest LR 1e-3 now has the lowest final loss,
0.0000110471, versus 0.234112 at the old start; the midpoint was previously best.
The improved small-LR path has 25.01% negative raw adjacent pairs versus zero
previously, while momentum stays positive. Whole-prefix B/A remains high despite
late negative raw means. The finite-budget ordering depends on initialization;
no causal alignment benefit or general LR ranking follows. All three fail joint
stationarity. Two selected deterministic starts supply no asymptotic/LLM result.

[Analysis012's AdamW LR comparison](../../analyses/012-2026-10-10-rosenbrock-adamw-lr/observations/001-learning-rate-comparison.md)
adds weight decay 0.1 and the geometric midpoint to the 2,000-update Rosenbrock
setup. The midpoint has the lowest final objective of the new decay cohort.
At matched LR 1e-2, decay increases final loss while greatly reducing negative
adjacent momentum pairs; at 1e-3 it reduces final loss. Thus positive momentum
alignment does not uniformly order objective outcomes, and the finite-horizon
decay response depends on LR here. All conditions fail joint stationarity.
Three deterministic paths; no no-decay midpoint control or general/LLM result.

[Analysis011's three-LR Rosenbrock comparison](../../analyses/011-2026-10-10-rosenbrock-lr/observations/001-learning-rate-comparison.md)
holds momentum/no decay fixed for 2,000 updates at LR 1e-2/1e-3/1e-4. Final
loss is 0.000141348/0.325413/4.138634. The fastest valley traversal has frequent
negative adjacent raw/momentum pairs and ongoing oscillation; the slower paths
stay positively aligned with less progress. None reaches sustained joint
stationarity. B/A measures prefix cancellation, without establishing a
convergence proxy or a causal loss benefit. One exact deterministic setup;
full histories and the requested single nine-panel PDF are verified.

[Analysis010's controlled Rosenbrock pilot](../../analyses/010-2026-10-10-rosenbrock-controlled/observations/001-controlled-results.md)
adds four exact2D trajectories with matched start/LR/beta2/horizon. In this
setup, first-moment memory lowers the final objective at both decay settings;
decoupled decay raises it at both momentum settings. The lowest-loss trajectory
has frequently negative adjacent momentum cosine, while decay gives positive
late momentum alignment at a higher objective floor. No condition sustains the
declared stationarity thresholds. [Exact temporal interaction](../../analyses/010-2026-10-10-rosenbrock-controlled/observations/003-temporal-interaction.md)
remains positive in the lowest-loss raw history despite mostly negative adjacent
cosines. These findings distinguish vector interaction from objective benefit;
one deterministic start/function/LR/beta2 is not a universal, asymptotic or LLM
result. Full histories, reviewed PDFs and linked step navigation are retained.

Run001 has verified descriptive observations:
[gradient evolution](../../runs/001-2026-10-03-pythia14m-adamw-gradient-baseline/observations/001-gradient-evolution.md)
and [temporal interactions](../../runs/001-2026-10-03-pythia14m-adamw-gradient-baseline/observations/002-temporal-interactions.md).
Broader interpretations remain for scientific review. Imported manuscript claims
and source-repository results are not new findings here.

Run002 adds a verified [paired raw-gradient/Adam-direction observation](../../runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/observations/001-adaptive-comparison.md):
adaptive median cosine and final B/A are higher in this trajectory, while A and B
remain separated. LR/decay are excluded from the diagnostic direction. This is
a local descriptive result; a broader agreed interpretation is still for review.

Use `F001-short-description.md` for the first supported, agreed interpretation.

[Analysis001](../../analyses/001-2026-10-03-model-scale-geometry/observations/001-scale-comparison.md)
adds verified 14M/31M/70M observations: full-pass median cosines and final B/A
decrease with size for both streams. Absolute raw/adaptive A/B move in opposite
directions, and phase/norm behavior is not uniformly monotonic. Predictability
remains inconclusive; no broader manuscript interpretation is promoted here.
Include claim, evidence/observations, coverage, uncertainty, limits, and manuscript
implications. Keep local observations even when they are inconclusive or do not
justify promotion.

[Analysis003](../../analyses/003-2026-10-04-raw-gradient-regimes/observations/001-raw-gradient-regimes.md)
adds the requested raw-gradient/loss reductions: negative-pair frequency and
unconditional intensity increase in the matched pilot cohort, while conditional
negative magnitude decreases and loss is lower at larger size. Run 001's 14M replay
removes strict unconditional-intensity ordering. Net all-pair raw interaction is
positive at every size; the lag-one squared-resultant estimate misses 98.06-98.54%.
The [manuscript assessment](../../analyses/003-2026-10-04-raw-gradient-regimes/REPORT.md)
proposes a restricted claim, without promoting a dimension-mediated performance
explanation, stability result or general scaling law. The pilot H003 reductions remain
separate from the completed H002 seed replication reported below.

[Analysis003 observation002](../../analyses/003-2026-10-04-raw-gradient-regimes/observations/002-schedule-and-recovery.md)
adds the user-requested schedule check. Identical nominal LR sequences co-occur
with different late alignment across sizes, and 70M returns to negative alignment
at the fixed LR floor. Decay may contribute to recovery, but training progress,
model-dependent dynamics and schedule history remain confounded. H004's causal
mechanism is untested; no schedule intervention was run.

Analysis002 completed six additional full passes, yielding three seeds per size.
[Seed replication](../../analyses/002-2026-10-04-scale-seed-replication/observations/001-seed-replication.md)
reports raw cosine: 3/3 seed blocks; raw B/A: 3/3 seed blocks; adaptive cosine: 3/3 seed blocks; adaptive B/A: 3/3 seed blocks. Data order is matched across sizes within seed and
varies with initialization between seeds. Seed1234 used RTX4090, new blocks RTX5090;
numerical hardware effects cannot be separated from seed effects across blocks.
All per-run values, reversals, A/B/norms and validation are retained; no predictive
law, stability result or causal learning benefit is established.

[Analysis003 observation003](../../analyses/003-2026-10-04-raw-gradient-regimes/observations/003-replicated-conclusions.md)
extends its exploratory F/D/M and raw-interaction reductions to the completed
three-seed cohort. Negative frequency and unconditional intensity increase,
conditional magnitude and validation loss decrease in 3/3 seed blocks; net
all-pair interaction remains positive in 9/9 runs. The user-supported restricted
conclusion is retained. Dimensionality and causal optimization mechanisms remain
unidentified. The [schedule proposal](../../analyses/003-2026-10-04-raw-gradient-regimes/SCHEDULE_EXPERIMENT.md)
tests late decay from shared checkpoints; no intervention was executed.

[Analysis004 observation001](../../analyses/004-2026-10-04-lr-schedule-intervention/observations/001-schedule-intervention.md)
now records the completed first intervention, seed1235 at14M/70M. Accelerated
decay reduces adjacent raw negativity and brings positive-smoothed alignment
earlier; held LR keeps alignment mostly negative and improves validation loss
at both sizes.70M recovery later reverses. F/D size ordering depends on schedule
in this pilot. This supports a restricted schedule-sensitivity/cancellation
observation, with cross-seed validation pending; it does not establish a
dimensionality mechanism, stability result or causal benefit of cancellation.
All new artifacts are verified and all six workers/volumes removed. The existing
original-recipe replicated conclusions and manuscript assets are preserved.


[Analysis005 observation001](../../analyses/005-2026-10-04-temporal-lag-profiles/observations/001-lag-profiles.md)
and [observation002](../../analyses/005-2026-10-04-temporal-lag-profiles/observations/002-all-lag-interaction.md)
record the first single-seed temporal-sketch test at14M/31M/70M. All three middle
windows have negative adjacent interaction but positive exact all-pair interaction;
lag3/4 rebound precedes a positive longer-lag background. Every measured window
has positive exact total interaction. Full-pass lag1 truncation misses97.94-98.57%
of squared cumulative raw-gradient norm. Early/late curves do not uniformly decay
smoothly toward zero over the measured horizon. This supports horizon-dependent
raw-gradient interference and tests the manuscript lag1 assumption directly.
Numerical replay sensitivity, one training seed and nonstationary window averages
limit generality; projection bands are not seed uncertainty. No dimensionality,
stability or performance mechanism follows. The [report](../../analyses/005-2026-10-04-temporal-lag-profiles/REPORT.md)
retains earlier replicated conclusions and proposes narrow wording. All owned
workers removed after artifact verification; multi-seed validation deferred.

[Analysis006 observation001](../../analyses/006-2026-10-07-cumulative-geometry-slopes/observations/001-slopes-and-loss.md)
adds exact A/B and quotient slopes for Run002. Both ratios rise over the last
100 full-sized updates; B increments and prefix alignments remain positive.
Raw B/A is nearly flat at 424-484 while B grows and training loss declines.
This supports distinguishing marginal accumulation from cumulative-ratio
flatness; it does not establish loss convergence or measure information storage.
Six PDFs, including loss-difference grids at lags1/2/5/20, and the source-preserving
reductions are verified locally.

[Analysis008 completed three-LR observation](../../analyses/008-2026-10-08-no-decay-momentum/complete-three-lr/observations/001-three-lr-results.md)
records the warmup-only Pythia-14M cohort, one matched seed and runtime. Final
validation losses are 4.764/6.232/6.636 at LR 1e-3/1e-4/1e-2. The large LR has
much larger raw norms and a smaller raw cumulative B/A (0.0448), with worsening
late loss even though every adjacent momentum and adaptive pair is positive.
Some sparse long-lag momentum pairs are negative. Momentum reduces cumulative
A substantially; comparison with raw B requires accounting for clipped inputs.
This supports distinguishing vector streams, magnitude accumulation and temporal
horizons, without a cross-seed performance law, stability criterion or causal
benefit of momentum. All three complete passes and five updated comparison PDFs
are verified; original interim figures, executed inputs and recovery state are
preserved. No manuscript change or additional experiment was launched.

[Analysis025](../../analyses/025-2026-10-17-pythia-descent-mechanism/REPORT.md) completed27 Pythia runs across
14M/31M/70M and three seed blocks. All18 preconditioning removals increased
fixed-probe loss at the unmodified parent's adaptive-step norm schedule. Paired
70M effects exceeded14M at both forks in each seed block. At update200, the70M seed mean has
a larger favorable linear prediction but a much larger nonlinear penalty after
removing preconditioning; this supports an optimizer-mediated contribution under
this norm schedule. It does not identify a unique200-250 trigger, an optimized
momentum LR comparison, a capacity effect, or Hessian curvature. All artifacts
are verified locally; three PDFs reviewed; task resources deleted; USD11.42 posted.
