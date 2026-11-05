# Consolidated findings

[Analysis043: fixed-step Spark interventions](../../analyses/043-2026-11-03-spark-fixed-step-interventions/FINDINGS.md)
completes 36 matched 64-update branches from exact warm state 1024 at the three
highest tested Adam rates. LR reductions lower MLP mean training loss about
45-47%, while slowing ResNet and Transformer progress despite smoother raw
diagnostics. Double-median clipping is inactive in all cases; median clipping
has mixed effects. Both decay values shrink norms without lowering mean/final
training loss. Fresh Spark controls and exact within-Spark replay qualify these
short responses; the failed historical ResNet cross-GPU gradient gate is retained.
One selected seed/state, unequal LR exposure and a short horizon limit scope;
no universal convergence proxy or held-out benefit is established. All raw
checkpoints and 13 reviewed vector figures are retained, with zero cloud cost.

[Analysis042: cooldown and cosine recovery](../../analyses/042-2026-11-03-mlp-lr-cooldown/REPORT.md)
compares five matched367-update MLP continuations from state657. Eight-update
hold plus eight-update cosine recovery lowers mean/final training loss2.26%/4.30%
relative to always-half here; abrupt hold is20.24%/3.25% worse. Repeated triggers
prevent any complete cosine recovery after the first reduction, keeping365/367
updates below base. No fixed-rate/exposure-matched control establishes incremental
timing value; the training gain is not a held-out gain. One selected development
seed/state, exact replay/resume, original UTC audit failure retained,57.75s local
work and zero cloud cost. Clean figures omit the one-step series only visually.

[Analysis037's three-case SGD comparison](../../analyses/037-2026-11-02-sgd-batch-extension/REPORT.md)
adds 348 verified ResNet/Transformer continuations and reuses 198 MLP continuations,
with three initialization blocks per case. Smallest trained-state batches weaken
negative adjacent alignment while retention means remain below 0.05. Transformer
batch 8 has nearly equal mean cosine from both starts but retention 0.110/0.042
and opposite mean training-loss changes. Adjacent geometry does not order
cumulative motion or loss progress. Selected rates/objectives, unequal exposure,
descriptive ranges and batch/runtime confounding in one ResNet block limit
attribution; no stability/generalization benefit is established.

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

[Analysis027's matched-state mini-batch experiment](../../analyses/027-2026-10-30-minibatch-noise/REPORT.md)
uses three CIFAR-10 MLP initializations, three checkpoints and five batch sizes
at LR 0.02, with all 5k-example reference gradients/losses/HVPs. Frozen variance
and noise-projection targets agree within 0.50%/0.74%; the cosine approximation
has maximum absolute error 0.00745 and detectable bias in 21/36 stochastic cells.
At the initial fork, batch 32 sampled cosine averages -0.141 while full gradients
along those same SGD states average +0.353. At the later fork, cosine closer to
zero does not improve R or loss uniformly. Offline noise correction reduces the
batch 32 median directional-HVP discrepancy 0.7467 to 0.0030. One objective/LR and
200-update forks establish diagnostic interpretation limits, not stability,
generalization or LLM batch requirements. Historical reconstruction does not
qualify; new matched controls replay exactly. Full evidence and five PDFs are verified.

[Analysis030's matched Adam LR pilot](../../analyses/030-2026-10-30-adam-lr-decisions/REPORT.md)
completed18 forks from six Pythia14M/70M states across three seed clusters.
At128 matched updates, doubling LR improves full-validation loss by0.023816--0.050583
relative to holding LR, and wins in every state. Half-rate branches have positive
early raw cosine and worse final loss; doubled-rate branches have negative cosine
and better loss. More positive gradient alignment is therefore not a performance
objective in this comparison. Per-update AdamW shrinkage is held constant.
The fixed-LR and loss-only selectors have zero observed regret; raw/joint feature
extensions give0.011137/0.016247. These prespecified pilot models do not establish
added diagnostic decision value. A common winning LR provides no evidence for
conditional selection, and probing all candidates has an additional compute cost.
One fork stage, small models and three seed clusters do not establish a scheduler,
a curvature estimator or a stability result. All180 files /121.692GB are local;
three PDFs reviewed and independent reductions verified. Task cloud deleted;
USD13.90 posted, provisional. No manuscript changes were made from these results.

## Analysis031: optimizer-state map and paired benefit

The local exploratory reanalysis predicts held-minus-candidate validation loss
at 128, conditional on model size and LR multiplier, with each seed held out.
Actual/adaptive accumulation ratios over the first 16 updates reduce MAE from
0.006643 to 0.004186/0.004174. The result concentrates in doubled-rate arms,
is sensitive to the observation window, and does not persist as a standalone
predictor of additional benefit 16-128. No tested initial-state summary improves
primary prediction. A common best action remains, so scheduling value is unresolved.
This is a candidate for independent validation, not a confirmed prediction law;
all feature/fold outcomes and four reviewed figures are in the
[report](../../analyses/031-2026-10-31-adam-state-map/REPORT.md). No new training.

- [MLP LR return at650 (Analysis045)](../../analyses/045-2026-11-04-mlp-lr-return/FINDINGS.md): abrupt0.75x restoration after a smooth half-LR period brings back large spikes; exact matched comparison, one seed/window.

- [MLP cosine70 restoration (Analysis046)](../../analyses/046-2026-11-04-mlp-cosine-rise70/FINDINGS.md): smooth restoration delays but amplifies the large spike and worsens training loss versus abrupt rise and held-half; matched single-seed evidence.

- [Cosine32 floors versus unchanged (Analysis047)](../../analyses/047-2026-11-04-mlp-cosine-floors/FINDINGS.md):0.75x lowers mean loss yet retains large late spikes at constant LR;0.5x suppresses post-decay excursions in this single MLP window.

- [Two-stage MLP cosine decay (Analysis050)](../../analyses/050-2026-11-04-mlp-two-stage-decay/FINDINGS.md): second decay650..682 reduces peak amplitude/recovery duration and mean loss versus held0.75x; first spike persists, later >10% excursions absent within800. Selected single-seed evidence.


## Persistent LR reductions across multiple spikes (Run223 / Analysis044)

[Controlled findings](../../analyses/044-2026-11-04-multispike-interventions/FINDINGS.md): half LR and cosine32 half lower mean training loss for MLP25.04%/26.30% and Transformer9.66%/10.68%, but raise it for ResNet27.20%/0.62%. Spikes are removed in MLP immediate-half, remain in larger models. Three fresh controls, exact same-worker/cross-worker replay; historical reuse failed. Single seed and selected windows, no generalization or universal threshold claim. All235 files/7 archives verified;3 vector figures reviewed. All task Pods deleted, estimatedUSD7.58 withinUSD15; billing provisional.

- [MLP global gradient clipping1.0 (Analysis051)](../../analyses/051-2026-11-04-mlp-global-clipping/FINDINGS.md): early peak reduction gives way to larger later spikes; mean loss+48.01% and maximum3.82x against exact matched unchanged control. Single-seed Spark evidence.

- [MLP clipping thresholds0.5 versus1.0 (Analysis052)](../../analyses/052-2026-11-04-mlp-clipping-thresholds/FINDINGS.md): smaller cap markedly reduces severe bursts versus1.0, yet mean and tail loss remain above unchanged; exact same-Spark single-window comparison.

[Combined MLP cosine decay and clipping](../../analyses/053-2026-11-05-mlp-cosine32-clip1/README.md) produced the same trajectory as cosine alone. The cap of 1 stayed inactive throughout this seed and window.
