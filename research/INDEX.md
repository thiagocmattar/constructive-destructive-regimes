# Research index

## MLP distributed-cap quantile and LR sweep, 8 November 2026

[H049 / Runs 286-288](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/013-distributed-cap-quantile-lr.md) completes 54 continuations across five quantiles, three LRs and three initializations. q25 absorbs 2x/4x LR increases (87.88%/94.75% lower mean loss than same-LR Adam), but increasing LR does not improve any quantile's across-seed mean over 1x. q50 at 1x has the lowest observed mean; held-out tradeoffs and residual oscillations remain. All 162 audits, six historical reproductions and four PDF pages pass. Raw evidence retained on Spark; compact verified outputs local after disk-limit recovery. USD 0 cloud cost; processes exited. Continue Analysis 065; next run 289 and unused H050.

## Local representation cap report, 8 November 2026

[Analysis 065 technical report](../analyses/065-2026-11-07-mlp-frozen-lr-double/report/local-displacement-cap-report.pdf)
consolidates the representation-informed stabilization method, numbered algorithm,
entry/distributed scopes, setup, cap-only results and MLP replication. Three new
manuscript-style figures use log loss and linear eta-kappa-hat, with independent
column scales. All 61,450 plotted values, 12 table rows and eleven rendered pages
verified; terminal test-loss limitations retained. Existing figures preserved.
LaTeX source and reproduction are local to 065; no new training or cloud work.

## Representation-informed transfer, 7-8 November 2026

[Analysis 065 / H048](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md) completes Runs 269-285.
Distributed local displacement caps lower mean training loss 22.85% / 16.22% /
26.00% for MLP / ResNet / Transformer; two further MLP seeds improve 19.43% /
32.08%. Reversals persist or increase, while terminal test loss is worse under
the cap in every setup. This supports movement-sensitive optimization control,
not a generalization or universal optimizer claim. Embedding LR boosts are mixed;
failed gates and directional controls are retained. All 68 new continuations,
35,452 updates, independent audits and 14 PDF pages verified. Five task pods
deleted; no retained cloud storage. Conservative cost USD 18.75 (billing
provisional), below USD 20. Continue in 065; next run 286, unused H049.

## Selective MLP update gates, 7 November 2026

[Analysis 065 / H047](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/006-selective-update-gates.md)
completes prefix calibration and ten branches (Runs 258-268). Neuron displacement
cap q75 lowers mean loss 10.42% and positive loss variation 79.37%; tail mean is
5.68% higher than unchanged Adam. It beats its paired first-layer norm control
by 4.63% in mean loss, while hidden/output reversals remain at every update.
Discarding large coordinates hurts mean loss; coordinate capping helps modestly.
Tighter neuron capping is worse, and its uniform norm control has the best new
full-window mean. No general superiority claim; one seed/fork. Two new PDFs,
270 retrieved files, independent audits and all three pages verified. Workers
exited; 385.42 s total, 21.907 GB retained on owned Spark, zero cloud cost.
Continue in 065; next run 269, unused H048.

## Selective MLP learning rates, 7 November 2026

[Analysis065 / H046](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/005-selective-layer-lr.md)
completes Runs252-257: first-layer/global/later-layer LR0.5x/0.25x from448, all
layers trainable. First-layer0.25x lowers mean loss7.06%, positive variation99.62%
and hidden reversals350/353 to1/353; last32 mean is15.79% higher than baseline.
Later-layer controls improve mean loss while retaining reversals; weaker damping
can postpone spikes. New three-page figure, previous figures preserved. All
source/retrieval/dense/50832 plotted-value/render checks pass. Workers exited;
49.75s successful execution,13.720GB retained on owned Spark, zero cloud cost.
Continue in065; next run258, unused H047.

## Ongoing MLP interventions, 7 November 2026

[Analysis064/065](../analyses/065-2026-11-07-mlp-frozen-lr-double/README.md) remains
one study; I1-I8 each compare with unchanged Adam. First-layer freezing suppresses
original excursions but does not uniformly improve training progress. LR0.02
from500 overshoots to107.099; halving through0.01/0.005/0.0025/0.00125/0.000625
ends at a strictly decreasing500-800 trajectory. At0.0025 a delayed peak appears
at707;0.00125 has only a0.006% first-step bump. The strict rule is recorded;
0.000625 is the first pass along this sequence, not the largest stable rate.
Runs247-251, full-state fork equality, all source/retrieval identities and dense
reductions verified. Four PDF pages reviewed. Workers exited,40.62s total,
10.034GB retained on owned Spark, zero cloud cost. No further runs after stopping.
Continue in065 until user requests another analysis; next run252, unused H046.
A separate [I6/I5 comparison](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/004-i6-i5-comparison.md)
uses independent linear loss axes, retaining all eight overview diagnostics.

## First-layer freeze intervention, 7 November 2026

[Run 244 / Analysis 064](../analyses/064-2026-11-07-mlp-freeze-first-linear/README.md)
freezes the first MLP weights/biases at 448 with all Adam settings unchanged.
Output reversals450-500 fall 51/51 to 4/51 and maximum output RMS falls 87.38%;
mean loss448-800 rises 22.27%. This supports a causal contribution to the observed
oscillation, not a unique first-layer cause. Figure includes eta kappa hat and
full/updated-layer gradient norms. Full-state qualification,54 input/21 retrieval
hashes, numerical checks and both pages pass. Worker exited,2.067GB retained on
owned Spark, zero cloud cost. Next Run 245 / Analysis 065 / H045.

## Architecture-wide oscillation tracing, 7 November 2026

The [five-page technical report](../analyses/063-2026-11-07-architecture-instability-tracing/report/instability-tracing-report.pdf)
consolidates the method, formal algorithm, findings and costs. MLP is newly fitted
on 450-500; fixed SVD patterns are explicitly distinguished from potentially
rotating motion. All seven report/companion pages reviewed; no new training.

[Analysis063](../analyses/063-2026-11-07-architecture-instability-tracing/README.md)
completes Run243 / H043 with a [formal tracing algorithm](../analyses/063-2026-11-07-architecture-instability-tracing/METHOD.md)
and standardized ResNet/CIFAR-10 and Transformer/WikiText-2 diagnostics. ResNet's
early reversals become more closely coupled to output in later stages; Transformer
output reverses while embedding motion can remain forward. Fixed probes preserve
sample/token structure; training objectives remain complete. No unique causal
source is established. Both cloud workers deleted after verified retrieval;
MLP causal interventions remain TODO. Next Run244 / Analysis064 / H044.


The [MLP companion](../analyses/063-2026-11-07-architecture-instability-tracing/observations/003-mlp-standardized-trace.md)
adds the matching two-page diagnostic over448--800 using all5000 training images
and all five learned representations. Prior numerical results reproduced; both
pages reviewed. No new training, cloud cost or causal intervention.

The [consolidated scientific section](../analyses/063-2026-11-07-architecture-instability-tracing/observations/004-consolidated-instability-tracing.md)
now separates motivation, per-update/window method, results and conclusions.
[Reusable workflow](INSTABILITY_TRACING.md), linked from AGENTS.md, covers probe
selection, stochastic training, computation/storage costs and verification.

## MLP gradient amplification, 7 November 2026

[Analysis062](../analyses/062-2026-11-07-mlp-gradient-amplification/observations/001-gradient-amplification.md)
completes Run237 / H042. The MLP has no RMSNorm; Adam denominators do not
collapse. Exact replay shows reduced cancellation between example gradients,
growing output oscillations and larger increments along already sensitive
directions. Crossed-Jacobian probes and exact output loss identities support
finite-step overshoot; no unique initiating cause established. All536 reference
pairs,15 anchors,57 inputs and28 artifacts verified; two vector figures reviewed.
Worker exited, about391MB retained on owned Spark, zero cloud cost.
[Run238 extension](../analyses/062-2026-11-07-mlp-gradient-amplification/observations/003-control-coherence-448-800.md)
adds the requested five-row control figure448--800 with norm-based intra-batch
coherence. All801 reference pairs and1,765 plotted values verified; worker exited,
about74MB retained on Spark, zero cloud cost. Run239 subsequently adds output-change alignment as final panel(g), using
outputs447..801; all2,471 values in the seven-row PDF pass verification. Exact
control qualification through800 and the extra update pass; worker exited,
about183MB retained on Spark, zero cloud cost. Next Run240 / Analysis063 / H043.

## MLP Adam spike internals, 7 November 2026

[Analysis061](../analyses/061-2026-11-07-mlp-adam-spike-internals/observations/001-spike-internals.md)
completes Run236 / H041. Exact Adam replay shows small parameter-norm changes
and nearly constant hidden RMS at peaks474/493/503/522. Static probes favor
output-sensitive hidden-feature motion and finite-step overshoot; restoring
norms leaves nearly all excess loss. Tanh attenuates changes, but unsaturated
entries carry almost all change energy. All536 replay loss/norm pairs and
state512 full tensors/RNG match;42 artifacts and1,372 plotted points verified,
two vector figures reviewed. Worker exited; about888MB retained on owned Spark,
zero cloud cost, no manuscript changes. Next Run237 / Analysis062 / H042.

## GD learning-rate extension, 7 November 2026

[Analysis060](../analyses/060-2026-11-07-mlp-adam-to-gd/observations/003-intermediate-gd-rate.md)
adds Runs234--235: GD0.03/0.04 mean losses0.0439842/0.102867; GD0.08 overflows
at547. Same original state512 and exactly qualified retained Adam control.
The current PDF shows0.02/0.03/0.04 with10,584 verified plotted scalars;
the0.08 comparison and original single-rate figure remain archived. All107
new artifacts verified;611 scientific updates plus16 qualification updates and
3 failure-audit replay updates. Workers exited; about408MB retained on owned
Spark including archives, zero cloud cost. Exploratory one-seed/window results;
no manuscript edits. Next Run236 / Analysis061 / H041.

## MLP beta2 and GD interventions, 7 November 2026

[Analysis059](../analyses/059-2026-11-07-mlp-beta2-reduction/README.md) completes
Run232: beta2=.95/.9/.5 raises window-mean training loss16.78%/18.16%/127.02%.
Native bias correction changes with beta2; original moments/clocks retained.
[Analysis060](../analyses/060-2026-11-07-mlp-adam-to-gd/README.md) completes
Run233: GD at.02 removes observed spikes but raises mean/tail loss2.30%/55.74%
versus continued Adam at.001. Exact state512 control replay passes in both.
All1152 scientific updates,80 artifacts and14,112 plotted scalars verified;
two vector PDFs visually reviewed. Workers exited; owned Spark retains about
525MB including archives, zero cloud cost. One seed/window; no manuscript edit.
Next Run234 / Analysis061 / H041.

## Intervention consolidation, 6 November 2026

Additional [Analysis058 Figure 08](../analyses/058-2026-11-06-intervention-consolidation/observations/008-spike-optimizer-zoom.md)
extends the manuscript spike zoom with cos(g_t,m_t), cos(g_t,d_t) and momentum/gradient
norm ratios. All 1,760 values and the seven-row render verified; original windows and
manuscript figure preserved. Nine PDFs now; no training or cloud cost.

7 November: [Analysis058 Figure 07](../analyses/058-2026-11-06-intervention-consolidation/observations/007-optimizer-overview.md)
extends the high-LR overview with momentum norm and adjacent moment/Adam-direction
cosines. All 21,495 plotted values pass read-back; original rows and Figure 00
are preserved. Eight PDFs now; no training, cloud cost or manuscript change.

[Analysis058](../analyses/058-2026-11-06-intervention-consolidation/README.md)
collects seven numbered paper-style intervention figures from retained evidence.
Immediate half LR, MLP schedule variants, clipping and Pythia schedules retain
their matched controls and full declared windows. All 69,354 plotted values and
seven vector PDFs pass numerical/visual checks. No new training, spending or
manuscript edit; the 6 November manuscript rollback remains intact.
Next analysis: 059; no run or hypothesis allocated.


## Current state

The active manuscript has been restored to its pre-review version (`89ec071`)
at the author's request. The reviewed PDF, TeX, figures and tables are preserved
in the [review-session archive](../manuscript/archive/2026-11-05-review-session/README.md).
No experiment or scientific evidence was changed; next Analysis058.

[Analysis057: Adam response and memory](../analyses/057-2026-11-05-adam-response-memory/observations/001-response-memory.md)
completes the Adam author-input manuscript revision from retained evidence.
All 794 rise-phase updates are locally descending; log-norm plus cosine worsens
all seven held-out-rate severity fits despite in-sample gains. Threshold/censoring
accounting, two new PDFs and the analytic memory bridge are verified. The revision
includes an explicit TODO and Runpod/Spark/local plan (core expected USD39-48,
proposed USD70 ceiling; pilot first). No new training or cloud spend.
Next analysis: 058; no run or hypothesis allocated.

[Analysis056: GD paired displacement](../analyses/056-2026-11-05-gd-paired-displacement/observations/001-paired-displacement.md)
completes the author-requested GD/SGD manuscript revision from retained evidence.
All nine runs have positive all-pair interaction despite seven with negative
adjacent interaction; paired factors locate cancellation without implying local
descent. Both offsets, nine lag endpoints and 546 SGD paths are checked.
Four new PDFs and the 16-page manuscript pass numerical and visual review.
No training or cloud cost. Next analysis: 057; no run or hypothesis allocated.

[Analysis055: Adam directional amplification](../analyses/055-2026-11-05-adam-directional-amplification/FINDINGS.md)
completes the nine-arm descriptive comparison and five reviewed PDFs.
Reversal has higher local matched future-rise frequency in six arms, with
three zero ties; support is limited and residual imbalance remains. Low-rate
controls reject a universal Q>2 alarm. All 9,216 states and 46,098 plotted points
pass independent checks. Argument/evidence map and tests2-5/>200M backlog are
organized separately. No fitted classifier, training, cloud or manuscript edit.

8 November: [Analysis055 timing/severity audit](../analyses/055-2026-11-05-adam-directional-amplification/observations/004-timing-severity-audit.md) verifies all-step, magnitude and 1-16-update lead-time structure with false alarms and independent raw checks. Results strengthen the single [paper consolidation map](../manuscript/consolidation/PLAN.md); manuscript and source histories remain unchanged. No new training or fitted predictor.

[Analysis043: fixed-step Spark interventions](../analyses/043-2026-11-03-spark-fixed-step-interventions/FINDINGS.md)
completed all 36 matched branches from warm state 1024, with 64 full-objective
updates each. LR reductions lower MLP mean training loss about 45-47% here,
while slowing ResNet/Transformer progress despite smoother raw diagnostics.
Double-median clipping never activates; decay reduces norms without lowering
mean/final task loss. All 347 transferred files, 45 endpoints and 13 PDFs pass
verification; exact within-Spark replay passes. Run221's ResNet cross-GPU gradient
gate failure remains retained, with no historical bitwise replay claim. The Spark
worker exited 06:52 BRT before cutoff; zero cloud cost, no manuscript edit or push.
H029 complete with model-dependent short responses. Next Run223 / Analysis044 / H030.

[Analysis042: cooldown and cosine recovery](../analyses/042-2026-11-03-mlp-lr-cooldown/REPORT.md)
completes five matched367-update arms. Cosine recovery improves mean/final
training loss2.26%/4.30% over always-half in this selected case; repeated triggers
keep365/367 updates below base. Abrupt hold remains inferior to always-half.
Clean four-arm figures omit the one-step series at the author's request; full
evidence retained. Exact replay/resume passes after excluding UTC audit metadata;
original failure retained. Active57.75s, zero cloud, no manuscript edit.
Next Run223 / Analysis044 / H030.

[Analysis041: temporary LR using $\eta\widehat\kappa_t$](../analyses/041-2026-11-03-mlp-temporary-lr/REPORT.md)
completed three 367-update continuations from the authenticated state657 parent.
The fixed1.8 rule reduces LR on18 updates with36 switches but gives worse mean/
final training loss and larger replacement excursions than fixed controls.
Always-half remains strongest in this selected case. Exact replay/overlap checks
pass; a Windows I/O failure/recovery and19 replayed updates are retained. Active
workers37.26s; zero cloud cost, no tuning or manuscript change.

[Analysis040: local matched MLP pilot](../analyses/040-2026-11-03-mlp-step-actions/REPORT.md)
completed all8 x128-update arms from the exact state657 fork; historical and
contemporary replay pass. Half LR and proposal clipping meet the training screen;
raw numerator/secant arms worsen excursions. Single seed,64.06s worker, zero
rental; all raw evidence retained locally. A7 deferred, no manuscript edit.

[Analysis038: Adam spike intervention design](../analyses/038-2026-11-03-adam-spike-intervention-design/PROTOCOL.md)
proposes eight matched arms per model, 32 updates before each formal pre-rise
state, with 128-update continuations. Source audit verifies three events and
675 extended-context rows; all 28 rise updates have a locally descending actual
step despite increasing loss. Intermediate Run215 recovery states were overwritten;
new common prefixes and untreated replay qualification are required. H028 proposed;
no intervention training or paid execution. Next Run223 / Analysis044 / H030.

[Analysis037: three-case SGD batch-size comparison](../analyses/037-2026-11-02-sgd-batch-extension/REPORT.md)
completed 348 new continuations plus 198 reused MLP continuations, with all three
initializations and sampling repetitions retained. The one-row mean-cosine figure and 16-page
manuscript pass numerical/visual review. Smallest trained-state batches weaken
reversal while retention means stay below 0.05; adjacent cosine alone does not
order retention or training loss. All 13 task Pods/disks deleted; 0 GB cloud storage;
USD 26.30 provisional / 4.663 hours within USD 40/six hours. Spark project 1.379 GB
retained with no task measurement active; old queue untouched. H023; no new
hypothesis. Next Run217 / Analysis038 / H028.

[Analysis036: controlled Adam spike anatomy](../analyses/036-2026-11-02-controlled-adam-diagnostics/README.md)
uses nine fixed-rate, 1024-update trajectories from matched initial weights.
The [current manuscript](../manuscript/draft/revisions/2026-11-03-original-argument-feedback-feedback/README.md)
addresses the next 30 author annotations: literal deletions, clearer raw-gradient
semantics under Adam, simpler equations, verified percentages and a compact
spike-boundary sketch. The 13-page review preserves the original introduction,
prior revisions and empirical assets; source, mathematical and page checks pass.
No overfull boxes or unresolved references remain.
The Adam evidence includes the loss overview, largest recovered spikes,
fixed-reference definition, 806-event density, grouped 129-event >10% anatomy
table and 127-point per-LR regression scatter. Pre-rise cosine is negative
in 117/127 defined events; all 129 endpoint gradient norms grow. Recovery is
required and the 10% threshold is operational. R-squared uses log peak/pre-rise
loss; temporal association, prospective prediction and causality remain distinct.
No new training, intervention, cloud resource or hypothesis.
A [multi-spike review figure](../analyses/036-2026-11-02-controlled-adam-diagnostics/observations/011-multiple-spikes.md)
now shows 19 complete >10% excursions with linear relative norms; the original
figure and manuscript are preserved (2026-11-05).

[Analysis035: backward fixed-endpoint profiles](../analyses/035-2026-11-02-eos-backward-anchors/REPORT.md)
implements rho_t(k)=cos(g_t,g_(t-k)), k=1-256, with one fixed endpoint per curve.
Run212 recomputes GD and cold Adam from nine identical saved starts; endpoints
756/2256/3256 follow exactly 256 updates. All 18 profiles / 4608 pairs and 90 local
vector audits pass; two PDFs reviewed. Historical forward/adjacent definitions
remain separate, and substantial later FP32 replay drift is explicitly recorded.
All 18 task Pods/storage deleted after verified retrieval; USD 9.26 compute provisional.
One seed, no stability or optimizer-benefit claim; no new hypothesis allocated.

[Analysis034: Adam from identical GD states](../analyses/034-2026-11-01-eos-adam-continuations/REPORT.md)
completed five-rate calibration and 27 matched-horizon cold-Adam forks.
Selected constant LRs: mlp 0.0003, resnet 0.001, transformer 0.001; 25 primary, two flagged reconstructions.
Exact raw-gradient anchored/adjacent cosines at every boundary, up to lag 2048.
Terminal 64 median adjacent cosine positive in 10/25 primary profiles.
Training-only short calibration; optimizer reset, one initialization, no stability
or independent optimizer-benefit claim. Four PDFs and numerical/vector checks pass.
All 21 task Pods/storage deleted after verified retrieval; USD 39.35 provisional.
Runs209/210/211; H026 conditional comparison complete. Next Run212 / Analysis035 / H027.

[Analysis033: exact long-lag profiles](../analyses/033-2026-11-01-eos-long-lags/REPORT.md)
completed 27 profiles / 36,096 anchored cosines, all through lag 256
and up to 2048; 25 primary, two previously unqualified Run062 starts.
Of 17 primary initially reversing profiles, 4 have terminal local half-gap <=0.1;
13 remain above it. Attenuation can rebound; no asymptotic or stability claim.
Runs206/207/208 use exact streaming and qualified checkpoint handoffs, without
compression. Four PDFs and numerical/vector/provenance checks pass.
All 20 task Pods/storage deleted after verified retrieval; USD 29.48 provisional.
Next Run209 / Analysis034; H020 remains active.

[Analysis032: exact gradient cosine through lag 10](../analyses/032-2026-11-01-eos-exact-lag10/REPORT.md)
completed 27 windows / 270 exact cosines near states 500/2000/3000 across nine
Analysis024 conditions. There are 21 exact saved starts, four qualifying
reconstructions and two failed Run062 replays retained separately. All 17 primary
negative-lag-1 profiles have negative odd / positive even lags through 10.
All 147 files / 7.072 GB are local and hash verified; three task pods/storage deleted.
Compute USD 1.05 provisional; posted billing pending. Next Run206 / Analysis033.
H020 remains active; one initialization and one anchored pair per lag.

[Analysis031: Adam state map and paired-benefit prediction](../analyses/031-2026-10-31-adam-state-map/REPORT.md)
reuses all 18 Analysis030 arms; 12 altered-rate outcomes and three seed clusters.
At update 16, actual/adaptive retention reduce held-seed final-benefit prediction
MAE by 37.0%/37.2% beyond model/LR means. No tested initial summary improves it;
window/target sensitivity and shared best action prevent a controller claim.
H025 is an exploratory candidate; H024 remains unresolved by this design.
Four PDFs pass visual review; 1,068 features, 1,956 predictions and 6,840 plotted
source values pass independent checks. No training, cloud cost, or manuscript edit.
Next Run205 / Analysis032 / H026.

[Analysis030: completed matched Adam LR pilot](../analyses/030-2026-10-30-adam-lr-decisions/REPORT.md)
completed18 forks /2,304 updates /4.83B input tokens from six Pythia14M/70M states.
The2x LR wins in all six states despite negative early raw-gradient cosine.
Fixed-LR and loss-only selectors have zero regret; raw/joint extensions have
mean regret0.011137/0.016247. H024 is unsupported in this pilot; no conditional
scheduling value is established. All180 files /121.692GB are locally verified;
162 features,72 predictions and three PDFs pass numerical/visual review.
Task compute/storage deleted October30 at21:49 BRT; USD13.90 posted, provisional.
No cloud storage or remaining compute; Spark untouched. Next Run205 / Analysis031 / H025.

[Analysis029: Pythia diagnostics across LR, schedule and size](../analyses/029-2026-10-30-pythia-adam-diagnostics/REPORT.md)
adds four reviewed vector figures from22 existing event files. Projected gradient
change reuses the SGD eta-kappa expression without an Adam curvature claim.
Incoming-gradient/prior-moment alignment is recovered from stored EMA geometry;
4,977 reconstructed clipped norms agree to max relative error3.40e-7.
All303 curves /192,399 values pass independent checks. The proposed next step is
matched LR forks evaluated by later loss; no training/cloud or manuscript edit.
Next Run187 / Analysis030 / H024.

[Analysis028: directional diagnostics under Adam](../analyses/028-2026-10-30-adam-diagnostics/README.md)
replaces the old optimizer/image section with a practical extension separating
raw gradients, moments, adaptive directions and actual parameter motion.
Two compact tables reuse 13 hash-verified event files; all 40 displayed values
and the geometric qualifications pass checks. Matched schedule branches show
alignment recovery without superior validation loss. Clipping limits the
raw-norm-to-momentum interpretation; diagnostic-driven scheduling remains a
proposal. Sections III--V and previous evidence are unchanged. The 17-page
PDF is reviewed; no training, remote work or spending. Next Run187 / Analysis029 / H024.

[Analysis027: mini-batch noise at matched training states](../analyses/027-2026-10-30-minibatch-noise/README.md)
completed frozen sampling and 297 matched 200-update forks across three MLP
initializations, three checkpoints and five batches. All full-gradient/loss/HVP
references cover 5k/5k examples. Frozen moment errors are below 0.75%; the cosine
approximation has small detectable bias. Noise correction reduces batch 32 median
HVP discrepancy 0.7467 to 0.0030. Historical reconstruction does not qualify; new
matched controls replay exactly. Section V now pairs the batch-size outcomes
with a temporal view explaining the full-batch endpoint; frozen calibration is
condensed and Eq. (19)'s scope clarified. All 1,140 plotted means pass independent
checks. The 20-page draft is reviewed;
all 15.192 GB of original evidence remain locally hash verified.
Task Runpod pod/volume deleted; approximately USD 0.87 provisional, zero cloud
storage. Spark and held queues untouched. Next Run187 / Analysis028 / H024.

[Analysis026: completed fixed-CE curvature experiment](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md)
completed30 paired conditions and300k updates on owned5070Ti. Common fully
FP64 full480-example Hessians give fixed/native curvature ratio 0.014885, 95% CI [0.0030771,0.072007];
Hard-collapse exposure qualifies 0/10 seeds. Matched-loss curvature is similar;
Adam-scaled curvature rises 2.41x. These are attained-state, conditional findings.
All evidence is local and hash verified; publication PDF reviewed.
Zero rental and no cloud resources created. Next Run185 / Analysis027 / H023.

[Analysis025: completed Pythia preconditioning experiment](../analyses/025-2026-10-17-pythia-descent-mechanism/REPORT.md)
completed27 runs and4,050 updates across three sizes and three seed blocks.
All18 norm-controlled preconditioning ablations increased fixed-probe loss;
70M effects exceeded14M in each paired block at both forks. The unique200-250
transition remains unresolved. All103.516GB are locally verified; three PDFs
reviewed. Task pods/volume deleted October17 at19:05 BRT; posted cost USD11.42/$25.

[Analysis024: completed3k GD LR calibration](../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/004-completed-calibration.md)
completed all nine conditions: MLP0.004/0.02/0.04, ResNet0.025/0.05/0.1,
Transformer0.05/0.1/0.2; seed0 and3000 full-GD updates per condition.
Five new runs122-126 finished October20 at13:47 Sao Paulo; four conditions
reuse059/062/120/121. All254 source files,26,991 pairs and279 spectra pass.
Transformer0.2 final train/test losses6.069747/5.858303; final B/A0.05413294,
last1k mean cosine-0.9936598. All scalar/recovery evidence is local and on Spark.
Training/retrieval ETC0; GPU idle at32C. The queue exited without continuation.
Monitor SSH failure recovered by direct collection; task monitor stopped.
Five new runs retain160 files/6.395GB per machine; redundant archives removed.
The final October20 PDF covers all nine conditions in4x3 panels,81,279 exact
points; all32 previous curves are unchanged. Source/array/render checks pass.
Six isolated1x3 companion PDFs now show log loss, raw/smoothed cosine,
smoothed q/norms and raw R_t;50-step trailing means are explicit.
[Observation005](../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/005-isolated-three-column-figures.md) records all54 verified curves and visual review.
Larger-model Hessians remain subsets. The [IEEE revision and manuscript integration](../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/006-ieee-figures-and-manuscript.md)
preserves all54 Y arrays, adds two composites and inserts SectionIV with
all nine endpoints and explicit scope limits. The [applied surgical revision](../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/008-applied-section-iv-revision.md) adds
local/all-lag evidence and a retrospective loss audit; all36 table rows and
26,991 loss pairs pass independent checks. The18-page draft is reviewed.
Single-initialization comparisons do not establish initialization robustness.
Owned Spark only, zero rental; old020/021 queues stay held.
Next Run154 / Analysis026 / H022.

[Analysis023: smallest-LR3k calibration](../analyses/023-2026-10-15-eos-smallest-lr-calibration/README.md)
completed MLP LR0.004 (Run059 prefix), ResNetLR0.1 (Run120) and
TransformerLR0.05 (Run121), seed0 and3000 full-GD updates each.
All79 retained files (3.23GB),8997 adjacent pairs and93 spectral states pass checks.
Final losses are0.3365/0.4898/6.6301; late1k mean raw cosines are
1/-0.9684/0.999995 and final B/A values0.7941/0.03737/0.7223.
Only ResNet has negative adjacent pairs in this window; first at update231.
Training finished October17 at01:05 Sao Paulo time; retrieval/reduction at01:10.
ETC0, owned Spark idle and local monitors closed. The reviewed MLP/ResNet3k
PDF now includes Transformer: three columns, four rows,27093 exact plotted points.
All eight previous MLP/ResNet curves are unchanged; only the norm row is log Y.
All later020/021 training stays held for review. No rental or manuscript edit.
Single-seed calibrations do not establish universal LR or stability effects.
Next Run154 / Analysis026 / H022.

[Analysis022: EOS long-horizon plan](../analyses/022-2026-10-15-eos-long-horizon-plan/README.md)
proposes MLP20k and ResNet/Transformer3k full-GD updates, three seeds/LR:
30 conditions including four complete MLP runs;26 new trajectories.
Measured production costs project348.94 Spark GPU-hours before allowance.
Two4090 cloud scenarios span55.6-109.3h at unmeasured4x-2x speedups;
historical120h/USD220 cloud proposal was not launched; staged023 supersedes it.
Analysis020 pending jobs stay held after023, as requested by the user;021 stays held. Next Run154 / Analysis026 / H022.

[Analysis021: EOS full-batch GD learning-rate comparison](../analyses/021-2026-10-15-eos-multitask-lr/REPORT.md)
holds 16 additional owned-Spark runs for replanning: three LRs and three
seeds per case, matched 500 updates, 11 reused prefixes. Extra compute 38.73h;
combined ETC about 56.5h (79 h allowance). The 3x3 figure and insertion after
Table 1 are prepared and mechanically/layout checked; final results remain
pending; no new updates started. Only075 and023 finish; later training held; no cloud rental.

[Analysis020: EOS task and noise extension](../analyses/020-2026-10-14-eos-multitask/REPORT.md)
completed owned-Spark calibration and three-seed frozen-gradient sampling on
CIFAR-10/MLP, CIFAR-10/ResNet-32 and WikiText-2/Transformer. The control has36
conditions and147,456 observations, retrieved/hashed. The October20 control
figure follows draft rho_t/R_128/eta*kappa_hat_t/epsilon_t/mathcal L; all27 seed
curves, nine means and ranges are unchanged. Frozen scope is explicit; no new run.
The reviewed seed0 GD overview now covers all three cases at500 updates:
loss, eta*lambda_max and eta*kappa_hat, with one existing LR per case.
Runs061/071/074,73 retained files,1,497 adjacent pairs and18 spectra pass checks;
the figure follows Table1 in the reviewed14-page report. Other Transformer seeds
and SGD remain held for review. Only the agreed075/120/121 can finish; no later training or rental.

[Analysis019: CIFAR-10 GD/Adam learning-rate sweep](../analyses/019-2026-10-13-cifar10-eos-lr/REPORT.md)
completed Runs059-066 on Spark:20,000 fixed updates each,common5k tanh MLP
initialization and full10k test evaluation. All160k rows and4,816 residual-checked
ordinary/preconditioned eigenpairs pass source/scalar/state/forward checks; four
PDFs are archived; the six-panel GD and ten-panel Adam full/early figures pass array/render review. Final train losses GD0.09688,0.05651,0.03204,0.01945,
Adam0.0007658,0.0008776,0.0007075,0.0006594 in increasing LR order. Actual training/diagnostics52.0min;
ETC0,peak allocation<0.35GiB; all outputs retrieved/hashed,owned Spark idle.
One frozen seed,no LR tuning; no global stability/convergence or alignment proxy
claim. Full scalar/recovery evidence retained locally/on Spark; no cloud rental.
Next Run154 / Analysis026 / H022.

[Run058: CIFAR-10 Edge of Stability calibration](../runs/058-2026-10-13-cifar10-eos-calibration/REPORT.md)
implements the confirmed5k tanh MLP with GD/Adam and raw/m/mhat/adaptive/update
geometry, kappa and ordinary/preconditioned Hessian spectra. Both owned GPUs
pass96short updates, math/state/hash verification. For20kupdates and Hessians
every100, early timings project5.2/8.0min on RTX5070Ti and6.2/8.4min on Spark;
peak allocation<0.35GiB. Plan about30min for a pilot pair; late eigensolver
cost/convergence remain unmeasured. No full trajectory/cloud rental launched;
844MB local/696MB Spark evidence retained; GPU idle. Next run059.

[Analysis018: eta-kappa with gradient noise](../analyses/018-2026-10-12-rosenbrock-kappa-noise/README.md)
reuses immutable Runs038-052: sigma0/0.01/0.1/1/10, three LRs, one paired
seed1234. The five-row overlay repeats the gradient-only quantity and includes
the measured normalized noise term separately. At sigma10 final100 means
1.050/1.072/1.095 become0.0186/0.00426/0.00270 with epsilon included.
All29,985 pairs,28,500 full means per metric and30 plotted curves pass checks;
both100/250-pair PDFs pass visual review. A separate250-pair version checks all52,500 plotted means. Noise correction removes the explicit difference,
but AdamW update mismatch remains; no curvature-estimator or stability claim.
Original PDFs preserved;2.66 MB derived histories retained locally outside Git.
No new optimizer run, cloud resources or billing; one seed limits inference.

[Analysis017: gradient-identity kappa](../analyses/017-2026-10-12-rosenbrock-kappa-identity/README.md)
reuses immutable Rosenbrock Runs055-057, with noise epsilon0. Broad100-pair
trends correlate with lambda_max (0.942/0.905/0.753), but amplitude differs;
the exact discrepancy separates AdamW step mismatch from nonlinear remainder.
A separate eta-kappa overlay preserves the original comparison and shows late
means1.790/1.755/1.748, with later transitions at smaller LR. All5,997 pairs,
5,700 complete means and three PDFs pass numeric/source/array/layout review.
No new optimizer trajectories, cloud resources or billing; this SGD-derived
quantity is a proxy on AdamW, not a verified Hessian estimator or stability test.

<!-- analysis015:start -->
[Analysis015: larger Spark learning rates](../analyses/015-2026-10-10-spark-high-lr/observations/002-completed-results-and-review.md)
completed Runs053/054 at LR0.02/0.04: 712 updates each, complete training and
validation, zero overflows. Final validation6.63438/5.19195 versus matched
Spark LR0.01 reference6.63649. Raw negative-pair fractions63.99%/62.59%;
momentum remains positive, with lower full-prefix B/A than the reference.
All120 retained files (15.88GB) are retrieved/hashed; both final PDFs pass
visual review and independent scalar/coverage/source checks. One seed1234,
one pass and LR-scaled decay limit interpretation. Training ETC0; Spark idle;
recovery states retained locally and on Spark, outside Git. No cloud rental.
<!-- analysis015:end -->

[Analysis014: Rosenbrock gradient noise](../analyses/014-2026-10-10-rosenbrock-gradient-noise/README.md)
completed Runs041-052: sigma0.01/0.1/1/10, three LRs, one paired seed1234,
2,000 updates each; sigma0 reuses038-040. Final loss rises and raw full-prefix
B/A falls with sigma at every LR in this realization. At sigma10 raw late mean
cosine is near0, momentum0.81-0.82, despite higher loss. All fail joint
stationarity; exact identities retain realized signal/noise cross terms.
All 24,000 new updates pass replay/independent optimizer/hash checks; five
navigation PDFs, five identity PDFs, a sweep PDF and a 5x3 norm grid pass review.
New raw histories18.63 MB retained locally; no cloud resources or billing.
One paired realization supplies no ensemble uncertainty or general noise law.

[Analysis013: Rosenbrock from (0,1)](../analyses/013-2026-10-10-rosenbrock-adamw-new-start/README.md)
completed Runs038-040 with Analysis012's exact recipe, changing only the start.
Final losses at LR 1e-2/midpoint/1e-3 are 0.000972263/0.000117958/0.0000110471;
the smallest LR now has the lowest final objective, versus the prior midpoint.
Its raw negative-pair fraction rises from zero to 25.01% despite much lower loss.
All fail joint stationarity. Three matched old/new pairs, replay/optimizer/
geometry/hash checks and the same styled nine-panel PDF are verified and reviewed.
The PDF marks each LR's first y-turn at updates 81/228/655 in all metric panels.
Raw histories 2.84 MB retained locally; no cloud resources or billing.

[Analysis012: Rosenbrock with weight decay](../analyses/012-2026-10-10-rosenbrock-adamw-lr/README.md)
completed Runs035-037: weight decay 0.1, LR 1e-2/0.003162277660/1e-3,
2,000 updates each. Final losses 0.001042468/0.000115599/0.234112; the
midpoint is lowest in this cohort. At 1e-2, decay raises final loss versus
the matched no-decay control despite mostly positive momentum cosine;
at 1e-3 it lowers final loss. All fail stationarity. The nine-panel PDF now uses
100-step cosine means and linear norm/A/B axes; all replay/optimizer/hash checks pass, including the Run030
prefix. Raw histories 2.91 MB retained locally; no cloud resources or billing.

[Analysis011: Rosenbrock learning rates](../analyses/011-2026-10-10-rosenbrock-lr/README.md)
completed the requested momentum/no-decay refinement: LR 1e-2/1e-3/1e-4,
2,000 updates each (Runs032-034). Final losses 0.000141348/0.325413/4.138634;
largest LR reaches the optimum neighborhood but continues oscillating, whereas
smaller LRs remain positively adjacent-aligned with less progress. All fail
the joint stationarity test. One large full-trajectory panel plus the exact
eight-panel diagnostic grid, including B/A ratios, is reviewed in a
single PDF. Replay/optimizer/geometry/hash checks pass; Run032 exactly matches
the Run028 prefix. Raw histories 2.85 MB retained locally; no cloud resources or billing.

[Analysis010: controlled Rosenbrock](../analyses/010-2026-10-10-rosenbrock-controlled/README.md)
completed four deterministic local CPU trajectories (Runs028-031),10,000 updates
each, varying momentum beta1=0/0.9 and decoupled decay=0/0.1 at matched
LR0.01/beta2=0.95/start(-1.2,1). All Analysis009 g/m/d metrics plus actual
displacements and exact temporal sums are retained. Every update/replay/hash
passed; eight PDFs and an offline linked step explorer passed visual/browser QA.
Final losses in run order are3.452e-6/0.02245/0.0009731/0.08264. None meets the
final100 sustained stationarity criterion. Lower loss with mostly negative
momentum cosine contrasts with the decay condition's positive late momentum
alignment; no alignment/convergence proxy or LLM transfer is established.
Raw histories17.13MB retained locally; no new cloud resources or billing.

[Analysis009 completed metric comparison](../analyses/009-2026-10-09-intermediate-lr/complete-seven-lr/README.md)
adds four intermediate warmup-only14M LR points (Runs024-027). All four completed
712 updates/1.492B input tokens and full validation at18:36-18:37 UTC, with zero
overflows and all g/m/d reconstruction checks passing. Their36 compact result
files are retrieved/hashed; eight seven-rate comparison PDFs passed visual review.
The combined raw-norm panel(e) is logarithmic; a new2x2 cumulative LR overlay
uses reversed plasma, with darker colors for larger LR. A new3x2 loss/raw-B/A/
alignment/norm view uses a50-update raw-cosine mean and10-update raw-norm mean.
A3x2 squared-geometry view compares A squared, B squared and their difference
for raw gradients and stored momentum, with independent logarithmic Y scales.
The latest4x2 PDF shows unsmoothed training loss followed by one norm panel per
LR: faint raw norms beneath a10-update mean, with fill to unsmoothed momentum.
Common log Y limits and all712 updates include the marked partial final batch;
source arrays, unsmoothed final/minimum metrics and other PDFs are preserved.
The current LR-versus-norm metric subtracts the trajectory minimum from the
final raw norm, using all712 updates: difference8.162 atLR0.01 versus0.978-2.085
elsewhere. The partial endpoint is labeled; the earlier ratio evidence is retained.
New validation losses are5.600857/5.237702/4.410370/4.318091 in increasing LR
order; LR0.00464159 is best among the four new hardware-matched conditions.
Its late momentum cosine0.852 is below0.956 atLR0.00215443, which has worse loss.
One-step alignment does not establish convergence/stability or long-lag effects.
The [manuscript interpretation](../analyses/009-2026-10-09-intermediate-lr/complete-seven-lr/observations/004-implications-for-manuscript.md)
audits full-prefix raw all-pair interaction: positive at six LRs, negative only
at0.01 (also without the partial tail); the late sign change is not an early-warning result.
Frozen initialization/order/recipe match the original Spark conditions, but GPU
architecture/runtime differ across old/new runs. Full recovery downloads and
cloud cleanup are ongoing. Two owned two-GPU RTX4090 workers and60GB volume
mja1prpve4 remain; rateUSD3.56/h combined, deadline20:30 UTC, bounded estimated
compute/storageUSD12.69 within the approvedUSD15 cap. Independent guard and
compressed-batch retrieval controller are active; evidence retained until verified.

[Analysis008](../analyses/008-2026-10-08-no-decay-momentum/complete-three-lr/README.md)
completed all three Pythia-14M full passes on DGX Spark: warmup-only LR 1e-3,
1e-4 and 1e-2 (Runs021–023). Every run covered 712 updates and 1.492B input
tokens, passed full geometry/coverage verification and was retrieved with all
sizes/SHA-256 verified. Five final comparison PDFs, six per-run PDFs and the
automatic joint figure passed visual review; six interim PDFs are preserved.
Final validation loss is 4.764/6.232/6.636: baseline LR 1e-3 is best in this
matched single-seed cohort. LR 1e-2 has much larger raw norms and clipping,
raw B/A 0.0448 and worsening late training loss despite positive adjacent
momentum/adaptive cosines. Momentum lowers A substantially; its B stays close
to raw B at the two smaller LRs but differs at the largest, where clipping matters.
All runs share exact frozen initial tensors, order and runtime; LR scales decay.
Sparse checkpoint profiles do not recover missing dense short-lag histories.

The sequential queue finished at 2026-10-09 12:59:25 UTC (09:59 Brasilia),
after about 15 hours. Training ETC is zero; the 13:03 UTC check found GPU idle
at 38 C. All conditions had zero overflow retries. The local retrieval controller
finished automatically; recovery states and about 21.62 GB of retrieved outputs
remain locally and on Spark, with shared inputs also retained remotely. No cloud
rental or new experiment was launched. Scientific inputs remain frozen;
next IDs are Run028 / Analysis010 / H009. No stability or causal mediation claim.

[The completed Run021/022 interim analysis](../analyses/008-2026-10-08-no-decay-momentum/interim-021-022/README.md)
has the requested six-panel comparison and two coarse checkpoint temporal
supplements; all three PDFs are reviewed. At equal full-pass coverage, LR 1e-3
reaches lower validation loss (4.764 vs 6.232) despite more negative raw pairs
(84.53% vs 26.30%). Momentum remains positively adjacent-aligned in both.
Raw/momentum and adaptive B/A order the conditions differently; one seed does
not validate a geometric performance/stability proxy. Dense short-lag histories
are unavailable; the supplements explicitly use saved vectors at updates 100–700.

The [focused 2x2 view](../analyses/008-2026-10-08-no-decay-momentum/interim-021-022/observations/003-loss-alignment-norms.md)
marks the first negative raw pairs at updates 54 and 385. Raw-norm fluctuations
increase near those events, while momentum remains smoother. The norm gap is
already present before each first negative pair; cosine sign alone does not
define its onset. All six interim PDFs passed visual review.

[Run020](../runs/020-2026-10-08-dgx-spark-calibration/REPORT.md) verified SSH and
the user's GB10 DGX Spark runtime at 14M/31M/70M. Nine calibration updates project
approximately 5.0/5.7/7.6 hours per existing full-pass recipe (one seed); peak
allocated memory is 5.45–8.52 GiB. Compact evidence verified, GPU idle, 9.48 GB
task-owned local state retained at calibration closeout. Calibration only.
ARM-generated initialization hashes differed; Runs021–023 subsequently use a
common frozen checkpoint for identical starts.
[Analysis007](../analyses/007-2026-10-08-spark-runpod-timing/README.md) compares
that calibration with completed matched Runpod passes: Spark update times are
3.45–3.68× RTX4090 and 4.32–4.80× RTX5090. Cloud times cover full passes;
Spark times remain short-calibration projections. Analysis007 itself launched
no new run.

[Analysis006](../analyses/006-2026-10-07-cumulative-geometry-slopes/REPORT.md)
reanalyzes Run002 slopes: last 100 A/B increments remain unequal and both B/A
ratios rise. Raw B/A is nearly flat at 424-484 while B grows and training loss
declines. A flat ratio means proportional accumulation, not absence of learning.
Six PDFs, including 3x2 loss-difference grids at lags1/2/5/20, are verified;
sources preserved.

[Analysis005](../analyses/005-2026-10-04-temporal-lag-profiles/REPORT.md) completed H005 at14M/31M/70M,seed1235: calibrated raw-gradient histories for all712 updates. Middle windows show negative lag1 and a positive longer-lag background, while exact all-pair interaction is positive in all nine windows. Full-pass lag1 truncation misses97.94-98.57% of squared cumulative raw-gradient norm. Early/late smooth decay to zero is not supported consistently over lags1-64. Six PDFs and two observations are complete; replays are numerically sensitive and do not add independent seeds. All six owned workers removed after verified retrieval; zero retained cloud storage, estimated USD6.54 provisional. Manuscript/source inputs preserved; multi-seed validation deferred.

The [Analysis005 size overlay](../analyses/005-2026-10-04-temporal-lag-profiles/observations/004-smoothed-size-overlay.md)
shows unsmoothed loss and 20-update means of eta*kappa_hat proxy, rho and raw norms;
update 712 excluded. Larger first-10 loss drops/norms coexist; first-50 mean norms
are non-monotonic across sizes. Exact checks pass; AdamW/minibatches limit interpretation.

A [separate unsmoothed 1-400 overlay](../analyses/005-2026-10-04-temporal-lag-profiles/observations/005-unsmoothed-first400.md)
shows all four raw variables with solid coloured curves; the earlier figure is preserved.

The [loss-difference variant](../analyses/005-2026-10-04-temporal-lag-profiles/observations/006-short-loss-difference.md)
uses an N=5 trailing mean of signed adjacent loss changes; the other panels remain raw.

The [log-loss-difference variant](../analyses/005-2026-10-04-temporal-lag-profiles/observations/007-log-loss-difference.md)
uses N=5 means of ln(L_t/L_(t-1)), approximately fractional loss changes; other panels stay raw.

The [split-range layout](../analyses/005-2026-10-04-temporal-lag-profiles/observations/008-split-update-windows.md)
uses four metric rows and columns 1-100/100-400, independent Y limits and unchanged source arrays.

The [column-smoothed revision](../analyses/005-2026-10-04-temporal-lag-profiles/observations/009-column-smoothing.md)
uses complete N=5 means for every metric at updates 1-100 and N=21 at 100-400, starting from raw scalars.

The [Adam-direction companion](../analyses/005-2026-10-04-temporal-lag-profiles/observations/010-adam-direction.md)
uses saved d cosines/norms and projected direction change q_d, with identical loss means and N=5/N=21 windows.

The [unsmoothed Adam-direction revision](../analyses/005-2026-10-04-temporal-lag-profiles/observations/011-unsmoothed-adam-direction.md)
removes extra averaging from q_d, rho_d and direction norms; only log-loss change retains N=5/N=21 smoothing.
The [200-250 inspection](../analyses/005-2026-10-04-temporal-lag-profiles/observations/012-loss-separation-200-250.md)
finds smaller models slowing while70M maintains its loss decline; matched schedule and geometry do not establish a mechanism.

[Analysis004](../analyses/004-2026-10-04-lr-schedule-intervention/README.md) completed
the first H004 intervention: seed1235,14M/70M, original/held/accelerated decay
from update356. Faster decay reduces raw negativity and brings positive alignment
earlier; held LR has the most negative alignment and lowest validation loss at
both sizes. At70M, positive recovery is temporary. F/D size ordering reverses
under held LR. [The restricted conclusion](../analyses/004-2026-10-04-lr-schedule-intervention/REPORT.md)
concerns schedule-sensitive cancellation, not a learning-quality/stability proxy.
All six artifact sets are verified and workers/volumes removed;88minutes,
estimated USD6.92 provisional (USD15 cap). Three PDFs are complete. Multi-seed
validation is deferred; source evidence and manuscript assets are preserved.

[Analysis003](../analyses/003-2026-10-04-raw-gradient-regimes/README.md) compares
raw-gradient cancellation and loss, with three pilot 1x2 figures and a
[completed replication audit](../analyses/003-2026-10-04-raw-gradient-regimes/observations/003-replicated-conclusions.md).
Larger models have lower validation loss, more negative adjacent pairs and higher
unconditional negative intensity, but smaller magnitude conditional on negativity,
in 3/3 seed blocks. Net all-pair raw interaction is positive in 9/9 runs; lag-one
truncation misses 97.90-98.57% of squared cumulative gradient norm. The restricted
[manuscript conclusion](../analyses/003-2026-10-04-raw-gradient-regimes/REPORT.md)
is retained. H003's dimensionality/performance mechanism remains untested;
manuscript and source-run evidence are unchanged.

[Observation002](../analyses/003-2026-10-04-raw-gradient-regimes/observations/002-schedule-and-recovery.md)
extends Analysis003 with the common LR schedule and phase-specific raw alignment.
Decay/recovery timing is compatible with H004, but the 70M trajectory returns to
negative alignment at fixed floor LR. Schedule and training progress remain
confounded in that observational check; the separate intervention is Analysis004.
The [recommended experiment](../analyses/003-2026-10-04-raw-gradient-regimes/SCHEDULE_EXPERIMENT.md)
branches update356 checkpoints at14M/70M into original, held and accelerated
decay. Analysis004 executes the user's selected first seed; the original
three-seed recommendation is reserved for later validation.

Runs 005-010 completed both additional seed blocks. [Analysis002](../analyses/002-2026-10-04-scale-seed-replication/README.md)
compares three seeds per size: raw cosine: 3/3 seed blocks; raw B/A: 3/3 seed blocks; adaptive cosine: 3/3 seed blocks; adaptive B/A: 3/3 seed blocks.
All new artifacts are verified; all six workers/volumes removed; estimated $10.11 provisional.
Initial/order seeds vary together; order is paired within seed. Three seeds support a
replication check, not a predictive scaling law.


Runs 003 (31M) and 004 (70M) completed both full MiniPile passes, 712 updates each,
with the paired raw-gradient/adaptive-direction metrics from Run 002 (14M).
All new artifacts were retrieved and verified; both workers and their volumes
are removed. [Analysis001](../analyses/001-2026-10-03-model-scale-geometry/README.md)
owns the matched scale comparison and four publication figures.

| Model | Raw median cosine | Adaptive median cosine | Raw final B/A | Adaptive final B/A |
| --- | ---: | ---: | ---: | ---: |
| 14M | 0.007220 | 0.966250 | 0.337132 | 0.704780 |
| 31M | −0.086252 | 0.952865 | 0.330751 | 0.665558 |
| 70M | −0.160566 | 0.931795 | 0.309380 | 0.609436 |

Full-pass alignment/ratio ordering is consistent across these three sizes.
Raw A/B decrease, adaptive A/B increase; raw median norms and phase-specific
raw cosines are not uniformly monotonic. H002's predictive scaling claim remains
inconclusive: one seed per size, no pre-specified functional form or held-out size.
The earlier one-seed comparison above is retained for reference; completed
replication is summarized at the top. The adaptive direction excludes LR/decay.
Main plots omit the final partial batch;
full data and terminal statistics retain it. Estimated new-run cost USD3.33
is provisional; no task-owned cloud storage remains.

## Earlier evidence

Run002 supports H001 descriptively: adaptive adjacent alignment and B/A exceed
raw-gradient values, while A and B retain a gap. [Run001](../runs/001-2026-10-03-pythia14m-adamw-gradient-baseline/README.md)
retains the first raw-gradient baseline and its temporal-interaction analysis.
Its replay differs from Run002; use Run 002's paired streams for the scale comparison.
No stability, causal learning benefit, or full SGD identity is established.
SGD remains deferred. Original manuscript assets are preserved.

## Navigation

- [Hypotheses](HYPOTHESES.md), [runs](../runs/INDEX.md),
  [analyses](../analyses/INDEX.md), [findings](findings/README.md)
- [Definitions](DEFINITIONS.md), [methods](METHODS.md), [metrics](METRICS.md),
  [model/data protocol](DATA.md)
- [Workflow](WORKFLOW.md), [compute and ETC](COMPUTE.md), [Runpod](RUNPOD.md),
  [artifact contract](ARTIFACTS.md), [plotting](PLOTTING.md)
- [Manuscript](MANUSCRIPT.md), [review context](REVIEW_CONTEXT.md),
  [bootstrap provenance](BOOTSTRAP.md)
- [Surgical changes proposal](../manuscript/2026-10-07-surgical-changes-proposal/README.md):
  evidence, draft changes, and proposed momentum/LR experiments; planning only.

Next run: 055; analysis: 019; hypothesis: H018. Preserve executed inputs and raw evidence.

## Resumed batch-size figure extension

[Analysis037](../analyses/037-2026-11-02-sgd-batch-extension/README.md) / Run214
prepare the requested three-case2x3 SGD figure. The user resumed execution on
Runpod; a new aggregate cost/time envelope is pending. Seven CPU tests and both
complete short worker queues pass, including sampled forks, replay, CPU audits,
verified copied artifacts and corruption rejection. Small ResNet FP32 replay
drift is recorded; Transformer short replay is exact. No production training or
paid allocation has started. The manuscript TODO remains open. Next Run215 / Analysis038; H023 extended,
no new hypothesis allocated.

## MLP control implementation milestone

[Run217](../runs/217-2026-11-03-mlp-adam-control-foundation/README.md) /
[Analysis039](../analyses/039-2026-11-03-mlp-control-foundation/README.md) implement
the A0-only local driver, pre-update completed-step observer, immutable labeled
forks and separate rolling recovery. Generated tensor/mock contracts pass;
Run215 remains unchanged. Preparation only: H028 untested, zero neural training,
intervention arms, remote execution or spending. Next Run223 / Analysis044 / H030.

## Adam intervention implementation milestones

[Run218](../runs/218-2026-11-03-mlp-adam-step-actions/README.md) /
[Analysis040](../analyses/040-2026-11-03-mlp-step-actions/README.md) implement
A0-A6/A8: prior-only raw/proposal clipping, raw-numerator ablation, historical
beta2 correction and transported-secant actions with authenticated fork/resume.
After explicit approval, the local pilot completed8 x128 branches and the657-update
prefix in64.06s. Exact historical/contemporary replay passes; A1 half LR and A4
proposal clipping meet the training screen, while A5/A8 create larger excursions.
[Empirical report](../analyses/040-2026-11-03-mlp-step-actions/REPORT.md). Single seed;
no remote work, rental or manuscript edit. A7 deferred. Next Run223 / Analysis044 / H030.

## Completed: Run223 / Analysis044 / H030

[Run223 / Analysis044](../analyses/044-2026-11-04-multispike-interventions/FINDINGS.md) completed all three frozen windows with fresh controls. Immediate/cosine half mean loss changes: MLP-25.04%/-26.30%, ResNet+27.20%/+0.62%, Transformer-9.66%/-10.68%. Spikes persist in larger models. Exact replay and235 files/7 archives verified;3 vector figures inspected. All task Pods deleted; estimatedUSD7.58 of15, billing provisional. No manuscript edit or push. Current next IDs remain Run228 / Analysis051 / H035.

## Completed follow-up: Run224 / Analysis045 / H031

[Analysis045](../analyses/045-2026-11-04-mlp-lr-return/README.md): returning both MLP schedules to0.75x at650 restores spikes; post-return mean loss is57.4%/45.4% above matching held-half controls. Exact pre-return history/replay and26 new transferred artifacts verified. Spark only. Run223 cloud cases remain active. Next Run225 / Analysis046 / H032.

## Completed follow-up: Run225 / Analysis046 / H032

[Analysis046](../analyses/046-2026-11-04-mlp-cosine-rise70/README.md): cosine70 restoration over650..720 delays the large MLP peak to716 but increases its height2.60x versus abrupt restoration. Mean651..800 loss is33.75% higher than abrupt and94.43% higher than held-half. Exact transition650 and19 new transferred artifacts verified; Spark only. Run223 remains active. Next Run226 / Analysis047 / H033.

## Completed follow-up: Run226 / Analysis047 / H033

[Analysis047](../analyses/047-2026-11-04-mlp-cosine-floors/README.md): cosine32-to0.75x lowers MLP mean loss17.24% versus unchanged but retains7 excursions, including a5.01x peak at659 while LR is constant. Cosine32-to0.5x lowers mean26.30%, with only one early excursion. Figures pair each intervention only with unchanged, in separate columns, as requested. Exact reuse/18 new artifacts verified; Spark only. Run223 remains active. Next Run227 / Analysis048 / H034.

## Figure synthesis: Analysis048

[Analysis048](../analyses/048-2026-11-04-mlp-three-paths-to075/README.md) assembles the requested three paths to0.75x from Runs223-226, each paired only with unchanged baseline.119 source artifacts and10584 plotted values reverified; vector figure inspected. No new training/hypothesis. Run223 remains active. Next Run227 / Analysis049 / H034.

## Figure synthesis: Analysis049

[Analysis049](../analyses/049-2026-11-04-mlp-direct-abrupt-overlay/README.md) provides the requested direct two-path overlay from Runs224/226, sharing the Run223 prefix.100 artifact checks and3528 plotted values verified; vector render inspected. No new training/hypothesis. Next Run227 / Analysis050 / H034.

## Completed follow-up: Run227 / Analysis050 / H034

[Analysis050](../analyses/050-2026-11-04-mlp-two-stage-decay/README.md): a second cosine32 decay from0.75x at650 to0.5x at682 reduces the first MLP peak48.77% and mean651..800 loss27.66% versus held0.75x; no >10% excursions after682. Exact prefix/update replay and20 new artifacts verified. Selected development timing, one seed, Spark only. Next Run228 / Analysis051 / H035.

## Completed follow-up: Run228 / Analysis051 / H035

[Analysis051](../analyses/051-2026-11-04-mlp-global-clipping/README.md): global norm clipping1.0 before Adam reduces the first MLP peak but worsens mean loss48.01% and maximum3.82x across512..800. Exact control reuse,73 artifacts verified, Spark only. Single seed/window; no universal rejection of clipping. Next Run229 / Analysis052 / H036.

## Completed follow-up: Run229 / Analysis052 / H036

[Analysis052](../analyses/052-2026-11-04-mlp-clipping-thresholds/README.md): clipping0.5 reduces MLP mean loss31.03% versus1.0 but remains2.08% above unchanged, despite a15.32% lower maximum. Exact replay,90 artifact checks, Spark only. Next Run230 / Analysis053 / H037.

[Run230 / Analysis053 / H037](../analyses/053-2026-11-05-mlp-cosine32-clip1/README.md) found zero activations of clipping at 1 under cosine32 decay to half LR. Exact agreement with cosine alone through state 800.

[Run231 / Analysis054 / H038](../analyses/054-2026-11-05-resnet-clip1/FINDINGS.md) found a 52% lower maximum and 28% lower mean training loss under clipping at 1, with higher late loss.

## Hidden-representation oscillation, 7 November 2026

[Analysis062 three-space extension](../analyses/062-2026-11-07-mlp-gradient-amplification/observations/010-gradient-output-hidden-svd-results.md)
compares raw-gradient and output/hidden-change SVDs using exact Run240 replay.
Hidden/output alignment correlation0.99847;10 visible directions per example
carry68.14% of hidden-change energy, but the null component also oscillates.
Existing-head transmission of hidden changes dominates output motion; no
unique initiating cause established. Verified/reviewed; Spark worker exited,
4.718GB retained, zero cloud cost. Next Run241 / independent Analysis063.

## Oscillation in the first hidden layer, 7 November 2026

[Analysis062 earlier-hidden extension](../analyses/062-2026-11-07-mlp-gradient-amplification/observations/011-first-hidden-oscillation.md)
uses exact Run241 replay to show oscillation already in the first tanh
representation: alignment correlation0.99590 with the second hidden layer,
shared dominant temporal SVD structure. Second tanh retains86.84--89.30%
of displacement norm; upstream representation changes dominate H2 motion.
Verified/reviewed; worker exited,5.457GB retained on owned Spark, zero cloud
cost. No unique initiating cause claimed. Next Run242 / independent Analysis063.

## Complete forward-layer trace, 7 November 2026

[Analysis062 all-layer extension](../analyses/062-2026-11-07-mlp-gradient-amplification/observations/012-all-layers-and-fixed-input.md)
uses Run242 to verify that inputs are exactly fixed while the first learned
linear representation already oscillates in step with outputs (r0.99101).
First tanh strongly attenuates displacement; less-saturated sample-neuron
coordinates carry most surviving motion. Changing W1 dominates that first
projection; no unique initiating optimizer mechanism established. Verified,
worker exited,4.592GB retained on owned Spark, zero cloud cost. Next Run243 /
independent Analysis063.
