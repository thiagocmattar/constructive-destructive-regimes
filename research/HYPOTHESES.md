# Hypothesis registry

H001 is the first user-directed experimental expectation. Manuscript/review
questions alone do not create experimental commitments.

| ID | Question / hypothesis | Status | Runs / analyses | Evidence / limits |
| --- | --- | --- | --- | --- |
| H001 | Bias-corrected Adam adaptive directions have higher adjacent alignment and cumulative B/A than raw gradients along the matched AdamW trajectory | Supported descriptively in one paired trajectory | [Run002](../runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/observations/001-adaptive-comparison.md) | Median cosine 0.966250 vs 0.007220; final B/A 0.704780 vs 0.337132. A and B retain a 29.52% relative gap. One seed; direction excludes LR/decay; no causal/stability conclusion |
| H002 | Adjacent cosine and cumulative A, B, B/A have predictable behavior across 14M, 31M, and 70M, for both raw gradients and adaptive Adam directions | Replication complete; predictive scaling law still inconclusive | Runs 002-010; [Analysis002](../analyses/002-2026-10-04-scale-seed-replication/README.md); [Analysis001](../analyses/001-2026-10-03-model-scale-geometry/observations/001-scale-comparison.md) | Strict full-pass pilot ordering: raw cosine: 3/3 seed blocks; raw B/A: 3/3 seed blocks; adaptive cosine: 3/3 seed blocks; adaptive B/A: 3/3 seed blocks. Three seeds, matched order within seed; initial/order seeds co-vary. Raw A/B/norm and phase reversals are retained. No fitted functional form or held-out prediction |
| H003 | Increasing model size reduces raw-gradient destructive-regime negativity/intensity, potentially through dimensionality and explaining better performance | Mixed descriptive evidence replicated; broad explanation unsupported, causal mechanism untested | [Analysis003 replication audit](../analyses/003-2026-10-04-raw-gradient-regimes/observations/003-replicated-conclusions.md), Runs002-010; Run001 replay sensitivity | Negative frequency and unconditional intensity increase, conditional magnitude and validation loss decrease in 3/3 paired seed blocks. Net all-pair raw interaction is positive in 9/9 trajectories. Restricted local-cancellation/loss conclusion retained; three seeds, hardware differences, architecture confounding and 14M replay sensitivity limit generality |
| H004 | LR decay contributes to late recovery of adjacent raw-gradient alignment, with size-dependent responses to the common optimizer recipe | Supported in first single-seed intervention; cross-seed validation pending | [Analysis004 observation001](../analyses/004-2026-10-04-lr-schedule-intervention/observations/001-schedule-intervention.md); Runs011-016; [earlier schedule observation](../analyses/003-2026-10-04-raw-gradient-regimes/observations/002-schedule-and-recovery.md) | Faster decay reduces F/D/M and brings positive-smoothed criterion65/101 updates earlier at14M/70M. Held LR has most negative alignment and lowest validation loss at both sizes.70M recovery reverses later; F/D size ranking is schedule-dependent. One seed, fixed trained prefix, LR-scaled decay/numerical variation; no convergence, stability or mediation claim. All workers removed; other seeds deferred |
| H005 | Temporal raw-gradient alignment decays smoothly early, oscillates during middle local cancellation, and decays again late | Mixed; first single-seed test complete, broader phase law unvalidated | Runs017-019; [Analysis005](../analyses/005-2026-10-04-temporal-lag-profiles/REPORT.md) | Middle negative short-lag trough/rebound observed in all three models; rebounds also occur early/late, with positive longer-lag backgrounds. Exact all-pair interaction is positive in all nine windows despite negative middle lag1; full-pass lag1 truncation misses 97.94-98.57% of squared raw-gradient sum. Four-map calibrated estimates, one training seed, recipe/hardware-sensitive replays; no periodicity, dimensionality, stability or performance mechanism established. Workers removed; extra seeds deferred |
| H006 | A/B slopes become equal; flat cumulative B/A tracks loss convergence and signals that new information is destroyed | Geometric equivalence unsupported; convergence predictor unvalidated; information storage unmeasured | Run002; [Analysis006](../analyses/006-2026-10-07-cumulative-geometry-slopes/REPORT.md) | Last 100 raw/adaptive B slopes remain positive and below A slopes; both ratios rise. Raw ratio nearly flat at 424-484 while B grows and training loss declines. Flat ratio requires proportional growth, and denominator accumulation reduces responsiveness. One seed, changing-batch loss, only initial/final fixed validation; no information-storage measurement |
| H007 | How do raw-gradient, stored Adam first-moment and adaptive-direction A, B, B/A, norms and adjacent cosine change under warmup-only LR 1e-4/1e-3/1e-2? | Completed exploratory comparison; one matched seed; no directional prediction | Runs021–023; [Analysis008 final results](../analyses/008-2026-10-08-no-decay-momentum/complete-three-lr/README.md) | Final validation loss at LR 1e-3/1e-4/1e-2 is 4.764/6.232/6.636. LR 1e-2 has much larger raw norms and lower raw B/A (0.0448); positive adjacent momentum/adaptive cosine coexists with worsening late loss and sparse negative long-lag momentum pairs. Momentum reduces A; raw/momentum B comparison is affected by clipping. Same frozen initialization/order/runtime, LR-scaled decay; no cross-seed, stability or causal mediation claim |
| H008 | How do loss and raw/momentum/adaptive geometry behave at four logarithmically intermediate warmup-only learning rates? | Completed exploratory metric comparison; recovery retrieval ongoing | Runs024-027; [Analysis009](../analyses/009-2026-10-09-intermediate-lr/README.md) | New four-run best validation4.318 atLR0.00464159, next4.410 atLR0.00215443. Late momentum cosine is0.852/0.956 respectively, so highest adjacent persistence does not minimize loss. Same frozen initialization/order/recipe; Spark-vs-Runpod confound; one seed; no convergence/stability threshold or causal mechanism |
| H009 | How do Analysis009 vector metrics relate to navigation of exact 2D Rosenbrock under matched momentum/decoupled-decay interventions? | Completed exploratory pilot; no directional hypothesis or general convergence proxy established | Runs028-031; [Analysis010](../analyses/010-2026-10-10-rosenbrock-controlled/README.md) | Final losses3.452e-6/0.02245/0.0009731/0.08264; all fail final100 stationarity test. Lower loss with frequently negative momentum pairs coexists with higher-loss positive-momentum decay trajectory. Four deterministic paths, same start/LR/beta2; all exact histories/replay/optimizer/hash checks pass; no LLM or asymptotic transfer |
| H010 | How do exact Rosenbrock loss navigation and raw/momentum geometry vary at LR1e-2/1e-3/1e-4 with momentum, zero decay and 2,000 matched updates? | Completed exploratory comparison; no directional prediction or convergence proxy established | Runs032-034; [Analysis011](../analyses/011-2026-10-10-rosenbrock-lr/README.md) | Final losses0.000141348/0.325413/4.138634. Largest LR travels fastest but keeps oscillating with negative adjacent pairs; smaller LRs remain positively aligned with less progress. All fail joint stationarity test. Exact Run028 prefix replicated; one deterministic function/start/beta2/horizon; no asymptotic or LLM transfer |
| H011 | How do Rosenbrock navigation and raw/momentum geometry behave with weight decay 0.1 at LR 1e-2, logarithmic midpoint and 1e-3 over 2,000 updates? | Completed exploratory comparison; no directional prediction or universal optimizer/LR ranking | Runs035-037; [Analysis012](../analyses/012-2026-10-10-rosenbrock-adamw-lr/README.md); matched endpoint controls032/033 | Midpoint has lowest new final loss 0.000115599; other losses 0.00104247/0.234112. At 1e-2, decay raises final loss despite reducing negative momentum pairs from 65.33% to 0.40%; at 1e-3 it lowers final loss. All fail stationarity; Run030 prefix exactly replicated. No midpoint no-decay control, asymptotic or LLM conclusion |
| H012 | How does the Analysis012 Rosenbrock AdamW LR comparison change when only the initial point becomes (0,1)? | Completed exploratory initialization comparison; no directional prediction or universal LR ranking | Runs038-040 paired with035-037; [Analysis013](../analyses/013-2026-10-10-rosenbrock-adamw-new-start/README.md) | Final losses0.000972263/0.000117958/0.0000110471 at decreasing LRs; smallest LR now best rather than midpoint. At1e-3 raw negative pairs rise from0 to25.01% despite much lower loss. All fail joint stationarity. Only two deterministic starts, one recipe/horizon; no alignment mechanism, asymptotic or LLM inference |
| H013 | How do Rosenbrock navigation and raw/momentum geometry change under additive isotropic Gaussian gradient noise sigma0/0.01/0.1/1/10? | Completed exploratory sweep; one paired seed; no directional prediction or ensemble law established | Controls038-040, Runs041-052; [Analysis014](../analyses/014-2026-10-10-rosenbrock-gradient-noise/README.md) | Final loss increases and raw full-prefix B/A decreases with sigma at each LR in this realization. Sigma10 raw late mean cosine near0 versus momentum0.81-0.82 despite worse loss. Exact identities retain signal/noise cross terms; all fail joint stationarity. Matched draws across sigma/LR, one seed, 2,000 updates; no ensemble monotonicity, causal stability or LLM inference |
| H014 | How do loss and raw/momentum/adaptive geometry behave at constant post-warmup LR0.02/0.04 versus matched Spark LR0.01? | Completed exploratory comparison; one seed; all artifacts verified and reviewed | Runs053/054; reference023; [Analysis015](../analyses/015-2026-10-10-spark-high-lr/README.md) | Both712-update passes completed with zero overflows. Final validation6.63438/5.19195 at LR0.02/0.04 versus6.63649 at matched LR0.01. Raw negative adjacent pairs63.99%/62.59%; all momentum pairs positive, but full-prefix momentum B/A below reference. One seed/order/pass and LR-scaled decay; no optimal LR, universal stability or causal alignment threshold established |
| H016 | Does the restatement gradient-identity kappa resemble largest Rosenbrock Hessian curvature along the Analysis016 AdamW trajectories? | Completed exploratory postprocessing; exact gradients, noise epsilon0 | Runs055-057 reused; [Analysis017](../analyses/017-2026-10-12-rosenbrock-kappa-identity/README.md) | Raw/100-pair correlations0.407/0.942,0.508/0.905,0.569/0.753 at decreasing LRs. Magnitudes differ; AdamW update mismatch dominates late discrepancy versus exact directional curvature. Separate eta-kappa means1.790/1.755/1.748 show similar late levels and different transition times. No validated curvature estimator, universal plateau or stability threshold |

| H017 | How does the eta-scaled gradient identity change with Rosenbrock Gaussian gradient noise sigma0/0.01/0.1/1/10, including its realized noise term? | Completed exploratory postprocessing; one paired seed | Runs038-052 reused; [Analysis018](../analyses/018-2026-10-12-rosenbrock-kappa-noise/README.md) | At sigma10 late gradient-only means1.050/1.072/1.095 become0.0186/0.00426/0.00270 when measured epsilon is added. AdamW mismatch persists versus directional curvature. All pairs/means and both100/250-pair PDFs reviewed; no ensemble law, validated curvature estimator or stability threshold |

| H018 | How do full-batch CIFAR-10 loss and raw/m/mhat/adaptive/update geometry relate to ordinary/preconditioned Hessian curvature across GD/Adam learning rates? | Completed exploratory single-seed fixed-horizon comparison | Runs059-066; [Analysis019](../analyses/019-2026-10-13-cifar10-eos-lr/README.md) | All160k updates and4,816 eigenpairs verified; four PDFs reviewed. Final train losses GD0.09688,0.05651,0.03204,0.01945,Adam0.0007658,0.0008776,0.0007075,0.0006594 in increasing LR order. Fixed20k horizon,one frozen initialization,full5k/10k coverage; no universal stability/alignment proxy or optimized method ranking |

| H019 | How do GD/SGD projected gradient change, alignment, sharpness and displacement vary across three EOS cases; how does measured real-data batch noise affect accumulation? | Active; replicated frozen controls complete; three-case seed0 GD overview reviewed; remaining training pending | Runs067-103; reference061; [Analysis020](../analyses/020-2026-10-14-eos-multitask/PROTOCOL.md) | Three initializations and32 sampling repetitions/case; greater frozen noise lowers positive cosine and B/A in seed means; ResNet retains strong initial signal. First500-step GD reversal at gradient109 for ResNet and456 for Transformer, absent for MLP in seed0; other seeds and SGD interpretation pending; no universal EOS, stability, convergence or generalization claim |

| H020 | How does constant GD learning rate change full-objective loss, measured maximum curvature and projected gradient change across the three EOS task/model cases? | Active; single-initialization 3k calibration and exact long-lag follow-up complete; broad sweeps held | Runs122-126 plus reused059/062/120/121; [Analysis024](../analyses/024-2026-10-17-eos-gd-lr-calibration/README.md); prior [Analysis023](../analyses/023-2026-10-15-eos-smallest-lr-calibration/README.md); [Analysis032](../analyses/032-2026-11-01-eos-exact-lag10/REPORT.md); [Analysis033](../analyses/033-2026-11-01-eos-long-lags/REPORT.md) | Three LRs per case,3000 updates,one frozen seed0; rates informed by prior calibration; subset Hessians for larger models; no initialization-robustness or global-stability claim; short-lag evidence is conditional on retained starts; two failed replay windows remain separately flagged. Exact anchored profiles through common lag 256, with extensions up to 2048, show nonuniform attenuation/rebounds; one pair per lag, no asymptotic/stability claim. |

| ID | Question / hypothesis | Status | Runs / analyses | Evidence / limits |
| --- | --- | --- | --- | --- |
| H021 | Does Adam coordinate preconditioning contribute to Pythia's model-size loss-rate separation, after controlling adaptive-step norm? | Complete; supports norm-controlled preconditioning contribution | [Analysis025](../analyses/025-2026-10-17-pythia-descent-mechanism/README.md), Runs127-153 | All18 ablations raise fixed-probe loss;70M effect exceeds14M in all3 seed blocks at both forks; unique200-250 onset not established |

| H022 | Does numerically fixed hard-target CE lead AdamW to lower common-objective Hessian curvature? | Supports raw-Hessian flattening in this recipe; exposure-qualified seeds0/10 | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md), Runs154-184 | fixed/native curvature ratio 0.014885, 95% CI [0.0030771,0.072007];30 trajectories/300k updates; full480-example FP64 Hessians; paired precision control and matched-loss audit; zero rental; no Pythia transfer claim |

| H023 | How does fresh minibatch sampling affect gradient alignment, projected change and actual displacement at matched trained states? | Completed conditional characterization; moment targets agree closely, cosine approximation has detectable bias, sampled/full-path alignment can differ in sign; three-case continuation extension verified | [Analysis027](../analyses/027-2026-10-30-minibatch-noise/REPORT.md), Runs185/186, [Analysis037](../analyses/037-2026-11-02-sgd-batch-extension/REPORT.md), Runs214/216 | Three MLP initializations; matched trained-state controls and complete-objective HVPs distinguish sampling from motion. Batch 32 median proxy discrepancy 0.7467 becomes 0.0030 after offline correction. No universal stability/generalization claim. Extension: three initializations per case, 348 new + 198 reused continuations; smallest trained-state batches weaken reversal with retention means below 0.05. Transformer batch 8 has similar cosine but different retention/opposite loss changes. Batch/runtime confounding in one ResNet block remains explicit |

| H024 | Do joint gradient/momentum/update diagnostics distinguish beneficial LR changes beyond loss and raw cosine? | Not supported in this pilot; no conditional LR-choice diversity | [Analysis030](../analyses/030-2026-10-30-adam-lr-decisions/REPORT.md), Runs187--204 | All six states favor2x LR; fixed-LR/loss baseline regret0, raw0.011137, joint0.016247. Positive raw cosine does not imply better loss. Three seed clusters, one stage; no validated scheduler. All evidence local; task cloud deleted |

| H025 | Can recorded optimizer-state summaries predict the magnitude of LR benefit beyond model size and action? | Exploratory candidate; no validated predictor | [Analysis031](../analyses/031-2026-10-31-adam-state-map/REPORT.md), reused Runs187-204 | Early actual/adaptive retention reduce final-benefit MAE about 37% across held-seed folds at 16; no initial feature improves primary prediction. Timing/target sensitivity, three seed clusters and outcome-known feature search limit inference. No controller claim |

<!-- Analysis034 records -->
| ID | Question / hypothesis | Status | Runs / analyses | Evidence / limits |
| --- | --- | --- | --- | --- |
| H026 | How do cold-Adam continuations change anchored and adjacent raw-gradient alignment at the same GD-trained states after training-loss LR calibration? | Completed conditional descriptive comparison | Runs209/210/211; [Analysis034](../analyses/034-2026-11-01-eos-adam-continuations/REPORT.md) | Same 27 retained states and GD horizons, 25 primary; five-rate 64-step calibration, one common rate per model; exact full gradients; optimizer reset, seed uncertainty unestimated; no independent method ranking or stability claim |
<!-- End Analysis034 records -->


<!-- H027 proposal -->
| ID | Question / hypothesis | Status | Runs / analyses | Evidence / limits |
| --- | --- | --- | --- | --- |
| H027 | How do raw, first-moment, adaptive-direction and actual-step geometry differ across smooth/oscillatory full-batch Adam LR conditions from identical initialization? | Complete descriptive study | [Analysis036 report](../analyses/036-2026-11-02-controlled-adam-diagnostics/REPORT.md), Runs213/215 | Calibration9 x128 and main9 x1,024; four streams and full split evaluations verified. One initialization, no Hessians or LR interventions. Raw-gradient projected change under Adam is a diagnostic, without a curvature, stability or convergence identification |
<!-- End H027 proposal -->

| ID | Question / hypothesis | Status | Runs / analyses | Evidence / limits |
| --- | --- | --- | --- | --- |
| H028 | Can matched Adam interventions reduce selected loss excursions without degrading subsequent optimization, and can that benefit transfer to a causal online policy? | Selected-fork treatments and combined hold/cosine promising; one-step rule unsupported | [Analysis040](../analyses/040-2026-11-03-mlp-step-actions/REPORT.md); [Analysis041](../analyses/041-2026-11-03-mlp-temporary-lr/REPORT.md); [Analysis042](../analyses/042-2026-11-03-mlp-lr-cooldown/REPORT.md) | Run218 A1/A4 pass the128-update screen. Run219 one-step rule is worse than fixed controls. Run220 eight-update hold/cosine lowers mean/final training loss2.26%/4.30% versus always-half, but retriggers keep365/367 updates below base. Abrupt hold remains worse. One selected development seed/state; no incremental timing or held-out gain. Exact controls/resume and original UTC audit failure retained |

| ID | Question / hypothesis | Status | Runs / analyses | Evidence / limits |
| --- | --- | --- | --- | --- |
| H029 | How do fixed-step gradient clipping, decoupled decay and LR schedules change optimization response at each model's highest LR? | Completed controlled response comparison; effects depend on model and regime | Run221 qualification / [Run222](../runs/222-2026-11-03-spark-matched-interventions/README.md); [Analysis043 findings](../analyses/043-2026-11-03-spark-fixed-step-interventions/FINDINGS.md) | 36 matched branches from warm state 1024, 64 updates each. LR reductions lower MLP mean loss about 45-47% but slow ResNet/Transformer; double-median clipping is inactive, decay shrinks norms without training-loss benefit. Exact within-Spark replay passes; failed ResNet cross-GPU gate retained. One seed/state and short horizon; no historical bitwise replay, universal stability or held-out claim |

Next hypothesis: `H030`. Record the operational meaning,
comparison, support/refutation criteria, and scope in the corresponding run
proposal. Suggested statuses: proposed, active, supported, unsupported,
inconclusive, withdrawn. Preserve superseded interpretations instead of rewriting
executed evidence. A run may address multiple hypotheses or be calibration-only.

## H030: persistent LR reductions across multiple large spikes

Status: completed; partial support with model-dependent effects. [Run223 / Analysis044](../analyses/044-2026-11-04-multispike-interventions/FINDINGS.md). MLP/Transformer mean loss improves but ResNet does not; spikes remain in both larger models. Exact replay passed, historical reuse failed, all235 files verified. Single-seed controlled experiment. Question: do immediate half LR and cosine32 to half reduce repeated loss excursions, including the largest historical region, without sacrificing training progress? Fixed windows MLP512..800, ResNet256..576, Transformer544..928; original full objectives and highest LRs. Qualified historical controls or fresh same-worker unchanged controls. Evaluate event counts/amplitudes/censoring alongside mean, final and last32-state loss; no generalization or universal stability claim.

## H031: does a smooth low-LR interval permit a later0.75x LR?

Status: unsupported for the selected MLP return at650. [Run224 / Analysis045](../analyses/045-2026-11-04-mlp-lr-return/FINDINGS.md). Both initial half-LR schedules develop new large spikes after an abrupt return to0.75x, despite exact matching histories through650. Mean651..800 loss rises57.4%/45.4% versus held-half. One seed/window; gradual restoration, other times and models untested.

## H032: can a70-step cosine rise prevent restoration spikes?

Status: contradicted for this selected MLP schedule/window. [Run225 / Analysis046](../analyses/046-2026-11-04-mlp-cosine-rise70/FINDINGS.md). Rise650..720 from0.5x to0.75x delays the peak but increases its height2.60x and post650 mean loss33.75% versus abrupt restoration. Exact transition state, one seed. Other timings/durations/models remain untested; endpoint test loss moves in a different direction from training-loss stability.

## H033: is cosine32-to0.75x sufficient to suppress repeated MLP spikes?

Status: partial loss benefit, spike suppression incomplete in this selected window. [Run226 / Analysis047](../analyses/047-2026-11-04-mlp-cosine-floors/FINDINGS.md). Mean loss17.24% below unchanged, but7 excursions and a5.01x peak while LR is constant. Exact same-Spark control reuse. Ongoing LR variation is not necessary for spikes here; no mechanism or universal threshold established.

## H034: can a second cosine32 decay suppress later spikes?

Status: supported within the selected MLP development window, with a residual initial spike. [Run227 / Analysis050](../analyses/050-2026-11-04-mlp-two-stage-decay/FINDINGS.md). Second decay650..682 from0.75x to0.5x lowers first peak48.77% and response mean27.66% versus matched held0.75x, with no >10% excursion after682. Exact observed pre-intervention replay; one seed, post-inspection timing, no universal stability or held-out benefit claim.

## H035: does global norm clipping1.0 suppress repeated MLP spikes?

Status: contradicted in the selected single-seed window. [Run228 / Analysis051](../analyses/051-2026-11-04-mlp-global-clipping/FINDINGS.md). Exact warm512..800; full-gradient clipping before Adam, unchanged LR. Active43.4% of updates; mean loss+48.01%, maximum3.82x despite smaller initial peak. Input norm cap does not bound actual Adam steps. Other thresholds/models and mechanisms remain untested.

## H036: does clipping0.5 improve on clipping1.0 and unchanged MLP training?

Status: partial support for peak reduction, no mean-loss benefit over unchanged. [Run229 / Analysis052](../analyses/052-2026-11-04-mlp-clipping-thresholds/FINDINGS.md). Same exact512..800 fork, LR0.001, seed0. Mean loss31.03% below1.0 but2.08% above unchanged; maximum15.32% below unchanged.238/288 clipped updates, oscillations remain. Post-inspection single-window evidence; no universal threshold claim.

## H037 | Does global clipping at 1 add to cosine32 decay to half LR?

[Analysis053](../analyses/053-2026-11-05-mlp-cosine32-clip1/README.md) found zero incremental effect in the selected MLP window. The maximum eligible gradient norm was 0.927 and all 288 update records and terminal model and optimizer states matched cosine alone.

## H038 | Does clipping at 1 suppress repeated ResNet loss spikes?

[Analysis054](../analyses/054-2026-11-05-resnet-clip1/FINDINGS.md) found lower peak severity and lower window-mean loss alongside more excursions above 10% and higher endpoint loss on this seed.

## H039: reducing Adam beta2 from a warm MLP state

Question: how do native beta2 changes to.95/.9/.5 affect repeated loss excursions
and training progress against retained.999 Adam? Completed descriptive comparison:
[Run232 / Analysis059](../analyses/059-2026-11-07-mlp-beta2-reduction/README.md).
Exact state512..800,seed0,LR.001; m/v/global clock retained, native bias-correction
denominator changes immediately. Mean/tail training loss is higher for all three;
test endpoint lower in this one seed. The0.5 branch has a long censored excursion.
No memory-only mechanism, generalization or universal stability inference.

## H040: switching warm Adam to plain GD at LR0.02

Question: does the specified GD switch suppress the selected MLP spikes while
maintaining training progress? [Run233 / Analysis060](../analyses/060-2026-11-07-mlp-adam-to-gd/README.md)
completed288 updates512..800; no loss increases or>10% excursions, but mean/tail
loss+2.30%/+55.74% against Adam(.9,.999),LR.001. Both optimizer and LR change;
one seed/window, exact retained-control and GD resume checks. Partial benefit
on excursion outcomes without mean/tail training-loss benefit. Next hypothesis:H041.

H040 extension,7 November: [Runs234--235 / Analysis060](../analyses/060-2026-11-07-mlp-adam-to-gd/observations/003-intermediate-gd-rate.md)
compare GD rates from the same warm parent. LR0.03/0.04 raise mean loss111.42%/
394.45% against Adam; both have an unrecovered excursion. LR0.08 overflows at547.
LR0.03 was requested after viewing earlier results, so this is exploratory rate
selection. The apparent smoothing benefit at0.02 does not extend uniformly to
larger GD rates. Within-GD comparisons vary LR alone; Adam comparisons change
both optimizer and LR. One seed/window; no universal threshold. H041 remains next.

## H041: internal origin of selected MLP Adam loss peaks

Question: do disproportionate weight/bias growth, activation magnitude or
saturation, or the final linear layer explain the last pre512 loss peak and
return? [Run236 / Analysis061](../analyses/061-2026-11-07-mlp-adam-spike-internals/observations/001-spike-internals.md)
replays the exact Adam control and examines four selected events. Evidence
favors output-sensitive hidden-feature displacement and finite-step overshoot.
Restoring parameter norms leaves nearly all excess loss; hidden RMS remains
similar, biases/head changes have small allocations, and tanh attenuates
changes rather than preventing loss oscillation. Static recombinations are
reference-dependent, not retrained interventions or unique causal shares.
One seed/window and declared post hoc probes; no universal instability claim.
Next hypothesis:H042.

## H042: small normalization scales versus coherent MLP gradient spikes

Question: could near-zero normalization inputs explain abrupt MLP loss peaks,
and what internal quantities spike with loss? [Run237 / Analysis062](../analyses/062-2026-11-07-mlp-gradient-amplification/observations/001-gradient-amplification.md)
finds no dynamic normalization and no sudden Adam denominator collapse. Exact
control replay shows growing, nearly reversing output changes and much less
cancellation between per-example gradients. Tanh attenuates backpropagation;
mean sensitivity changes little. Crossed Jacobian/direction probes favor larger
increments along already sensitive directions. All87 actual steps initially
point downhill,42 increase loss; exact output-space terms quantify overshoot.
This does not uniquely identify the oscillation's initiating cause or exclude
existing Adam preconditioning/momentum. Exploratory one seed and selected window.
Next hypothesis:H043.

H042 measurement extension: [Run238 / Analysis062](../analyses/062-2026-11-07-mlp-gradient-amplification/observations/003-control-coherence-448-800.md)
adds the user-defined norm-based coherence across448..800 and a five-row figure.
The unchanged Adam control is exactly replayed; the earlier squared-energy
coherence is preserved as a distinct statistic. No new causal hypothesis or
optimizer intervention. Next hypothesis remainsH043.

H042 output-space extension: Run239 measures cos(F_(t+1)-F_t,F_t-F_(t-1)) over
448..800 for all5000 examples and10 raw outputs. Added as panel(g) in Analysis062;
exact control qualification, full-example reductions and extra-step audit pass.
This describes temporal prediction reversals, not per-example cosine or a new
causal test. Next hypothesis remainsH043.


## H043: architectural localization of loss-associated representation reversal

Question: does the MLP's early representation reversal extend to ResNet/CIFAR-10
and Transformer/WikiText-2, and where does output-like temporal structure become
observable? [Run243 / Analysis063](../analyses/063-2026-11-07-architecture-instability-tracing/README.md)
completes descriptive fixed-probe traces under unchanged Adam. ResNet first-conv
motion reverses, but its output-alignment correlation is 0.578 versus 0.991 at
the last residual stage. Transformer at update558 has embedding/first-attention/
output cosines +0.390/-0.672/-0.920. Fixed-input checks, exact Grams/SVD, native
observer parity and replay qualification pass. This does not identify a unique
causal origin; attention internals, local-parameter transmission and matched
interventions remain open. One seed, selected windows and declared probes.
Next hypothesis:H044.

## H044: first-layer updates contribute to the MLP oscillation

[Run 244 / Analysis 064](../analyses/064-2026-11-07-mlp-freeze-first-linear/README.md)
freezes first-linear weights/biases at 448 while preserving native Adam state,
clocks, LR and later-layer updates. Output reversals450-500 fall 51/51 to 4/51;
maximum output RMS falls 87.38% and loss at 474 falls 46.43%. Extended mean loss
448-800 is 22.27% higher. Supports a causal contribution of first-layer motion to
this oscillation, without identifying a unique initiating mechanism or a uniform
optimization benefit. First layer contains 93.57% of parameters; no matched
later-layer freeze or rescue control. One seed/fork; no generalization inference.
Low-dimensional concentration persists after damping. Next hypothesis:H045.

## H045: learning-rate changes with the first MLP layer held fixed

[Runs245-246 / ongoing Analysis064-065](../analyses/065-2026-11-07-mlp-frozen-lr-double/README.md)
retain the freeze from448 and set LR0.002 or0.02 at500. Each complete intervention
is compared with unchanged Adam, with I1 as a supplementary incremental reference.
I2 preserves spike suppression but mean loss448-800 remains20.58% above baseline.
I3 overshoots to107.099 at504 and ends800 at0.308132 versus baseline0.0302502.
Exact first-layer constancy shows that later-layer updates can generate large
excursions at the stronger rate. This is not a unique-cause result or threshold
estimate. One seed/fork; full data, unchanged native moments/clocks, verified
qualification, retrieval and dense reductions. Keep iterating this same analysis.
Next unused hypothesis:H046; next run247. No new analysis without user request.

H045 adaptive extension: [Runs247-251](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/003-lr-halving.md)
test0.01/0.005/0.0025/0.00125/0.000625 from the same frozen state500. Loss maxima
are12.5043/2.58580/0.0724478/0.0283371/0.0283354. At0.0025 the peak is delayed to707;
0.00125 has only a0.006% first-step bump, while0.000625 strictly decreases through800.
The user requested halving until no peak and confirmed stopping. Conservative
rule: no adjacent increase or excess over loss500 above1e-8. This is finite-window
selection, not a universal threshold or optimal-LR result. All interventions
remain compared with unchanged Adam in the same analysis. Next run252; H046 unused.

## H046: selective damping can preserve first-layer learning

[Runs252-257 / Analysis065](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/005-selective-layer-lr.md)
compare first-layer/global/later-layer LR reductions at0.5x/0.25x from untouched448.
First-layer0.25x keeps all layers trainable, reduces mean loss7.06% and positive
variation99.62%, and lowers first-hidden reversals350/353 to1/353. Its last32 mean
is15.79% higher than baseline and its sustained-target crossing is later, so
there is no uniform optimization advantage. Later-layer reductions retain most
reversals yet improve average loss. First-layer0.5x and global0.5x develop late
excursions. Supports location-sensitive suppression with continuing feature
learning, not a unique cause or universal LR threshold. One seed/fork, unequal
actual step norms, no generalization claim. All checks pass; next unused H047.

## H047: selective first-layer increments and representation budgets

[Runs 258-268 / ongoing Analysis 065](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/006-selective-update-gates.md)
test coordinate discard/cap at prefix q90 and neuron preactivation-displacement
RMS caps at q90/q75/q50, plus one absolute first-layer norm-matched uniform
control per selective arm. All start at untreated 448, native LR 0.001 everywhere,
all gradients/moments/clocks advance; only first-layer increments are transformed.
Thresholds use only 416-447. Neuron q75 improves mean loss 10.42%, excursion maximum
39.43% and positive variation 79.37%; it beats its paired control 4.63% in mean
loss but retains all hidden/output reversals and has a 5.68% higher tail mean
than baseline. Discarding large coordinates worsens mean; coordinate capping
helps 2.26%. Stronger neuron capping loses to its uniform control, which has the
best new branch mean. Supports budget-dependent amplitude damping and a local
benefit of neuron selection, not universal superiority to uniform scaling,
unique causation, or generalization. One seed/fork. All checks pass; next H048.

## H048: representation-informed selective updates across architectures

[Runs 269-285 / Analysis 065](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md) test coordinate discard/cap,
local affine-displacement caps, diagnostic gates, movement budgets with/without
slow recovery, probe-loss acceptance, and larger Transformer embedding LR.
All branches compare with unchanged Adam and preserve complete native objectives.
Thresholds use eight untreated preceding updates; all gradients/moments/clocks
advance. The common distributed q50 cap lowers mean loss 22.85% (MLP seed 0,
448-1152), 16.22% (ResNet, 264-576), and 26.00% (Transformer, 552-928).
MLP seeds 1/2 improve 19.43%/32.08% without retuning. Reversals persist or
increase; negative cosine is not a sufficient intervention target. Test loss
at the terminal state is worse for this cap in every setup. Coordinate rules,
diagnostic gates and slow recovery do not all transfer; embedding 2x is nearly
neutral in mean loss and 4x worsens it. No universal optimizer, unique initiating
layer, held-out improvement, or online SVD-controller claim. Larger models use
one seed/fork; GPU-dependent Transformer controls are compared within workers.
All inputs/retrieval/audits and 14 PDF pages verified. 68 new continuations,
35,452 updates; all five task pods deleted, no cloud storage retained. Conservative
compute/storage estimate USD 18.75 within USD 20; billing provisional.
Continue Analysis 065. Next run 286; next unused hypothesis H049.

## H049: distributed local caps at larger MLP learning rates

[Runs 286-288 / Analysis 065 results](../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/013-distributed-cap-quantile-lr.md). Five quantiles (25/50/75/90/99) at Adam LR 0.001/0.002/0.004, uncapped controls at each LR, three pinned initializations. Fixed eight-update 1x budgets; matched 448 forks; full 5000-image objective and 10000-image test set. All 54 continuations (38,070 updates) completed on owned Spark. Fixed q25 caps reduce mean training loss 87.88%/94.75% versus uncapped 2x/4x, but neither LR improves the across-seed mean over q25 at 1x. The lowest observed mean is q50 at 1x (24.79% below original Adam; positive variation 80.66% lower). Tight caps can increase smaller oscillations and worsen test loss. Three initializations, one continuation window; no generalization superiority claim. All dense checks, 162 retained audits, six historical reproductions and four PDF pages pass. Raw evidence stays on Spark after a local disk-limit recovery; compact verified outputs and scalar histories are local. Zero cloud cost; task processes exited. Continue Analysis 065; next run 289 and unused H050.

## H050: conventional intervention coverage across the three architectures

**Active.** Test the architecture-dependent effects of late native beta2 reductions (.999 to .95/.9/.5) and global raw-gradient clipping (.5/1) on the existing fixed-window Adam excursions. Runs 289/290 extend beta2 to ResNet/Transformer; Run 291 extends clipping to Transformer. Matched factors, original forks, complete objectives, precision, native moment/clock semantics, validation and retention are frozen in their configurations. No universal stability or generalization hypothesis is assumed. Continue Analysis 065; all prescribed outcomes will be reported. Next unused H051.
