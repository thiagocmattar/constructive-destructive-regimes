# Analysis registry

## Analysis058: matched intervention figure consolidation

Additional [Figure 08](058-2026-11-06-intervention-consolidation/observations/008-spike-optimizer-zoom.md)
recreates the manuscript spike zoom with gradient/momentum and gradient/Adam
cosines and momentum/gradient norm ratios. All 1,760 values verified; original windows and
1,056 values preserved. Nine publication PDFs; no new training or manuscript edit.

7 November extension: [Figure 07](058-2026-11-06-intervention-consolidation/observations/007-optimizer-overview.md)
adds momentum norm and adjacent momentum/Adam-direction cosines to Figure 00.
Eight PDFs now; 21,495 extension values verified, including 12,285 unchanged
original-row values. Original seven figures and evidence remain preserved.

[Analysis058](058-2026-11-06-intervention-consolidation/README.md) consolidates
seven numbered publication PDFs (00--06): high-rate overview, immediate half LR,
MLP floors/return, return styles, two-stage decay, MLP/ResNet clipping, and
Pythia schedules. Existing raw evidence only; 69,354 plotted scalar checks,
48,006 exact original-plot matches, 63 source hashes and seven visual reviews
pass. Pythia norms/cosines support the projected-change row; raw tail retained.
No training, cloud work or manuscript change. Next analysis: 059.


`000-template/` is a template, not Analysis 000.

| Analysis | Question | Source runs / other inputs | Status | Observations / figures |
| --- | --- | --- | --- | --- |
| [001](001-2026-10-03-model-scale-geometry/README.md) | H002: geometry versus model size | Runs002/003/004 | Complete; matched cohort verified | [001: Scale comparison](001-2026-10-03-model-scale-geometry/observations/001-scale-comparison.md), four PDFs |
| [002](002-2026-10-04-scale-seed-replication/README.md) | H002: replicate scale ordering across seeds | Runs 002-010; three seeds per size | Complete; nine trajectories verified; new cloud closed | [001: Seed replication](002-2026-10-04-scale-seed-replication/observations/001-seed-replication.md), four PDFs |
| [003](003-2026-10-04-raw-gradient-regimes/README.md) | H003/H004: raw-gradient negativity, loss, LR/recovery and restricted manuscript claim | Runs002-010; Run001 replay; working draft | Complete local pilot/schedule/replication audit; experiment recommended, no launch | [001: Raw-gradient regimes](003-2026-10-04-raw-gradient-regimes/observations/001-raw-gradient-regimes.md), [002: Schedule/recovery](003-2026-10-04-raw-gradient-regimes/observations/002-schedule-and-recovery.md), [003: Replicated conclusions](003-2026-10-04-raw-gradient-regimes/observations/003-replicated-conclusions.md), three 1x2 PDFs; [schedule design](003-2026-10-04-raw-gradient-regimes/SCHEDULE_EXPERIMENT.md); [manuscript assessment](003-2026-10-04-raw-gradient-regimes/REPORT.md) |
| [004](004-2026-10-04-lr-schedule-intervention/README.md) | H004: single-seed LR intervention after a matched prefix | Runs011-016; parents005/007 | Complete; six branches verified; all new cloud closed; multi-seed validation deferred | [001: Schedule intervention](004-2026-10-04-lr-schedule-intervention/observations/001-schedule-intervention.md), three1x2 PDFs; [conclusion](004-2026-10-04-lr-schedule-intervention/REPORT.md) |
| [005](005-2026-10-04-temporal-lag-profiles/README.md) | H005: temporal lag profiles by training phase | Runs017-019 | Complete; three full histories verified; all owned cloud closed | [001: Lag profiles](005-2026-10-04-temporal-lag-profiles/observations/001-lag-profiles.md), [002: All-lag interaction](005-2026-10-04-temporal-lag-profiles/observations/002-all-lag-interaction.md), [003: Training overview](005-2026-10-04-temporal-lag-profiles/observations/003-training-overview.md), [004: Smoothed size overlay](005-2026-10-04-temporal-lag-profiles/observations/004-smoothed-size-overlay.md), [005: Unsmoothed first400](005-2026-10-04-temporal-lag-profiles/observations/005-unsmoothed-first400.md), [006: Short loss difference](005-2026-10-04-temporal-lag-profiles/observations/006-short-loss-difference.md), [007: Log-loss difference](005-2026-10-04-temporal-lag-profiles/observations/007-log-loss-difference.md), [008: Split update windows](005-2026-10-04-temporal-lag-profiles/observations/008-split-update-windows.md), [009: Column smoothing](005-2026-10-04-temporal-lag-profiles/observations/009-column-smoothing.md), [010: Adam direction](005-2026-10-04-temporal-lag-profiles/observations/010-adam-direction.md), [011: Unsmoothed Adam direction](005-2026-10-04-temporal-lag-profiles/observations/011-unsmoothed-adam-direction.md), [012: Loss separation](005-2026-10-04-temporal-lag-profiles/observations/012-loss-separation-200-250.md), fifteen PDFs; [conclusion](005-2026-10-04-temporal-lag-profiles/REPORT.md) |
| [006](006-2026-10-07-cumulative-geometry-slopes/README.md) | H006: A/B slopes, ratio plateaus and loss convergence | Run002 attempt02; original paired figure rows | Complete local analysis; sources preserved; geometric equivalence unsupported | [001: Slopes and loss](006-2026-10-07-cumulative-geometry-slopes/observations/001-slopes-and-loss.md), [002: Loss-difference grid](006-2026-10-07-cumulative-geometry-slopes/observations/002-loss-difference-grid.md), [003: Matched-lag grids](006-2026-10-07-cumulative-geometry-slopes/observations/003-matched-lag-grids.md), six PDFs; [answer](006-2026-10-07-cumulative-geometry-slopes/REPORT.md) |
| [007](007-2026-10-08-spark-runpod-timing/README.md) | Operational timing: Spark versus historical RTX4090/RTX5090 workers | Runs002–010 and Run020 | Complete; matched recipe/coverage verified; no launch | Three-size duration and update-time comparison; Spark is 3.45–3.68× / 4.32–4.80× slower respectively |
| [008](008-2026-10-08-no-decay-momentum/README.md) | H007: warmup-only LR scale and actual Adam first-moment geometry | Runs021–023; Run002 frozen initialization and recipe | Complete; three full passes retrieved/verified; one matched seed; owned Spark retained | [Final results](008-2026-10-08-no-decay-momentum/complete-three-lr/README.md), five updated PDFs; six per-run PDFs and automatic joint figure reviewed; six interim PDFs preserved |
| [009](009-2026-10-09-intermediate-lr/README.md) | H008: four intermediate warmup-only LR points, same raw/momentum/adaptive diagnostics | Runs024-027; matched Run021 initialization and recipe | Completed metric comparison; recoveries downloading; cloud cleanup pending; USD15 cap | [Seven-rate results](009-2026-10-09-intermediate-lr/complete-seven-lr/README.md); eight PDFs reviewed, including loss/norms per LR with smoothed raw traces and final raw norm minus minimum versus LR; best new validation4.318 atLR0.00464159; hardware limitation disclosed |
| [010](010-2026-10-10-rosenbrock-controlled/README.md) | H009: controlled Rosenbrock momentum/decay factorial and linked loss navigation | Runs028-031; exact objective and full float64 histories | Complete; 40,000 local updates; replay/optimizer/geometry/hash checks passed; no cloud | [Four observations](010-2026-10-10-rosenbrock-controlled/observations/INDEX.md); eight PDFs reviewed; offline step explorer tested |
| [011](011-2026-10-10-rosenbrock-lr/README.md) | H010: Rosenbrock momentum/no decay at three LRs; simplified full trajectory and eight diagnostics | Runs032-034; exact 2,000-update histories | Complete; 6,000 local updates; replay/optimizer/geometry/hash checks passed; no cloud | [001: LR comparison](011-2026-10-10-rosenbrock-lr/observations/001-learning-rate-comparison.md); single nine-panel PDF reviewed |
| [012](012-2026-10-10-rosenbrock-adamw-lr/README.md) | H011: Rosenbrock weight decay 0.1 at LR 1e-2/midpoint/1e-3 | Runs035-037; endpoint controls032/033; Run030 prefix | Complete; 6,000 local updates; replay/optimizer/geometry/hash checks passed; no cloud | [001: Decay LR comparison](012-2026-10-10-rosenbrock-adamw-lr/observations/001-learning-rate-comparison.md); single nine-panel PDF reviewed |
| [013](013-2026-10-10-rosenbrock-adamw-new-start/README.md) | H012: Analysis012 AdamW/LR recipe from(0,1), changing only start | Runs038-040; paired controls035-037 | Complete; 6,000 local updates; replay/optimizer/geometry/hash checks passed; no cloud | [001: Initial-point comparison](013-2026-10-10-rosenbrock-adamw-new-start/observations/001-initial-point-comparison.md); single nine-panel PDF reviewed |
| [014](014-2026-10-10-rosenbrock-gradient-noise/README.md) | H013: Rosenbrock Gaussian gradient-estimation noise sigma0/0.01/0.1/1/10 at three LRs, one paired seed | Controls038-040, new Runs041-052 | Complete; 24,000 new local updates; all replay/optimizer/geometry/hash checks passed; no cloud | [Seven observations](014-2026-10-10-rosenbrock-gradient-noise/observations/INDEX.md); 12 PDFs reviewed |
| [015](015-2026-10-10-spark-high-lr/README.md) | H014: stress higher warmup-only LR0.02/0.04 with loss and raw/momentum/adaptive geometry | Runs053/054; completed matched Spark reference023 atLR0.01 | Complete; both712-update passes verified/retrieved; two PDFs visually reviewed | [Results and review](015-2026-10-10-spark-high-lr/observations/002-completed-results-and-review.md); validation6.63438/5.19195 at LR0.02/0.04; no cloud rental |
| [017](017-2026-10-12-rosenbrock-kappa-identity/README.md) | H016: gradient-identity kappa versus Hessian and separate eta-kappa overlay | Immutable Runs055-057; no new optimizer run | Complete; all5,997 pairs and5,700 full means checked; three PDFs reviewed; local only | [Observations](017-2026-10-12-rosenbrock-kappa-identity/observations/INDEX.md); normalized late means1.790/1.755/1.748; original comparison unchanged |

| [018](018-2026-10-12-rosenbrock-kappa-noise/README.md) | H017: eta-kappa identity across Gaussian gradient noise, retaining realized epsilon | Immutable Runs038-052; five sigmas, three LRs, one paired seed | Complete; 29,985 pairs and28,500 means per metric checked; two PDFs (100/250 pairs) reviewed; no optimizer run | [001: Noise overlay](018-2026-10-12-rosenbrock-kappa-noise/observations/001-noise-overlay.md); gradient-only late sigma10 means1.050/1.072/1.095 versus corrected0.0186/0.00426/0.00270; original PDFs preserved |

| [019](019-2026-10-13-cifar10-eos-lr/README.md) | H018: full-batch CIFAR-10 GD/Adam LR ablation and geometry versus Hessian spectra | Runs059-066; frozen Run058 inputs | Complete; eight20k-update Spark trajectories,160k rows and4,816 eigenpairs verified; six-panel GD and ten-panel Adam full/early PDFs reviewed; four detailed PDFs archived | [Results](019-2026-10-13-cifar10-eos-lr/REPORT.md), [original observation](019-2026-10-13-cifar10-eos-lr/observations/001-learning-rate-sweep.md), [GD figure](019-2026-10-13-cifar10-eos-lr/observations/002-gd-six-panel.md), [Adam views](019-2026-10-13-cifar10-eos-lr/observations/003-adam-ten-panel.md); one seed; no cloud rental |

| [020](020-2026-10-14-eos-multitask/README.md) | H019: EOS real-data cases, GD/SGD geometry, measured noise and sharpness | Calibration067, frozen control068, staged training069-103; reference061 | Control complete: 3 seeds/case,36 conditions,147,456 observations; seed0 GD3x3 overview complete and reviewed,500 updates per case; remaining training running | [Report](020-2026-10-14-eos-multitask/REPORT.md), [control observation](020-2026-10-14-eos-multitask/observations/001-frozen-real-data-noise.md); no cloud rental; one LR/case; frozen-noise figure uses draft rho_t/R_128/eta*kappa_hat_t/epsilon_t/mathcal L, unchanged values and explicit frozen-state scope; no new training; observation001/002 |

| [021](021-2026-10-15-eos-multitask-lr/README.md) | H020: three-LR full-GD comparison across the EOS cases | New 104-119; reused 059/061/062 and Analysis020 GD | Held for replanning; 27 conditions, three seeds/rate, 500 updates; extra 38.73h compute | [Protocol](021-2026-10-15-eos-multitask-lr/PROTOCOL.md), [status](021-2026-10-15-eos-multitask-lr/REPORT.md); 3x3 paper figure and Table 1 insertion prepared; no rental |

| [022](022-2026-10-15-eos-long-horizon-plan/README.md) | Model-specific EOS horizons and bounded compute plan | Reuses019/020; no new run allocated | Prospective: MLP20k, ResNet/Transformer3k, three seeds/LR; 30 conditions,26 new; cloud spending pending | [Plan](022-2026-10-15-eos-long-horizon-plan/README.md), [qualified costs](022-2026-10-15-eos-long-horizon-plan/estimate.json); 348.94 Spark GPU-hours; no rental |

| [023](023-2026-10-15-eos-smallest-lr-calibration/README.md) | Smallest-LR3k full-GD calibration on Spark | Reused Run059; new120/121, seed0 | All three3k calibrations complete;79 files/8997 pairs/93 spectra verified; Three-case4x3 PDF reviewed; no subsequent training | [Protocol/status](023-2026-10-15-eos-smallest-lr-calibration/README.md), [MLP result](023-2026-10-15-eos-smallest-lr-calibration/observations/001-mlp-3k-prefix.md); no rental or manuscript change |

| [024](024-2026-10-17-eos-gd-lr-calibration/README.md) | One-initialization3k GD LR calibration | Nine conditions; reused059/062/120/121; new122-126 | Complete9/9;254 source files/26,991 pairs/279 spectra verified; final4x3 PDF reviewed,81,279 points; previous32 curves unchanged; ETC0 | [Completed calibration](024-2026-10-17-eos-gd-lr-calibration/observations/004-completed-calibration.md), [final figure](024-2026-10-17-eos-gd-lr-calibration/observations/003-completed-three-model-overview.md), [six isolated1x3 figures](024-2026-10-17-eos-gd-lr-calibration/observations/005-isolated-three-column-figures.md), [IEEE review and manuscript SectionIV](024-2026-10-17-eos-gd-lr-calibration/observations/006-ieee-figures-and-manuscript.md);54 Y curves unchanged; two composites; [applied SectionIV revision](024-2026-10-17-eos-gd-lr-calibration/observations/008-applied-section-iv-revision.md),36 table rows and26,991 loss checks;18-page draft reviewed; no new training or rental |

| Analysis | Question | Source runs / other inputs | Status | Observations / figures |
| --- | --- | --- | --- | --- |
| [025](025-2026-10-17-pythia-descent-mechanism/README.md) | H021 Pythia descent mechanism: fixed probe, direction substitutions and norm-matched preconditioning ablation | Runs127-153; 3 sizes x3 seed blocks | Complete;27 runs verified;3 PDFs reviewed; task resources deleted; USD11.42 posted | [Results](025-2026-10-17-pythia-descent-mechanism/REPORT.md), [observation](025-2026-10-17-pythia-descent-mechanism/observations/001-preconditioning-effects.md) |

| [026](026-2026-10-23-fixed-ce-curvature/README.md) | H022 fixed-CE curvature | Runs154-184;10 paired seeds,three loss-arithmetic arms | Complete30/30,300k updates; verified ordinary/scaled spectra; zero rental | [Conclusions](026-2026-10-23-fixed-ce-curvature/REPORT.md), [figure](026-2026-10-23-fixed-ce-curvature/figures/fixed_ce_curvature.pdf); fixed/native curvature ratio 0.014885, 95% CI [0.0030771,0.072007]; collapse-qualified seeds0/10 |

| [027](027-2026-10-30-minibatch-noise/README.md) | H023 sampling noise at matched MLP training states | Runs185/186; 184,320 frozen draws, 297 forks, 1,188 HVP pairs | Complete; Section V uses batch-size outcomes and a temporal view; 1,140 plotted means verified; frozen calibration condensed; original diagnostics retained; all evidence retrieved; task cloud deleted, USD0.87 provisional; Spark untouched | [Report](027-2026-10-30-minibatch-noise/REPORT.md) |

| [028](028-2026-10-30-adam-diagnostics/README.md) | Practical Adam extension: raw gradients, moments, adaptive directions and informed scheduling | Existing Analyses004/008/009; Analysis025 mapped for later development | Complete manuscript synthesis; 13 event files and 40 displayed values verified; 17-page draft reviewed; no new training or cloud | [Argument and idea map](028-2026-10-30-adam-diagnostics/README.md), [table evidence](028-2026-10-30-adam-diagnostics/observations/001-adam-extension.md); Section VI and two compact tables |

| [029](029-2026-10-30-pythia-adam-diagnostics/README.md) | Pythia Adam diagnostics across LR, schedule and size; useful next experiment | Analyses009/004/002, 22 event files; no new training | Complete; four vector PDFs reviewed; 303 curves / 192,399 values verified; input-versus-moment alignment reconstructed for4,977 pairs | [Assessment and proposed test](029-2026-10-30-pythia-adam-diagnostics/REPORT.md), [figure observation](029-2026-10-30-pythia-adam-diagnostics/observations/001-figures.md) |

| [030](030-2026-10-30-adam-lr-decisions/README.md) | Matched Adam LR decisions and diagnostic information beyond loss | Six Pythia14M/70M checkpoints, three seeds; Runs187--204 | Complete;18 runs verified, three PDFs reviewed; task cloud deleted; USD13.90 posted, provisional | [Results and reviewed figures](030-2026-10-30-adam-lr-decisions/REPORT.md) |

| [031](031-2026-10-31-adam-state-map/README.md) | H025 optimizer state and paired LR benefit | Analysis030 / Runs187-204; 12 paired altered arms, three seed clusters | Complete exploratory map; four reviewed PDFs and independent reductions/predictions verified; no training or rental | [Report](031-2026-10-31-adam-state-map/REPORT.md); early actual/adaptive retention candidate, sensitive to window and target |

| Analysis | Question | Source runs / other inputs | Status | Observations / figures |
| --- | --- | --- | --- | --- |
| [032](032-2026-11-01-eos-exact-lag10/README.md) | Exact anchored gradient cosine through lag 10 at three trained states | Run205; nine Analysis024 conditions, seed 0 | Complete; 27 profiles / 270 values, 25 primary; two failed MLP replays flagged | [Report](032-2026-11-01-eos-exact-lag10/REPORT.md), two reviewed PDFs; 7.072 GB verified; task cloud deleted; USD 1.05 compute provisional |

| [033](033-2026-11-01-eos-long-lags/README.md) | Exact long-lag damping across nine EOS conditions | Runs205/206/207/208, seed0 | Complete; 27 profiles, 25 primary; common 256, maximum 2048; all task cloud deleted | Four reviewed PDFs, exact CPU/GPU/vector audits; USD 29.48 provisional |

<!-- Analysis034 records -->
| Analysis | Question | Source runs / other inputs | Status | Observations / figures |
| --- | --- | --- | --- | --- |
| [034](034-2026-11-01-eos-adam-continuations/README.md) | Calibrated Adam from identical EOS GD checkpoints | Runs209/210/211; Run205 starts and Analysis033 GD reference | Complete; 27 matched horizons, exact anchored/adjacent raw cosines, 25 primary | Four reviewed PDFs; verified retrieval; task cloud deleted; USD 39.35 provisional |
<!-- End Analysis034 records -->


<!-- Analysis035 records -->
| Analysis | Question | Inputs | Status | Outputs |
| --- | --- | --- | --- | --- |
| [035](035-2026-11-02-eos-backward-anchors/README.md) | Backward fixed-endpoint gradient cosine, separately for GD and Adam | Run212; nine directly saved Run205 starts; largest source GD rates and Analysis034 calibrated Adam rates | Complete; 18 profiles / 4608 pairs; endpoints 756/2256/3256, lags 1-256 | Two reviewed PDFs, scalar/CPU/vector audits, explicit historical replay discrepancies; task cloud deleted; USD 9.26 provisional |
<!-- End Analysis035 records -->

<!-- Analysis036 preparation -->
| Analysis | Question | Source runs / other inputs | Status | Observations / figures |
| --- | --- | --- | --- | --- |
| [036](036-2026-11-02-controlled-adam-diagnostics/README.md) | Controlled fixed-rate Adam and loss-spike anatomy | Runs213/215; Analysis024/029/034 | Complete: 806 recovered excursions; 129 above 10%; 127 defined regression points; integrated in the 18-page draft | [Event definition](036-2026-11-02-controlled-adam-diagnostics/loss_spike_definition.md); [event data](036-2026-11-02-controlled-adam-diagnostics/spike_dataset/loss_spikes.csv); [manuscript integration and evidence](036-2026-11-02-controlled-adam-diagnostics/observations/010-manuscript-spike-anatomy.md); [multi-spike review figure](036-2026-11-02-controlled-adam-diagnostics/observations/011-multiple-spikes.md): 19 excursions, linear relative norms; source figures/data preserved; earlier exploration in archive/ |
<!-- End Analysis036 preparation -->

Next analysis: `056-YYYY-MM-DD-slug`. Analysis numbering is independent of runs;
an analysis may use manuscript or external inputs without referencing a run.

## Active batch-size extension

| Analysis | Question | Sources | Status | Records |
| --- | --- | --- | --- | --- |
| [037](037-2026-11-02-sgd-batch-extension/README.md) | H023: three-case batch-size figure extension | Runs214/216; 198 unchanged Analysis027 MLP continuations | Complete; 348 new forks, six references, Spark bridge, raw/math checks and manuscript review pass; all 13 task Pods/disks deleted, USD 26.30 provisional | [Report](037-2026-11-02-sgd-batch-extension/REPORT.md); [figure observation](037-2026-11-02-sgd-batch-extension/observations/001-three-case-sgd.md); compact mean-cosine figure |

## Adam spike intervention proposal

| Analysis | Question | Sources | Status | Records |
| --- | --- | --- | --- | --- |
| [038](038-2026-11-03-adam-spike-intervention-design/README.md) | H028: can matched interventions mitigate selected Adam spikes while preserving loss progress? | Three highest-LR Run215 arms and Analysis036 events | Design for author evaluation; source audit verified; no intervention training or paid execution | [Protocol](038-2026-11-03-adam-spike-intervention-design/PROTOCOL.md); eight-arm pilot, earlier forks, safeguards and later online confirmation; [source observation](038-2026-11-03-adam-spike-intervention-design/observations/001-source-audit.md) |

## MLP control foundation

| Analysis | Question | Inputs | Status | Records |
| --- | --- | --- | --- | --- |
| [039](039-2026-11-03-mlp-control-foundation/README.md) | H028 control implementation contracts | Run217; unchanged Run215 inputs | Verified preparation only; no neural experiment | Observer clocks, unchanged Adam, immutable fork/recovery and exact mock resume |

## Adam intervention implementation contracts

| Analysis | Question | Inputs | Status | Records |
| --- | --- | --- | --- | --- |
| [040](040-2026-11-03-mlp-step-actions/README.md) | H028 selected-fork MLP intervention pilot | Run218; exact Run215 replay, same authenticated fork | Complete8 x128 branches; single-seed exploratory | [Report](040-2026-11-03-mlp-step-actions/REPORT.md); A1/A4 pass training screen, A5/A8 negative; A7 deferred |

## Temporary LR outcome

| Analysis | Question | Inputs | Status | Records |
| --- | --- | --- | --- | --- |
| [041](041-2026-11-03-mlp-temporary-lr/REPORT.md) | H028: temporary reduction using $\eta\widehat\kappa_t$ | Run219, authenticated Run218 state657;367 updates per condition | Complete;18 reductions/36 switches, worse progress than fixed controls | Predeclared zoom and full view, direct clock/replay checks, failed control attempt preserved |

## Cooldown and cosine recovery

| Analysis | Question | Inputs | Status | Records |
| --- | --- | --- | --- | --- |
| [042](042-2026-11-03-mlp-lr-cooldown/REPORT.md) | H028: persistence and gradual LR recovery | Run220; five matched367-update arms from state657 | Complete; cosine mean/final training loss2.26%/4.30% below always-half;365/367 updates below base | [Clean four-arm figure](042-2026-11-03-mlp-lr-cooldown/figures/focal_four_arm.pdf); raw/clock/replay checks and audit correction retained; one development seed |

## Fixed-step intervention responses

| Analysis | Question | Inputs | Status | Records |
| --- | --- | --- | --- | --- |
| [043](043-2026-11-03-spark-fixed-step-interventions/README.md) | H029 fixed-step clipping, decay and LR schedules | Run221 qualification / Run222 matched highest-LR Spark controls, all three cases | Complete: 36 branches, 13 reviewed PDFs, 34,326 source-value checks and 45 verified endpoints | [Eight findings](043-2026-11-03-spark-fixed-step-interventions/FINDINGS.md); MLP LR benefit, model-dependent responses; failed cross-GPU gate retained |

## Analysis044: multi-spike interventions (2026-11-04)

[Analysis044](044-2026-11-04-multispike-interventions/FINDINGS.md) completed the three-case comparison and3 verified vector figures. Mean loss changes immediate/cosine half: MLP-25.04%/-26.30%, ResNet+27.20%/+0.62%, Transformer-9.66%/-10.68%. Fresh controls, exact replay;47,456 plotted values checked. No universal suppression or historical bitwise replay claim.

## Analysis045: MLP LR return

[Analysis045](045-2026-11-04-mlp-lr-return/README.md) compares Run224 return-to0.75x at650 against Run223 unchanged and held-half controls. Two-column five-row vector figure and source-verified values; large renewed transients, single-seed scope.

## Analysis046: MLP cosine70 restoration

[Analysis046](046-2026-11-04-mlp-cosine-rise70/README.md) compares gradual restoration with unchanged, held-half and abrupt-rise controls. Three paired columns, five diagnostic rows, verified raw values; delayed but larger spike and worse training loss.

## Analysis047: baseline-only cosine floors

[Analysis047](047-2026-11-04-mlp-cosine-floors/README.md) shows cosine32-to0.75x and0.5x side by side, each only against unchanged LR. Five raw diagnostic rows and independent scales;0.75x reduces mean loss but retains large late spikes.

## Analysis048: three paths to0.75x

[Analysis048](048-2026-11-04-mlp-three-paths-to075/README.md) places direct cosine32-to0.75x, abrupt restoration and cosine70 restoration side by side, each only against unchanged LR. Five raw diagnostic rows, independent scales; existing verified evidence only. No training.

## Analysis049: direct-decay versus abrupt-restoration overlay

[Analysis049](049-2026-11-04-mlp-direct-abrupt-overlay/README.md) directly overlays cosine32-to0.75x and cosine32-to0.5x followed by an abrupt rise at650, as requested. Five raw rows, no baseline curve, existing verified data only.

## Analysis050: two-stage decay versus two controls

[Analysis050](050-2026-11-04-mlp-two-stage-decay/README.md) pairs the new trajectory with unchanged LR and cosine32 held at0.75x. The second decay reduces the first spike and post650 mean loss; no >10% excursion after682 in this window.

## Analysis051: MLP global clipping response

[Analysis051](051-2026-11-04-mlp-global-clipping/README.md) compares Run228 global norm clipping1.0 with unchanged same-Spark control. Early peak reduced, later instability worse; five-row vector figure and73 artifact checks.

## Analysis052: MLP clipping thresholds

[Analysis052](052-2026-11-04-mlp-clipping-thresholds/README.md) compares0.5 and1.0 each against unchanged LR, including raw and clipped norms. Smaller cap reduces peak severity but does not improve mean loss over baseline.90 artifact checks; independent panel ranges.

[Analysis053](053-2026-11-05-mlp-cosine32-clip1/README.md) compares the combined policy, decay alone and clipping alone with the matched MLP baseline.

## Analysis055: Adam directional amplification (2026-11-05)

[Analysis055](055-2026-11-05-adam-directional-amplification/README.md) organizes the argument, evidence map and deferred work, and completes the nine-arm descriptive test. Five reviewed vector PDFs cover full trajectories, joint geometry and fixed context matching. Six primary arms favor reversal and three tie at zero, with sparse/limited support; low-rate controls reject a universal alarm. All 9,216 states, 812 events, 4,534 old labels and 46,098 plotted points pass checks. No classifier, new training, cloud work or manuscript edit. See [findings](055-2026-11-05-adam-directional-amplification/FINDINGS.md).

## Analysis056: GD paired displacement and manuscript revision

[Analysis056](056-2026-11-05-gd-paired-displacement/README.md) reanalyzes all nine
Analysis024 runs, nine Analysis035 backward profiles and 546 Analysis037/027
SGD continuations. C/D and the observed-length zero-cross-term reference
separate immediate cancellation, cumulative interaction and objective progress.
Both pairing offsets and two-update losses pass independent checks. Four new
vector PDFs and the revised 16-page paper are reviewed; no training/cloud work.
Next analysis: 057. No new hypothesis or run.

## Analysis057: Adam response, memory and manuscript revision

[Analysis057](057-2026-11-05-adam-response-memory/README.md) reuses frozen
Run215 / Analyses036 and 055 to add incidence, eight severity specifications,
112 fits and complete-event cross-alignments. All 794 rise-phase updates are
locally descending. Two PDFs and the 18-page manuscript pass numerical and
visual review; the analytical memory bridge awaits neural HVP calibration.
[Observation001](057-2026-11-05-adam-response-memory/observations/001-response-memory.md)
records limits; the manuscript revision retains a TODO and staged compute plan.
No new training or cloud cost. Next analysis: 058; no new run or hypothesis.

[Analysis054](054-2026-11-05-resnet-clip1/README.md) compares training loss, raw gradient geometry and clipped inputs on ResNet over states 256 through 576.

## Analysis059: MLP beta2 reductions

[Analysis059](059-2026-11-07-mlp-beta2-reduction/README.md) compares Run232's
three beta2 reductions with retained Adam. Native moment/clock continuation;
bias correction changes at the fork. Mean losses rise16.78%/18.16%/127.02%,
with one long censored excursion for0.5. Three-column figure matches Analysis058
Figure04's rows with beta2 last; all10,584 plotted scalars and render verified.

## Analysis060: warm Adam-to-GD switch

[Analysis060](060-2026-11-07-mlp-adam-to-gd/README.md) compares plain GD at.02
with Adam at.001 from identical state512. No loss increases or>10% excursions
under GD, but mean/tail loss+2.30%/+55.74%. Both optimizer and LR change.
All3528 plotted scalars and vector render pass. Both analyses use owned Spark,
one seed/window, exact control reuse and no new full controls. Next analysis:061.
