# Evidence map and analysis sequence

| Source | What is available | Role and limitation |
| --- | --- | --- |
| [Run215](../../runs/215-2026-11-02-controlled-full-batch-adam/README.md) | Nine matched-initialization full-objective Adam arms, three cases/rates,1024 updates each; raw scalar geometry, loss, identities and final recovery | Primary cohort here; one initialization per case, runtime differences; scalar histories cannot recover missing intermediate full tensors |
| [Analysis036](../036-2026-11-02-controlled-adam-diagnostics/README.md) | Fixed-reference812 candidates/806 recovered,129 recovered >10%, current anatomy/scatters and winning multi-spike figure | Loss-only event source and motivating illustration; selected events/peak norms do not establish prediction |
| [Historical temporal analysis](../036-2026-11-02-controlled-adam-diagnostics/archive/PEAK_TEMPORAL_ANALYSIS.md) | Quiet-state eight-update/10% future labels,4534 chronological test rows; loss/norm/Q models and false alarms | Existing development evidence and label audit; different episode convention; all inspected, no untouched confirmation |
| [Analysis024](../024-2026-10-17-eos-gd-lr-calibration/README.md) | Three-case full-GD LR trajectories with recorded curvature and gradient geometry | Grounding for deterministic interpretation; ordinary/subset Hessian scopes differ |
| [Analysis027](../027-2026-10-30-minibatch-noise/README.md) | Frozen sampling, matched MLP SGD continuations,1188 HVP probe pairs | Noise/curvature calibration at declared objectives; not Adam preconditioning validation |
| [Analysis037](../037-2026-11-02-sgd-batch-extension/REPORT.md) | Three cases, three initializations, matched batch-size/start-state continuations; full endpoint losses | Directional attenuation support; does not establish Adam spike suppression or a universal improvement from noise |
| [Analysis029](../029-2026-10-30-pythia-adam-diagnostics/REPORT.md) and [002](../002-2026-10-04-scale-seed-replication/README.md) | Pythia14M/31M/70M stochastic AdamW histories, three-seed size cohort, LR/schedule branches and raw geometry | Practical-setting bridge; clipping/schedules/noise change interpretation; no >200M demonstration |
| [Analyses038-043](../043-2026-11-03-spark-fixed-step-interventions/FINDINGS.md), [044-053 registry](../INDEX.md) | Existing fork/action/clipping/schedule results, including negative and model-dependent outcomes | Reuse before proposing new controls; isolated selected-window interventions do not validate an online classifier/policy |

Historical forecasting uses trailing-minimum episode conventions and a separate
future-rise label. The current event table uses fixed pre-rise excursions from
initialization. Neither set of 129 counts should be silently substituted for the
other. This analysis audits future labels against raw losses without fitting
the archived models again.

## Ordered work and evidence ladder

| Step | Status at authorization | Evidence added |
| --- | --- | --- |
| 1A Complete nine-arm timelines | Execute here | Coverage, repetition, counterexamples; no selection on a successful spike |
| 1B Joint direction/norm-ratio map | Execute here | Where ordinary and future-rise states lie in the same observed geometry |
| 1C Similar-norm/context descriptive comparison | Execute here | Whether differences survive fixed local matching, and where common support fails |
| 2 Freeze confirmatory target/availability | Backlog | An operational future event and deployable clock |
| 3 Fair predictor/component comparisons | Backlog | Incremental information beyond norm and recent loss |
| 4 Unseen-trajectory validation | Backlog | Generalization, calibration, lead time and false alarms |
| 5 Action utility | Backlog | Whether acting improves full-trajectory loss/progress/cost tradeoffs |
| Optional >200M/HVP comparison | Backlog | Measured implementation cost and distinct diagnostic utility |

Step1's descriptive frequency is not a calibrated probability. Steps3-4 are
needed for an incremental predictive claim; step5 is needed for an intervention
benefit claim. A future run/analysis gets a new sequential ID when implemented.
