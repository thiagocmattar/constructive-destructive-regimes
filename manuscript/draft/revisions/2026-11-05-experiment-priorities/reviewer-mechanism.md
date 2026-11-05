# Independent optimization reviewer: experiment priorities

5 November 2026. Read independently of the other new reviewer reports. This is an adversarial assessment of experiments, not a manuscript edit, execution plan, or request to launch. No training or new scientific reduction was performed for this review.

## Verdict

**Do not open a separate large mechanism campaign. The highest expected return from new training is independent Adam initialization replication, with a few fixed-state diagnostics attached only after a cheap calibration.** First exploit the existing actual-update records and previously collected fixed-objective counterfactuals. They address concrete reviewer objections at almost no compute cost. HVP calibration has more genuine contribution upside than another clipping/schedule comparison, but also substantially greater implementation and interpretation risk. A direction-by-length sweep is scientifically clean yet lower priority: much of what it could establish is already admitted in the paper's stated scope.

My likely review objection is: *The measurements are mathematically sound, but do their empirical distinctions persist beyond selected trajectories, and what do they resolve beyond a loss curve, a step norm, and a first-order directional derivative?* Additional plots of reversal or another successful intervention do not answer that objection.

There are two different objectives here:

- **Credibility:** independent initialization coverage, application to actual increments, complete rather than selected comparisons, and robustness to observation horizon.
- **Contribution upside:** a discriminating account of when optimizer memory or finite-step effects makes the cheaper response statistic disagree with an independently measured curvature quantity.

The first has a substantially higher probability of improving the paper. The second is a bounded opportunity, not a prerequisite for the present diagnostic claim.

## What is already available and should not be commissioned again

[Analysis025](../../../../analyses/025-2026-10-17-pythia-descent-mechanism/REPORT.md) already contains **81 frozen-state Pythia measurements and seven candidates at each state: 567 one-step records**, besides the 18 sustained coordinate-scaling ablations. Candidates include native/reconstructed Adam, removal of coordinate scaling, refreshed first-moment direction, removal of both components, and half/double native step lengths. All share a fixed 128-sequence probe objective. Candidate-specific retained fields include actual displacement norm, probe-gradient norm, descent cosine, first-order decrease, endpoint loss, and the measured nonlinear residual. There are three seed blocks and nine measurement times per size. The current paper emphasizes only the sustained branches.

This is already a strong fixed-objective counterfactual experiment. Repeating it would not create a new mechanism result. Its pre-step temporal gradient history is shared across all candidates and therefore **cannot rank those candidates by itself**. Nor is the identity `D = P - R`, where R is computed using the observed endpoint loss, an independent validation of explanatory or predictive value. The role of this evidence is to demonstrate specific failures of step length or first-order direction as sufficient descriptions, with complete state/seed coverage.

[Analysis027](../../../../analyses/027-2026-10-30-minibatch-noise/REPORT.md) already compares sampled and complete gradients on the **same SGD states**, with three MLP initializations, frozen sampling controls and independent HVP calculations. The reported batch-32 comparison has 288 HVP probe pairs; the complete study has 1,188 paired derivative evaluations. Repeating a generic stochastic-versus-fixed-objective demonstration on that MLP would duplicate completed science.

[Analysis056](../../../../analyses/056-2026-11-05-gd-paired-displacement/README.md) already checks both adjacent pairing offsets and nine fixed-endpoint lag profiles. A new offset-only sensitivity test is redundant. The separate three-seed GD references in [Analysis027](../../../../analyses/027-2026-10-30-minibatch-noise/README.md) and [Analysis037](../../../../analyses/037-2026-11-02-sgd-batch-extension/README.md) are valuable reuse opportunities before launching new GD trajectories. Their 2,000-update horizon and reference recipes must remain explicit; they do not become independent replications of every rate in the original 3,000-update sweep.

## Ranked tests

### 1. Actual-update and horizon audit from retained histories

**Return:** high credibility per unit effort; moderate contribution upside. **Recommendation:** do first, before any new GPU work.

The paired factors are currently demonstrated most directly on GD gradients, while the Adam conclusions require a careful distinction between raw gradients and applied motion. Test whether the empirical account survives that transfer rather than merely citing the algebraic generalization.

**Minimum design.** On all nine Run215 trajectories, compute the already defined paired cancellation C, residual-path factor D, R and Q for the four recorded streams: raw gradient, stored first moment, adaptive direction, and actual parameter increment. Use prespecified full prefixes ending at updates 256, 512, 768 and 1,024; use both valid pair offsets with boundary singletons. Compare the full tables with adjacent cosine, norm ratios and R alone. Report whether raw-gradient reversal corresponds to cancellation of applied motion, and whether the relative roles of local cancellation and residual turning change with horizon. These are descriptive, within-trajectory comparisons, not independent replicated observations or forecasts of loss.

Reuse the three-initialization GD reference histories from Analyses027/037 for the same endpoint quantities where their logged fields permit it, at matched reference endpoints. Check the manifest and measurement schema before assuming every prefix or vector product exists. This can test seed consistency of the GD geometry at the existing fixed rate without reproducing the full rate sweep.

**Available now.** A directly inspected Run215 metric row contains norm, A, B, energy, adjacent cosine, all-pair and lag-one terms for every stream, including actual increments. Thus the prefix calculation is genuinely recoverable from scalars. It does not require final-checkpoint replay. The current exact pairing mathematics has already passed independent tests; the new scientific question is transfer across streams and horizons.

**Important boundary.** An arbitrary window displacement is not obtainable by subtracting prefix norms. For a cropped window, D/R/Q need the displacement vector between its endpoints or a recorded window-sum norm. C alone can be reconstructed from adjacent pair norms. Pythia Runs127?135 record actual-increment norm and adjacent cosine but do **not** record actual-increment cumulative B in their scalar event schema. Do not claim a full actual-update C/D study for those runs from those scalars. Their raw/moment/adaptive prefix streams and checkpoint-defined displacement intervals have different coverage.

**What would count against the interpretation?** If C/D conclusions reverse with prefix length, or raw cancellation largely disappears in actual increments, that limits the proposed regime language to its named stream and horizon. It is a useful result, not a failed run to exclude. If C/D never distinguishes cases beyond mean cosine and R in the retained cohort, it remains a transparent accounting device rather than an empirically indispensable diagnostic.

**Effort.** Low implementation and CPU reduction only. No new gradient evaluations. Keep this as one auditable analysis, not another set of large training runs. The all-state Analysis025 candidate summary is a companion completeness check, not a new validation claim.

### 2. Sparse Adam HVP calibration, attached to replication if it is commissioned

**Return:** highest mechanism/contribution upside, but uncertain. **Recommendation:** conditional addition to independent-seed Adam work; no separate full-grid training campaign.

Appendix C currently separates the normalized raw response into a frozen-time preconditioned proxy, a first-moment correction, and a finite-step discrepancy. It openly states that this has not been calibrated on neural trajectories. A small empirical calibration could determine whether this bridge explains a recurrent difference between diagnostics, rather than being unused formal machinery.

**Minimum design.** First qualify a smooth tanh-MLP pilot. At a fixed, loss-independent grid such as states 256, 512, 768 and 1,024 of the highest-rate arm, retain the complete objective gradient, Adam state and exact incoming proposal. For any subsequently approved new initialization blocks, apply the identical grid without selecting only spikes. At each state compute one complete-objective `H*g`; it provides both the frozen proxy and its memory correction. Compare the independently measured next-gradient response with each approximation. Then hold the current state, objective, moments, direction and bias-correction clock fixed, and evaluate endpoint gradients along that direction at length multipliers 1, 1/2 and 1/4. The full-length endpoint can be the ordinary next state when it qualifies. Do not advance moments separately for each shortened proposal.

**Discriminating outcomes.** Does the memory correction improve agreement over the frozen proxy across ordinary as well as reversing states? Does the remaining discrepancy shrink with step length, as a local curvature approximation would require on the smooth pilot? Report per-state signed/absolute error and scale-sensitive numerical tolerances, not only a pooled correlation dominated by large responses. Compare with a first-order/no-curvature reference and a starting-Hessian mixed projection. This is an independently measured derivative test, not re-evaluation of the same algebra.

**Negative result.** A memory correction that does not improve agreement, or an error that fails to contract above numerical resolution, would argue against promoting this appendix into a central mechanism. Keep the bounded diagnostic interpretation and stop. A negative calibration is not a reason to add more architectures or select more favorable states.

**Assumptions and feasibility.** The smooth MLP supports the stated path-regularity interpretation. The controlled ResNet and WikiText Transformer use ReLU; their residual combines possible activation-boundary effects, finite-step variation and numerics, and cannot automatically be named smooth curvature variation. The Hessian and gradient must belong to the same complete objective and fixed physical partition. A fixed-probe gradient cannot be substituted into the training-gradient moment recurrence.

Run215's original middle checkpoints were overwritten, as established by [Analysis038's source audit](../../../../analyses/038-2026-11-03-adam-spike-intervention-design/PROTOCOL.md); its README's original checkpoint cadence is not evidence that all states remain. Final states and audit gradients do not recover historical HVPs. This therefore requires new diagnostics at retained suitable states or a qualified replay/new replication, not a scalar-only reanalysis.

**Effort.** Medium implementation risk; potentially low incremental compute if attached to already justified runs. Four anchors require four HVPs plus at most eight additional full-gradient evaluations per trajectory, in addition to its ordinary next-step measurements. HVP cost and numerical fidelity are unmeasured for this recipe; benchmark the smooth pilot before expanding. The historical full-objective Transformer was much slower than the MLP, so counting HVPs as cheap minibatches would be misleading. No current dollar or wall-clock estimate is justified from this review.

### 3. Frozen-state direction-by-length response, only if the scalar-scale objection becomes central

**Return:** moderate scientific information, lower expected acceptance return than independent Adam seeds. **Recommendation:** defer in the minimal campaign.

The 18 existing branches remove coordinate scaling at the native Adam adaptive-component length. A skeptical reader can reasonably ask whether poor scalar scale explains the harm. The manuscript already explicitly says that these results do not compare against independently tuned momentum, so this is **not an unaddressed correctness flaw**. Extra curves might refine the explanation without improving the main contribution materially.

**Minimum bounded test, if needed.** Use the retained Pythia-14M/70M checkpoints, three seeds and states after updates 175/225: 12 states. Form native, no-coordinate-scaling and refreshed-moment proposals from the same next training batch and optimizer clock, then evaluate fixed-direction loss curves at four lengths (1, 1/2, 1/4, 1/8). This is at most 144 endpoint probe evaluations and no sustained continuations. The native next proposals correspond to updates 176/226: existing probes recorded at updates 175/225 were evaluated from states 174/224 and are not the identical parents.

Specify whether alpha scales the whole realized displacement or only its adaptive component, because decay must not silently change between candidates. The simplest line test scales the whole constructed displacement, records its actual norm and uses the unchanged fixed probe. This is a frozen-direction diagnostic, not a new optimizer. Keep the existing native/reconstructed numerical application control.

**What new fact could result?** Different loss curves might show that substitute directions become useful only at much shorter lengths, or remain inferior over the prespecified range. Neither establishes superiority of retuned momentum or long-run benefit. A higher measured nonlinear residual at a full step is already known from Analysis025; reproducing it alone earns little. Adding directional HVPs would permit independent comparison to the starting quadratic model, but would expand this test beyond its current narrow return.

**Negative result and limits.** If shorter substitute steps eliminate the apparent disadvantage, the interpretation shifts toward a direction-dependent scalar-scale mismatch. That remains consistent with the current qualified paper. If the gap persists, it strengthens only the bounded local comparison. A few alpha points do not estimate an optimizer's best achievable performance, and the same probe must not serve as both tuning set and independent validation of a new method.

**Effort.** Medium, lower than new training but not free: the probe has 262,016 targets, uses FP32/math SDPA/microbatch one, and cannot be priced as one eighth of the mixed-precision training update simply from token counts. The 45 retained recovery states and verified input/probe machinery make this feasible. It does not require new model training.

## What I would explicitly not prioritize

- **A new stochastic monitoring campaign:** the core sampled-versus-same-objective distinction is already tested deeply for the MLP, and Pythia has fixed-objective candidate evidence. A new adaptive stochastic extension would add implementation and objective-definition issues. It is justified only if the paper elevates a stochastic Adam mechanism beyond its present claim.
- **A larger model merely to reproduce negative cosine:** scale alone would not distinguish mechanisms or repair replication of the controlled studies.
- **More clipping, decay, or orthogonalization arms:** existing interventions already establish that smoother symptoms need not improve outcomes. A new method would change the scope.
- **Another identity or synthetic-quadratic check:** these are necessary implementation gates, not new scientific evidence; the existing suites already contain extensive mathematical verification.
- **Pair-offset-only or threshold-only checks:** relevant sensitivities have already been executed. New robustness work should target independent initializations, meaningful horizons or applied increments.

## One compact campaign

1. Finish the retained-history actual-update/horizon audit, including the existing three-seed GD references and complete Analysis025 controls. Stop if it exposes a scope problem and report the result before choosing training.
2. If new training is authorized, spend the main effort on the narrowly selected independent Adam seed/rate replication proposed by the validity reviewer, not on more mechanisms. Fix the primary descriptive conclusions and experimental unit before execution.
3. Attach the sparse HVP/shortened-step calibration only after the smooth MLP pilot qualifies and only at the fixed anchors of those already justified runs. Failure to calibrate ends the extension. Do not automatically add the direction-by-length or stochastic campaigns.

The primary deliverable should be a compact seed-by-condition evidence table and, only if the calibration earns it, one mechanistic figure. This is a closing experiment program with a stopping rule, not a new research branch.
