# Highest-return tests for the diagnostic paper

5 November 2026. Reviewer simulation and experiment prioritization requested after the submission-synthesis revision. **Assessment and planning only:** no manuscript edits, new result reductions, training, cloud actions or spending. The evaluated paper and evidence are pinned in [reviewed_sources.json](reviewed_sources.json). This plan does not allocate a run, analysis or hypothesis ID.

## Recommendation

**Do the retained-data audits first. If one new experiment is chosen, prospectively test raw-gradient clipping versus actual Adam motion across new seeds and fixed checkpoints. Attach a small smooth-model HVP calibration only if the authors want to strengthen the memory/curvature interpretation.**

The return criterion is the extent to which a test could change a skeptical reviewer's judgment, relative to implementation, compute and scope cost. This is a qualitative assessment, not an estimated acceptance probability. More evidence can strengthen credibility without making the underlying identities more novel.

| Priority | Test | What it can improve | Additional execution |
| --- | --- | --- | --- |
| 1 | Existing three-seed GD references, actual-Adam geometry, and complete Pythia counterfactual audit | Replication, horizon robustness, and diagnostic usefulness beyond selected examples | Local reductions only; no new training |
| 2 | Fixed-checkpoint, independent-seed clipping comparison | A reproducible practical consequence of distinguishing optimizer input, motion and objective progress | Six parent trajectories and 18 paired fork comparisons; 10,752 main full-objective updates; qualification is additional |
| 3, conditional | Nine smooth-MLP HVP anchors with fixed-direction shortening | Whether the Adam memory correction is empirically material and useful as an approximation | Nine HVPs and 27 trial endpoint gradients, plus common anchor gradients and qualification |
| Deferred | Pythia direction-by-length response surfaces; function-space cancellation; broad event-replication grid | Secondary mechanism or broader conceptual scope | Separate campaign only if its specific claim becomes central |

The strongest remaining objection is substantive: the mathematical ingredients are established, so the paper must demonstrate what its combined diagnostics reveal. The answer should be a repeated, falsifiable distinction, not another validation of R=CD or a larger collection of oscillating curves. Close literature already treats trajectory directionality and Adam mechanisms: [Singh et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37771cc0be272368102a37f202bb88d8-Abstract-Conference.html), [Bai et al., ICML 2026](https://proceedings.mlr.press/v306/bai26o.html). These are reasons to target the added diagnostic value precisely.

## 1. First use evidence already collected

### A. Replication and temporal scale of the central GD result

The main three-rate, 3,000-update GD sweep has one initialization per model. However, [Analyses027](../../../../analyses/027-2026-10-30-minibatch-noise/REPORT.md) and [037](../../../../analyses/037-2026-11-02-sgd-batch-extension/REPORT.md) already retain **three-initialization, 2,000-update full-GD references for each model**, plus complete-objective continuations from trained states. This coverage was collected for the SGD study; it can test the paired geometry without buying new GD trajectories.

Use all nine references at fixed prefixes T={200,500,1000,2000}, and the nine 200-update full-GD continuations starting at state 2000. Report C, D, R, Q, adjacent interaction, actual displacement agreement and loss change for both pair offsets. Keep all seed values and aggregate within model. The rates are MLP .02, ResNet .1 and Transformer .2: this is a fixed-rate replication, not replication of the original whole sweep.

The substantive questions are whether saturated negative cosine repeatedly hides different local cancellation/residual coherence, and whether negative adjacent but positive all-pair interaction survives on late reset segments. A full prefix can be dominated by early motion. Failure on late segments bounds the horizon of the observation; it does not invalidate the identity. Runtime differences remain part of the block caveat.

Arbitrary-window resultant B cannot be recovered by subtracting two prefix norms. Use genuinely reset streams or authenticated vector endpoints. Existing scalar records are sufficient for the stated prefixes and reset continuations; do not presume dense gradient histories.

### B. Carry the accounting through to actual Adam updates

[Run215](../../../../runs/215-2026-11-02-controlled-full-batch-adam/README.md) already stores norms, adjacent cosines, cumulative A/B/energy and cross-stream quantities for actual increments. Apply the same C/D/R/Q accounting to all nine trajectories at fixed prefixes T={128,256,512,1024}; compare with the raw-gradient stream at correctly aligned update boundaries.

This tests whether the paper's central accounting adds a useful empirical distinction after optimizer transformation, beyond the present verbal warning that raw gradients are different. Compare the information from cosine alone, R alone, and C/D together; report losses separately. Recover C from adjacent norm/cosine pairs and B from the actual prefix records. Do not describe R=CD as prediction, recover arbitrary windows from scalar prefix differences, or label a raw-gradient sum as Adam displacement.

### C. Audit every retained Pythia candidate, not only the striking example

[Analysis025](../../../../analyses/025-2026-10-17-pythia-descent-mechanism/REPORT.md) contains **81 frozen-state probes x seven candidates = 567 records**, in addition to 18 sustained branches. The seven candidates already include optimizer-component substitutions and native half/double steps. For every size, seed and preselected time, tabulate first-order versus measured finite-step loss rankings and actual displacement norms. Report reversals of those rankings, heterogeneity outside the highlighted interval, and native/reconstructed-native agreement.

The outcome is whether a diagnostic distinction repeats across the complete retained cohort, and where it does not. The identity 'finite change = linear term + residual' is accounting, not independent evidence of a curvature mechanism. A pre-step temporal cosine is shared by every candidate at a frozen state, so it cannot rank those alternatives by itself. Native half/double steps do not constitute retuning the substituted directions.

For all three audits, first inspect whether the requested complete-cohort reductions already exist. Reuse them if so. Do not create another analysis or figure merely to restate an existing table. Otherwise freeze a new analysis contract, retain source hashes and all failures, and produce one compact evidence table per question. Three seed blocks support individual effects and ranges, not precise population confidence intervals. These audits have not been executed in this planning task.

## 2. Best new experiment: prospective clipping confirmation

**Reviewer objection addressed:** the practical diagnostic payoff may depend on one initialization and hand-selected intervention windows. The original MLP clipping example shows increased applied motion despite bounded raw input, while the ResNet response differs. Test that distinction with frozen choices on new initializations.

### Minimal grid

- Models/objectives: the existing smooth tanh MLP on its prepared 5,000 CIFAR-10 images, mean half-squared error; existing fixed-normalization-group ResNet-32 on all 50,000 CIFAR-10 training images, mean cross-entropy.
- Parents: three initializations per model, high original Adam rate .001 for MLP and .003 for ResNet, 1,024 updates. Seed0 is a contemporary replay of development evidence; seeds 1/2 are genuinely new Adam initialization conditions. That is six parents and four new model/initialization conditions, not six new independent confirmations.
- Interventions: fixed post-update states 256, 512, 768; unchanged versus raw global norm clipping at1 before both Adam moments; 128 updates per arm. The first proposal after state 256 is update 257.
- Size: 2 models x 3 seeds x 3 forks = 18 paired comparisons; 36 fresh branches. Conservative main-training arithmetic is 6 x 1,024 + 36 x 128 = 10,752 updates. Exact within-environment replay may qualify the parent segment as the unchanged control, reducing execution by up to 2,304 updates. Budget conservatively for fresh controls until qualification. Calibration and replay qualification may require extra optimizer updates; 10,752 is not an all-inclusive execution ceiling.

Hold initial tensors, data and ordering, objective reductions, accumulation/BN groups, Adam betas (.9, .999) / epsilon 1e-8, precision, warm optimizer/RNG states, parameter scope and update budget fixed. Parents have no clipping, schedule or decay. Generate each new initialization once, hash it, and use the same checkpoint for its treatment/control. Use one qualified runtime per model; preserve discrepancies rather than promising historical bitwise equivalence across devices.

This 128-update grid is a new prospective test of the diagnostic implication, not literal replication of the old 288/320-update selected windows. Clip 1 and the high rates are development choices, now frozen. An inactive clip or an absent excursion remains a result; do not move the fork, change the threshold or tune the rate to recover the desired behavior.

### Outcomes and what the diagnostics must add

Primary quantities are the paired 128-update mean training loss, mean actual-increment norm ratio, maximum actual-increment norm ratio and clipping-active fraction. Report each separately. Prespecified secondary quantities are endpoint loss, last 32 mean loss, maximum loss, and first-step applied-motion ratio from the identical warm state. The first step separates an immediate optimizer response from later changes involving diverged parameter and moment histories; it does not by itself explain the entire branch.

Log raw, clipped-input, first-moment, adaptive-direction and actual-increment quantities with explicit clocks. Reset the branch geometry at the fork, retain all per-step norms/products and cumulative vectors, and compute **actual-update C/D/R/Q** for both pair offsets over the common 128-update interval. Include cosine-only and step-norm-only summaries alongside them.

The added-value question is concrete: does the joint account distinguish increased step length, within-pair cancellation and residual accumulation across treatment/control, where cosine or norms alone conflate them? Loss establishes the objective consequence separately. Do not fit R from C and D, or regress an exact loss residual back onto its defining terms, and call that validation. A result showing no useful additional distinction is a legitimate negative finding.

Keep initialization as the replication unit. The three forks share a parent. Show every seed-by-fork effect; then equally average the three prespecified forks within each seed. Report seed values and ranges within each model, with new-seed confirmation separated from seed 0 development replay. Do not pool the two objectives, bootstrap steps/events as independent, or interpret inactive clipping as evidence that active clipping is harmless. Preserve nonfinite and incomplete conditions in coverage.

A positive result strengthens a practical scientific statement: optimizer-input control can differ from applied-motion control, and their objective effects depend on the observed setting and horizon. A heterogeneous or negative result tells us how narrow the existing example is. Neither establishes that clipping is generally good/bad, that a diagnostic-guided policy improves training, nor that the Transformer/all-rate Adam anatomy is replicated.

### Cost and retention

This is a bounded campaign using existing drivers, not the earlier 27-trajectory/HVP program. Historical same-project timing places a 1,024-update ResNet Adam trajectory at about 0.743 GPU-hours on an RTX4090. The conservative ResNet portion above is 5,376 updates, or about 3.9 such GPU-hours before setup, added instrumentation, checkpoint I/O, transfer and verification. This is a historical scaling scenario, not a measured ETC or current price. MLP training was much cheaper in the existing local runs, but its new end-to-end cost must also be calibrated. No dollar estimate or cloud envelope is authorized here.

Before any launch, freeze input/model/data/source hashes and the complete metric contract; verify resume/control equality; calibrate the instrumented recipe; choose checkpoint retention and compute/storage ceilings. Retain the six parent initial/final states, all 18 forks, optimizer/RNG/data position and diagnostic accumulators, branch scalar records and terminal recovery states. Verify retrieval sizes/hashes before worker cleanup. Any cloud plan needs live pricing and explicit task-level authorization; old budgets in other records are not this campaign's budget.

## 3. Optional small calibration, not a second large campaign

If the authors want to move Appendix C from an analytical qualification toward a measured diagnostic result, use the three smooth-MLP parents at the same three fixed anchors: **nine HVP anchors**, plus endpoint gradients at frozen direction scales {1,1/2,1/4}, giving 27 trial endpoints. Compute the shared complete-objective anchor gradient and H*g, and preserve the post-recursion moment/preconditioner and clock. Numerical qualification and any repeated measurements are additional work, not hidden in those counts.

Compare the observed projected gradient response with the published single-HVP proxy and with its measured memory correction. The corrected expression and direct mixed-direction Hessian projection are algebraically equal: that comparison checks implementation, not a second scientific hypothesis. The actual empirical tests are whether the correction materially improves agreement across the fixed anchors, and how the remaining error behaves as the step shrinks. Report signed/absolute errors, numerical resolution and null zero-denominator cases. Both improvement and worsening remain visible.

The smooth MLP avoids the unqualified classical Hessian interpretation across ReLU boundaries. Existing [Analysis027](../../../../analyses/027-2026-10-30-minibatch-noise/REPORT.md) already supplies substantial SGD noise/HVP validation; repeating it adds little. If the Adam correction is negligible, worsens agreement, or cannot be resolved numerically, retain the appendix's current qualified status. Do not commission more models to obtain a positive story.

## Reviewer disagreement and final resolution

The independent reports are [significance](reviewer-significance.md), [validity](reviewer-validity.md), and [mechanism](reviewer-mechanism.md). They are simulated internal reviews, not journal verdicts.

Initially the significance reviewer preferred a Pythia direction-by-length response surface over new seeds, arguing that replication cannot fix novelty. The mechanism reviewer preferred independent Adam replication because the paper already qualifies its native-scale ablation. On challenge, the significance reviewer agreed that the length sweep mostly strengthens a secondary explanation and would barely alter the judgment of the central C/D contribution.

Both supported the fixed-calendar clipping grid as a stronger combined test of replication and diagnostic usefulness. The significance reviewer added an essential condition: measure actual-update C/D/R and compare what they reveal with cosine and step norms; otherwise the work is mainly a larger clipping case study. The final recommendation adopts that condition. A final read-only methods review checked the root design, independence and update/HVP counts; its calibration-count caveat is incorporated above. The validity reviewer treats pure Adam event replication as an alternative if that headline becomes the authors' priority, not an additional mandatory campaign.

The most conceptually ambitious alternative is a small parameter-to-function cancellation study. It might show whether parameter reversal corresponds to repeated changes in outputs, but it introduces a new measurement space and scientific question. It is worth retaining as a future direction, not adding to this submission by default.

## Stopping rule

Complete the reuse audits and reassess which empirical objection remains. If new training is chosen, freeze one clipping comparison, complete all admitted cells and report unfavorable results alongside favorable ones. Add only the nine-anchor smooth calibration if the authors explicitly want the associated stronger claim. Stop after those answers; do not expand thresholds, schedules, seeds or model sizes in response to an inconvenient result.

This replaces an open-ended experiment menu with a decision sequence. It does not promise that additional runs will make established mathematics novel or guarantee acceptance. The deliverable should be a clearer, replicated scientific distinction that earns its place in the diagnostic argument.
