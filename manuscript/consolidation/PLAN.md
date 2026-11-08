# Paper anatomy and closure map

- **Central argument:** learning updates reinforce or reverse earlier changes. Curvature, amplification, sampling and accumulation explain their consequences; representation tracing connects them to functional changes and interventions.
- **Title:** *Constructive and Destructive Training Regimes: A Framework for Interpreting SGD Stability*.
- **Structure:** retain the six sections, theory subsections and four GD/SGD figures. Develop characterization → tracing → intervention **inside Section V**.
- **Reading contract:** this is the complete working map. Links provide source evidence; no other consolidation document is required.

## I. Introduction — the gaps and what we contribute

- **1. What does a destructive regime mean for learning progress?**
  - **Prior work → gap:** [EoS](https://arxiv.org/pdf/2103.00065) explains oscillatory progress; [Chee–Toulis](https://proceedings.mlr.press/v84/chee18a.html)/[Pesme](https://proceedings.mlr.press/v119/pesme20a.html) diagnose stationarity. A local reversal label does not resolve accumulated motion or objective progress.
  - **Our contribution:** constructive/destructive characterization from SGD updates, connecting local interaction, gradient magnitude and cumulative displacement.
  - **Demonstration:** all nine GD cases have positive total pair interaction; seven simultaneously have negative adjacent interaction. **Home:** III opening/III-D → IV Figures 2–3.

- **2. What can existing gradients tell us about curvature under sampling?**
  - **Prior work → gap:** [Barzilai–Borwein](https://doi.org/10.1093/imanum/8.1.141) supplies secants; gradient-product theory includes noise. Neural trajectory interpretation still requires separating directional curvature, norm change and sampling.
  - **Our contribution:** the alignment–curvature relation, gradient-only proxy and explicit noise contribution, with direct derivative and batch-size validation.
  - **Demonstration:** 1,188 HVP comparisons; at batch 32, an offline full-gradient noise correction reduces median proxy discrepancy from .7467 to .0030. **Home:** III-A/B/C → IV Figures 1/4 and compact supporting validation.

- **3. When does reversal become a loss excursion?**
  - **Prior work → gap:** EoS and [Bai et al.](https://arxiv.org/html/2506.04805v2#S1) explain productive oscillation and Adam spike dynamics. Reversal alone does not distinguish descent from excessive movement; its functional consequence remains to be measured.
  - **Our contribution:** a local GD relation combining reversal and norm amplification, followed by empirical loss accounting and its functional interpretation under Adam.
  - **Demonstration:** 26,991 GD transitions delimit the approximation; within the measured continuation windows of three MLP seeds, every meaningful baseline loss rise has beneficial residual correction outweighed by squared output movement. **Home:** III-E → IV validation → V interpretation.

- **4. Where does instability appear in representations, and what should be controlled?**
  - **Prior work → gap:** [Catapults](https://proceedings.mlr.press/v235/zhu24h.html) studies output/NTK dynamics; [Kosson](https://arxiv.org/html/2410.23922v1#S6) studies representation change and warmup. We need event-level localization and tests of which local effects can be limited while retaining progress.
  - **Our contribution:** fixed-probe tracing of activation increments and temporal modes, followed by per-unit local-displacement caps and discriminating controls.
  - **Demonstration:** structured motion across three architectures; cap training-mean reductions of 22.85%/16.22%/26.00% in the selected MLP/ResNet/Transformer comparisons, plus MLP replication. **Home:** V-B/C.

- **Edit:** retain the four current contribution components; extend empirical characterization with tracing/interventions. Credit established identities and representation-control precedents; claim the connected explanation, capability and tested findings.

## II. Stability Analysis of SGD — give each predecessor a clear role

- **Keep the four existing subsections:** local curvature; minibatch stochasticity; curvature/noise/EoS; practical large-model stability.
- **Add precise comparisons:** connect those subsections to the four gaps; define regime/stability operationally.
- **Cap distinction:** Kosson's RRC correction uses layer scaling from a signal-to-noise model of **relative** response; ours directly caps per-unit **absolute** affine RMS against a past budget. Both fix local inputs (§6/Appendix B.6).

## III. Constructive and Destructive Regimes — retain the analytical spine

- **Opening: update decomposition.** Keep exact SGD displacement, all temporal pair interactions, and the local constructive/destructive sign definition.
- **A. Directional curvature.** Keep the boxed curvature–alignment connection and the distinction between reversal and amplification.
- **B. Hessian-free estimator.** Keep `eta*kappa_hat = 1 − rho*||g_current||/||g_previous||`; explain its GD directional/path-average meaning and secant antecedent.
- **C. Sampling.** Keep the noise term, finite-population variance and cosine approximation; separate fixed-state sampling from changing SGD trajectories.
- **D. Normalized displacement.** Keep R as net displacement divided by path length, with the existing example. Low R does not measure useless learning.
- **E. Loss decrease and excursions.** Keep the local quadratic criterion and angle/norm examples; distinguish an individual loss rise from a recovered spike.
- **Finish:** label exact/approximate statements and assumptions; retain notation and undefined-value conventions. Add [024](../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/007-section-iv-critical-review.md)/[027](../../analyses/027-2026-10-30-minibatch-noise/REPORT.md) validation. **New training: none.**

## IV. Empirical Characterization — four questions, four existing figures

- **Cases:** CIFAR-10 MLP, 5,000 examples/half-squared error; CIFAR-10 ResNet-32, 50,000 examples/cross-entropy; WikiText-2 Transformer, 2,088,600 supervised tokens/token-weighted NLL.
- **Figure 1 — How do regimes emerge with LR and curvature?**
  - Keep [024's loss/proxy/spectral display](../draft/figures/analysis024/gd_loss_projected_change_spectral_clean.pdf): nine GD conditions, three model/data cases, 3,000 updates.
  - Add concise HVP/noise validation in supporting material; disclose full-MLP versus subset spectral coverage.
- **Figure 2 — How can destructive alignment coexist with progress?**
  - Keep [reversal, gradient norm and R](../draft/figures/analysis024/gd_reversal_norm_retention.pdf); explain local reversal alongside positive all-pair accumulation and loss progress.
- **Figure 3 — How are interactions organized over time?**
  - Keep [035's backward lags](../draft/figures/analysis035/gd_backward_compact.pdf): endpoint 2256, lags 1–256. Explain alternating/reinforcing longer lags.
- **Figure 4 — What changes with minibatch size?**
  - Keep [037's batch comparison](../draft/figures/analysis037/sgd_batch_size_compact.pdf): 546 trajectories, three cases, three initializations. Use frozen-state controls to separate sampling from changed paths.
- **Finish:** connect results to III; state seeds, coverage and exposure. **New GD/HVP/lag/batch experiments: none.**

## V. Raw-Gradient Diagnostics under Adam — characterize, trace, intervene

- **Opening — preserve the observable, explain its meaning.**
  - Adam determines the traversed direction; the preceding raw gradient determines the projection. Use the existing Appendix A identity.
  - The same computation remains meaningful; GD's loss/stability thresholds do not automatically transfer.

### A. Characterization: one high-LR case per model

- **Display:** [058's high-LR overview](../../analyses/058-2026-11-06-intervention-consolidation/figures/00-high-lr-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf).
- **Three-row peak table:** use the already selected largest loss-defined excursions in [036](../../analyses/036-2026-11-02-controlled-adam-diagnostics/README.md):
  - MLP, LR .001: pre-rise/peak/recovery **689/703/729**.
  - ResNet, LR .003: **415/422/482**.
  - Transformer, LR .003: **787/794/855**.
- **Columns:** loss, rho, gradient-norm ratio and eta-kappa-hat at pre-rise and peak; incoming/outgoing update and first-availability labels.
- **Timing:** the current gradient precedes the outgoing update but describes the completed incoming step. Extract the table; retain event/censoring counts; perform check **A** below. **Characterization needs no training.**

### B. Tracing: follow the functional oscillation through the network

- **Algorithm:** fixed probe → output increments → gradient/output association → hidden-layer increments → cosine/amplitude → temporal SVD and signed coefficients.
- **One combined display:** loss/output motion; layer heatmaps; energy spectrum; leading coefficients. Follow [065 Figure 07, page 2](../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/07-distributed-displacement-and-controls.pdf) for readable layer grouping.
- **Use the existing [063 event fits](../../analyses/063-2026-11-07-architecture-instability-tracing/observations/005-technical-report-event-window.md):**
  - MLP 450–500: all 5,000 examples; output rank95 = **5**.
  - ResNet 280–310: one intact 1,000-example normalization group; rank95 = **16**.
  - Transformer 552–600: 64 sequences/2,240 supervised positions; rank95 = **15**.
- **Interpretation:** structured functional motion and architecture-specific localization; use 061/062 for hidden-motion/amplification explanations.
- **Finish:** assemble algorithm/figure using matched windows, explicitly distinct from V-A's peaks. **New tracing: none.**

### C. Intervention: test what the interpretation makes actionable

- **Present in the author's order:** LR schedules → clipping → beta2 → local-displacement cap.
- **Existing evidence:** [058](../../analyses/058-2026-11-06-intervention-consolidation/README.md) supplies LR/clipping; [059](../../analyses/059-2026-11-07-mlp-beta2-reduction/README.md) recovers the late beta2 reductions; [065](../../analyses/065-2026-11-07-mlp-frozen-lr-double/README.md) supplies caps and controls.
- **Cap algorithm:** scale proposed weight/bias increments by `min(1, tau/s)`, where s is per-unit local affine RMS and tau is a positive past-calibrated budget. Preserve Adam state. The bound is local; the controller uses no SVD.
- **Main evidence:** cap loss/motion display plus one comparison table; include the existing norm-matched control, MLP replication and completed 54-arm quantile×LR ablation in support.
- **Interpretation:** selective amplitude allocation can help while reversals persist. Terminal test loss worsens under the cap in all three cases; larger tolerated LR did not improve learning over the same cap at its native LR.
- **Finish:** use experiment **B** for direct ranking. Describe late beta2 switching with retained state separately from training throughout with lower beta2.

## VI. Conclusion and appendices — close the same argument

- **Conclusion:** C/D formulation → curvature/noise/loss interpretation → GD/SGD characterization → Adam's functional expression → intervention findings.
- **Appendices:** derivations and approximation scope; HVP/loss checks; full event population; probe/calibration contracts; complete controls and held-out results.
- **Abstract:** update after these sections; replace outdated “future interventions” with completed findings.

## Remaining empirical work — two bounded decisions

### A. Can we support the requested pre-update warning?

- **First, existing data:** [055](../../analyses/055-2026-11-05-adam-directional-amplification/observations/002-direction-context.md) has 7,204 eligible quiet decisions, including 645 positives. **Primary target:** loss ≥10% above the current loss within eight updates.
- **Nested comparison:** fixed L2-logistic models using loss history; then adding gradient norm/growth; then adding cosine. The decisive test is whether cosine adds value beyond the same loss/norm features.
- **Evaluation:** freeze features/fitting/threshold rules; use chronological train/validation/test blocks, an eight-update exclusion gap and training-only normalization.
- **Success:** higher average precision **and** recall at ≤5 false alarms per 100 eligible negative decisions. Choose thresholds on validation; report test results per model and lead times to first increase/escalation/peak.
- **Stop rule:** if the criterion fails, report the finding and retain the descriptive diagnostic; no repeated horizon/threshold search.
- **Validated prediction requires:** the frozen pipeline on two fresh initializations per architecture, 1,024 unchanged high-LR updates each—six trajectories. Otherwise report retrospective feasibility.

### B. Complete the direct intervention comparison

- **Recommended new comparison:** native Adam, native half-LR, global raw-gradient clipping at norm 1 before moments, and distributed q50 cap; add beta2=.95 for the MLP comparison.
- **Matched seed-0 cases:** MLP **448→1152, LR .001**; ResNet **264→576, LR .003**; Transformer **552→928, LR .003**.
- **Match:** fork tensors, optimizer/RNG, objective, hardware, precision and V-B probes. Calibrate caps on eight untreated pre-fork updates; retain moments/clock and native bias correction for beta2 switching.
- **Reuse qualified matches only; otherwise rerun the paired baseline.** At most **13 post-fork arms** before reuse; budget replay/calibration separately.
- **Outputs:** full-window and final-32-state mean loss, maximum, positive loss variation, output motion, control activation fraction; complete held-out evaluations at fork, fork+32 and endpoint.
- **Stop:** complete fixed arms or record divergence; no threshold retuning. Report inactive clipping without generalizing to effective clipping. Keep unmatched historical comparisons separate.
- **Execution:** continue Analysis 065 with new run IDs; prepare cost/duration/storage/verification before paid execution.

## Closure sequence and completion test

- **1 — Write I–IV from retained evidence:** the theory, core experiments and validations already exist.
- **2 — Produce V-A/B:** extract the peak table, run check A, assemble the tracing algorithm/display.
- **3 — Finish V-C:** execute comparison B if retaining the planned direct ranking; integrate all outcomes and existing ablations.
- **4 — Rewrite abstract/VI and finalize:** verify equations, claims, citations, figures and reproducibility; compile and inspect the manuscript.
- **Submission item:** resolve the supplied TNNLS decision's revised-resubmission restriction before submission; it does not change this scientific anatomy.
- **Map completion criterion:** every contribution has a cited gap, manuscript home, existing evidence and concrete closing action; the only possible new training is specified above. **This map is sufficient to proceed to paper closure.**
- **Current boundary:** this iteration changes the consolidation map only. Manuscript editing and experimental execution are the next phase.
