# Consolidation plan: from gradient geometry to representation-informed intervention

**Status: iteration 1 proposal, 8 November 2026. No manuscript changes approved or applied.**

## 1. Recommended paper and why

The strongest current paper follows one question: **How can observations of
optimization instability be connected to changes inside a learned model, and
what does that connection tell us about interventions?**

Proposed thesis:

> Consecutive-gradient geometry describes the response along a training
> trajectory, but it does not by itself locate the associated functional motion
> or determine whether suppressing it improves learning. We connect this
> diagnostic to fixed-probe representation-motion measurements, trace their
> direction, magnitude, and temporal structure through neural networks, and test
> local displacement control against unchanged optimization and targeted controls.

This retains the author's constructive-destructive foundation and develops its
representation-level motivation. It gives the tracing algorithm a central role,
with the intervention study testing what the diagnostic makes actionable.
It does not reduce the paper to a catalogue of Adam plots.

Three plausible framings were considered:

| Framing | Fit to evidence | Recommendation |
|---|---|---|
| Universal representation-based stability or acceleration principle | Requires a general representation/learning bridge, stronger novelty verification, and evidence beyond current full-objective continuations. Higher nominal LR did not improve capped learning progress. | Keep as a research ambition, not the present paper's established claim. |
| Measurement and intervention: from gradient response to representation motion | Fits the formal diagnostic, qualified tracing workflow, cross-architecture observations, and informative positive/negative interventions. | **Use as the main narrative.** |
| A new optimizer that outperforms Adam | Current held-out tradeoffs, unmatched historical comparators, selected forks, and unmeasured isolated overhead prevent this claim. | Treat local caps as a tested intervention, with an optional stronger method evaluation. |

The literature already links instability, feature learning, and representation
change. Thus the motivating gap cannot be that optimization research ignores
learning representations. A narrower candidate gap is an operational connection
between gradient-response events, layer-resolved functional motion, and
controlled intervention in the tested settings. Whether that combination is
substantively new remains a comparison task, especially against relative
representation-change control. See [the literature map](evidence/LITERATURE.md).

## 2. Contributions and argument map

These are three contribution families, not a claim that every equation or plot
is independently novel. Established identities and prior algorithms receive credit.

| ID | Question / candidate gap | Proposed contribution | Main evidence | Closest literature and boundary |
|---|---|---|---|---|
| C1 | What does constructive/destructive gradient geometry measure, and what remains meaningful beyond GD? | An explicit measurement framework separating adjacent interaction, accumulated motion, and loss; the same scalar becomes a normalized projected gradient response along the optimizer's realized path. | Existing derivations; GD/SGD characterization; loss-approximation and all-lag checks; Adam context. | Edge of Stability, spectral/Barzilai-Borwein secants, Adam stability. The mixed-Hessian identity and displacement algebra alone are not independent novelty; no optimizer-independent stability threshold follows. |
| C2 | Where is loss-associated oscillatory motion visible within the network, at what amplitude, and with what temporal structure? | A reproducible fixed-probe tracing algorithm joining outgoing representation differences, exact temporal Gram/SVD summaries, and architectural traversal. | Analysis 063: MLP, ResNet, Transformer; observation invariance and numerical checks; selected loss-defined windows. | Feature learning at/beyond EoS, low-dimensional training dynamics, representation-comparison methods. Newness must be in the specific capability and evidence, not SVD itself. No unique causal source is established by tracing. |
| C3 | Does using representation motion to allocate an update improve the trajectory, and what is lost by optimizing reversal alone? | A local displacement-cap intervention and a controlled study separating motion amplitude, reversal, and progress, including unsuccessful directional controls and held-out tradeoffs. | Analyses 064/065; matched MLP norm controls; cross-architecture caps; three-seed MLP quantile/LR sweep. | Kosson et al.'s relative representation-change control, clipping, LR changes, beta2 intervention. Existing evidence supports bounded training-trajectory benefits, not a generally superior optimizer or faster generalization. |

The supporting chain is **C1 supplies the diagnostic → C2 localizes functional
motion → C3 tests an intervention informed by that motion**. This is a narrative
relationship, not a proof that the scalar diagnostic causes or uniquely predicts
the measured representation changes.

## 3. Proposed paper map and section elements

Stable element IDs connect this outline to [claims](evidence/CLAIMS.md),
[artifacts](evidence/ARTIFACTS.md), and [tasks](work/TASKS.md). Figure/table IDs
below are storyboard IDs; manuscript numbering will be assigned only when editing.

| Section | Elements and their function | Evidence / planned display | Change from active draft |
|---|---|---|---|
| I. Introduction | E01: observed problem; strongest existing explanations; precise unresolved question; thesis and C1-C3; scope. | Literature L01-L12; central result previews without claiming forecasting or acceleration. | Rewrite only after the argument and novelty comparison are accepted. |
| II. Background and positioning | E02: spectral stability/EoS; secant geometry; Adam preconditioner dynamics; feature-learning and representation-change predecessors. | Short role-based related work; full comparison remains in consolidation records. | Current Section II is SGD stability background; retain essential mathematics and broaden the comparison. |
| III. Gradient interaction and realized-path response | E03: sample/batch definitions; E04: exact all-lag displacement decomposition distinguished from the local lag-one diagnostic; E05: cosine/norm scalar; E06: GD curvature assumptions, finite-step/noise residuals; E07: arbitrary-update projection and availability timeline. | Compact derivations plus an information-timing table (T1). Full proof and counterexamples in appendix. | Preserve the conceptual SGD derivation; remove interpretations that turn low displacement ratio into retained learned information or negative adjacent alignment into cumulative destruction. |
| IV. Controlled GD and SGD characterization | E08: matched GD rates; E09: loss/curvature approximation audit; E10: frozen minibatch versus moving SGD distinctions. | F1 GD geometry/progress; F2 batch effect; setup table T2; concise approximation/all-lag caveat. Analyses 024, 027, 033-037, 056 as applicable to selected assets. | Compress current experiments around explicit tests of Section III. Avoid repeating every historical exploration. |
| V. Adam events and representation-motion tracing | E11: one high-LR trajectory per architecture; E12: selected event characterization with correct timing; E13: tracing definitions and Algorithm A1; E14: architectural heatmaps; E15: window-specific spectra and signed coefficients. | F3 high-LR overview; F4 tracing composite; T3 compact event/coverage/spectrum summary. Analysis 063 algorithm and Analysis 057 descent-direction context. | Deep restructuring of current Adam section. Make tracing the methodological center, not an appendix after optimizer plots. |
| VI. Representation-informed interventions | E16: local affine displacement score and Algorithm A2; E17: unchanged Adam and comparator placement; E18: effect on loss, displacement and reversal; E19: norm/directional controls; E20: replication, quantile/LR and held-out limits. | F5 distributed cap + motion; F6 comparator/control outcomes with held-out results; T4 concise method/outcome table. Analyses 058/059/064/065. | New central section. A unified ranking awaits matched comparisons; historical studies remain clearly separated if no reruns are selected. |
| VII. Discussion and conclusion | E21: what transfers as a definition versus a tested finding; E22: informative failures; E23: cost, stochastic/probe limits, held-out tradeoffs; E24: bounded conclusion. | Explicit scope paragraph; no new evidence introduced here. | Replace optimizer-general promises with the supported contribution selected by the author. |
| Appendices / supplement | Exact derivations; all-lag/finite-step residual details; probe and calibration contracts; per-seed results; negative controls; complete event census; extra model-size/schedule evidence; reproducibility instructions. | Existing artifacts linked in the artifact map. | Preserve useful evidence without forcing every experiment into the main argument. |

This is a compression target: six main figures, two short algorithms, and up to
four compact tables, combining T3/T4 or moving detailed coverage to the supplement
if needed. It is not an instruction to paste six full report pages into the paper.
Final page allocation follows a verified TNNLS article-type/format check. The
official author page could not be retrieved in this pass, so no hard page limit
is asserted here.

## 4. Figure and artifact decisions

- **F1/F2:** Keep the best existing GD/SGD assets and only the comparisons that
  test the retained theory. Bring the already completed all-lag and one-step
  loss audits into the explanation. A low displacement ratio alone cannot
  establish worse learning.
- **F3:** Use Analysis 058 Figure 00 as the starting point, as requested: one
  high-rate case per architecture, with loss, projected response, cosine, and
  gradient norm. Gray event intervals remain retrospective. The dashed value
  two is not an Adam stability theorem.
- **F4:** Recompose Analysis 063's measured event windows into a compact shared
  layout inspired by Analysis 065 Figure 07 page 2. Show loss, layer motion
  cosine and RMS, spectral concentration, and selected signed coefficients.
  Reuse the fresh MLP 450-500 SVD consistently; do not mix its spectra with
  correlations from 448-800. Use absolute RMS anchors beside normalized heatmaps.
- **F5:** Use Analysis 065's distributed-cap figure as the intervention starting
  point. Keep baseline/cap motion maps on the same baseline-derived magnitude
  scale, so normalization cannot hide a treatment amplitude reduction.
- **F6/T4:** Show mean/tail training loss, excursions, and held-out outcomes with
  uncertainty appropriate to the available seeds. Include a compact failed
  directional-control or norm-matched comparison: it directly tests the idea
  that suppressing reversal is the objective. Keep unmatched historical LR,
  clipping, and beta2 results in separate panels/rows until standardized.

Algorithm A1 is the fixed-probe tracing procedure. Algorithm A2 directly caps
local affine responses; it does **not** use the temporal SVD basis. Each needs
explicit inputs, timing, scope, zero cases, computation/storage cost, and outputs.
See [the artifact map](evidence/ARTIFACTS.md) for the exact source paths.

## 5. What the evidence changes in the raw ideas

| Original intended direction | Treatment in this plan |
|---|---|
| High curvature means rewriting previously learned structure | Preserve as a motivating hypothesis. Neither gradient cancellation nor representation change measures loss of learned information without an additional operational test. |
| The diagnostic principle applies to any optimizer | Yes as a realized-path projected gradient response under the stated fixed-objective regularity assumptions. GD curvature, SGD noise, actual update cancellation, and Adam stability remain different claims. |
| We can detect the peak before applying its update | Build the requested event table, but label it characterization. Current-gradient diagnostics describe the incoming step; future peak forecasting needs prospective tests on alarms and non-events. |
| A single low-dimensional mode reverses across architectures | The observed motion has concentrated fixed-window structure, with different spectra. Output top-one energy is 53.02% / 24.03% / 41.65% in the selected MLP / ResNet / Transformer windows. |
| LR reduction hurts learning by preventing basin transitions | Existing mean-loss effects are mixed, and basin transition/fractality was not measured. State the observed tradeoffs; do not assert that mechanism. |
| Local caps allow faster learning at larger LR | The latest sweep supports tolerance to larger nominal LR. Every tested quantile has worse across-seed mean training loss at 2x/4x than at its own 1x. |
| A generally better intervention | Distributed caps improve selected mean training trajectories, while measured terminal test losses worsen in all cross-architecture setups. Make this part of the result, not a hidden footnote. |

These are corrections to claim strength, not changes to the author's scientific
question. The raw ideas remain intact under `human/`.

## 6. What remains before manuscript editing and submission

**First, finish consolidation without new training.** Resolve the closest-work
comparison (P01), timing/algebra and wording (P02), source/comparator inventory
(P03), and event-table specification (P04). Agree the figure storyboard (P05)
and contribution scope. Much of the needed evidence already exists; freezing,
selective LR, directional controls, and the MLP cap/LR sweep are completed work.

**Recommended scientific addition if C3 is a substantial method claim:** a
standardized same-fork comparator study (E01), with independent evaluation seeds,
replication beyond the MLP, a second declared fork, repeated held-out measurement,
and isolated intervention overhead (E02). Include Transformer clipping, which
is missing from the existing matched full-batch clipping study. Compare with
the closest representation-change control, or narrow the cap's novelty claim
if an exact comparable implementation is not feasible. Resolve the beta2
transition contract before designing that arm; the existing switch changes
native bias correction as well as memory behavior.

Probe robustness (E03) is needed for stronger localization/transfer claims.
Ordinary minibatch evaluation, spike-specific spectral controls, and prospective
prediction (E04-E06) are conditional on retaining those broader claims. They
should not expand the paper automatically. The minimum bounded paper can omit
forecasting and universal optimizer claims; doing so does not excuse an
unresolved closest-predecessor comparison or missing evidence for its actual claims.

No cloud launch or spending budget is implied by these proposals. Before any
selected run, prepare its exact comparison, revisions, seeds, retained diagnostics,
cost/ETC, storage and cleanup plan. Assign fresh run IDs only after checking the
then-current registry; keep ongoing intervention work in Analysis 065.

## 7. Decisions for the next consolidation iteration

1. **Main contribution emphasis:** adopt the recommended measurement/tracing/
   intervention narrative, or make a stronger general learning principle the
   main goal with its additional formal and empirical requirements?
2. **Intervention scope:** present a bounded controlled case study, or strengthen
   C3 through standardized comparisons and replication before manuscript edits?
3. **Prediction:** keep prospective spike prediction outside this submission,
   or make it a distinct contribution requiring E06?

Recommended answers are: the proposed narrative; strengthen C3 with a focused
matched evaluation; keep forecasting outside the core. These are proposals for
author review. The next step is to revise this plan from that feedback, not to
start editing the manuscript.
