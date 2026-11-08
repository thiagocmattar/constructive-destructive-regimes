# Argument, contribution and manuscript-element map

**Iteration 2 — 8 November 2026. Author structure retained; proposed positioning.**

This is the affirmative counterpart to [CLAIMS.md](CLAIMS.md). Its purpose is to
make the value of the author's work explicit: what question it answers, what the
closest literature already establishes, and what formulation or evidence makes
the answer useful. It follows the supplied [review skill](../human/review-skill/SKILL%20%282%29.md).
It does not choose a different paper or apply prose to the manuscript.

## 1. Positioning thesis

**Constructive/destructive regimes provide a connected account of how learning
updates interact: curvature, amplification, sampling and temporal accumulation
explain their consequences, while representation tracing makes their functional
expression and possible intervention targets observable.**

The intended contribution is the explanatory framework together with its
analytical connections, empirical characterization and investigative capability.
Simple mathematical ingredients can support that contribution when they answer
consequential questions together. The paper should demonstrate those answers,
give established ingredients their proper attribution, and state exactly what
the experiments establish.

### Proposed positioning paragraph

Training changes a model through a sequence of updates that can reinforce or
partially reverse preceding changes. Curvature-based stability analyses explain
important oscillatory behavior, including progress beyond the nominal stability
threshold, while gradient-product diagnostics and secant methods already use
information from successive gradients. We study a complementary question: how
do direction, magnitude, sampling and temporal accumulation jointly distinguish
constructive and destructive training regimes, and how do those regimes appear
in the model's changing representations? Starting from SGD updates, we connect
their interaction to directional curvature and finite-step loss change, obtaining
a gradient-only diagnostic with an explicit sampling interpretation. Controlled
GD and SGD experiments characterize these relationships across the retained
model/data cases. The same observable then measures gradient response along
Adam trajectories, where layerwise tracing reveals structured functional motion
and motivates interventions on its amplitude and location.

This paragraph is a proposal for strengthening the existing introduction, not
a replacement introduction. Its literature premises are supported by
[EoS](https://arxiv.org/pdf/2103.00065),
[Chee and Toulis](https://proceedings.mlr.press/v84/chee18a/chee18a.pdf),
[Pesme et al.](https://proceedings.mlr.press/v119/pesme20a/pesme20a.pdf), and
[Barzilai–Borwein](https://doi.org/10.1093/imanum/8.1.141).
The connection to layerwise intervention is this project's synthesis, evaluated
below against the closer representation and Adam literature.

## 2. Gap → contribution → evidence

The gaps below are specific questions addressed by this manuscript. A comparison
with selected papers is not proof that no other publication addresses the same
question. Priority claims require the remaining P01 checks in [TASKS.md](../work/TASKS.md).
Reference IDs resolve in [LITERATURE.md](LITERATURE.md).

| Component | Closest knowledge and specific question | Contribution to articulate | Direct support and significance | Scope / remaining comparison |
|---|---|---|---|---|
| **C1. Constructive/destructive formulation** | EoS/Central Flows (L01/L12) explain productive oscillations; gradient-product diagnostics (L13/L14) study transitions using successive gradients. How should local reinforcement/reversal be read together with accumulated parameter motion and loss progress? | An explicit temporal-geometric characterization from SGD updates, separating gradient magnitude, local interaction and full-path accumulation. | Exact displacement decomposition in III;024/035 and supporting005/019/056 show local reversal, structured lags and continued motion/progress. It supplies a common interpretation for apparently conflicting training signals. | The norm expansion and gradient inner product are established mathematics. Distinct value lies in the connected characterization and demonstrated explanatory use, not invention of an inner product or proof of lost knowledge. |
| **C2. Curvature connection and proxy** | BB secant geometry (L11), gradient-product adaptation (L13–L15), and EoS sharpness (L01) provide related quantities. What does the observed angle-plus-norm change measure, and when can it diagnose a destructive regime without Hessian evaluation? | Connect the regime boundary to directional curvature; give the finite-step/path-average meaning and an explicit sampling term. | Boxed III-A relation, III-B estimator, Appendix A;017/018 controls and027 independent HVP/noise checks. Existing gradients make the quantity accessible for tracking the formulated regime. | Credit reciprocal BB1 equivalence under plain GD. Path-average directional curvature differs from a largest eigenvalue and from a pointwise estimate. Claims of measured speed advantage need cost evidence. |
| **C3. Reversal, amplification and loss** | EoS/catapults (L01/L04) already allow nonmonotonic progress. A reversal sign alone does not resolve whether a particular update decreases loss or creates a large excursion. | Relate reversal and norm amplification to the local GD loss change, then examine their functional expression under Adam. | III-E/Appendix A;024 approximation audit;036/055/057 Adam event characterization;062/065 functional motion and exact squared-loss accounting. This explains why suppressing every reversal is the wrong evaluation target. | GD quadratic relation is exact in its stated model, approximate for nonlinear finite steps. Adam's function-space MSE identity is separate evidence, not a transfer of the GD theorem. |
| **C4. Sampling and batch effects** | Stochastic gradient-product theory already includes noise (L13/L14). How much of observed alignment is sampling, and how does changing batch size alter the subsequent training path? | An explicit noise contribution connected to frozen-state validation and moving-SGD characterization. |020/027 isolate fixed-state sampling;027 tests derivative/noise terms;037 extends moving SGD to three cases and three initializations. This links the formula to a practical source of ambiguity in regime measurement. | Fixed-state cosine approximation and evolving negative-alignment attenuation answer different questions. A batch-mean gradient cannot separately identify per-sample energy and intra-batch interaction. |
| **C5. Controlled characterization** | The literature establishes oscillation and progress; the formulated components must be measured together in the actual model/data cases. | Characterize LR-dependent emergence, norm growth, displacement retention, lag structure and batch responses in the original IV sequence. |024 nine full-GD trajectories;035 exact backward profiles;037 three-case batch extension. Positive all-pair interaction can coexist with negative adjacent interaction. | Three architecture/task cases establish observed breadth, not a universal scaling law. Full-GD LR comparison has one initialization per case; sampling extension has three. |
| **C6. Interpretation beyond SGD** | Adaptive EoS and Bai et al. (L02/L03) analyze transformed curvature/preconditioner dynamics. What can the same raw-gradient observable tell us along Adam's chosen path? | Preserve its computation while identifying the traversal direction and raw-gradient projection separately. | Existing Appendix A arbitrary-update identity;028/029/034/036;025 norm-matched preconditioning intervention and055/057 event evidence. It makes the diagnostic portable with an explicit algorithm-specific meaning. | Same computation does not imply same stability threshold or raw-gradient/update alignment. Baydin's general hypergradient (L15) uses the derivative of the actual update rule and must not be conflated with a raw–raw gradient statistic. |
| **C7. Representation tracing** | Catapults (L04) connect spikes, output/NTK dynamics and features; SVCCA/CKA (L08/L09) compare representations. Where and in what temporal subspace does measured functional motion appear throughout these networks? | A reproducible algorithm for consecutive activation increments, direction/amplitude, exact temporal Gram/SVD analysis and graph-aware layer comparison; associated findings and interpretation. |061/062 development,063 formal method and cross-architecture evidence. This turns the representation-learning motivation into measurements that can guide targeted tests. | Specific combination/capability must be compared with nearest methods before priority language. Temporal SVD is retrospective; descriptive localization does not by itself identify the initiating cause. |
| **C8. Informed intervention** | LR/clipping/beta changes act on different quantities; representation-change control already exists (L05). Does controlling the local functional effect add value beyond reducing a scalar step norm? | A local affine-response cap and matched tests of amplitude allocation, with schedules/clipping/beta evidence explaining the design. |064 freezing;065 selective LR, displacement controls, norm-matched control, three-model cap results, MLP replication and quantile×LR sweep. It tests an actionable implication of the tracing interpretation. | Exact formula comparison with L05 remains necessary for a method-novelty claim. Training-window improvements coexist with held-out tradeoffs; common-fork comparisons are needed for a direct method ranking. |

## 3. How the original four contribution bullets can become more explicit

Keep the four bullets in the [active introduction](../../draft/sections/01-introduction.tex)
as the organizing form. The following are content instructions, with candidate
wording, rather than an approved replacement:

1. **Characterization (C1/C5).** “We characterize constructive and destructive
   training regimes through the magnitude and temporal interaction of SGD
   updates, and distinguish local reversal from cumulative displacement and
   objective progress.” Position against EoS/Central Flows and gradient-product
   diagnostics, then point to the exact decomposition and IV results.
2. **Curvature and sampling (C2/C4).** “We connect consecutive-gradient alignment
   and norm growth to directional curvature, obtaining a gradient-only
   diagnostic with an explicit sampling contribution, and evaluate the
   connection with derivative and batch-size controls.” Credit the secant
   antecedent beside the estimator; explain the regime interpretation and
   empirical validation as the contribution being made here.
3. **Loss behavior (C3).** “We show how reversal and amplification combine in
   the local GD loss relation, separating oscillatory descent from upward loss
   excursions, and examine the corresponding finite-step behavior empirically.”
   Keep the angle examples and quantitative approximation scope. The functional
   Adam findings later deepen this interpretation.
4. **Empirical extension and use (C5–C8).** “We characterize these relationships
   across GD, minibatch SGD and Adam, trace functional oscillations through
   network representations, and evaluate representation-informed interventions
   against unchanged baselines and targeted controls.” The accompanying text
   names the tracing algorithm, findings and cap rather than leaving completed
   experiments as future work. This bullet can be split editorially if needed,
   without changing the paper's hierarchy.

The gap, reference and evidence should appear immediately around the relevant
contribution. A reader should not have to reconstruct the novelty claim from a
separate limitations section. Conversely, citations do not need to be repeated
inside every bullet if the preceding positioning paragraph makes the relation
unambiguous.

## 4. Positive results that make the argument concrete

### C2/C4: the curvature/noise formulation has direct tests

[Analysis 027](../../../analyses/027-2026-10-30-minibatch-noise/REPORT.md)
contains 1,188 paired comparisons against independently computed HVP quantities.
At batch 32, the raw proxy's median absolute discrepancy is .7467; explicitly
accounting for the measured noise term reduces it to .0030. The corresponding
corrected medians are .0033/.0034/.0025 for batches 128/512/2048. Exact frozen
variance/noise-projection checks agree within .50%/.74%.

This is useful affirmative evidence for the decomposition. The correction uses
an offline full-gradient reference, so it is not an already deployed online
estimator. The cosine approximation has small absolute errors but detectable
bias in 21/36 cells. A concise supporting presentation can retain this distinction
without restoring the dense HVP presentation previously removed by the author.

### C1/C5: local destruction and net progress coexist in a measurable way

The [024 audit](../../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/007-section-iv-critical-review.md)
finds positive final all-pair interaction in all nine GD trajectories, including
seven with negative cumulative adjacent interaction. Within every case, absolute
displacement increases with LR while the retained fraction of path length falls.
This makes the local-versus-cumulative distinction explanatory: the formulation
accounts for different aspects of the same trajectory. Analysis056 supplies
additional evidence, but its rolled-back paired notation need not return.

### C3/C7: the functional account connects back to the original loss question

[Analysis061](../../../analyses/061-2026-11-07-mlp-adam-spike-internals/observations/001-spike-internals.md)
and [Analysis062](../../../analyses/062-2026-11-07-mlp-gradient-amplification/README.md)
trace output-sensitive hidden motion and distinguish gradient coherence from
per-example magnitude. [065 observation009](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/009-what-the-negative-controls-show.md)
then evaluates the exact squared-loss change in output space: across the three
MLP seeds, all 377/371/379 meaningful baseline loss increases have a negative
residual–output-change term outweighed by the positive squared-motion term.
This provides a direct functional account of overshoot in those trajectories.
It is distinct from merely observing that a raw-gradient angle is negative.
This accounting uses outgoing updates 448–1152, including the extra terminal
transition 1152→1153; it is not the same denominator as the ordinary loss-mean tables.

### C7/C8: tracing changes what can be tested

[063](../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/004-consolidated-instability-tracing.md)
localizes structured motion across architectures. The top temporal mode is
prominent but not universally sufficient; the [short event-window fits](../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/005-technical-report-event-window.md)
need output ranks 5/16/15 for 95% energy in MLP/ResNet/Transformer, respectively.
The MLP fit here uses 450–500, distinct from the earlier 448–800 fit. The first projection is
important in the MLP, while the Transformer's embedding is comparatively steady.
This supports an architectural investigation rather than one universal layer story.

[065](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md)
reports lower selected-window mean training loss with distributed caps across
the three setups and two additional MLP seeds. The [MLP layer-norm-matched
control](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/009-what-the-negative-controls-show.md)'s
−3.43% versus the cap's −22.85% supports the value of allocation in
that comparison. Persisting reversals and held-out tradeoffs make the measured
outcome more informative: successful control concerns useful functional motion,
not a requirement that all diagnostic signals become positive.

## 5. Element-by-element placement in the existing paper

These **M** identifiers are manuscript-planning elements; they are separate from
the P/E/W task IDs. They exhaust the elements named in [PLAN.md](../PLAN.md).

| Element / location | Object and purpose | Formulation, analysis or result to retain/add | Presentation / task |
|---|---|---|---|
| M01 / I | Motivation and contributions | Existing learning-update argument; cited predecessor/question/contribution chain above. | Preserve author-edited introduction; P01, W01. |
| M02 / II | Literature foundation | Existing four subsections; secant/gradient-product predecessors and closest Adam/feature-motion work in their relevant places. | Selective additions, no new survey structure; P01. |
| M03 / III opening | Exact displacement and regimes | Existing all-lag squared-displacement identity; magnitude/alignment; local sign definition. | Keep equations and terminology; P02. |
| M04 / III-A/B | Curvature and Hessian-free estimation | Boxed connection, estimator, path-average identity, assumptions and inexpensive computation. | Keep both subsections; compact027 validation in support; P02/P03. |
| M05 / III-C | Sampling contribution | Exact unnormalized noise relation, finite-population variance, fixed-state cosine approximation. | Tie to 020/027/037; P02/P03. |
| M06 / III-D/E | Accumulation and loss | R as path retained in displacement; quadratic/local loss criterion, norm-ratio and angle examples. | Preserve both subsections;024 audit and all-lag interpretation; P02. |
| M07 / V opening | Optimizer-general computation | Existing mixed-projection identity: Adam chooses traversal, preceding raw gradient defines projection. | Intuitive explanation; refer to Appendix A without new notation; P02. |
| M08 / IV | GD loss, proxy, spectral context |024 measured trajectories and coverage distinctions. | Existing Figure1 retained; P03. |
| M09 / IV | Norm/R and temporal lags |024 regime/norm/retention followed by035 backward-lag oscillation. | Existing Figures2/3 retained, same order; P03. |
| M10 / IV minibatch subsection | Batch effects |037 complete three-case extension;020/027 support separates frozen sampling and moving paths. | Existing Figure4 retained; P03. |
| M11 / V-A | High-rate characterization |058 Figure 00 with one high-rate trajectory/model;036 loss-defined events. | F3 plus compact event definition; P04. |
| M12 / V-A | Peak timing and detection objective | State/update/loss table with eta-kappa-hat, rho, norm growth, first rise, peak, recovery. | T1/T3; P04 then E06 only for the stronger forecasting claim. |
| M13 / V-B | Tracing algorithm |063 fixed probe/context, consecutive increments, exact Gram/SVD, cost/coverage. | Algorithm A1; P05. |
| M14 / V-B | Layerwise motion | Output-to-layer direction/amplitude heatmaps with architecture-appropriate stage grouping. | F4 based on065 Figure 07 page 2; P05. |
| M15 / V-B | Temporal subspace findings | Spectrum, cumulative energy and signed coefficients with matched window/basis. | F4 companion panels;063 event report; P05. |
| M16 / V-B | Functional interpretation |061/062 internal investigation, correlation, output-sensitive motion;065 squared-loss decomposition. | Concise synthesis linked to III-E; P03/P05. |
| M17 / V-C | LR intervention |058 persistent-halving trajectories and subsequent progress. | F6/T4; compare against unchanged baseline; P03. |
| M18 / V-C | Clipping and beta2 |058 clipping,043 earlier Transformer control,059 recovered beta2 result; distinguish switching from training with lower beta2. | F6/T4; missing matched comparison identified by P03/E01. |
| M19 / V-C | Local-displacement cap |065 affine rule, calibration, optimizer-state treatment, direct-response guarantee and representative effect. | A2/F5; literature comparison P01, geometry P02. |
| M20 / V-C | Intervention evidence |064/065 freezing, selective LR, norm/directional controls, cross-case results, MLP replication and quantile×LR. | Compact comparison plus supporting ablations; P03/E01. |
| M21 / VI | Main contribution | C/D formulation, curvature/proxy, loss and sampling connections. | Lead conclusion with original scientific contribution; W01. |
| M22 / VI | Empirical lessons | Regime sign, accumulation and objective progress are related but distinct measured aspects. | Synthesize IV rather than enumerate figures; W01. |
| M23 / VI | Extension and practical implication | Adam response, tracing capability, functional findings and intervention tests. | Replace outdated future-work status; W01. |
| M24 / VI | Actual open questions | Broader prediction/replication/online deployment and held-out outcomes tied to precise missing tests. | Local scope statement and targeted future work; E01–E06 as applicable. |

Appendices retain the exact path identities, approximation error, derivations
and measurement conventions. Add concise validation and tracing/intervention
contracts there as needed. Complete event populations and ablations remain
available even when main Adam displays are consolidated.

## 6. Applying the skill in the way requested

The argument is evaluated on its positive explanatory content. Each component
should give the reader a useful distinction, a measurable relationship or a
new investigative capability. Qualifications belong beside the specific claim;
they are not a substitute for explaining why that claim matters.

The current synthesis is strongest when it answers a sequence of concrete
questions: what reverses; how curvature and sampling affect that observation;
when amplification changes its loss consequence; how local interactions
accumulate; where functional oscillation appears; and which controls help in
the observed setting. These are the author's questions and existing structure.
The remaining literature comparison should sharpen this sequence and its
attribution, rather than replace it with a different central thesis.
