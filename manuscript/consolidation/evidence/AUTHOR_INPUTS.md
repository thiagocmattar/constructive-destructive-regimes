# Author-input incorporation record

**Iteration 2 — 8 November 2026.** The author's scientific direction and current
six-section manuscript govern the consolidation. This record distinguishes
faithful incorporation from the local scientific qualifications needed to make
the intended argument precise.

## 1. Complete supplied-input coverage

All six original files are preserved in `human/`; sizes and hashes are in
[provenance.json](../provenance.json). The review-skill archive's main skill,
three reference guides, worksheet, literature-matrix template and agent metadata
were inspected. The main skill was also supplied as a standalone Markdown file.
Nothing was installed globally or changed inside the supplied originals.

| Input | Author intent | Incorporation in this iteration |
|---|---|---|
| [task.md](../human/task.md) | Main structural document, gap/contribution/reference table, section elements, artifact placement and actionable remaining tasks; separate supplied inputs from generated plans. | [PLAN](../PLAN.md), [ARGUMENT_MAP](ARGUMENT_MAP.md), [ARTIFACTS](ARTIFACTS.md) and [TASKS](../work/TASKS.md). M01–M24 identify manuscript elements; P/E/W identify tasks. |
| [central-argument-raw-idea.md](../human/central-argument-raw-idea.md) | Learning/representation motivation; constructive/destructive formulation; curvature, cheap proxy, loss criteria, GD and batch characterization; ideas useful across learning problems. | C1–C5 remain the foundation in I–IV. C6–C8 extend that argument in V. Each component has a closest-work/question/contribution/evidence row. |
| [adam-section-restructure.md](../human/adam-section-restructure.md) | Same observable beyond SGD; one high-LR case/model; peak table; tracing; LR, clipping, beta and local-cap comparisons. | The three requested blocks are consolidated inside V, with the existing mixed-projection identity as opening. Lost beta2 result recovered in 059; Transformer clipping status corrected using 043. |
| [learning-instability-tracing-idea.md](../human/learning-instability-tracing-idea.md) | Preserve the investigative sequence from loss/gradient to outputs, correlations, hidden representations and SVD; algorithm, findings, interpretation and intervention motivation are contributions. | C7 and M13–M16 retain all four contributions.061/062 preserve development;063 supplies the unified method;065 Figure 07 page 2 guides proposed visual aggregation. |
| [review skill](../human/review-skill/SKILL%20%282%29.md) and [archive](../human/review-skill/research-framing-codex-skill.zip) | Strong literature-grounded positioning: simple formulations can support a compelling contribution when the gap, argument and evidence are explicit. | Affirmative thesis and contribution wording, exact predecessor comparisons, positive evidence recovered from older analyses, and a separate support audit. The skill's optional framing-choice exercise does not override the author's chosen structure. |

The author's subsequent messages are controlling instructions: read all human
inputs, navigate the repository/workflow, conduct a thorough consolidation,
stay close to the current paper, and converge within the supplied structure.
The next review is about completeness and precision of that consolidation,
not choosing a new central emphasis.

## 2. Central argument: what is preserved and what is made precise

| Author idea | Preserved scientific role | Precise treatment |
|---|---|---|
| Optimization geometry should connect to the learning process that produces it. | Opening motivation; representations are part of the intended explanation. | Credit existing feature-learning and catapult accounts, then state the specific measurement/intervention question. Do not claim the field never studies representations. |
| C/D behavior follows directly from SGD's learning updates. | Central formulation and terminology in III. | Exact displacement retains all temporal interactions; adjacent sign defines the local regime. Parameter cancellation motivates investigating functional rewriting but does not measure semantic forgetting. |
| High curvature connects to reversal of preceding changes. | Boxed III-A relationship and its interpretation. | State directional/path scope, sampling terms and norm ratio. Connect representation language to the later measured activation changes. |
| Cheap curvature/instability observable is a contribution. | III-B remains a central subsection. | Attribute secant antecedents; emphasize regime interpretation, derivative/noise tests and accessibility. Avoid replacing it with new notation or reducing it to a caveat. |
| Loss shape depends on model/problem and interventions can alter the trajectory. | Motivation for joint geometry/function analysis and comparative controls. | Landscape visualizations support architecture sensitivity. Fractality, a particular basin escape and premature convergence are not established for these runs. They remain possible motivations to investigate, not reported mechanisms. |
| Contributions should offer principles useful beyond one setup. | Each C1–C8 card states transferable question/capability. | Separate general mathematical identities under stated assumptions, reusable measurement procedures, and cross-case empirical observations. Three cases do not prove universality. |

## 3. Adam restructuring: every requested block

| Request | Material found / planned placement | Remaining concrete work |
|---|---|---|
| Keep the same raw-gradient diagnostic for modern optimizers. | Existing Appendix A already gives the arbitrary-update mixed-Hessian identity; V opening/M07. | Explain traversal versus projection intuitively. Qualify smoothness and sampled gradients; retain original notation. |
| One high-rate characterization per model using058 Figure 00. | V-A/M11; the named PDF exists and was inspected. | Retain complete nine-arm/population context in support when replacing the main overview. |
| Table of eta-kappa-hat, rho, norm growth for peaks, including before-update values. | V-A/M12;036/055/057 supply histories and timing contracts. | P04 constructs the selected-event table; distinguish incoming-step response from future target. E06 describes the time-correct forecasting test if making the stronger detection claim. The objective is retained, not silently dropped. |
| Follow characterization into instability tracing. | V-B/M13–M16. | Align the selected windows; the older largest-event examples and newer tracing windows are not automatically the same peaks. |
| LR schedules may suppress excursions but also change subsequent learning. |058 persistent-half comparisons, M17. | Show case-dependent training outcomes. Mean loss improves for MLP/Transformer and worsens for ResNet in the retained comparison; no blanket claim that halving LR hurts. |
| Clipping may induce oscillation; check Transformer. |058 clipping, and043 fixed-fork Transformer clipping already exists; M18. | Identify the missing selected tracing/cap-window comparison, rather than rerun as though no Transformer clipping result existed. |
| Find the beta-reduction experiment and compare with the cited Adam paper. |059 late beta2-switch study, M18. | Show native bias correction/moment treatment. Distinguish a late switch from training throughout with lower beta2; do not call it a direct refutation. |
| Present local-displacement cap and compare approaches. |065 rule, controls, cross-architecture effects, MLP replication and quantile×LR; M19/M20. | Explain what is controlled, connect to tracing, compare exact nearest method and specify only missing matched forks. Include training and held-out outcomes. |
| Standardize or rerun if necessary. | P03 first builds a comparability ledger; E01 identifies the smallest missing comparisons. | Proposed scientific inputs require explicit run records and the normal execution envelope before billable work. This iteration launches nothing. |

## 4. Tracing: retain the intellectual development

The tracing argument follows the author's reasoning rather than starting from a
controller: observe loss/gradient changes, measure output increments on fixed
inputs, compare directional signals, move through hidden layers, analyze temporal
subspaces, and test an intervention motivated by that localization.

| Requested contribution | Treatment |
|---|---|
| Algorithm | A1 makes probe/context, increment timing, stage coverage, Gram/SVD computation and outputs explicit, using063/METHOD.md and the repository tracing workflow. |
| Findings | Report architecture-specific amplitude/reversal patterns and temporal energy concentration. The original gradient/output correlations are .776/.705/.879; the shorter MLP event fit is .6420. Avoid turning a particular window's >.7 observation into a universal threshold. |
| Interpretation | Structured low-dimensional motion is supported. One mode is prominent but does not explain all motion across architectures; the measured output rank95 is 5/16/15. Backward inspection localizes motion, not a proved backward causal propagation. |
| Visuals | F4 uses the readable layer grouping/heatmap idea in065 Figure 07 page 2, combined with spectrum/coefficient panels drawn from the same window. Trace the measured graph and distinguish residual branches/normalization context. |
| Motivation for interventions | Explain how layer-specific motion leads to freezing/selective-LR/cap tests, and why norm-matched and directional controls discriminate interpretations. Cap decisions use local affine scores, not the retrospective SVD basis. |
| Connection to mechanistic interpretability | Retain as a useful adjacency to investigate. Position the concrete measurement capability against catapults, representation similarity and representation-change control before naming a new field. |

## 5. Earlier author annotations remain part of the specification

Read alongside the new human notes:

- [First annotated argument review](../../draft/revisions/2026-11-03-original-argument-feedback/feedback.md): C18/C20 require C/D and destructive-interference emphasis; C23/C26 preserve the boxed curvature relation and the estimator subsection; C29 explains R as path translated into net displacement; C35/C36 preserve loss explanations through angle and norm growth; C40 fixes the IV figure sequence.
- [Second annotated review](../../draft/revisions/2026-11-03-original-argument-feedback-feedback/feedback.md): C06/C07/C09 simplify repeated equations and proxy exposition; C14/C15 require intuitive Adam raw-slope semantics; C20/C26/C29 request affirmative spike descriptions and percentages.
- [Current manuscript record](../../README.md): the November6 restoration and author-edited introduction govern the active draft. November5 paired-displacement/Adam-memory alternative prose was rolled back. Its evidence can be used without reinstalling that prose or its notation.

## 6. What changed from consolidation iteration1

1. Restored C/D, curvature/proxy, loss, noise and original controlled experiments
   as the central argument. Tracing and caps extend it within V.
2. Replaced the proposed seven-section/three-contribution hierarchy with the
   existing six sections and an explicit map of eight components into the
   current four contribution bullets.
3. Restored all four current IV figures to the main plan and recovered positive
   validation in027 and functional loss evidence in061/062/065.
4. Replaced broad experiment requirements with a ledger of completed work and
   precise missing comparisons. Kept the author's pre-update detection objective
   as a question with a concrete validation path.
5. Added affirmative literature positioning and full input traceability. The
   mathematical audit remains supporting work rather than the organizing story.

No manuscript edits are applied. Consolidation continues until the author is
satisfied with the plan of modifications and requests implementation.
