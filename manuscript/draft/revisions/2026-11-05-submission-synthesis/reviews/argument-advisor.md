# Independent argument advisor and adversarial review

Date: 2026-11-05. Scope: the active manuscript sources and 18-page PDF, the original submitted abstract and two referee reports, the editor decision transcribed in `manuscript/2026-10-03 review/feedback.md`, research definitions and methods, the two November 5 author-input revisions, the November 3 convergence record, and selected existing experimental reports. This is an editorial/scientific assessment, not an independent replication of every numerical result. No training or manuscript edits were performed.

## 1. Recommended central argument

**Training updates interact across time, but local cancellation, cumulative parameter motion, and objective improvement answer different questions. A diagnostic account must measure all three and identify the vector stream being measured.**

The constructive/destructive framework should organize those distinctions. The most convincing evidence is not that negative cosine is bad. It is that strong local reversals can leave a small but coherently accumulated residual, achieve the best loss within a controlled learning-rate sweep, and remain present when an adaptive optimizer moves downhill to first order. Conversely, making alignment more positive can worsen equal-budget loss. Together, these findings explain what the proposed measurements tell an investigator and why a single sign is insufficient for designing an intervention.

This retains the original scientific object: temporal interactions of training updates. It changes the interpretation of destructive interaction from an assumed optimization defect to an observable geometric phenomenon whose consequences must be assessed. That is a response to the actual rejection and to the new evidence, not a replacement project.

Recommended title: **Constructive and Destructive Training Regimes: Diagnosing Gradient Interaction and Parameter Motion**. A shorter alternative is **Constructive and Destructive Training Regimes: A Multiscale Diagnostic Account**. The former says more concretely what readers receive. Remove “Interpreting SGD Stability”: the paper does not establish a general stability criterion, and that title recreates the central expectation that the original review found unmet.

### Motivation in the reader's terms

Loss and gradient norms tell us whether the objective is improving and how steep its local slope is. They do not say how successive parameter changes combine. A training trajectory can repeatedly undo part of its preceding update while continuing to improve. The practical question is therefore: **where does motion cancel, what residual motion accumulates, and how does the resulting trajectory change the objective?** Answering it makes future intervention studies interpretable: suppressing a visible oscillation is useful only if the resulting objective response justifies the change.

The opening should reach this problem within one paragraph. Scaling laws can remain brief context, but a long scaling-law opening makes readers expect a large-model efficiency contribution that the current main experiments do not establish.

### Literature gap, carefully bounded

Do not claim that previous optimization work is only local or cannot study temporal interactions. EoS, central-flow theory, trajectory-directionality studies, secant methods, hypergradient methods, and recent Adam-spike work all occupy parts of that space.

The defensible gap is a **measurement and interpretation gap**: how to read available gradient and update histories jointly across adjacent, cumulative, and objective scales, and which tempting inferences fail under sampling and adaptive transformations. The paper supplies an explicit accounting, quantitative case studies, and controlled counterexamples. Calling this a wholly new curvature estimator, a theory of stability, or the first account of useful oscillation would weaken the case.

### Three contributions that can survive the evidence

1. **An exact accounting of temporal cancellation.** Retain all cross-step terms and factor normalized displacement into cancellation within neighboring pairs and accumulation of pair residuals. Use a zero-cross-term reference to distinguish a small path ratio from negative total interaction. The algebra is simple; the contribution is its diagnostic organization and the empirical questions it enables.
2. **An optimizer-aware interpretation of observed gradient response.** Relate consecutive alignment and norm change to the classical secant directional curvature under smooth GD, state the finite-step loss approximation and sampling contribution, and distinguish raw gradient response from the moment, adaptive direction, and applied increment under Adam. Any memory/proxy identity is an analytical qualification, not a validated replacement for an HVP baseline.
3. **A controlled characterization and its implications for interventions.** Establish the coexistence of local reversal, positive total interaction, and objective progress in the controlled sweeps; quantify sampling dependence; and use Adam/Pythia and matched interventions to demonstrate that improving an alignment statistic does not reliably improve loss. This establishes boundaries for future methods without proposing one.

The contribution should be a reusable way to ask and answer the diagnostic question, supported by nontrivial evidence, rather than an inventory of equations plus caveats.

## 2. Evidence hierarchy and manuscript allocation

| Evidence | What it contributes to the central claim | Recommended role |
| --- | --- | --- |
| Analysis024 + Analysis056, nine 3,000-update GD paths | Seven negative adjacent-interaction totals coexist with nine positive all-pair totals; highest tested rates achieve lowest final training losses; pairing identifies immediate cancellation and residual accumulation | Primary empirical result, with one overview and one quantitative pairing/reference display |
| Analysis035 + Analysis056, backward lag profiles | Longer lags are structured; dropping them is empirically unjustified; alternating signs alone do not establish drift | One compact illustration supporting the exact accounting; avoid a lag atlas |
| Analysis037, three initialization blocks and moving SGD continuations | Batch size affects sampled geometry and the path; alignment does not order objective progress | Primary robustness/interpretation result, with sampling unit and equal-update comparison explicit |
| Analysis036 + Analysis057, full-objective Adam trajectories | Reversal/amplification describe excursions, while all 794 rise-phase applied steps remain descending to first order | Compact adaptive-optimizer extension; the current strongest discriminator of raw-gradient versus update semantics |
| Analysis057, component/held-out-rate comparisons | Norm growth explains much event severity; direction's incremental descriptive value is fragile | Retain the negative finding concisely; demote the original product scatter and extended regression discussion |
| Analysis025, Pythia 14M/31M/70M norm-matched removal of scaling | At fixed native adaptive-step norm, changing coordinate scaling worsens all 18 fixed-probe continuation outcomes; the applied direction and nonlinear response matter | Strong optional practical bridge, already replicated across three seed/data-order blocks; one compact table or paragraph |
| Analysis030, six Pythia LR-fork states | The doubled LR wins all six final full-validation comparisons despite negative early raw cosine | Clean counterexample to treating positive cosine as an optimization target; report the six matched outcomes without reviving classifier framing |
| Analysis043, 36 matched short branches | The same intervention can smooth diagnostics but improve MLP and slow ResNet/Transformer; clipping duty and decay effects qualify general claims | A compact discussion/table of fixed-half responses closes the diagnostic-to-intervention argument |
| Analysis040--042 | Selected-state intervention and trigger explorations, including failures | Keep research record; no need to place every attempt in the paper |
| Earlier Pythia size/seed descriptions | Separation of raw, moment and adaptive streams under realistic clipping/schedule/decay | Reuse only the smallest clear bridge needed; avoid an independent scale-law narrative |

The current paper gives approximately six pages to Adam events, regressions and an uncalibrated theoretical bridge. That allocation makes the paper read like a spike-characterization submission. A diagnostic paper can instead present the loss-only segmentation once, retain the main anatomy and actual-step result, and devote saved space to existing matched intervention evidence. No need to add every old Pythia figure.

Select evidence by whether it answers the three measurement questions, not by how expensive it was to collect or how recent it is. Preserve all omitted evidence in its owning analyses.

### Concrete empirical statements with existing support

- In the highest-rate GD conditions, normalized displacement is only 2.6%, 3.7%, and 5.4%, yet total cross-step interaction is positive and final training loss is the lowest of the tested rates. The magnitude reference explains why a small ratio is not itself “lost learning.”
- The pair factors differ: residual accumulation is substantially more coherent for the Transformer than for the MLP/ResNet in the reported paths. This is useful quantitative information beyond an almost saturated cosine of -1.
- In Adam, a rise interval can contain exclusively first-order-descending applied steps. Calling those “uphill updates” based on raw-gradient reversal is incorrect.
- In Analysis043, fixed-half LR moves Transformer mean raw cosine from -0.299 to +0.589 while increasing mean training loss by 3.234%. The same fixed-half change lowers MLP mean training loss by 46.824% and raises ResNet mean training loss by 4.838%. These are one-state, one-initialization, 64-update responses, not universal rankings.
- Analysis025 supplies three seed/data-order blocks and a common fixed probe, but matches the adaptive component's norm schedule, not the complete rounded increment including weight decay. State that boundary if used.

## 3. Verified prior-work positioning

These checks are selective overlap checks, not an exhaustive novelty search.

| Work | Established scope | Required distinction |
| --- | --- | --- |
| [Barzilai and Borwein (1988)](https://doi.org/10.1093/imanum/8.1.141), with [original paper copy](https://pages.cs.wisc.edu/~swright/726/handouts/barzilai-borwein.pdf) | Two-point step sizes from secant equations | For GD, `kappa_hat = (s^T y)/(s^T s)` is the reciprocal first BB step where defined. Credit this beside the main estimator, not only in the bibliography. The diagnostic application is the claim. |
| [Mishkin et al. (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1ac83203e88eb6cf6b30642f0239b932-Abstract-Conference.html) | Path-dependent directional smoothness, convergence rates, adaptive steps | Present this paper's observed signed response and multiscale measurements, not a new general convergence guarantee or first path-dependent curvature view. |
| [Singh et al. (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37771cc0be272368102a37f202bb88d8-Abstract-Conference.html) | Macro/micro directionality, oscillations, effects of momentum and decay | This is the closest framing overlap. Explain how weighted cumulative interactions, the zero-cross-term reference, and adjacent-pair factorization answer a narrower quantitative question. Do not claim that coexistence of micro-oscillation and macroscopic direction is new. |
| [Bai et al. (ICML 2026)](https://proceedings.mlr.press/v306/bai26o.html) | Adaptive-preconditioner decoupling, spike stages, directional curvature predictor, neural/Transformer evidence | A general Adam-spike mechanism is already occupied. The current study contributes a qualified raw-response reading and empirical distinctions. A new HVP calibration campaign is required only if approximation fidelity/superiority is claimed. |
| [SPAM (2025)](https://arxiv.org/abs/2501.06842) | An optimizer using spike-aware clipping and momentum reset, evaluated in LLM training | Do not imply the paper introduces intervention through gradient monitoring. A diagnostic paper need not compete as an optimizer if it claims no optimization improvement. |
| [Li and Liu (ICML 2022)](https://proceedings.mlr.press/v162/li22q.html) and [Lei, author manuscript](https://leiyw.github.io/paper/TPAMI2025.pdf) | High-probability optimization/generalization analysis under assumptions on gradients/noise | Acknowledge the original review's concern and distinguish observed displacement identities from probabilistic iterate/stationarity bounds. Do not describe the identities as norm-control guarantees. |

Lei's article metadata needs care: the original review used 2025, whereas the [author's current publication list](https://leiyw.github.io/) records the final TPAMI issue as 48(3), 2104--2117, 2026. The [institutional record](https://hub.hku.hk/handle/10722/366758) reports a 2025 early publication and DOI 10.1109/TPAMI.2025.3621591. This is a metadata distinction, not two papers.

Suggested main-text novelty sentence:

> We use established gradient-product and secant geometry to quantify how local reversal combines into cumulative motion, then test how that accounting relates to objective progress under sampling and adaptive optimization.

Suggested Singh comparison:

> Prior trajectory-directionality studies show that local oscillation can coexist with macroscopic direction. Our measurements separate the path length canceled within adjacent pairs from the coherence of their residuals, and compare both with a zero-cross-term reference and the measured loss response.

Avoid claiming that this exact pairing construction is globally novel without a broader literature search. It can be presented as an explicit diagnostic factorization introduced for this analysis.

## 4. Section narrative and concrete revisions

### Abstract and introduction

Start with the measurement problem, state the three levels, then report two decisive findings and their implication. Use no more than three contribution bullets. End on what the reader can diagnose, not a list of required future experiments. Remove the current claim that reversal and growth jointly “produce loss increases” outside its local quadratic model. The paper documents compatibility and qualified relations; it does not establish a general neural spike mechanism.

The introduction should proceed: practical ambiguity; existing trajectory/curvature work; precise diagnostic question; accounting; findings; implications. Replace vague statements that heuristics are poorly understood with a concrete ambiguity the experiments resolve.

### Background

Rename “Stability Analysis of SGD” to “Curvature, Sampling, and Temporal Geometry.” Keep the quadratic distinction between sign reversal at one and contraction/loss behavior at two. Include BB, Singh, Mishkin, and the probabilistic-bounds distinction in the relevant paragraphs. Condense the pretraining anecdotes to establish relevance, not scale validation. Do not overstate exact covariance's cost as the impossibility of online approximations.

### Methods

Lead with what each diagnostic measures. Put the exact all-pair decomposition and `R = C D` close together, before the curvature approximation, so the central contribution is visible early. Define constructive/destructive **local interactions** by the sign; a persistent regime should be described over a stated window rather than leaving “persistent” as an unquantified classifier. Make the existing-gradient secant identity explicit and cite BB. Give the distinction between smooth GD, sampled gradients, and Adam response once in a compact scope paragraph. Keep detailed derivations in the appendix.

Change “Conditions for Loss Decrease and Spike Formation” to “Local Relation to Objective Change.” Its present ending silently bridges a one-step GD approximation to later Adam spike events. The section should end by saying why objective changes must still be measured.

### GD/SGD results

The November 5 structure is substantially stronger than the original. Preserve its pairing, all-pair reference and loss comparison. Fix “range from 1.97 and 1.99” to “range from 1.97 to 1.99.” Explain that secant near two follows algebraically from opposite, equal-length gradients; it is not independent confirmation. One lag figure is sufficient to defeat the old lag-one approximation. The SGD results should close by illustrating that even similar observed cosines can accompany opposite full-objective changes.

### Adam and existing interventions

Use the sequence: gradients versus actual updates; controlled event example/population; all first-order-descending rise steps; interpretation boundaries; existing matched intervention evidence. The population event analysis remains valuable, but the event-product fit should not carry the contribution: its incremental held-out-rate result is negative. One sentence and an appendix summary preserve that honest conclusion.

Move the lengthy Bai discrepancy derivation and bound to an appendix, with a short main-text statement that optimizer memory and finite-step variation alter the raw response. Its empirical calibration is an optional future question. Restore a compact Pythia bridge if the retained sources can support a clear table without reopening a whole scale story.

Add a short discussion on interventions. The user's sparsity and orthogonal-step ideas can be mentioned only as untested examples, if helpful; do not turn them into a method proposal. A more general useful statement is that a future intervention should demonstrate its effect on cancellation, cumulative motion and task loss under matched budgets. An orthogonalized step can reduce cancellation while also deleting a useful descent component, so geometry alone cannot select it.

### Conclusion

Lead with the learned result: local cancellation is compatible with cumulative constructive interaction and useful objective progress. Then state how sampling and adaptive transformations alter its interpretation. Close by saying that this diagnostic separation supplies an empirical basis for evaluating future interventions. Do not end with a mandatory independent-initialization/HVP/large-scale shopping list; state the current coverage and unresolved claims proportionately in the discussion.

## 5. Adversarial review: what could still cause rejection

| Objection | Severity | Response that the current revision can make |
| --- | --- | --- |
| “The central mathematics is norm expansion and a product of ratios.” | Major, unresolved by polish alone | Acknowledge mathematical simplicity. Demonstrate diagnostic value through concrete findings that a single cosine/norm does not provide, and compare directly with trajectory literature. Do not sell elementary identities as a difficult theorem. |
| “Useful oscillations and micro/macro directionality were known already.” | Major | Credit EoS and Singh explicitly. Center quantitative localization, cumulative-interaction sign, and controlled diagnostic/outcome dissociation. The strength of that advance remains a referee judgment. |
| “You call a regime destructive while it performs best.” | Major if semantics remain vague | Define destruction as local geometric cancellation. Keep “productive reversal” explicit. Never equate it with lost knowledge or inefficiency. |
| “The Adam story is weaker than Bai et al. and lacks curvature validation.” | Major if mechanism/proxy claims headline | Narrow to observable response and optimizer-stream distinctions; demote uncalibrated proxy analysis. Existing actual-step and intervention findings support that scope. |
| “Most controlled sweeps use one initialization.” | Material limit | State the unit of replication; use the existing three-block SGD and Pythia evidence where relevant. Do not convert 129 dependent events into 129 independent runs. No writing revision can create absent replication. |
| “Low overhead and scale benefit are unmeasured.” | Avoidable | Claim no extra gradient evaluation/HVP for the minimal statistic, and specify retained vector/reductions. Do not claim measured efficiency, negligible overhead, or superior fidelity per cost. |
| “This is a catalog of counterexamples without a positive contribution.” | Major | Give the reader a clear diagnostic procedure: identify vector stream; inspect adjacent magnitude/direction; account for residual/cumulative motion; measure objective response; evaluate interventions with matched outcomes. Show what each stage newly resolves. |
| “The same evidence is described three times.” | Writing risk | Make each figure answer one question. Remove duplicate overview, density, severity, and ablation explanations where they no longer serve the chosen claim. |

The original rejection is not fully answered by adding experiments. Its decisive issue was that the promised stability/norm-control contribution did not follow from the algebra. The current evidence supports a more defensible diagnostic contribution, but novelty and significance remain genuine risks. Repeated simulated positive verdicts cannot resolve them.

## 6. Stopping rule for this revision

Freeze the diagnostic scope and existing evidence set. Complete the source-to-claim map, direct literature positioning, coherent rewrite, numerical/reference checks, and visual review. Ask adversarial reviewers to challenge the exact final claims and repair errors or omissions they identify. Stop when the draft has no known unsupported claim or unresolved internal contradiction; record persistent significance/coverage reservations honestly.

The November 5 experiment plan should remain a future program, not the submission dependency list. It would turn the Adam proxy/memory relation into a separately testable empirical contribution, but that is a scientific choice beyond closing the present diagnostic paper. No new spending, prediction task, orthogonalization algorithm, sparsity intervention, or large-model campaign follows from this review.

My assessment is **credible diagnostic-paper direction after substantial focus and synthesis; acceptance is not established**. The strongest path toward submission is a compact paper in which each result answers the same question, with the existing controlled interventions making the diagnostic implications tangible.
