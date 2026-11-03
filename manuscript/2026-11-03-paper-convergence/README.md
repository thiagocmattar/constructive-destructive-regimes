# Paper evidence and submission plan

Snapshot: 3 November 2026, America/Sao_Paulo. This is a consolidation and a proposed closing plan, not a manuscript revision or an experiment launch.

The strongest paper available here explains what consecutive training gradients reveal about the geometry traversed by optimization, distinguishes productive reversal from amplified excursions, and shows how familiar stabilization choices change those observations. Close that argument before expanding the experiment tree. A diagnostic-triggered controller is a possible additional contribution whose value still needs a matched comparison.

The recommended order is: **position the geometry precisely; recover relevant existing evidence; test incremental information; run a small intervention only if it resolves a remaining claim; freeze the paper.** More model sizes and more retrospective spike plots have lower priority.

Use this page as the decision entry point, [NEXT_ACTIONS.md](NEXT_ACTIONS.md) as the bounded work queue, and [evidence_map.csv](evidence_map.csv) to find all 38 numbered analyses. The project has 216 numbered run folders; a folder count is not a count of independent completed experiments. [source_manifest.json](source_manifest.json) pins the reviewed sources and relevant chat snapshots. Relative repository links below resolve from this folder.

## The paper in one paragraph

Consecutive-gradient interactions provide an inexpensive observation of training geometry. Under smooth, full-objective GD, the gradient-difference statistic measures directional curvature averaged along the completed step. Adjacent cosine identifies reinforcement or reversal, while gradient magnitude shows the strength of that response. Controlled trajectories demonstrate that reversal can coexist with sustained progress and substantial cancellation of parameter motion. Sampling and Adam change the interpretation: noisy gradients mix geometric response with sampling, and Adam follows a transformed direction. Adam spikes nevertheless exhibit recurring raw-gradient reversal and amplification. Existing Pythia schedule interventions show that these regimes can be changed, while also demonstrating that more positive alignment does not guarantee better validation loss. The practical question is therefore when intervention improves the loss/cost tradeoff, rather than how to keep cosine positive.

This paragraph is the proposed narrative, with its claims restricted to the evidence below. It does not assert that a new controller has been validated.

## Current manuscript and work sessions

The current [14-page manuscript](../draft/revisions/2026-11-03-regime-writing-revision/revised-manuscript.pdf) has eight figures and two tables. Its [verification record](../draft/revisions/2026-11-03-regime-writing-revision/verification.json) matches the PDF hash `34a1d851293c4d3fc8bad5e15f2608d66380e743ee1697ac47aeb9130c8ec789`. Its main sequence is geometry, GD/SGD characterization, and Adam spike anatomy. The exact curvature identity is already present in Appendix A; moving its statement into the main argument is an editorial option, not a missing mathematical result.

The chat began in the neighboring `language-diagnostic-training` repository, which was retired on 29 October. This report belongs to `constructive-destructive-regimes`. The older harness's FineWeb ladder, Ricker results, and archived experiments are not evidence for this paper.

| Chat title, preserved as shown in the app | Recorded endpoint | What to carry forward |
| --- | --- | --- |
| Design spike intervention tests | Latest turn completed; chat not loaded. Analysis038 is a design, with no intervention training. | Existing protocol and replay audit; do not duplicate its planning. |
| Extend EOS Adam training | Latest turn completed; chat not loaded. Latest work restores curvature in the 14-page revision. | This chat has evolved into the current manuscript task; its title is not proof of active training. |
| Review manuscript comments | Latest turn completed; chat not loaded. Its last inspected revision used a forecast table. | Historical forecast results exist, but that presentation has since been replaced. |
| Plan checkpoint gradient analysis | Latest turn completed; chat not loaded. Analysis035 completed backward profiles. | Backward anchors are complete; do not rerun them because the chat title says “plan.” |
| Revisar testes de calibrabilidade | Latest turn completed; chat not loaded. Last exchange concerns Analysis031. | Falling retention does not identify an LR that is too small. |
| Review paper for PAMI | One completed simulated-review turn. | Novelty, Adam consistency, non-spike controls, replication and overhead objections. |
| Review Paper for PAMI | Two completed turns: review and revision strategy. | Prioritize geometry and counterfactual intervention; verify cited literature. |
| Review Paper for TNNLS | Two completed turns; the visible review request and report actually address PAMI. | Useful objections, not an independent TNNLS editorial assessment. |

These are app/history and local artifact observations, not a live audit of every remote process. Run215 and Runs214/216 record completed work and task-cloud closeout. Analysis038 records no launch. Older queues remain held in the research index. There is no evidence in these inspected sessions of a currently running new intervention experiment. No chat was messaged, resumed or stopped for this review.

## Evidence that belongs to the argument

Evidence labels below refer to the stated comparison: **strong**, **mixed**, **suggestive**, **not supported**, or **contradicted**. Mechanistic response, quality benefit, scope and compute cost are separate claims.

| Question | Existing evidence and counterfactual | Conclusion and paper role |
| --- | --- | --- |
| Can negative adjacent alignment coexist with progress? | [Analysis024](../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/004-completed-calibration.md): three models, three constant rates, matched initialization within model, 3,000 full-GD updates. | **Strong within these trajectories.** Seven conditions develop nearly opposing adjacent gradients; the highest tested rate has lower final training loss within each model. Keep as the controlled foundation; one initialization limits robustness. |
| Does local reversal determine the entire path? | [Analysis024 audit](../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/007-section-iv-critical-review.md) and [Analysis035](../../analyses/035-2026-11-02-eos-backward-anchors/REPORT.md). Final all-pair interaction is positive in all nine GD conditions, despite negative lag-one interaction in seven. | The equivalence is **contradicted**. Keep one explicit local/all-lag distinction. Detailed lag atlases can stay in the research record. Backward, forward-anchor and moving-adjacent plots answer different questions. |
| Does the statistic identify curvature under stochastic sampling? | [Analysis027](../../analyses/027-2026-10-30-minibatch-noise/REPORT.md): three MLP initializations, frozen sampling and matched moving paths, full-gradient and HVP references. At batch 32, median proxy/HVP discrepancy changes from 0.7467 to 0.0030 after measuring the explicit noise term. | **Strong evidence for the conditional noise account.** An HVP validation already exists. The correction requires offline full-gradient references; it is not available for free from two sampled gradients. Large nominal batch does not ensure small relative noise. |
| Does reducing reversal improve SGD? | [Analysis037](../../analyses/037-2026-11-02-sgd-batch-extension/REPORT.md): 546 continuations including 198 reused MLP paths, three initialization blocks per case, matched 200-update horizons and full controls. | A universal improvement claim is **not supported**. Smaller batches weaken trained-state reversal, but retention stays low. Almost equal Transformer cosine accompanies opposite loss changes at different starts. Equal updates imply unequal exposure. Keep one compact sampling figure. |
| What recurs around Adam spikes? | [Analysis036](../../analyses/036-2026-11-02-controlled-adam-diagnostics/README.md), Run215: nine 1,024-update full-objective Adam trajectories, one initialization per model. 812 candidates, 806 recovered, six censored; 129 recovered excursions exceed 10%. | **Strong descriptive counts within this dataset; suggestive broader scope.** Norms grow from pre-rise to peak in 129/129; pre-rise cosine is negative in 117/127 defined cases. These are dependent events, not 129 seed replications. Keep overview and selected anatomy. |
| Do cosine and amplification explain severity separately? | The current [signed-product scatter](../../analyses/036-2026-11-02-controlled-adam-diagnostics/observations/008-rho-gradient-rise-peak-scatter.md) has 127 defined points and seven within-rate fits, using the gradient norm at the eventual peak. | **Suggestive descriptive association.** Reported R-squared 0.38–0.94 is on log peak/pre-rise loss, in the fitting sample. Incremental value over amplification alone remains to be checked for this event definition. It is not a prospective predictor. |
| Has future-loss information been tested already? | [Historical temporal analysis](../../analyses/036-2026-11-02-controlled-adam-diagnostics/archive/PEAK_TEMPORAL_ANALYSIS.md) and [tables](../../analyses/036-2026-11-02-controlled-adam-diagnostics/archive/paper_evidence/TABLES.md): expanding-prefix fits, five later blocks, 24-state gap, same nine trajectories. | **Suggestive temporally held-out evidence.** On 4,534 decisions, AP is 0.260 for loss history, 0.303 with gradient level/growth, and 0.454 with projected change. This is historical exploratory evidence, withdrawn from the current presentation; it supplies neither new seeds nor an already validated online policy. |
| Can schedules move the measured regime? | [Analysis004](../../analyses/004-2026-10-04-lr-schedule-intervention/observations/001-schedule-intervention.md): checkpoint-matched Pythia-14M/70M original, held and accelerated-decay branches, one seed. | **Strong conditional mechanistic treatment effect; mixed practical implication.** Faster decay recovers positive-smoothed alignment 65/101 updates earlier, but held LR has the lowest validation loss at both sizes. This directly answers the advisor's schedule question. |
| Is there already a practical AdamW bridge? | [Analysis002](../../analyses/002-2026-10-04-scale-seed-replication/README.md) and [Analysis029](../../analyses/029-2026-10-30-pythia-adam-diagnostics/REPORT.md): Pythia-14M/31M/70M, three seed blocks, complete MiniPile passes, accumulation, clipping, decay and scheduling. | **Strong descriptive replication within that recipe.** Use the existing bridge before requesting a larger model. These experiments do not validate stochastic curvature estimation or a spike controller. Model width/head and cross-block hardware vary. |
| Do raw gradients describe Adam's applied direction? | [Analysis025](../../analyses/025-2026-10-17-pythia-descent-mechanism/REPORT.md): 18 norm-controlled coordinate-scaling ablations versus native AdamW across sizes/seeds. | **Strong conditional evidence that preconditioning matters.** All 18 worsen the fixed-probe loss. A favorable first-order direction can still incur a damaging finite step. Keep as supporting mechanism, not an additional main storyline about scale. |
| Should negative raw cosine trigger an LR decrease? | [Analysis030](../../analyses/030-2026-10-30-adam-lr-decisions/REPORT.md): 18 branches from six states, two sizes, three seed clusters. | The simple practical rule is **contradicted in this pilot**: 2x LR wins all six states despite negative early cosine. Fixed-LR/loss-only selection has zero observed regret; diagnostic extensions do worse. This is a valuable negative control, not a failed experiment to hide. |
| Does retention supply a ready scheduling rule? | [Analysis031](../../analyses/031-2026-10-31-adam-state-map/REPORT.md) reuses those branches. At update 16, actual/adaptive retention reduces held-seed final-benefit MAE by about 37%. | **Suggestive and sensitive to target/window.** No tested initial-state summary improves prediction; other windows weaken the result; all states share the same best action. Keep as backlog, not a deployed controller. |
| Have spike interventions been demonstrated? | [Analysis038](../../analyses/038-2026-11-03-adam-spike-intervention-design/PROTOCOL.md) audits three events and specifies matched forks. | **Not supported yet: proposed only.** Intermediate Run215 recovery files were overwritten. A new common prefix and untreated replay qualification are prerequisites. |

Peripheral Rosenbrock, CE-arithmetic and early scale investigations remain indexed in the CSV. They inform interpretation and controls but need not become additional sections. Analysis026's CE result, for example, concerns attained states under changed arithmetic and matched-loss controls; it should not silently become evidence that this diagnostic flattens a landscape.

## The mathematical claim to protect

Use completed-state notation here to avoid a one-step indexing ambiguity: `g = grad L(theta)`, `s = theta_next - theta`, `g_next = grad L(theta + s)`, and `y = g_next - g`. The manuscript uses a different preupdate gradient index; map it explicitly in any new code or caption.

For a smooth fixed objective,

\[
y=\int_0^1 H(\theta+us)s\,du,\qquad
\kappa_{\rm sec}=\frac{s^Ty}{\|s\|^2}
=\int_0^1\frac{s^TH(\theta+us)s}{\|s\|^2}\,du.
\]

For ordinary GD, `s = -eta*g`, so

\[
Q_{\rm raw}=1-\frac{g^Tg_{\rm next}}{\|g\|^2}
=1-\rho\frac{\|g_{\rm next}\|}{\|g\|}
=\eta\kappa_{\rm sec}.
\]

This is a genuine Hessian-free observation of one path-averaged directional quantity. It is also exactly the secant curvature underlying the first Barzilai–Borwein step size. [Barzilai and Borwein](https://doi.org/10.1093/imanum/8.1.141) must be credited explicitly. “A newly invented curvature estimator” is a weak novelty claim; the empirical characterization, unified interpretation, measurement discipline and incremental usefulness must carry the contribution. A scalar directional measurement is not recovery of the Hessian, its spectrum, or every aspect of the surface.

For Adam, the raw projection contains a bilinear response between `g`, the Hessian and Adam's direction. The actual-step secant above remains a directional path average under full-objective gradients. A useful future comparison is

\[
Q_{\rm step}=-\frac{s^Ty}{s^Tg},\qquad s^Tg<0,
\]

against `Q_raw`, with actual applied increments, including rounding and any decay. This comparison needs the appropriate vector products; endpoint scalar histories do not automatically contain them. Under stochastic gradients, `y` also contains endpoint sampling differences.

The distinction between reversal and a harmful step has an exact reason. The loss change is

\[
\Delta L=g^Ts+\int_0^1(1-u)s^TH(\theta+us)s\,du.
\]

It weights curvature differently from the unweighted secant average. A trapezoidal approximation gives `Delta L approximately s^T(g + g_next)/2`, or `s^Tg*(1 - Q_step/2)` for a descending step. Thus the threshold at two is exact for a quadratic, but does not give a general nonlinear loss-sign guarantee. The current highest-rate GD Transformer sign discrepancy, 44.2% of 2,999 transitions, limits the loss-sign approximation without refuting the smooth-GD secant identity.

Preserve these distinctions in the abstract and discussion:

- `rho < 0` defines adjacent raw-gradient reversal. It does not by itself define optimization failure, an excessive LR, or a non-descending Adam direction.
- Positive alignment can occur with poor progress; negative alignment can occur with lower validation loss. The target is useful progress at acceptable excursion cost.
- `Q_raw`, cosine and the adjacent norm ratio are algebraically coupled. Plotting all three is interpretation, not three independent validations.
- The diagnostic becomes available after the current backward pass and describes the preceding movement. It can guide the next update; it cannot repair an update already taken without rollback.
- One retained gradient suffices for cosine and the raw statistic. Cumulative retention needs another running vector. The complete multi-stream diagnostic suite has a larger memory footprint. Wall-time overhead is still unmeasured.

## What the advisor actually prioritized

The supplied 19 October meeting is an automatically generated transcript. Treat the scientific content as discussion evidence, not as instructions authorizing today's training or as verbatim technical notation. Timestamps below refer to that source.

| Meeting passage | Research direction | Status now and implication for closing |
| --- | --- | --- |
| 11:58–15:59 | Connect practical spikes/recovery heuristics to cheap diagnostics computed from ordinary gradients. | This remains the opening motivation. Add measured overhead and a precise connection to current monitoring baselines. |
| 15:59–23:30 | Replace the old S1/S2/S3 exposition with displacement, magnitude, alignment, curvature and noise; retain longer lags. | Implemented in the current formulation. Do not restore an unsupported lag-one truncation. |
| 23:30–28:27 | Explain loss excursions through reversal, amplification and noise; distinguish the local thresholds one and two. | Implemented with approximation scope. Keep “destructive descent” as a central result, rather than equating destructive with divergent. |
| 28:27–34:31 | Use the MLP, ResNet and Transformer GD comparisons to illustrate the analysis. | Analysis024 is complete. These are three model–dataset cases on two datasets, not three independent modalities or proof of architecture invariance. |
| 34:31–37:06 | Place a focused empirical characterization after the derivation so concepts become concrete. | Current section flow largely does this. Preserve the progression; avoid alternating every formula with a full experiment. |
| 38:00–43:56 | Go directly to Adam and distinguish raw gradient, temporal momentum and preconditioned step. Preserve older material rather than expanding every optimizer formula. | Analysis028/029 and current Adam section supply the bridge. Positive momentum cosine is filtering/persistence evidence, not a stability theorem. |
| 43:56–46:32 | Consider a practical language-model setting; batch/noise ablation could be secondary. | Pythia/MiniPile evidence already exists. Analysis027/037 goes well beyond the suggested batch ablation. Do not launch another broad study to satisfy an already covered suggestion. |
| 46:32–49:08 | Prioritize Adam experiments and writing, let the advisor read, then discuss schedules and what interventions the diagnostic suggests. | A discussion grounded in existing matched schedule branches satisfies the minimum intent. A complete new optimizer or controller was not a prerequisite imposed in this meeting. |
| 50:14–51:22 | Finish and submit this paper. | Establish stopping criteria and a bounded final package. |
| 51:22–52:27 | ICML/Christmas deadline discussed for a separate paper with another group. | Do not import that deadline into this manuscript without a new author decision. |

Two transcript expressions need interpretation rather than literal implementation. Its suggestion that a positive cosine means LR is too small is a hypothesis, narrowed by the subsequent Pythia controls. Its spoken “R below -0.2” refers contextually to rho: normalized displacement R is nonnegative. An offline centered moving average used in an old plot also cannot become an online trigger without replacing it with a causal trailing statistic.

## What to retain from the simulated reviews

The three ChatGPT conversations are brainstorming/review simulations, not independent referee decisions or acceptance forecasts. The chat titled TNNLS contains a PAMI request. Their repeated objections are useful because they identify the same missing comparisons; the repetitions do not increase evidential strength. The earlier actual TNNLS rejection concerns the preserved original submission and is summarized separately in [REVIEW_CONTEXT.md](../../research/REVIEW_CONTEXT.md).

| Objection | What the repository already answers | Remaining response worth doing |
| --- | --- | --- |
| The curvature statistic is a classical secant quantity. | Current draft cites BB and proves a path-average identity. | State the exact equivalence prominently. Compare against prior work rather than claiming novelty from Hessian-free computation alone. |
| Raw-gradient thresholds do not transfer to Adam's loss sign. | Current text distinguishes a bilinear response; Analysis038 finds all 28 selected rise updates are first-order descending. | Compare raw and actual-step diagnostics when suitable products are available; explain finite-step behavior. A momentum-conflict-only repair would miss those 28 rises. |
| No evidence beyond loss/norm monitoring. | Historical blocked forecasts already compare loss, norm and projected-change additions. | Audit and summarize them with their original target; test current event-severity components separately. Independent validation remains open. |
| Peak-based R-squared is being read as forecasting. | Current text calls the scatter retrospective. | Reduce its narrative weight. Test amplification alone versus the product before assigning the cosine incremental explanatory value. |
| Sampling is not separated from geometry. | Analysis027 already has full-gradient, frozen-sampling and HVP controls. | Recover a compact validation result or clear companion citation. Extending the claim to stochastic Adam requires a separate test. |
| Scope is only tiny full-batch Adam with one seed. | Pythia multi-seed AdamW evidence exists outside the current compiled story. | Use one compact practical bridge; it does not retroactively replicate the full-batch spike dataset. New confirmation should address the specific remaining claim. |
| “Cheap” has no measured cost. | O(p) storage/work and no extra backward are correct algorithmic facts. | Benchmark the actual selected diagnostic, with/without capture, time and peak memory. Do not quote full multi-stream overhead as the cost of a single-buffer method. |
| Censoring and pseudoreplication weaken statistics. | Raw current dataset retains six censored excursions; two exceed 10% and are excluded from recovered anatomy. | Keep denominators explicit; include censored outcomes in intervention evaluation and use trajectory/seed-level uncertainty. |

The historical forecast's 129 episodes are **not** the current 129 recovered >10% excursions. The older definition starts at state 64, uses a trailing 16-state minimum and a 10% onset, and retains incomplete recovery. The current definition uses the immediate pre-rise level, searches from initialization and conditions the anatomy table on recovery. Their numerical equality is accidental. Never merge their counts, lead times, AP values or censoring denominators.

### Literature that changes the positioning

This is a focused overlap check, not an exhaustive novelty review.

- [Barzilai and Borwein, 1988](https://doi.org/10.1093/imanum/8.1.141): secant-based gradient step sizes. Credit the exact relationship.
- [Mishkin et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1ac83203e88eb6cf6b30642f0239b932-Abstract-Conference.html): path-dependent directional smoothness and adaptive gradient methods. Distinguish the observed quantity and empirical question.
- [Bai et al., ICML 2026](https://proceedings.mlr.press/v306/bai26o.html): Adam second-moment decoupling, spike stages and a gradient-directional curvature predictor. The general connection between Adam spikes and curvature is already occupied. Explain exactly what this observational protocol adds, and compare suitable baselines before claiming superiority.
- [SPAM](https://arxiv.org/abs/2501.06842): spike-aware clipping and momentum reset for LLM training. A new clipping heuristic alone does not establish novelty. A diagnostic paper need not implement a complete competing optimizer unless it claims optimizer performance.
- [Zhu et al., ICML 2024](https://proceedings.mlr.press/v235/zhu24h.html): catapults, feature learning and generalization. Spike suppression must be evaluated with subsequent progress; a smaller peak is not automatically better training.

## Proposed manuscript sketch

This is a structure for the next author discussion, not a request to restore all removed appendices.

| Part | Reader's question and main claim | Evidence and space decision |
| --- | --- | --- |
| Introduction | What useful training geometry can we observe using ordinary gradients? | Lead with productive reversal versus damaging amplification. Separate established interpretation from proposed intervention. |
| Positioning | What is inherited from secant methods, gradient products, hypergradients, EoS and Adam-spike studies? | One direct comparison paragraph/table; explicitly state the secant equivalence and Bai overlap. |
| Observable geometry | What exactly do rho, Q and retention measure, and when? | Short path-average proposition in the main text, proof in the existing concise appendix. Distinguish full GD, sampled gradients and adaptive steps. |
| Controlled characterization | Can reversal remain productive, and how does sampling change it? | Compact GD loss/geometry and batch-size views. Keep retention as supporting geometry. Move the long lag atlas to linked research material if space is needed. |
| Adam excursions | What repeats, what precedes the pronounced rise, and what is only retrospective? | Keep the trajectory overview and three selected spike cases; condense the anatomy table. KDE and peak-regression panels are first candidates for removal from the compiled paper. |
| Practical implications | How do schedules/clipping/moments act on the observed components? | Add a compact Pythia schedule comparison from existing matched branches. If a bounded intervention supplies useful new evidence, use one paired result here. A historical forecast table is optional and must retain its separate definition. |
| Conclusion | What is now established, and which practical decision is still open? | Cheap observable geometry, controlled recurrence and concrete heuristic hypotheses. Claim an effective warning/controller only if its dedicated validation passes. |

The current abstract devotes substantial space to 129 events and R-squared. Prefer spending that space on the exact scope of the geometric observation and the decisive controlled finding. Keep quantitative anatomy if it supports that hierarchy. A compact paper need not place every discarded figure in an expanding appendix: preserve complete research sources in the repository companion.

## Experiment decisions and stopping rule

[NEXT_ACTIONS.md](NEXT_ACTIONS.md) specifies the comparisons. The closing sequence is deliberately conditional:

1. **Use existing evidence first.** Finish the novelty/claim map, compare the current severity descriptor with its components, and audit the archived forecast package before rebuilding an equivalent classifier. Recover the existing noise/HVP result and Pythia schedule contrast.
2. **Choose whether intervention is needed for this paper's claim.** Analysis038 already has a rigorous eight-arm proposal. For a closing study, a reduced pilot can test untreated Adam, half LR, raw-gradient clipping, actual-step clipping and Armijo. This is a proposed narrower design, not an amendment silently applied to its frozen source.
3. **Separate action efficacy from trigger value.** A retrospectively selected fork tests whether an action can help that state. Only whole-trajectory, past-only policies with norm/loss controls test whether the diagnostic chooses useful times.
4. **Confirm only the winning question.** If the claim is a useful online trigger, evaluate new independent trajectories and an AdamW setting. If the paper remains interpretive, prioritize replicated geometry/cost evidence and a precise discussion. Do not commit to every branch merely because it is feasible.

For the user's proposed conditional clipping, use current and preceding measured gradients: `rho < 0 and ||g_current|| > ||g_previous||`. This is a reasonable candidate to test. Global clipping leaves the current pair's angle unchanged, and in Adam it alters both moment streams without directly bounding the applied parameter step. Compare it with the same clipping action using a norm-only gate, always-on clipping, a loss-based gate, and an untreated control. Add a trigger-agnostic LR comparator. Calibrate thresholds on development histories and report intervention counts/cost; apparent gains from simply taking smaller steps need to be visible.

Stop adding experiments when the paper can support its chosen claim with: an explicit prior-work distinction; one controlled geometry result; sampling qualification; an optimizer-consistent Adam interpretation; evidence for incremental value if claimed; measured overhead; appropriate independent coverage; and a compact discussion of interventions with their actual status. If cosine adds no value over norms for the chosen target, preserve that negative result and narrow the contribution instead of searching indefinitely for a favorable threshold.

My venue recommendation is to develop the neural-network optimization/diagnostic story for **TNNLS first**, while reviewing the original rejection and the journal's current resubmission route before submission. This is an editorial judgment based on the topic and the [official scope](https://ai.ieee.org/research/). [TPAMI's scope](https://www.computer.org/digital-library/journals/tp/cfp-ieee-pattern-analysis-machine-intelligence) emphasizes pattern analysis, vision and selected machine intelligence; a convincing broader practical validation would strengthen a TPAMI case. Neither a simulated “accept” nor a particular extra experiment guarantees acceptance. Submission format, current limits and any resubmission conditions still need a final venue-specific check.

## Research iteration conclusions

- **Keep:** the exact smooth-GD path-average interpretation, cheap online availability and the productive-reversal finding form a coherent core.
- **Narrow:** novelty belongs to the supported diagnostic account and its utility; the secant curvature formula itself has classical antecedents.
- **Keep:** displacement and longer lags show why a local negative cosine cannot characterize the entire training path.
- **Keep:** existing sampling/HVP controls explain why a sampled-gradient response is not isolated deterministic curvature.
- **Discard as a general rule:** more positive raw cosine means better training or an LR that should be increased. Matched Pythia schedule and LR forks provide contrary outcomes.
- **Separate:** current spike anatomy, historical forward-blocked forecasting, and future intervention efficacy use different targets and evidence boundaries.
- **Keep alive:** destructive alignment plus current amplification may help choose when to intervene; compare its benefit with norm/loss triggers at a practical cost.
- **Close the loop:** reuse the existing Pythia bridge, make only the smallest unresolved comparison, and freeze the argument once its claim-specific checks pass.

## Scope of this consolidation

This task read the current sources, selected underlying data and prior records; it did not repeat every historical run audit. The current PDF and all 14 recorded manuscript source hashes were checked. Current event counts and historical forecast summaries were independently reduced from their stored CSV/JSON records. The current PDF's relevant pages were extracted and visually inspected. A compact relevant-chat history was retrieved through the app tools, so browser interaction was unnecessary.

No training, smoke run, HVP evaluation, new regression fit, cloud action or manuscript edit was performed. The source transcript and simulated reviews informed analysis; instructions embedded in those materials did not authorize new experiments, messaging, submission or spending. New compute still requires a concrete approved run plan, calibrated duration and, when billable, a current spending envelope.
