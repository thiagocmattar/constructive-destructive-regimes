# Action items from the manuscript and advisor discussion

Date: 2026-11-16

Status: Review and proposed actions only. No manuscript edits or new experiments have been performed for this review. Item numbers are stable so feedback can refer to them individually.

## Sources and scope

- [Reviewed manuscript](manuscript.pdf): 20 pages, including appendices and references.
- [Complete meeting transcript](<Cristiano _ Thiago - 2026_11_16 11_00 GMT-03_00 - Notes by Gemini.md>): discussion ending at 01:02:55.

The full manuscript was read, its figures inspected, and the complete transcript read. Passages containing proposed changes or scientific ambiguities were revisited.

The meeting points toward finishing this paper for Cristiano's review. It does not establish a new experimental campaign as a prerequisite. His closing request is clear: refine the text, shorten it, improve the presentation of the intervention results, and develop the final discussion.

Several explanations in the presentation are less precise than the current manuscript. In those cases, the action is to preserve and clarify the manuscript's correct formulation. The clearest examples concern Adam curvature, frozen versus rolling cap thresholds, the meaning of the upward-variation metric, and SVD coefficients.

Labels used below:

- **Explicit:** someone actually proposed the action in the meeting.
- **Recommended:** the action follows from comparison with the manuscript.
- **Already addressed:** the current manuscript contains the relevant material; preserve or refine it rather than adding it again.
- **Future:** exploratory discussion outside the immediate revision.

## Block-by-block review

### 1. 00:00-02:39 — Weight-space programming and interpretability

- **Context, not an agreed manuscript addition.** The opening discussion establishes a shared interest in connecting weights with learned functions.
- Do not expand the related-work section into weight-space programming merely because it appeared in the conversation.
- Preserve the useful motivation already present: parameter-space dynamics leave questions about changes in the network's responses unanswered.

### 2. 02:39-06:47 — Historical antecedents and the curvature contribution

- **Explicit:** acknowledge the older optimization literature rather than claiming the underlying secant construction is new.
- **Already addressed:** Section II-B identifies the relationship with BB1; the bibliography includes the original 1988 Barzilai–Borwein paper.
- **Recommended:** audit the abstract and contribution statement so novelty resides in the diagnostic framework, interpretation, empirical investigation, and connection to functional changes—not in discovering gradient differences as curvature information.
- Check the original sources before finalizing claims about gaps in Bai et al. and other related work. The meeting's enthusiasm is motivation, not verification of priority.
- Avoid claiming that the inexpensive statistic directly replaces every curvature quantity used in prior Adam analyses: the measured directions and quantities differ.

### 3. 06:47-08:35 — Central Flows as motivation

- **Explicit:** use the unresolved connection between optimization dynamics and internal representations to motivate the paper.
- **Already addressed:** the introduction and Sections II-A/II-C make this connection.
- Refine the transition into a concrete question: *where does oscillation appear in the network's responses, and what changes should an intervention limit?*
- Verify the precise scope of the cited open question. Avoid broadening it into “previous work ignores representations.”

### 4. 08:35-09:33 — Shorten before handing over

- **Explicit:** perform an editorial reduction before sending Cristiano the draft.
- The discussed “17 pages” corresponds approximately to the main narrative; the supplied PDF has **20 pages including appendices and references**.
- No exact page target was agreed. Shorten by removing duplication and reducing explanatory overhead.
- Preserve the assumptions needed to interpret the results, especially where the presentation itself revealed confusion.

### 5. 09:33-11:25 — Preserve the four-contribution structure

- **Already addressed:** the introduction lists the framework, curvature/diagnostics, representation oscillations, and intervention.
- **Recommended:** make these four contributions form one argument: characterize update interactions → inspect their functional expression → test control of that expression.
- Present the cap as evidence that representation displacement is a useful intervention target in these cases.
- The connection to the other sparsity project is intellectually relevant, but it is not an instruction to import that project's terminology or results.

### 6. 11:25-13:17 — Verify assisted writing and retain completed presentation improvements

- **Explicit:** review the generated prose and its scientific claims before sending the paper.
- **Already addressed:** Figure 1 uses comparable smoothing; Table I summarizes architectures, objectives, and coverage.
- Preserve the qualification that the solid directional-curvature curves and dashed sharpness curves measure different quantities, with different data coverage for some models.
- Do not turn their visual resemblance into a claim of equality or precise agreement in onset time.
- Treat these existing improvements as completed work, rather than adding them again to the revision queue.

### 7. 13:17-14:45 — Clarify curvature under Adam

- **High-priority clarification:** two gradients evaluated across an Adam step do not make their raw-gradient alignment, by itself, curvature along Adam's applied direction.
- **The manuscript already distinguishes this correctly:** Eqs. (16)–(17) use the actual parameter increment; Eq. (28) projects the gradient response onto the preceding raw gradient.
- Make this distinction easier to retain when entering Section V. The continued use of $\eta\hat\kappa_t$ under Adam creates an avoidable reading burden.
- Consider a distinct symbol or a compact reminder for the Adam projected-response statistic.
- Preserve the smooth, fixed-objective and full-gradient conditions on the exact identity.

### 8. 14:45-15:48 — Keep the Adam overview focused

- **Already addressed:** Figure 5 displays the highest learning rate for each case and normalizes gradient norms relative to initialization.
- Retain its role as an overview of loss, reversal, and gradient growth.
- Make clear that the overview is selected from the stated experimental grid.
- Preserve the distinction between the figure's excursions exceeding 10% and Table II's next-step increases of at least 0.1%.

### 9. 15:48-18:37 — Explain what the warning results establish

- Keep the timing explicit: information available at state $t$ is used to warn about $t\rightarrow t+1$.
- Preserve the class-imbalance baselines and definitions of accuracy, precision/hit rate, and lift.
- The two equivalent conditions should remain identified as **one rule expressed in two ways**, not two independent successful diagnostics.
- **Correct the spoken “100% retrospective” interpretation:** exact loss-sign equivalence applies to full-batch quadratic GD under the stated conditions. It does not generally apply to nonlinear training or Adam.
- The manuscript already supplies counterevidence to universal exactness in Appendix B, Table IV. Keep that qualification visible.

### 10. 18:37-19:36 — State that no classifier was trained

- Cristiano's question reveals a presentation gap: “classifier” suggests a fitted model.
- **Recommended:** explicitly call these *fixed decision rules*, evaluated on recorded trajectories, with no learned classifier.
- State the target once and consistently as a relative increase of **0.1%**.
- Consider adding warning counts and recall to the supporting table. Precision alone does not show how many increases the rule misses.
- This is a reporting improvement; the meeting does not require developing a new predictive model before review.

### 11. 19:36-22:00 — Make the representation measurements easy to reconstruct

- **Already addressed mathematically:** Eqs. (29)–(30) define the activation differences, cosine, and RMS displacement.
- Preserve the distinction between examples, spatial/token positions, and features.
- Keep $P_\ell=N_\ell m_\ell$ explicit; the transcript's dimensional description contains slips.
- Explain that the cosine concerns the entire flattened change. Negative aggregate cosine does not mean every neuron or example reverses.
- If simplifying this section, retain the concrete intuition: compare the same responses to the same inputs before and after successive updates.

### 12. 22:00-24:01 — Interpret displacement retention carefully

- **Recommended:** retain the functional analogue of normalized parameter displacement; it provides an important bridge between the first and second halves of the paper.
- Correct the spoken interpretation that low $R_T$ means the weight norm is not changing. It measures net displacement relative to path length.
- Likewise, low representation retention does not mean the function is unchanged.
- Do not translate 1–4% retention into “only 1–4% of learning is useful.” The manuscript correctly rejects that interpretation.
- Preserve the orthogonal-motion example, or an equally short explanation, because low retention alone does not establish cancellation.

### 13. 24:01-25:05 — Explain SVD energy without calling it explained variance

- **Already addressed:** Section V-B explains that the change matrix is uncentered and that squared movement includes persistent drift.
- Keep the distinction between singular values, squared singular values, spatial patterns, and temporal coefficients.
- Consider compressing the SVD tutorial while preserving these distinctions.
- The essential reader takeaway is: a few fixed patterns can reconstruct much of the observed movement, including its changing sign and amplitude.

### 14. 25:05-27:25 — Separate observed timing from proposed mechanisms

- Preserve the observed differences across architectures: widespread MLP reversal versus less persistent reversal at earlier ResNet and Transformer sites.
- **Recommended:** maintain the exact timing language already in the paper. The first displayed output reversal precedes the **10% loss marker**, which is not necessarily the onset of the excursion.
- Do not infer propagation direction or a unique initiating layer from synchronized heatmaps.
- Do not infer tanh's numerical attenuation from independently normalized RMS rows. The manuscript already explains why that comparison needs unnormalized quantities.
- Treat movement between minima, gradient propagation into earlier layers, and representation reorganization as hypotheses unless directly tested.

### 15. 27:25-29:38 — Quantify “low dimensional” accurately

- The presentation's approximate “ten components explain 95%” should not become a general result.
- **Existing numerical anchor:** Table VI reports output ranks for 95% squared movement of **5, 20, and 15** for MLP, ResNet, and Transformer in the selected windows.
- Explain concentration relative to the observed window length and measurement space.
- Retain the orthogonal and unit-row controls: concentration can partly reflect a few large changes.
- Avoid equating concentrated movement with redundant knowledge, redundant neurons, or a low-dimensional learned function globally.

### 16. 29:38-33:02 — Interpret temporal overlap correctly

- **Already addressed:** Eq. (35) compares temporal subspaces, allowing sites with different feature dimensions to be compared.
- Make clear that an overlap of 0.5 is not “this layer causes 50% of the output oscillation.”
- Preserve the distinction between shared timing and shared feature directions.
- The plots can motivate candidate intervention sites, but they do not establish which sites are optimal to intervene on.
- The finer site-level analysis mentioned in the meeting can remain supporting material; no expansion to every site was requested.

### 17. 33:02-35:56 — Correct the explanation of the signed coefficient

- **Important explanatory correction:** $\alpha_{t,j}$ is a signed coefficient, $\sigma_jU_{t,j}$; it is not an eigenvalue.
- Singular values are nonnegative. The sign-changing coefficient is what demonstrates movement in opposite directions along a fixed pattern.
- The plotted normalization is division by $\sqrt P$.
- Keep the efficiency interpretation bounded: extensive oscillatory movement motivates intervention, but neither the SVD nor retention directly measures wasted computation.
- The manuscript's equations are already correct; the revision should make them easier to explain accurately.

### 18. 35:56-36:59 — Relate representation oscillation to the loss peak

- **Author decision, 2026-11-16:** dismiss Cristiano's cross-correlation suggestion.
- **Retained action only:** overlay the loss curve on a secondary vertical axis in Figure 7's signed eigenmode-coefficient oscillation panels, using the same training-step axis and window so the loss peak is visible alongside the oscillations.

### 19. 36:59-38:02 — Explain the bridge from modes to per-unit control

- The discussion correctly notes that a few modes do not necessarily mean a few neurons.
- **Recommended:** state explicitly that the cap does not identify or suppress SVD modes. It limits direct per-unit changes.
- Explain the bridge as a motivated intervention hypothesis: large functional movement suggests testing limits on local displacement.
- Preserve the statement that the cap does not compute an SVD during training.

### 20. 38:02-39:03 — Fix the repeated c notation

- **Explicit, still outstanding:** rename $c_{t,j}$ in Eq. (36), since $c_t^\ell$ already denotes representation-change cosine.
- Use a distinct symbol consistently in the score definition, scaling rule, calibration equation, algorithm, and supplement.
- Prefer including the site index where it helps distinguish site-specific budgets and per-unit scores.
- This is a straightforward editorial correction with no scientific change.

### 21. 39:03-41:59 — Resolve frozen versus rolling thresholds

- **Highest-priority method clarification:** the meeting describes a rolling quantile whose threshold changes as the distribution changes.
- **The reported method is different:** Eq. (38) calibrates each site's budget from the eight preceding untreated updates and freezes it during treatment.
- State plainly what varies: current inputs, proposed updates, scores, and scaling factors. The calibrated budget stays fixed.
- Distinguish the budget $\tau_\ell$ from the applied scaling factor $\gamma_{t,j}$.
- The remark about reusing $n/m$ also calls for a notation check. The current equation already gives the calibration length explicitly as eight updates; any generalized window length should use a separate symbol.
- A rolling-budget controller would be a different intervention requiring its own evidence. It should not be introduced through a prose edit.

### 22. 41:59-43:09 — Correct the meaning of the upward-variation metric

- **High-priority explanatory correction:** the presentation calls this quantity a peak and then the maximum loss.
- In Eq. (39), $\mathcal P$ is **the sum of all positive consecutive loss changes** over the continuation.
- Keep three distinct outcomes: mean loss, accumulated upward loss variation, and final loss.
- Use those names consistently in Table III, Figure 8, and the discussion.
- Do not introduce a maximum-loss metric merely to match the spoken explanation.

### 23. 43:09-45:53 — Present the intervention tradeoffs precisely

- Emphasize the demonstrated result: every tested cap quantile improves mean training loss and upward variation against unchanged Adam in each case.
- Preserve the endpoint exception: **ResNet q50 ends above the control**, while q25 and q75 improve it.
- Replace interpretations such as “kills learning,” “gets trapped,” or “premature convergence” with the observed finite-budget result: some interventions suppress fluctuations while attaining worse loss.
- Preserve the final-32-step comparison, which checks sensitivity to the oscillatory endpoint.
- Do not claim a universal best quantile, matched intervention strength, generalization improvement, or faster wall-clock training.

### 24. 45:53-47:46 — Redesign Figure 8

- **Explicit:** Cristiano favors retaining the tradeoff visualization, but finds it too cluttered.
- His suggestions include fewer displayed interventions or representative/best settings; yours includes graying out less important points.
- **Recommended implementation:** retain all settings as subdued points, highlight the cap settings and unchanged Adam, and label only the points essential to the comparison.
- If selecting “best” settings, define the selection criterion. With two outcomes, there may be no single best setting in a family.
- Keep the full numerical grid in Table III or the supplement.
- If showing a Pareto frontier, describe it as the frontier among the tested settings—not a general optimizer frontier.
- Preserve readable axes and an explicit explanation of the vertical scale near zero.

### 25. 47:46-48:49 — Retain Figure 9's central result

- **Recommended:** keep the functional before/after comparison in the main paper. It connects the intervention back to the paper's central question.
- Its message is that large changes become smaller and training outcomes improve while reversals remain.
- Maintain common normalization within each baseline/treatment RMS pair; independent normalization would obscure the treatment effect.
- Preserve the disclosure that q25 is an illustrative selection and q50 was prespecified.
- The figure supports “eliminating reversal was unnecessary for these improvements,” not “oscillation is necessary for learning.”

### 26. 48:49-51:10 — Separate evidence from the multiobjective hypothesis

- Cristiano explicitly calls the idea that some oscillation may be necessary a speculation.
- The proposed three objectives—task loss, representation compactness, and exploration—are a future research framing.
- **Recommended:** use them, at most, in a short discussion of open questions.
- Do not present the current study as establishing a multiobjective theory of learning.
- The approximate “70% dead neurons” remark needs architecture, site, activation definition, and data coverage before it can support a manuscript claim. It is not established by this paper's presented evidence.

### 27. 51:10-53:38 — Record sparsity and exploration as future experiments

- **Future:** test whether sparsity constraints alter oscillation, displacement, and loss outcomes.
- **Future:** investigate activity-dependent mechanisms that increase the responsiveness of underused neurons.
- **Future:** consider differential-evolution-inspired perturbations as an exploration mechanism.
- These are distinct hypotheses, not interchangeable explanations of the cap's effect.
- None was specified sufficiently in the meeting to become a required experiment before Cristiano's review.

### 28. 53:38-56:40 — Handle biological inspiration carefully

- **Explicit suggestion:** Cristiano thinks a neuroscience connection could strengthen the discussion.
- The useful connection is regulation of activity or sensitivity, presented as an analogy and a research direction.
- A biological claim would need appropriate scientific sources; the book discussion alone is insufficient.
- Resolve the bias-sign ambiguity before describing any mechanism. Increasing an additive bias in $z=w^\top x+b$ differs from decreasing a firing threshold in $z=w^\top x-\vartheta$.
- The present displacement cap should not be described as implementing homeostatic regulation.

### 29. 56:40-59:56 — Compress the paper's central message

- The discussion repeatedly returns to synthesis: a substantial investigation should admit a precise central statement.
- **Recommended anchor:** *Across the studied trajectories, limiting large local representation changes improves training-loss outcomes while reversals persist, connecting update-interaction diagnostics to a concrete intervention target.*
- Use that statement to decide what the introduction promises and what the conclusion delivers.
- Retain the four contributions, but make their relationship more prominent than their individual technical machinery.

### 30. 59:56-01:00:46 — Develop a final discussion

- **Explicit:** Cristiano asks for a substantive discussion incorporating the implications raised in the meeting.
- The current conclusion summarizes findings well, but only briefly develops those implications.
- Add a compact discussion, or reshape the closing section, around:
  - what functional measurements reveal beyond loss and whole-network gradients;
  - why controlling movement magnitude can help despite continued reversal;
  - what remains unknown about beneficial oscillation, exploration, and representation organization;
  - what the experiment's scope permits: finite windows, limited initializations, selected cases, and measured computational cost.
- Keep the speculative material visibly separate from demonstrated findings.

### 31. 01:00:46-01:02:55 — Follow the agreed review workflow

- **Explicit:** prepare a polished PDF for Cristiano to print and annotate.
- The stated target was Wednesday, November 18, with delivery before the weekend as the broader intention.
- After his comments, revise and then consider sending the paper to Braga.
- A change of writing platform is not required; local compilation and PDF exchange fit the agreed workflow.
- This conversation does not authorize sending the PDF automatically.

## Coordinated manuscript editing pass

| Manuscript area | Concrete action |
|---|---|
| **Abstract** | Shorten the method inventory; foreground the central finding and intervention evidence. Preserve the distinction between the exact identity and empirical warnings. |
| **Introduction** | Keep the four contributions, but reduce detailed results and qualifications that are repeated later. Make the functional question lead naturally into the intervention. |
| **Related work** | Retain the closest comparisons: oscillatory dynamics, secant diagnostics, representation measurements, and update control. Verify the gap claims against the originals. |
| **Section III** | Preserve the mathematical distinctions. Consider shortening the successive displacement expansions and repeated explanations, while keeping the exact/approximate boundary clear. |
| **Section IV** | Preserve the result that adjacent cancellation coexists with positive all-pair interaction. Keep sampling at fixed parameters distinct from evolving SGD trajectories. |
| **Section V-A** | Clarify fixed rules, timing, the target threshold, and the meaning of the Adam statistic. Keep the class-imbalance baseline prominent. |
| **Section V-B** | Shorten the SVD tutorial; improve the interpretation of coefficients and overlap. Overlay the loss curve on a secondary vertical axis in Figure 7's signed eigenmode-coefficient oscillation panels so the loss peak is visible. |
| **Section VI** | Fix the score notation, emphasize frozen calibration, simplify Figure 8, and retain Figure 9. Explain the upward-variation metric consistently. |
| **Closing discussion** | Interpret the findings without turning exploration or biological analogies into established mechanisms. |
| **Appendices/supplement** | Preserve the nonlinear loss-sign audit, spectral controls, endpoint sensitivity, and cost evidence. These resolve foreseeable questions and should not disappear during shortening. |

## Recommended execution order

These are proposed revision tasks, not completed changes. Checkbox IDs are separate from the block numbers above.

- [ ] **R1 — Resolve scientific communication discrepancies:** Adam curvature, retrospective exactness, frozen budgets, the upward-variation metric, SVD coefficients, and retention.
- [ ] **R2 — Fix notation**, especially the duplicated $c$.
- [ ] **R3 — Simplify Figure 8** while preserving the complete comparison.
- [ ] **R4 — Add the secondary loss axis** to Figure 7's signed eigenmode-coefficient oscillation panels, showing the loss curve and its peak over the same training-step window (item 18).
- [ ] **R5 — Write the final discussion** around the demonstrated result and clearly separated open questions.
- [ ] **R6 — Shorten the abstract, introduction, and repeated explanations** across the body.
- [ ] **R7 — Verify novelty and literature-gap statements** against their sources.
- [ ] **R8 — Check the compiled PDF at print size**, including figure labels, captions, cross-references, and consistency with the supplement.
- [ ] **R9 — Prepare the review PDF for Cristiano**, with any genuinely unresolved choices identified briefly.

The new seeds, rolling controller, sparsity interventions, exploration mechanisms, biological framing, and larger-scale studies should not become an expanding prerequisite for this handoff. The meeting supports a focused revision of the existing paper.
