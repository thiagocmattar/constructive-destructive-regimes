# Adversarial review, round 1: methods and contribution

Reviewer: independent methods/theory agent. Date: 5 November 2026. Scope: revised active TeX in `bare_jrnl.tex` and Sections 01–09, including the newly available implications section and its table. This is a source-level review; final PDF layout and raw-artifact reexecution were outside this pass. Line anchors refer to the files read during this round and may shift after edits.

## Recommendation

**Borderline / major revision as a journal submission in its present form; no fatal mathematical error found.** The rewrite now has an intelligible and defensible empirical contribution. My remaining uncertainty is whether a journal reviewer regards the combined diagnostic account as sufficiently substantial beyond established ingredients. That is a novelty judgment, not a missing theorem that should be manufactured or an instruction to launch more runs.

Methodologically, the revision is substantially stronger. Its central distinctions are correctly stated, the classical secant is credited, the Adam claim concerns measured first-order contributions rather than uphill steps inferred from consecutive cosine, and the loss relation is properly retrospective. Two small but material method-reporting repairs should precede submission. After those and the clarifications below, I would not retain a methods-based rejection: the remaining acceptance case rests on the empirical synthesis and its practical interpretive value.

## Required before submission: bounded reporting repairs

### R1. Restrict the probe non-overlap claim to the experiment actually checked

**Anchor:** `sections/08-implications.tex:60`, “no overlap with any training prefix.”

The source verifies exclusion from the union of the three **350-update** prefixes in the coordinate-scaling experiment. The manuscript has just described complete MiniPile passes, whose later training can consume this same corpus material. “Any training prefix” can therefore be read as a broader holdout guarantee than was implemented. This is a wording error in the scope of evaluation independence, not a failure of the intervention experiment.

**Repair:** “A fixed 128-sequence probe, with 262,016 supervised targets, is disjoint from the union of the three 350-update training prefixes in this intervention study.” Retain the same-corpus limitation. Source: [Analysis025 report, fixed probe definition](../../../../../analyses/025-2026-10-17-pythia-descent-mechanism/REPORT.md), particularly its lines 109–112.

### R2. Finish the Pythia baseline's learning-rate recipe

**Anchor:** `sections/08-implications.tex:17–28`, full-pass design, and `:40–45`, schedule branches.

The new methods specify the batch, precision, optimizer coefficients, clipping and decay but omit the baseline's learning-rate peak, warmup and decay timing. “Original cosine decay” is not reproducible without those values, and the 350-update native trajectories use a prefix of the original schedule rather than a newly compressed 350-update schedule.

**Repair:** add one compact recipe sentence, plus identify the 350-update native runs as using the original schedule clock. The immutable config records peak `0.001`, minimum `0.0001`, warmup fraction `0.01`, and `gpt_neox_v1_pre_step` semantics; verify the actual warmup/decay step endpoints in the runner or retained LR series before stating their rounded update numbers. The source observations place the LR floor at the original update575 neighborhood and the accelerated floor near466, so do not assume an ordinary cosine reaching its minimum only at712. [Run005 config](../../../../../runs/005-2026-10-04-pythia14m-seed1235/config.json), [schedule observation](../../../../../analyses/004-2026-10-04-lr-schedule-intervention/observations/001-schedule-intervention.md).

## Nonblocking scientific clarifications

### N1. Connect smoothness qualifications to the actual architectures

**Anchors:** `sections/03-methods.tex:223`, `sections/04-gd-experiments.tex:44`, `sections/07-gd-diagnostics.tex:5–18`.

“Smooth full-batch GD” correctly qualifies the identity, and the Adam appendix explicitly mentions activation boundaries. However, the measurement appendix does not tell the reader that ResNet and the retained Transformer use ReLU. A reader can still miss why the exact smooth-path interpretation applies directly to the tanh MLP but requires qualification for the other two cases.

**Repair:** add one measurement-scope sentence: the latter two models use ReLU, so endpoint projected changes are measured directly and are not asserted to be exact classical-Hessian path averages across activation boundaries. No new HVP comparison is required for the paper's diagnostic claim.

### N2. Separate sign-of-approximation from reliable nonlinear loss-sign inference

**Anchor:** `sections/03-methods.tex:325–330`, “crossing two predicts the loss sign only under the stated approximation.”

This is substantially improved and is not false. One further sentence would prevent readers from treating a crossing near two as meaningful merely because the objective is smooth: the inferred sign is reliable only when the predicted change dominates the approximation error. The appendix bounds the order of the remainder but no empirical neural sign-accuracy claim is made. Preserve that distinction.

### N3. Use direct language for the observed effect

**Anchors:** `bare_jrnl.tex:441`, “smoother alignment”; `sections/08-implications.tex:104–105`, “smoother raw-gradient history.”

The main table measures more-positive cosine and changed loss. Smoothness usually means smaller temporal variation, a different statistic. The source has variation measurements, but they are not reported in that table. “More positive alignment” is the clearest supported shorthand. Similarly, “smaller norms” should identify raw, clipped-input, or parameter norms if retained in the abstract.

### N4. One grammar correction in the principal GD numbers

**Anchor:** `sections/04-gd-experiments.tex:48–49`, “mean scaled secants from 1.97 and 1.99.”

Replace with “mean scaled secants range from 1.97 to 1.99.” This is a readable statement of a range; it should not be interpreted as a fixed equality with the mean norm ratio.

## Contribution stress test

The strengthened claim succeeds when its parts are kept together:

- The `C/D` and zero-cross-term quantities answer a question that adjacent cosine alone cannot answer: where path length disappears and whether all-pair interaction is positive or negative.
- The nine GD trajectories supply a concrete separation between negative adjacent interaction, positive complete-path interaction and observed objective progress. This is more informative than another negative-cosine plot.
- Adam's 794 negative gradient/increment products exclude a particular explanation of the observed rise intervals: that the update simply points uphill at the starting states. The exact loss-minus-linear accounting explains what remains unresolved without claiming a causal mechanism.
- The Pythia and warm-state interventions show why the preceding distinctions affect conclusions: an altered angle, peak or endpoint can disagree with another specified objective outcome.

The remaining weakness is that several headline lessons are familiar individually. The manuscript must make the **joint diagnosis** easy to recognize. The new final subsection in Section08 largely accomplishes this. A concise pointer there to `C`, `D`, `Q`, current-gradient/update alignment and the chosen loss horizon would make the reader's resulting workflow still more concrete. Do not add an optimizer proposal, universal alarm, or a new experiment solely to compensate for modest theoretical novelty.

## Checks that passed

- Main displacement, paired factorization, zero-reference and secant formulas are consistent, including degenerate denominator handling.
- The stochastic noise expectation is appropriate under the stated fresh independent sampling; the same-state cosine approximation is not misused as a negative-alignment reference.
- Adam gradient, moment, state and outgoing-step clocks are consistent. A rise interval is not described as 794 individual positive loss differences.
- The exact finite-step loss integral has the correct weighting and is restricted to smooth objectives; the measured residual remains separately defined for nonsmooth cases.
- The memory recurrence, finite-time alternating expression, discrepancy decomposition and bound remain algebraically consistent under their assumptions.
- The title, abstract, introduction and conclusion now agree on a diagnostic contribution rather than a general stability theorem, new secant estimator or validated spike predictor.
- Event dependence, one-seed full-batch scope, multiple-seed SGD/Pythia scope, selected intervention windows and distinct outcome horizons are disclosed.

No new neural experiment is required to repair any issue identified in this round. Final evidence-table checks and PDF review remain necessary before treating this as the completed submission artifact.
