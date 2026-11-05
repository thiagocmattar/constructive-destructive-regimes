# Reviewer assessment: significance and diagnostic usefulness

5 November 2026. Independent skeptical TNNLS/TPAMI-style assessment. This is a recommendation, not an execution authorization or an acceptance forecast. No manuscript text, scientific inputs, registry, or training resource was changed.

## Judgment before requesting more work

I would still have a substantial significance objection. The revised paper is careful and its main distinctions are valid, but correctness is not yet a compelling answer to "what does this let us understand that close prior work does not?" The norm identity, secant, productive oscillations, and first-order-versus-finite-step distinction are established ingredients. Repeating the current sweeps with more seeds would strengthen confidence without resolving this objection.

The best route is a compact demonstration in which the diagnostics distinguish competing explanations of a concrete observation, followed by the smallest missing control that could overturn that explanation. It need not predict spikes, improve an optimizer, or discover a new instability law.

The literature comparison is consequential. Singh et al.'s [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37771cc0be272368102a37f202bb88d8-Abstract-Conference.html) already studies macro/micro trajectory directionality and oscillations across vision and language tasks. Bai et al.'s [ICML 2026 paper](https://proceedings.mlr.press/v306/bai26o.html) already offers an Adam second-moment mechanism and directional-curvature diagnostic. These primary sources were checked for this review. Neither an additional zigzag plot nor a generic Adam moment ablation would, by itself, establish a distinctive advance here.

The repository contains more of the relevant experiment than the manuscript presents. In particular, [Analysis025](../../../../analyses/025-2026-10-17-pythia-descent-mechanism/REPORT.md) has 81 same-state probes with seven candidates each, not only the 18 sustained branches. Its candidates already remove coordinate scaling, replace the moment with the current input, remove both, and change native step length. [Analysis027](../../../../analyses/027-2026-10-30-minibatch-noise/REPORT.md) already validates noise-corrected secants against independently computed same-objective HVPs. [Analysis040](../../../../analyses/040-2026-11-03-mlp-step-actions/REPORT.md) already compares raw-gradient clipping with proposal clipping. A proposal to perform those comparisons again is not a high-return experiment.

## Ranked recommendations

### 1. Complete a falsification audit of the existing same-state diagnostic evidence

**Objection:** The practical payoff could be a collection of selected counterexamples. Does the diagnostic explain a repeated distinction across the retained cohort, or only the chosen update-200 example?

**Already known:** Analysis025 reports that a norm-matched unpreconditioned direction can have a better linear prediction but a much worse realized probe loss. The example at 70M/update200 is striking. It also reports harm before the motivating interval, and all 18 sustained ablations worsen their endpoint. This is much stronger evidence than generic trajectory correlations, but the current paper mainly displays the latter sustained result.

**Minimal test:** Re-reduce all 81 states and all seven candidates without filtering on loss-rise events. For each size and seed, report paired differences versus reconstructed native Adam in (a) first-order decrease, (b) finite-step decrease, and (c) the residual separating them. Tabulate how often improving the linear projection reverses the observed finite-step ranking; show every preselected time, including 100/150 and 300/350. Separately report the native half/double-step responses. Check actual applied norms, decay contributions, and native-versus-reconstructed agreement before describing norm-controlled effects. Aggregate within seed before making cohort summaries; candidates and times are dependent measurements.

The discriminating question is **whether the conclusion survives outside the illustrative state and what competing explanation the magnitude controls leave open**. Do not count the identity D=P-R itself as a successful empirical test. Do not claim that the controls already retune the unpreconditioned direction: they change native Adam's length only.

**If positive:** The paper can substantiate a repeated diagnostic distinction: a seemingly better first-order direction fails at the realized scale, and coordinate transformation changes the finite-step penalty. This supplies a clear case of explanatory usefulness, including its boundary across states.

**If negative/heterogeneous:** Restrict the mechanism to its measured states and explain the variation. Do not buy a larger campaign to restore the preferred story. Heterogeneity may identify the one missing control in recommendation 2.

**Return and novelty:** Highest return per effort; zero new training. It establishes scope and practical explanatory value rather than mathematical novelty. It is a new complete-cohort test/reduction, not a new dataset or independent replication. Existing HVP calibration can be audited alongside it if needed, but it must not be presented as a new experiment.

**Stop rule:** If the entire advertised discriminator is already tabulated across every state in the retained outputs, verification plus an evidence map is enough; no duplicate reanalysis or new registry is needed merely to produce another figure.

### 2. Test direction versus step length at the same retained Adam states

**Objection:** Removing preconditioning at the native Adam norm may intentionally give an unpreconditioned direction an unsuitable scalar step. Does the diagnostic locate the failure at the realized radius, or is the paper implying an intrinsically inferior direction?

**Already known:** Analysis025 controls adaptive-component norm and includes half/double *native Adam* steps. It does not provide a common dose-response curve for each substituted direction. Its own report explicitly leaves retuned scalar momentum open. The new control must address this gap, not repeat the existing seven candidates.

**Minimal distinguishing design:** Use the retained post-update175 and post-update225 parents: three sizes by three seed blocks by two states gives 18 anchors. At each anchor compute the original next training gradient and updated moment state once, then compare four directions: native adaptive direction, unpreconditioned moment, preconditioned current clipped input, and current clipped input. For each use adaptive lengths {1/8,1/4,1/2,1,2} times the native length. This is 360 counterfactual endpoint evaluations plus numerical controls, with no continued training beyond obtaining the shared next-step ingredients. Use the identical fixed probe and loss reduction from Analysis025, common decay displacement, exact parameter restoration, and explicit rounding controls. Check checkpoint/gradient clocks: a saved state after225 is not automatically the old pre-update225 probe.

This is a diagnostic response surface, **not a proposed optimizer or an evaluated learning-rate selector**. Any best dose selected on the probe is an oracle descriptive minimum over that fixed grid, not out-of-sample method performance. If the minimizing dose lies at an endpoint, report unresolved range rather than expanding adaptively until a desired ranking appears.

**If a smaller unpreconditioned step recovers or beats native loss:** The original large penalty was conditional on using native Adam's scale; the diagnostic identifies a direction/length mismatch. This would be a useful, sharper scientific finding even though it weakens a broad preconditioning-superiority reading.

**If native scaling retains an advantage across the fixed range:** This supports a stronger local directional-efficiency conclusion on those states. It still does not compare converged or independently tuned momentum training.

**Return and novelty:** Best bounded *new measurement* for the current optimizer-aware argument. It distinguishes explanations using the existing three-seed, three-size cohort and avoids the previous plan's large fresh training campaign. The finite-step idea is established; the controlled empirical discrimination is the contribution.

**Cost/scope traps:** Count full probe-forward passes, restored parameters and transfers; "360 evaluations" is not a measured runtime estimate. Check retained parents before promising replay. Do not turn the test into an LR search, longer training contest, or universal Adam mechanism claim.

### 3. A small parameter-to-function cancellation check, only if functional relevance remains a central objection

**Objection:** A coordinate-dependent parameter-path statistic may diagnose parameter arithmetic without saying whether the model's predictions are repeatedly undone. The present paper correctly disclaims this implication, but that also limits its significance.

**Already known:** Parameter C/D/R and actual-step measurements are thorough. They do not measure cancellation of changes in outputs on fixed inputs. Loss alone cannot supply that missing vector-valued observation.

**Minimal distinguishing design:** Replay a short, fixed 64-update window from already retained trained GD checkpoints for the MLP and one ReLU model, using the existing three initialization blocks where they qualify. Evaluate a fixed, declared probe at every state. Apply the same exact cancellation accounting to successive output changes, separately from parameter increments. For the squared-error MLP use its actual output vector; for cross-entropy use class-centered logits to remove the irrelevant additive logit offset. This representation choice is a new measurement decision and must be explicit, with no cross-model pooling. Use a fixed probe selected independently of observed cancellation, equal weighting, deterministic evaluation, and an identity test that the increments telescope. Both pair offsets and a fixed two/four/eight-update blocking sensitivity are enough; no search for the most favorable scale.

**If local cancellation and coherent residual accumulation also appear in the output stream:** The paper gains a bounded connection to repeated changes in model behavior, while retaining a clear distinction from loss progress and generalization.

**If they disappear or reverse:** The limitation becomes empirically demonstrated: large parameter reversal can correspond to different functional motion. That outcome is scientifically useful but may force a narrower significance argument.

**Return and novelty:** Potentially more conceptually valuable than another seed sweep, but lower immediate return than recommendations 1/2 because it adds a new measurement space and must be positioned carefully. This is a diagnostic bridge, not evidence of forgotten knowledge or harmful interference.

**Cost/scope traps:** Existing scalar logs cannot recover intermediate outputs. Use a short authenticated replay, not reconstructed historical gradients assumed exact. The probe does not establish population-level behavior. Avoid Transformer full-vocabulary logit storage in the first test. Do not make this a new representational-learning project.

### 4. Targeted independent-seed Adam replication, after the diagnosis is fixed

**Objection:** The event anatomy and universally descending rise-interval directions use one initialization per full-batch model. Hundreds of events cannot replace independent initializations.

**Already known:** SGD and Pythia have three blocks. Some GD reference streams are also already replicated in Analyses027/037; they should be audited before asking for new GD seeds. None of these substitutes for the particular full-batch Adam claim.

**Minimal design:** Freeze the existing highest-rate Adam recipe and loss-only event definitions; add two initialization seeds for each of the three models, six 1024-update trajectories. Preserve all ordinary intervals, event thresholds and censoring. Main outcomes should be run-level incidence/occupancy, the fraction of rise-interval actual steps with negative current-gradient dot product, and norm/cosine phase summaries. No severity prediction fits, post hoc LR adjustment to obtain spikes, or new HVP program is needed. If a model has no qualifying events, that is informative coverage.

**If replicated:** The key observed Adam distinction is less plausibly an initialization accident. **If not:** Replace absolute statements with heterogeneous run-level findings and characterize the exception. New seeds cannot make an incorrect general statement correct by majority vote.

**Return and novelty:** High robustness value, low novelty value. Six targeted trajectories have a clearer return than the prior 27-trajectory plan, which addresses several claims the current paper does not make. Numerical environment consistency and recovery-state retention still need execution preparation.

## What I would not request

- A bigger model solely to reproduce negative cosine, an additional lag atlas, or a fresh validation that R=C D algebraically.
- Spike classifiers, more severity regressions, or held-out forecasting competitions: these do not match the scientific question.
- Activation sparsity, orthogonalized-step methods, or a diagnostic-driven controller: each creates a new methods paper and a substantial tuning burden.
- Another generic moment/preconditioner ablation before accounting for Analysis025's 567 retained one-step records.
- A 27-arm replication/HVP campaign before establishing whether a small missing direction/length control changes the explanation.

## Priority decision

First finish the complete-cohort diagnostic test with existing evidence. The largest conceptual upside is the parameter-to-function transfer test; it also has the greatest risk of widening the paper. Within the current optimizer-aware diagnostic scope, I favor the common-state direction/length response surface only if the audit shows that the native-radius ambiguity is consequential to the intended claim. It can distinguish an explanation, uses replicated existing parents, and has a useful outcome in either direction. Another reviewer could reasonably rate this as a textbook response surface with little claim upgrade; if the paper is content with its existing explicitly native-scale conclusion, I agree the new sweep is unnecessary and independent Adam seeds become the next investment. Keep function-space transfer as a higher-upside optional extension and targeted Adam seeds as a robustness investment. No amount of additional computation guarantees that a novelty-focused reviewer will be persuaded; the relevant question is whether the evidence answers a sharper scientific question rather than making the same plot more densely populated.
