# Research program for temporal geometry and training interventions

Owner: the coordinating agent, with scientific direction and final design decisions retained by the author. Updated 3 November 2026, America/Sao_Paulo.

The author has expanded the objective from a minimum closing revision to a stronger research contribution, including new experiments where they provide better evidence. The earlier [evidence consolidation](../2026-11-03-paper-convergence/README.md) remains a useful inventory. This program replaces its conservative ordering as the forward plan. Existing results guide measurement and controls; they do not limit the quality or scope of the new study.

The proposed paper investigates **when temporal gradient geometry makes curvature information useful for the next adaptive step, and whether that information improves the tradeoff between loss excursions, progress and compute**. The target contribution is a tested account of observability and intervention. Acceptance and the absence of future novelty criticism are editorial outcomes; the plan targets the specific weaknesses that a strong reviewer could test.

## The scientific bet

There are three connected questions:

1. **What is oscillating?** Can short histories distinguish bounded alternating motion, a growing oscillation envelope and slow net progress more informatively than adjacent cosine and gradient norm alone?
2. **When is old curvature useful?** Can history and the current adaptive direction identify when a secant observation from a completed step is informative about the next Adam step?
3. **Does acting on that information help?** Can an intervention preserve productive motion and improve the loss/cost tradeoff compared with shrinking steps, norm-based controls and the closest adaptive-step methods?

These are hypotheses to test. The current manuscript establishes neither a reliable transport criterion nor a validated policy. The longer-lag observations are important motivation: they expose persistent alternation and rebound beyond the adjacent pair. Their next role is to explain or improve a measurable decision, rather than to supply more striking curves.

A possible working title is *Temporal Gradient Geometry for Curvature Estimation and Training Interventions*. Title and abstract should follow the winning result. The current draft stays intact while the new evidence is developed.

## Contributions that would distinguish the paper

| Candidate contribution | What must actually be demonstrated | What would weaken it |
| --- | --- | --- |
| A temporal account of productive and growing oscillation | New controlled trajectories connect norm-weighted temporal geometry to subsequent objective progress and independently defined envelope growth. | A strong norm/history baseline explains the same behavior, or the apparent effect is only phase alternation. |
| An account of when secant curvature can inform Adam | Held-seed measurements show which current-direction/history features predict curvature or action error beyond a last-secant and current-pair comparator. | Direction rotation alone explains everything; longer history adds cost with no consistent value. |
| A useful intervention based on that account | Same-action trigger tests and independent whole trajectories show a favorable excursion/progress/cost tradeoff against strong controls. | Smaller native steps reproduce the gain, or suppressed peaks come with worse equal-budget progress. |

The paper need not win all three hypotheses. A strong negative result can delimit when cheap curvature is usable. It must not be sold as a successful controller if the intervention comparison fails. A changed scientific outcome leads to a deliberate claim decision, not an endless search for a favorable threshold.

## What is already known and what remains open

The “optimal step from rho” chat is scientifically useful. For current raw gradient `g`, Adam direction `d`, preceding applied step `s`, and raw-gradient difference `y`,

\[
\widehat\kappa_{\rm sec}=\frac{s^Ty}{s^Ts},\qquad
\widehat\alpha=\frac{g^Td}{\widehat\kappa_{\rm sec}\|d\|^2}
=\underbrace{\frac{s^Ts}{s^Ty}}_{\rm BB1}\frac{g^Td}{\|d\|^2}.
\]

For smooth full-objective gradients, the first quantity is an exact curvature average along the completed step. The second transports that average to a different direction/state. Its current-direction quadratic reference is `alpha_star = (g^T d)/(d^T H d)`, when slope and directional curvature are positive. This is a one-dimensional step estimate, rather than a reconstruction of the full Newton update.

The missing information matters even on a quadratic. For `f_M(x)=x1^2/2 + M*x2^2/2 + x2`, the observed segment `(a,0) -> (0,0)` and its endpoint gradients are identical for every M. The historical secant is one, while the current second-coordinate optimum is `1/M`. A history of steps that never explores that direction cannot identify its curvature.

With a fixed Hessian and past steps/differences `S,Y`, `HS=Y`. Writing the current direction as `d=Sa+r`, with `r` perpendicular to the history span,

\[
d^THd=a^TS^TYa+2a^TY^Tr+r^THr.
\]

The last term is unobserved. Long almost-alternating history can have nearly rank-one span. Nonlinear curvature drift and stochastic endpoint noise add uncertainty even within the span. These are elementary facts that motivate a new empirical question; they are not presented as newly invented quasi-Newton theory.

The [mathematical review](reviews/control.md) derives a direction-rotation/path-drift bound and explains its assumptions. Unknown Hessian bounds prevent calling a cosine-only criterion certified. [The NumPy preflight](mathematical_preflight.py) validates algebra and counterexamples; it supplies no neural-network result.

## Literature boundaries

| Prior work | Consequence for the new claim |
| --- | --- |
| [Barzilai–Borwein](https://doi.org/10.1093/imanum/8.1.141), limited-memory quasi-Newton and multisecant methods | Secant curvature, storing S/Y and using it to approximate Newton are established. |
| [Directional Smoothness](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1ac83203e88eb6cf6b30642f0239b932-Abstract-Conference.html) | Path-dependent curvature/smoothness and adaptive gradient steps already have theory. |
| [Bai et al., ICML 2026](https://proceedings.mlr.press/v306/bai26o.html) | Adam spike mechanisms, second-moment decoupling and gradient-direction curvature predictors overlap directly with the old broad story. |
| [GALA](https://arxiv.org/html/2506.08419v1), [Adam Algorithm with Step Adaptation](https://www.mdpi.com/1999-4893/18/5/268), [adaptive BB training](https://arxiv.org/abs/2205.13711) | A scalar curvature/alignment LR formula is insufficient novelty. Compare faithful numerical conventions and evaluation costs. The last source has a 2022 arXiv posting and a DICTA 2019 publication record. |
| [Greedy Alignment Principle](https://arxiv.org/html/2512.06370v3) | Gradient autocorrelation already informs optimizer/momentum selection. A first-use-of-history claim would be indefensible. |
| [Adam Rod Flow](https://arxiv.org/html/2605.06821v1) | Slow motion and oscillatory extent have close existing structure; merely renaming a center/envelope decomposition adds little. |
| [SPAM](https://arxiv.org/abs/2501.06842) and [catapults](https://proceedings.mlr.press/v235/zhu24h.html) | Clipping/reset methods and potentially useful spikes are established. Count subsequent training and validation progress. |

The [novelty review](reviews/novelty.md) gives primary sources, boundaries and experiment suggestions. It is a focused overlap audit, not proof that no other relevant method exists. Before claiming a competitive optimizer, read and reproduce the closest selected baseline's full specification and code. H_ADAM publisher retrieval was rate-limited; its indexed algorithm is enough to identify overlap, not to certify a faithful implementation.

## The new experiment sequence

The [execution specification](EXECUTION.md) and [experiment matrix](experiment_matrix.csv) make the comparisons concrete. Each stage releases the next stage only when its measurement and scientific question are clear.

| Stage | Question and new evidence | Exit decision |
| --- | --- | --- |
| E0 Design validation | Analytic/synthetic secant identities, direction-transport failure and bounded-cycle controls. | Completed in this planning task; mathematical preflight passes. |
| E1 Geometry and action calibration | Fresh predetermined trajectories/windows; ordinary states plus stress conditions; current-direction HVP checks and frozen-state scalar loss-response grids. | Determine whether the secant estimate, direction coherence or added history predicts a useful action. |
| E2 Matched intervention pilot | Original eight Analysis038 arms plus a cap-only transported-secant reference, from identical complete optimizer states. Include both selected-event and ordinary-state checks. | Identify action effects and progress costs; do not infer online timing efficacy. |
| E3 Intervention timing | Same action triggered by norm, loss, current-pair and the selected temporal criterion; compare intervention budgets fixed on development data. | Establish whether the diagnostic adds useful timing beyond simpler controls. |
| E4 Independent confirmation | Freeze one proposed method, strong baselines and all thresholds; new complete trajectories, including ordinary/no-spike runs and stochastic AdamW. | Support or reject the final practical claim at equal tokens and measured compute. |
| E5 Cost and manuscript | Measure minimal and history-enabled capture separately; rewrite the compact paper around the actual winning result. | Freeze claims, figures, reproducibility package and submission route. |

E1 and the first E2 mechanics can use the owned MLP GPU. That is an inexpensive place to validate state restoration and test more initializations. It is not a substitute for independently collected ResNet/Transformer and realistic AdamW confirmation.

## Temporal measurements that earn their cost

Record the raw gradient, clipped input, stored moments, adaptive proposal and actual applied increment separately. A completed-state clock must identify what is available before the next update. Record loss and scalar norms at every decision state.

The minimal history probe includes lag-one alignment, adjacent norm growth, same-parity norm growth `||g_t||/||g_(t-2)||`, and the norms of `(g_t+g_(t-1))/2` and `(g_t-g_(t-1))/2`. These pair quantities follow from the existing adjacent inner product and norms; some add only scalar history. They become slow/oscillatory interpretations only after validation. Test them against a comparably expressive baseline using recent norms and losses, rather than assigning an advantage to a nonlinear feature map by construction.

For a selected richer test, use exact histories K=2/4/8/16 to measure direction coverage, conditioning/rank and secant consistency. Keep raw-gradient and actual-step histories distinct. Analyze a small number of predeclared long-lag windows, up to 256, to connect to the original observation. Use common endpoints/horizons and independent initializations; an anchored curve is not a stationary autocorrelation estimate. The initial exact-history implementation can be an offline scientific probe. It does not justify a one-buffer online cost claim for a method that needs many vectors.

A necessary control is a bounded period-two trajectory: normalized-gradient motion on `x^2/2` cycles between 0.25 and -0.75 at step one. Adjacent reversal plus norm growth repeatedly fires, while the two-step envelope stays constant. Its loss still has >10% recovered excursions, so it is a control for progressive envelope/runaway claims, not a false positive for the existing excursion detector. In neural models, select ordinary bounded-oscillation windows using loss criteria independent of the proposed diagnostic and report their later progress.

## What would count as a convincing result

The key result is a **new independent comparison**, not a larger retrospective R-squared. For example: a frozen temporal criterion identifies when a secant step is unreliable beyond current direction rotation alone, and using that criterion reduces harmful loss cost while preserving or improving equal-budget progress relative to strong norm/step controls.

Keep four targets separate: curvature-estimation error; one-step scalar action quality; event prediction; and whole-trajectory policy utility. A gain in one is not automatically a gain in the others. Treat the seed/trajectory as the replication unit. Use six unseen seeds for the inexpensive primary controlled confirmation where feasible, with three-seed transfer initially descriptive; expand the primary practical cohort if variability and budget require it. Three events or repeated spikes from one seed do not constitute independent confirmation.

Use six-seed coverage as an initial design target, not a power calculation. Choose a primary outcome and any noninferiority margin before the test cohort, using development variability to assess whether more replication is needed. Report paired seed outcomes and uncertainty; preserve failures, divergent runs, censored events and no-spike histories. Do not replace an unfavorable seed or choose the best window from confirmation.

If gains come entirely from step scale, say so and retain the best simple controller as the comparator. If history improves geometric calibration but not training, the contribution is a reliability/diagnostic result. If the added history fails to improve either, keep the temporal observations as supporting characterization and reconsider the method before spending on a broader campaign.

## Budget and calendar gates

At 11:59 Sao Paulo on 3 November, a read-only Runpod API query returned **US$23.6127** credit. The account also reports US$0.016/hour unrelated existing spend, which this task will preserve. Two stopped resources were left untouched. Current stock is scarce: the quoted 4090/5090 entries were unavailable, while A40 and PRO4500 SE had low stock. Catalog observations are in [account_preflight.json](account_preflight.json); requery before allocation.

| Funding scenario | Proposed cumulative task ceiling | Scientific purchase |
| --- | ---: | --- |
| Existing balance only | US$20, leaving about US$3.61 reserve at the snapshot | Owned-GPU development, bounded paid qualification and selected matched exploratory cloud cases. |
| Author later adds US$50 | US$65, leaving about US$8.61 against the original balance plus deposit | Independent confirmation of the surviving claim and faithful key comparator, subject to measured runtime. |
| Author later adds US$100 | US$110, leaving about US$13.61 | Stronger seed coverage plus realistic AdamW confirmation, with subcaps after calibration. |

The extra deposits are scenarios, not money already authorized for immediate spending. Budget additional cloud funding only after the first result identifies the unresolved claim. No paid resource was created in this planning task.

Historical 4090 times imply about US$5.33 ordinary compute for the two paid cases of a nine-arm pilot, plus local MLP; Armijo, setup, new capture and transfers are additional. This is an arithmetic scenario on historical throughput and a presently unavailable GPU, not a live completion quote. The first paid stage has a US$20 aggregate ceiling, at most two workers, a 12-hour wall bound and retrieval headroom. Calibration must decide whether the exact selected design fits current hardware. The [feasibility review](reviews/feasibility.md) shows the arithmetic and alternatives.

The first actionable packet is a new controlled MLP development pilot, specified in [pilot_plan.json](pilot_plan.json): nine arms, one exploratory seed, 657 prefix plus 1,152 branch updates, zero cloud charge. Historical ordinary arithmetic is tens of seconds; allow a provisional 5–15 minute local mechanics/calibration window, with a hard 30-minute local bound. The modified method's runtime has not been measured. This packet is a design specification; the production driver and measured launch readiness remain explicit tasks.

## Execution ownership and immediate work

The coordinating agent owns the decision ledger, integration, code review, budget accounting and final evidence map. Three delegated reviews covered novelty/temporal geometry, mathematical/control design and execution feasibility. Each has reported sources and limitations. No new user-owned chat or cloud job was needed to obtain those reviews.

The next implementation belongs in a new numbered run, never in executed Run215. Reuse verified model/objective code while adding: a pre-update observer; immutable fork states; explicit moment-safe interventions; exact selected temporal products; and rollback/cost accounting. Focused tests must cover clock availability, independent moment recurrences, clipping placement, beta2 history correction, actual increments, rollback and resume. A mechanics smoke and timing calibration precede production. Details and the proposed run plan are in EXECUTION.md.

The current request authorizes planning, delegation and preparation, and provides an available credit envelope. The chosen scientific comparison still needs the explicit run-plan confirmation required by the user's operating guide before training. There is no need to reconfirm already authorized reading, tests or reversible preparation. This planning turn completes those independent tasks and leaves a concrete first-run specification for that decision.

## Manuscript structure after the new evidence

1. Practical question: recognize and control costly excursions without suppressing productive oscillation.
2. Established secant observation and the precise limits of curvature inference under adaptive motion and sampling.
3. New temporal characterization and independent calibration of its useful information.
4. Same-state intervention and controls that isolate timing, direction and scale.
5. Frozen realistic confirmation and actual compute/storage overhead.
6. Implications for schedules, clipping and momentum, with conclusions matched to the results.

The current GD/SGD evidence remains valuable support. Existing Pythia data can motivate the practical setting and provide development context, while new cleanly controlled trajectories supply the claims that require independence or new instrumentation. Keep the full research record; promote only the comparisons that advance the chosen argument. TNNLS is a natural initial target for the optimization/learning-system account; TPAMI becomes a stronger target if the new practical validation and novelty comparison support the broader case. Verify current submission and prior-rejection conditions when the final claim is frozen.

## Research iteration conclusions

- The ambition has expanded: new evidence quality takes precedence over minimizing the number of runs or preserving every old figure.
- The scalar “optimal Adam step” has a valid secant interpretation and close prior work; a new formula alone is insufficient as the novelty anchor.
- Long-lag persistence is interesting, but normalized cosine omits amplitude and can add little independent direction coverage.
- Curvature transport across changing Adam directions is a concrete, falsifiable question connecting the temporal and intervention ideas.
- Bounded recovered excursions, growing envelopes and lasting loss harm need distinct outcomes and controls.
- The local MLP permits broad development and independent seeds at very low compute cost; cloud credit should purchase missing scientific information.
- The existing credit can support a bounded exploratory stage; extra US$50/100 scenarios should purchase confirmation of the surviving claim.
- The next paper should follow the strongest independently tested result, with acceptance claims replaced by explicit reviewer-facing comparisons.
