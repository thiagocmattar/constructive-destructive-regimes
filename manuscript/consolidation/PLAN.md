# Paper anatomy and closure map

- **Central argument:** learning updates reinforce or reverse earlier changes. Their direction, amplification and accumulation connect optimization geometry to changes in the learned function; measuring those changes motivates selective control.
- **Keep the current paper:** title, Sections I–VI, III's theory spine and four GD/SGD figures. Develop **characterization → tracing → intervention** inside V.
- **Decision from this review:** the evidence supports this explanatory paper. Validated forecasting, a universal initiating layer and general optimizer superiority require additional evidence.
- **Reading contract:** this is the complete executive map. The current revision follows the author's request to stay close to `manuscript/draft/bare_jrnl.pdf`, explain complex ideas fully, and place numerical detail in figures/tables. Links provide provenance, not additional required reading. The [complete manuscript rewrite](draft/manuscript.pdf) and [supplement](draft/archive/2026-11-18/supplementary.pdf) are now ready for author review inside consolidation; transfer to the active manuscript remains a later step.
- **Local assets:** [section-indexed evidence package](evidence/README.md); [display/table/algorithm map](evidence/ARTIFACTS.md).
- **Figure convention, 14 November:** training-progress axes use **Training step t = completed optimizer updates**. Incoming representation changes are placed at their endpoint; lag/batch axes retain their units. The complete draft assigns Figures **1–4 to GD/SGD, 5 to Adam characterization, 6–7 to tracing, and 8–9 to interventions**. Supplementary Figures S1–S24 retain supporting comparisons. Current publication variants are in `draft/figures/`; preserved evidence PDFs may retain the former labels. Numerical arrays, event windows and reductions are unchanged.

## I. Introduction — motivation before gaps

- **Opening sequence, following the current introduction:**
  - Longer training horizons and constrained compute make excursions and hyperparameter sensitivity consequential. Retain the scaling/efficiency motivation; the present experiments quantify dynamics.
  - Oscillatory trajectories can keep learning. Reducing the learning rate can suppress an excursion while slowing subsequent progress; removing reversals can also hurt.
  - Curvature describes geometric risk. The practical question is **what changes in the learned function, when that change becomes excessive, and which part should be controlled**.
  - Introduce the route: gradient interaction → inexpensive curvature/noise interpretation → functional tracing → controlled intervention. Avoid unmeasured claims about forgetting, fractal landscapes or compute savings.

- **Contribution 1 — connect local reversal to cumulative progress.**
  - **Known:** [EoS](https://arxiv.org/pdf/2103.00065) establishes oscillation with progress; [Chee–Toulis](https://proceedings.mlr.press/v84/chee18a.html) and [Pesme](https://proceedings.mlr.press/v119/pesme20a.html) use gradient products for stationarity.
  - **Gap addressed:** a local sign does not explain how successive interactions accumulate during learning.
  - **Response/test:** constructive/destructive interpretation, exact all-pair displacement and normalized displacement R; nine GD cases have positive total pair interaction, seven have negative adjacent interaction. **III opening/D → IV Figures 2–3.**

- **Contribution 2 — recover directional curvature from existing gradients.**
  - **Known:** spectral stability analyses characterize sharp directions using Hessian eigenvalues.
  - **Gap addressed:** obtain curvature along the actual training step from quantities already available to the optimizer, with a precise geometric interpretation and explicit sampling error.
  - **Response/test:** introduce the actual-step estimate for any optimizer, including Adam; derive its smooth path-average identity, pointwise error bound and explicit noise term. The SGD specialization links it to alignment/norms; **1,188 directional-HVP pairs in a tanh MLP** provide an independent derivative reference. **III-A/B/C → IV Figures 1/4 and validation.**

- **Contribution 3 — distinguish reversal, amplification and loss consequences.**
  - **Known:** [Bai et al., §§4–5](https://arxiv.org/html/2506.04805v2#S4) relates Adam spikes to preconditioner dynamics.
  - **Gap addressed:** gradient reversal alone does not determine a finite-step loss change or identify what must be damped.
  - **Response/test:** explicit GD loss condition and optimizer-dependent response interpretation; exhaustive Adam characterization separates direction, growth, severity and timing. **III-E → IV checks → V-A.**

- **Contribution 4 — trace and control functional movement.**
  - **Known:** [Catapults](https://proceedings.mlr.press/v235/zhu24h.html) connects output/NTK dynamics and feature learning; representation-aware update methods already exist.
  - **Gap addressed:** locate which stages participate in an observed excursion and test whether selective movement control preserves progress.
  - **Response/test:** tracing on the same reference examples, architecture-specific participation and the distributed q50 affine-displacement cap. The completed 36-condition comparison gives q50 mean-loss changes **-32.18%/-9.42%/-19.84%** for MLP/ResNet/Transformer, with explicit spike, endpoint and held-out tradeoffs. Compare LR → beta2 → clipping → cap from the same starting state in each model. **V-B/C.**

## II. Stability Analysis of SGD — position each contribution precisely

- **Retain four subsections:** local curvature; minibatch stochasticity; curvature/noise/EoS; practical large-model stability.
- **Temporal diagnostics:** credit [hypergradient adaptation](https://arxiv.org/abs/1703.04782) and gradient-product stationarity tests. Our contribution is the connected interpretation and empirical tests; cosine and the displacement identity are established ingredients.
- **Adam:** Bai's boundary concerns **preconditioned curvature**, including momentum conventions. Raw-gradient response alone does not establish preconditioner decoupling.
- **Tracing:** [SVCCA, §2](https://proceedings.neurips.cc/paper/2017/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf) and [CKA](https://proceedings.mlr.press/v97/kornblith19a.html) study representation similarity. Specify our object: signed **consecutive changes**, matched examples, event timing and stage participation.
- **Closest cap comparisons:**
  - [Kosson, §6/Appendix B.6](https://arxiv.org/html/2410.23922v1#S6): layer-scale correction of **relative** representation change; ours directly caps absolute per-unit affine RMS.
  - [AGC, §4/Eq. (3)](https://proceedings.mlr.press/v139/brock21a/brock21a.pdf#page=4): unit-wise gradient/weight norm clipping **before** optimizer transformation; ours acts on the actual Adam increment in an input-dependent metric.
  - [FOOF, §4/Eqs. (5–6)](https://proceedings.mlr.press/v162/benzing22a/benzing22a.pdf#page=6): solves for neuron-space descent using input second moments; ours radially truncates the existing proposal, without an inverse.
- **Positioning:** replace “prior work ignores representations” with these concrete differences. Do not claim the first representation-aware optimizer or first unit-wise clipping rule.

## III. Constructive and Destructive Regimes — exact object, assumptions, clock

- **Opening/D:** retain all-pair SGD displacement, R = displacement/path length and longer lags. Negative adjacent cosine describes a destructive step; persistence describes a regime. Cancellation does not by itself measure erased knowledge, wasted compute or generalization.
- **A/B - make cheap curvature estimation prominent:**
  - Retain the original alignment derivation. For plain SGD, **eta*kappa_hat_t = 1 - rho_t*||g_t||/||g_(t-1)||**.
  - Introduce the optimizer-independent version using the applied increment **u_(t-1) = theta_t - theta_(t-1)**: **kappa_hat_step,t = u_(t-1)^T(g_t-g_(t-1))/||u_(t-1)||^2**.
  - Under a smooth full objective, this equals the directional Hessian curvature averaged along that completed step, regardless of the optimizer. Starting-point error is at most **L_H*||u_(t-1)||/2**. Sampling adds **u^T(delta_noise)/||u||^2**.
  - **O(p) reductions and storage; no additional backward pass or HVP.** Retain the preceding gradient and actual increment. This is a scalar directional estimate, not a reconstructed Hessian or largest eigenvalue.
  - **Adam distinction:** existing Adam panels measure **Q_t = 1 - rho_t*||g_t||/||g_(t-1)||**. Their values are unchanged; they characterize reversal and amplification. They are not relabeled as actual-step curvature, which needs the applied Adam increment. The generic identity is derived here; no new Adam curvature experiment is asserted.
- **C — sampling:** retain noise and finite-population variance. [027](evidence/04-gd-sgd/observations/27-REPORT.md) reduces batch-32 median HVP discrepancy **.7467 → .0030** with offline full-gradient noise correction. These are dependent MLP measurements across states/seeds/batches, not 1,188 independent experiments. Less negative noisy cosine does not imply improved loss.
- **E — loss and indexing:**
  - State t means t completed updates, and g_t is evaluated at theta_t. Both two-gradient quantities are available after backward, before t→t+1, and describe the **incoming** step. Use eta*kappa_hat in III–IV; introduce Q separately in V for the raw-gradient combination under Adam.
  - Quadratic GD: **L_t−L_(t−1) = eta*||g_(t−1)||²*(Q_t−2)/2**. A strict rise requires Q_t>2, equivalently rho_t<−1/r_t; this implies reversal and growth, whose conjunction alone is weaker.
  - For nonlinear losses retain the approximation and residual. Outgoing prediction is a separate test; activation boundaries limit smooth expansions.
  - Compute Q directly from the inner product: previous zero norm makes Q undefined; current zero norm gives Q=1 while cosine is undefined. Never turn undefined cosine into a zero observation.

## IV. Empirical Characterization — preserve the four figures

- **Cases:** CIFAR-10 MLP/5,000 examples/half-squared error; CIFAR-10 ResNet-32/50,000 examples/cross-entropy; WikiText-2 Transformer/2,088,600 supervised tokens/token-weighted NLL.
- **Figure 1 — LR, curvature, loss:** [nine GD conditions over 3,000 updates](draft/figures/gd_loss_projected_change_spectral_clean.pdf).
  - **Central message:** existing gradient norms and alignment expose the directional-curvature signature of smooth and oscillatory GD.
  - **Evidence:** oscillatory conditions have late Q near two while loss continues to improve; smooth low-rate MLP/Transformer conditions can have Q near zero despite nonzero leading curvature. The two curvature quantities need not coincide.
  - **Role:** make III's inexpensive directional Hessian estimate concrete. The independent HVP comparison below tests its derivative interpretation; this spectral overlay supplies geometric context.
  - **Read precisely:** both curvature curves use the same centered 101-step smoother, with sparse spectra interpolated before smoothing. Spectra cover the full MLP objective and subsets for ResNet/Transformer. Smoothed crossings do not locate onset or establish a forecast.
- **Figure 2 — reversal with progress:** [norms, alignment and R](draft/figures/gd_reversal_norm_retention.pdf).
  - **Central message:** persistent local reversal can coexist with cumulative movement and optimization progress.
  - **Evidence:** across the three cases, high-rate reversal and small R coexist with lower endpoint loss and greater absolute displacement; read this figure alongside Figure 1 and the endpoint table.
  - **Role:** separate direction, magnitude and accumulation. R measures displacement relative to path length; a small R alone does not measure poor learning. The separate all-pair audit establishes positive total interaction in all nine cases despite negative adjacent interaction in seven; that result is not inferred from R alone.
- **Figure 3 — temporal organization:** [backward gradient alignment](draft/figures/gd_backward_compact.pdf), endpoint 2256, lags 1–256 with a 1–32 detail view.
  - **Central message:** reinforcing and opposing directions recur beyond adjacent steps; lag-one alignment cannot summarize the trajectory's temporal geometry.
  - **Evidence:** a small cosine at one separation can be followed by substantial alignment or opposition at larger lags. The early detail makes sign alternation visible.
  - **Role/boundary:** motivate the all-lag formulation and test lag-one dominance. Each point is one endpoint/lag pair from a recomputed continuation; this is neither an all-pair average nor evidence of monotone decay or repeated traversal of one minimum.
- **Figure 4 — sampling:** [546 SGD/GD continuations](draft/figures/sgd_batch_size_compact.pdf), three initializations, forks at initialization and after 2,000 GD updates.
  - **Central message:** sampled-gradient alignment depends jointly on batch size and the training state from which learning continues.
  - **Evidence:** the smallest batches weaken trained-state reversal in all three cases; at the initial MLP fork, the smallest batch gives negative alignment while full GD gives positive alignment. Sampling does more than uniformly attenuate an existing cosine.
  - **Role/boundary:** connect III's noise analysis to evolving trajectories. Changing the batch also changes the visited states; the frozen-state control below isolates sampling. Shading spans three initialization means, and equal updates entail unequal data exposure. Alignment alone does not rank loss progress.
- **Supporting figure — [frozen sampling calibration](draft/figures/frozen_calibration.pdf):**
  - **Central message:** sampling alone changes the observed gradient geometry at fixed parameters.
  - **Evidence:** across saved MLP states, variance and noise projection match moment predictions within maximum relative discrepancies of **0.50%/0.74%**. The cosine approximation's maximum absolute error is **0.00745**, but its target misses 21/36 stochastic confidence intervals.
  - **Role/boundary:** isolate sampling noise and distinguish exact moment identities from the approximate cosine formula; this control contains no learning trajectory.
- **Supporting figure — [directional-HVP validation](draft/figures/sgd_proxy_validation.pdf):**
  - **Central message:** accounting for realized sampling noise sharply improves agreement with independently measured directional curvature.
  - **Evidence:** batch-32 median discrepancy falls **.7467 → .0030**; all stochastic batches improve. Raw paired markers supply the derivative comparison, while smoothed curves provide context.
  - **Role/boundary:** full gradients make this an offline validation. The 1,188 dependent pairs in one MLP do not establish an online correction or a largest-eigenvalue estimate. [Source and reductions for both validation figures](evidence/04-gd-sgd/observations/27-001-controlled-minibatch-noise.md).
- **Separate finite-step check:** [024's residual audit](evidence/04-gd-sgd/observations/24-007-section-iv-critical-review.md) finds **1,325/2,999 loss-sign disagreements** for the high-LR Transformer. Directional-curvature validation and accuracy of the quadratic loss approximation are separate tests.
- **Closure:** existing evidence suffices. Label single-initialization GD versus replicated minibatch results; no additional GD training is needed.

## V. Raw-Gradient Diagnostics under Adam

- **Opening:** retain the Appendix A identity. For a fixed smooth objective and actual increment u, delta_g = integral H(theta+a*u)u da. Adam chooses u; the previous raw gradient supplies the projection. The same observable transfers as a **mixed Hessian response**, with different semantics from GD curvature or actual-update alignment.
- **Empirical scope:** controlled full-objective Adam trajectories for the three tasks; ResNet/Transformer representation probes cover subsets. These cases do not validate stochastic Adam deployment.

### A. Characterization — distill the timing/severity audit here

- **Figure 5 — Adam excursions:** [three-model high-LR overview](draft/figures/00-high-lr-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf).
  - **Central message:** loss excursions must be characterized through direction, amplification and timing together; reversal alone does not determine their occurrence or size.
  - **Evidence:** the loss, Q, cosine and relative-norm rows show how gradient amplification varies within reversing regimes. Model-specific timing motivates the exhaustive comparison below rather than a universal threshold inferred from selected peaks.
  - **Role:** extend the III-E distinction between reversal and loss increase to Adam, then ask where the motion appears in the learned function. Here Q is a projected raw-gradient response along Adam's actual increment, with the semantics stated at the start of V.
  - **Read precisely:** gray bands are selected using loss alone: filled bands show baseline-to-peak rises of recovered >10% excursions; outlined bands retain unobserved recovery. They are event annotations, not diagnostic alarms. Q's zero-based viewport leaves the retained negative values outside view.
- **Population check:** [055's exhaustive audit](evidence/05-adam/a-characterization/observations/55-004-timing-severity-audit.md) covers **9,207 decisions across nine arms**. Reversal is a broad antecedent; amplification is more selective and differently timed across models. In the three displayed high-LR arms, one-step-lead Q≥2 covers **172/454, 21/95, 81/152** increases and also precedes decreases.
- **Severity comparison:** counts below are signals available before an individual **≥10% relative loss-increase step**, not independent spikes.

| Signal | MLP: 168 rises | ResNet: 12 | Transformer: 26 |
|---|---:|---:|---:|
| rho<0 | 149 | 12 | 26 |
| r>1 | 110 | 11 | 22 |
| Both | 105 | 11 | 22 |
| Q≥2 | 82 | 7 | 22 |

- **Timing:** Transformer Q coverage becomes **22/26 → 25/26 → 26/26** at leads 1→2→3; lead-3 precision is **26/147**, versus base frequency **26/1,008**. MLP's three ≥50% rises escape the current Q flag.
- **Supporting figure — [severity, lead time and warning memory](draft/figures/adam_forward_and_severity.pdf):**
  - **Central message:** coverage depends on the target's size and time horizon; broader coverage must be weighed against false alarms and the target's base frequency.
  - **Severity, page 1:** compare reversal, growth, their conjunction and Q across minimum relative step increases. Read coverage and precision together; rare extreme rises cannot establish a reliable threshold.
  - **Forward time, pages 2–4:** distinguish an individual increase at a fixed future offset, net rise from the decision state, and any rise within a horizon. These answer different questions; a longer search window offers more opportunities for a positive target.
  - **Non-rising states, page 5:** test whether associations persist when the current step is not already rising. This restriction does not guarantee that the trajectory is outside an ongoing excursion.
  - **Memory, page 6:** retaining a recent flag captures more next-step rises but also flags more decreases; each target is counted once. This characterizes a persistent regime, without establishing a high-precision warning system.
- **Claim:** descriptive pre-update association, one seed/model. The angle/norm condition equals Q>2 algebraically; it is not a second validating signal. Forecasting is optional further work.

### B. Tracing — turn the visual pattern into a reproducible anatomical result

- **Algorithm 1:** fixed samples and stochastic context → consecutive activation changes → layerwise cosine/RMS → temporal Gram/SVD → signed coefficients and output association.
- **Dimensions:** D_l is T×P_l; each row flattens all matched examples/tokens/features of one activation change. Do not average samples, center or normalize the primary rows. Raw gradients form their own undifferenced stream.
- **Current fixed-window cases:** both selected figures use the same completed-step windows. MLP/Transformer retain the [original fits](evidence/05-adam/b-tracing/observations/63-005-technical-report-event-window.md); ResNet uses the [fresh larger-event fit](evidence/05-adam/b-tracing/observations/63-008-resnet-major-peak-temporal.md).

| Case / completed steps / probe | Anatomical finding | Output rank95 / fitted rows |
|---|---|---:|
| MLP / 451–501 / all 5,000 examples | First projection reverses; [tanh attenuates transmission](evidence/05-adam/b-tracing/observations/062-development-summary.md); all five learned stages reverse throughout. | 5 / 51 |
| ResNet / 390–450 / intact 1,000-image BN group | Output-like reversal strengthens through residual depth; first/final-stage alignment-series correlation with output is .764/.995. | 20 / 61 |
| Transformer / 553–601 / 64 sequences, 2,240 positions | At completed step 559 (outgoing update 558): embedding cosine +.390, attention output −.672, network output −.920. | 15 / 49 |

- **Support for directional concentration:** [current-window Gram controls](evidence/05-adam/b-tracing/tables/spectral-controls.md) give top-mode energy **53.0/27.3/41.7%**, versus **7.8/9.9/16.7%** with identical step norms and orthogonal directions. Unit-row rank95 is **5/33/29**: direction and amplitude both contribute, with substantial magnitude weighting in ResNet and Transformer. These controls separate their contributions descriptively; they are not a statistical null test.
- **Connect to loss:** use raw MLP outputs and cross-entropy log probabilities. In the original ResNet/Transformer tracing windows, correct-label readouts reproduce probe loss by their negative mean, and their alignment histories correlate **.943/.946** with full-output histories; those auxiliary correlations are historical, not newly fitted ResNet 390–450 statistics. The current ResNet capture independently reproduces every displayed full/probe loss. The MLP exact decomposition provides a stronger loss connection.
- **Figure 6 — loss and functional motion:** [three-model loss and stage heatmaps](draft/figures/00-three-model-loss-representation-motion.pdf).
  - **Central message:** the observed oscillation has a distributed expression in representations: stages reverse and undergo movement bursts with architecture-specific participation and timing around the loss excursion.
  - **Complementary rows:** loss shows the objective change, cosine shows direction reversal, and RMS shows movement magnitude. Read the signs and bursts together; negative cosine alone does not identify excessive movement. The case findings above specify which stages participate, including the Transformer's positive embedding cosine while later stages reverse.
  - **Larger ResNet event:** [revised trace](evidence/05-adam/b-tracing/observations/63-007-resnet-major-peak.md) replaces the +31.9% peak at 293 with **0.1071 at 404 → 0.4407 at 415 (+311.5%)**, recovering at **439**. First-convolution displacement RMS peaks at 415; the other five endpoints peak at 416. First-convolution/final-stage alignment correlations with output are **.764/.995** in this window.
  - **Temporal evidence:** ResNet's first displayed negative output cosine is **391**, before the selected rise at **405** and +10% crossing at **406**; that earlier reversal is not a unique onset. Transformer reversal at **556** precedes its selected first increase at **558** and +10% crossing at **561**. MLP is already reversing at entry, so its onset is unobserved. Blue/black markers align the first displayed reversal/+10% crossing across all rows.
  - **Read precisely:** 5/6/6 measured endpoints; completed steps **451–501 / 390–450 / 553–601**. ResNet uses the same qualified native continuation and probe as the prior trace, not bitwise replay of the original Adam overview. RMS divides each stage by its own window maximum; full/probe losses expose subset coverage. These selected windows establish participation and timing, without identifying a unique initiating layer or validating a forecast.
- **Temporal-structure companion:** [spectra, stage overlap and signed output coefficients](draft/figures/01-three-model-temporal-structure.pdf). Fits now match Figure 6 at completed steps **451–501 / 390–450 / 553–601**. ResNet is freshly refitted over 61 changes after an exact replay of all 187 control losses from 264 through 450; all six stage cosine/RMS histories match Figure 6.
  - **Central message:** oscillatory training has a structured expression in representation space: much of the motion concentrates in leading modes, temporal structure is shared across depth, and the dominant output component repeatedly reverses.
  - **Energy — structure beyond gradients:** cumulative energy sums squared singular values of the first k modes as a fraction of the total. For output changes this measures squared movement; the raw-gradient comparison uses gradients themselves. The middle row gives each stage's top-one/top-ten fractions. Ten modes capture **99.5%/87.1%/90.8%** of output-change energy in MLP/ResNet/Transformer; this does not measure loss explained or information retained. ResNet raw-gradient top-ten energy is **94.7%**, versus **87.1%** for output changes in the paired update window.
  - **Overlap — shared timing across depth:** top-ten overlap measures how closely a stage's dominant temporal subspace matches the output's. The pattern generally becomes more output-like toward later stages; first learned-stage overlaps are **.75/.57/.46**. ResNet progresses through **.567/.630/.822/.985/.994/1.000** at the six displayed endpoints. This supports shared temporal structure across measured depth, without asserting uniformly high or universally monotonic overlap. Output self-overlap is one by construction.
  - **Signed coefficient — repeated reversal with changing amplitude:** alpha_1/sqrt(P) projects each output change onto one fixed fitted pattern and scales by the square root of the output-coordinate count. Repeated sign changes show back-and-forth motion along that pattern; its absolute value is the RMS magnitude of that component. Growing and shrinking swings add the amplitude dimension; Figure 6 supplies their temporal context relative to loss.
  - **Fresh ResNet coefficient:** the fixed leading output pattern contains **27.3%** of movement energy and changes sign **33/60** adjacent fitted pairs. This differs from **21/61** negative full-vector cosines: tiny component flips may carry little movement, and the cosine includes a predecessor outside the fit. Gradients are the undifferenced vectors used for the displayed updates, at states 389–449.
  - **Contribution/boundary:** leading-mode energy concentration plus signed reversals establishes structured functional oscillation. Saying “destructive interference concentrates in a few dimensions” would additionally require measuring negative cross-step contributions within those modes. Shared temporal overlap alone does not establish causal propagation through layers.
- **Supporting-record role:** previous all-stage plots and the detailed tracing report retain layer-level provenance behind these two selected views. The paired baseline/cap comparison remains in V-C.
- **Interpretation/cost:** SVD energy is time-permutation invariant and rank≤T automatically; window lengths/probes differ across architectures. The coefficient time series supplies the ordering that the spectrum alone lacks. Spike-specific concentration needs quiet-window contrasts. Rolling cosine/RMS costs O(P); full-window Gram costs O(T²P), eigendecomposition O(T³), and is retrospective.

### C. Intervention — test what the functional interpretation buys

- **Completed comparison:** 36 initialization-0 conditions / 11,904 total continuation updates (10,912 treated). Every family uses the complete common fork and the same horizon: **MLP 512→800 / ResNet 256→576 / Transformer 544→928**. Fixed cap budgets use outgoing untreated states **f−8,…,f−1**; the first treated update is **f→f+1**. [Setup and execution record](evidence/05-adam/c-intervention/EXPERIMENTAL_PLAN.md).
- **Exactly four families, in the author's order:**
  - **Immediate LR reduction:** native multipliers **.75/.5/.1**; no ramp or cosine schedule. 0.1x LR removes 99.6%/99.0%/100% of positive loss variation in MLP/ResNet/Transformer, but raises ResNet mean loss by 89.5%. At 0.5x, mean loss falls 25.0%/11.4% in MLP/Transformer and rises 27.2% in ResNet. Smoothness and learning progress are distinct outcomes.
  - **Late beta2 reduction:** **.95/.9/.5**, retaining moments/clock; the new value enters both recurrence and native bias correction. All three switches increase MLP mean loss and positive variation. ResNet is mixed: beta2=.9 lowers mean loss 3.5% while increasing positive variation 63.3%. Transformer mean loss improves 9.6-16.8%, but beta2=.95 increases positive variation 7.1%, and all three switches worsen terminal test loss by 29.3-40.2%.
  - **Global gradient clipping:** full accumulated raw-gradient L2 thresholds **.5/1**, before Adam. ResNet .5 is now included. Both thresholds increase mean loss and positive variation in MLP and ResNet. Transformer clipping at .5 reduces positive variation 87.7% and mean loss 17.8%, while increasing terminal test loss 18.3%. A raw-gradient threshold does not impose a bound on Adam parameter updates or propagated functional movement.
  - **Distributed displacement cap:** **q25/q50/q75** of the same past calibration pool. **q50 remains the main representation-inspired intervention**; the other quantiles test this method's sensitivity. All three quantiles lower mean training loss in all three cases. q50 has the lowest observed mean among the tested settings in MLP and Transformer; ResNet q25/q75 improve mean, positive variation and final/late loss more than q50. Quantile sensitivity is nonmonotone, and q50 terminal test loss increases 0.53%/2.42%/27.52%. No setting wins every metric.
- **q50 versus unchanged Adam, same windows:** positive PV reduction means a smaller sum of upward loss changes; negative loss change means lower loss. All changes are percentages.

| Case | PV reduction | Mean loss change | Final loss change | Last-32 change | Test loss change |
|---|---:|---:|---:|---:|---:|
| MLP | +87.75% | -32.18% | -71.27% | -32.56% | +0.53% |
| ResNet | +2.77% | -9.42% | +20.02% | +21.47% | +2.42% |
| Transformer | +96.23% | -19.84% | -23.31% | -38.97% | +27.52% |

- **V-C integrated in the complete consolidation draft, 14 November:** [full manuscript](draft/manuscript.pdf), with **Figures 8–9, Table III, Algorithm 2, and Equations 21–25**. The [full-paper section](draft/sections/05c-interventions.tex) connects directly to V-A/B and incorporates explicit matrix dimensions and the practical cap computation. Limiting local displacement can reduce excursions and improve mean training loss while reversals persist. The [previous three-page excerpt](draft/archive/2026-11-18/intervention-review.pdf) remains preserved. Numerical and visual checks pass; the active manuscript and original evidence remain unchanged.

| Source display | Placement | Job in the paper / current status |
|---|---|---|
| [24 — loss-variation/learning tradeoff](draft/figures/24-main-intervention-tradeoff.pdf) | **Main figure, full width; verified** | All 33 treated settings and three unchanged-Adam references; nine-point labels, every adverse point, and explicit metric definitions. Earlier layout 18 is preserved. |
| [25 — paired q50 stage heatmaps](draft/figures/25-main-q50-functional-motion.pdf) | **Main figure, full width; verified** | Connect V-B to V-C: smaller extreme jumps coexist with reversals. Loss row, all 5/6/6 stage endpoints and full horizons retained; nine-point labels and visible aligned references. Earlier layout 19 is preserved. |
| [Compact outcome table](evidence/05-adam/c-intervention/tables/matched-main-outcomes.tex) / [full detailed tables](evidence/05-adam/c-intervention/figures/23-matched-intervention-outcomes.pdf) | **Main table complete**; detailed tables in supplement | Twelve setting rows and three model groups; all 144 percentage cells checked in the compiled PDF. Terminal test loss remains beside training outcomes, with absolute Adam references. |
| [17 — four-family loss matrix](draft/figures/17-matched-intervention-loss.pdf) | Supplement, first comparison figure | Preserve every trajectory and the shared log-loss scale within each model. Valuable audit of timing and excursions; its 12 panels duplicate the main numerical comparison. |
| [20 — motion under every setting](draft/figures/20-matched-all-setting-stage-heatmaps.pdf) | Supplement | The 11-page grid checks whether the q50 interpretation survives other settings; main text cites the full comparison. |
| [21 — full held-out trajectories](draft/figures/21-matched-heldout-loss.pdf) | Supplement; **terminal test values stay in main table/text** | Five fixed evaluation states with complete split coverage. Sparse held-out samples cannot establish the absence or timing of held-out spikes. |
| [22 — updates and controller activity](draft/figures/22-matched-update-and-controller-activity.pdf) | Supplement | Establish what each control actually changes, including unequal effective strengths. Raw-gradient clipping and actual Adam updates remain distinct. |
| [Component-cost table](evidence/05-adam/c-intervention/tables/matched-costs.tex) | Supplement; **cost limitation stays in main text** | Separate gradient, probe, optimizer and controller costs; preserve host and instrumentation caveats. |

- **Main comparative figure:** x = mean-loss change over f+1,…,e; y = reduction in **PV = sum max(L_t−L_(t−1),0)**. Left/up is favorable relative to unchanged Adam at the origin. All nine cap/model combinations lower both mean training loss and positive variation; the lowest observed mean in each model belongs to a cap setting. **PV measures accumulated upward loss change, not spike count**; lowering it alone can reward slower training. A finite-window mean is not terminal convergence. Describe the tested grid, without connecting settings into an inferred frontier or claiming optimally tuned competitors.
- **Main outcome table:** rows = Adam; LR .75/.5/.1; beta2 .95/.9/.5; clipping .5/1; cap q25/q50/q75. Each model shows **PV reduction ↑ / mean training-loss change ↓ / final training-loss change ↓ / terminal test-loss change ↓**, relative to unchanged Adam. Notes specify common windows, one initialization/model and q50 preselection. All 12 absolute Adam reference values retain percentage context, especially ResNet's small terminal training loss. Absolute outcomes, last-32 loss and validation details remain in supporting tables. All 12 rows compile at 9-point body text in the three-page review, with no resizing.
- **Main functional figure:** loss; unchanged/capped change cosine; unchanged/capped displacement RMS; 5/6/6 stage endpoints. q50 reduces the largest output RMS jump by 83.2%/23.1%/62.9% (MLP/ResNet/Transformer). Total output-motion energy changes by -88.6%/+39.7%/-55.4%, while reversal fractions are 89.2%/66.9%/95.8% of updates, versus 99.3%/36.9%/36.7% under Adam. The central message is smaller extreme functional jumps with continued oscillation; cumulative movement can increase.
  - Both RMS maps divide by the **same unchanged-Adam maximum for that stage**, not activation magnitude; cosine uses [−1,1]. Stages are measured endpoints, not averages of every module.
  - Incoming changes are placed at their completed-step endpoint, f+1,…,e. Dashed lines mark the peak of the baseline's largest **relative** loss excursion in each fixed window; they are descriptive references, not treatment triggers.
  - The ResNet control reproduces the previously selected 390-450 tracing window. q50 lowers its maximum loss from 0.4407 to 0.1736 (60.6%), while the whole 256-576 continuation has only 2.8% lower positive variation because later oscillations remain. This is a secondary readout in the already selected window; whole-horizon outcomes remain primary. The MLP comparison uses 512-800, after its earlier 451-501 tracing window.
- **V-C reading order:** (1) question from tracing and the specified cap/Algorithm 2; (2) common-fork setup and four intervention families; (3) joint comparison with the tradeoff figure and outcome table; (4) functional explanation with paired heatmaps; (5) endpoint, held-out, cost and replication limits. Keep ResNet q50's **+20.0% final / +21.5% last-32 loss**, its **2.8% PV reduction**, and Transformer q50's **+27.5% terminal test loss** in the result discussion. They define the scope of the contribution. Preserve q50 as the main setting despite q25/q75 doing better on ResNet. No held-out setting or endpoint selection.
- **Typesetting verified:** new main figures use **9-point ordinary labels** at 7.15-inch source width, consistent with [IEEE graphics guidance](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/file-formatting/). The heatmap is compacted to 5.7 inches high, with stage abbreviations defined in the caption, distinct line styles and outlined reference markers. The tradeoff retains all outliers and no callout box obscures a point. The complete IEEE-style manuscript has **14 pages** and its separately numbered supplement has **33 pages**; both compile without warnings and every page has been inspected. This is the author-review layout, not a verified TNNLS page-limit claim. The previous three-page V-C review remains unchanged.
- **Mechanistic bridge:** MLP's exact output-space identity is **delta_L = <F−Y,delta_F>/N + ||delta_F||²/(2N)**. For output-space loss rises above 1e-8, all 161 unchanged-Adam rises and all 70 q50 rises have a negative linear term outweighed by positive quadratic movement; LR x0.5 has 11 such rises. The exact identity shows how excessive output displacement can outweigh a corrective linear contribution, even when the latter remains favorable. Cross-entropy uses its own readout, not this squared-loss identity.
- **Algorithm 2 — [past-calibrated local cap](evidence/05-adam/c-intervention/algorithm.md):** current inputs H and native stored increments U,v give **D = H Uᵀ + 1vᵀ**. For each output unit, radially scale its entire weight row and bias by **min(1,tau/RMS(D_j))**. One fixed budget/module; zero score leaves the increment unchanged, zero budget disables that module's cap. Adam moments/clock advance natively; untargeted parameters retain native increments.
- **Geometry and targets:** scores use the input second-moment metric without an inverse. Target both MLP hidden affines, all 31 ResNet convolutions, and two QKV/four feed-forward Transformer affines. The direct affine displacement is bounded on the current probe; incoming motion, interactions, full-network output and loss are not bounded. [Equations and calibration rule](evidence/05-adam/c-intervention/equations.md).
- **Reporting and verification:** 108 exact full-gradient/probe/native-update audits; complete calibration pools and stored-increment bound checks; common eight-update fingerprints across workers. Full timing rows and the [editable component-cost table](evidence/05-adam/c-intervention/tables/matched-costs.tex) separate gradient, probe, Adam and controller cost. In the recorded MLP q50 arm, controller time is **41.8 ms/update**, versus **4.4 ms/full-gradient boundary**; this implementation does not support a cheap-intervention or time-to-loss advantage. Host/warm-up and instrumentation differences preclude a clean cross-arm speed ratio. Memory is the process high-water allocation through an arm, not an isolated method peak. [Captions, sources and limits](evidence/05-adam/c-intervention/observations/65-016-matched-intervention-matrix.md).
- **Historical support:** the earlier [q50 MLP replication](evidence/05-adam/c-intervention/background/figures/11-local-cap-mlp-replication.pdf) and [quantile × LR ablation](draft/figures/14-distributed-cap-quantile-ablation-mlp.pdf) retain their original windows and interpretations. They do not turn this new matrix into a three-initialization comparison. Earlier conventional/cap figures remain [background](evidence/05-adam/c-intervention/background/README.md).
- **Claim boundary:** these are controlled full-objective continuations at one initialization/model. Settings have unequal effective strength and tuning grids; held-out tradeoffs remain explicit. The comparison supports case-specific intervention effects, not a unique causal source, universal superiority, stochastic deployment or generalization benefit. A late retained-moment beta2 switch is not lower beta2 from initialization. The 36 conditions are not 36 independent replications; variation across updates, stages or quantiles cannot supply seed-level confidence intervals. Matched initializations 1/2 are the priority extension for a claim of reliability across initializations; the existing historical MLP replication has different windows.
- **Deferred:** initializations 1/2 and the cap × LR × initialization grid remain [TODOs with separate budgets](evidence/05-adam/c-intervention/TODO_ABLATIONS.md); no additional representation-inspired method enters the paper.

## VI. Conclusion and appendices

- **Conclusion:** interaction → curvature/noise → functional expression → selective control. Preserve the explanatory scope; distinguish training dynamics from generalization and computational efficiency.
- **Appendices:** proofs/clock/general applied-step curvature, HVP and finite-step residuals, complete Adam audit, probes/stochastic controls, q50 calibration/replication and held-out outcomes. Archived exploratory methods are not automatically appendix content.
- **Abstract last:** replace outdated “future interventions” with completed findings and their measured scope.

## Closure work — concrete order and stopping criteria

- **1. Assemble existing evidence — complete inside consolidation, no new training.**
  - The introduction and stability survey follow the active draft's progression. III–IV retain the original argument, examples and four figures, with the setup table, validation evidence and the general actual-step curvature extension.
  - V-A uses exhaustive severity/timing evidence; V-B supplies the tracing on the same reference examples and temporal structure; V-C supplies **two main figures + one compact outcome table**. Its comparison retains LR → beta2 → clipping → distributed q50. The [complete draft](draft/manuscript.pdf) has reconciled numbering and references; [supplement](draft/archive/2026-11-18/supplementary.pdf) contains proofs, controls, every setting, and separate historical replication.
  - State/update notation, losses, coverage, initializations, and source provenance are explicit. All 416 protected evidence/human/original-manuscript files remain unchanged.

- **2. Matched intervention comparison — complete.**
  - All 36 initialization-0 conditions, common forks/horizons, native runtime controls and every requested setting are verified. The full grid retains unfavorable results and unequal effective strengths.
  - The same ResNet control reproduces the larger traced excursion; MLP's new window starts after its earlier tracing window. Historical windows remain separate.
  - No further runs are required to report this bounded comparison. Initializations 1/2 add 72 continuations; the gross cap × LR grid has 162. Both remain deferred with [separate budgets](evidence/05-adam/c-intervention/TODO_ABLATIONS.md).

- **3. Method reporting — complete for the present scope.**
  - Shared stage probes, fixed past-only budgets, activity/stored-increment checks and full held-out evaluations at five fixed states accompany every condition.
  - Positive loss variation, mean/final/late loss, actual updates and component timing are supplied. Memory is a process high-water mark; controller time alone is not deployable overhead.
  - Full-paper typography, compact table, captions, algorithms and supplement are complete and verified in consolidation. The one-initialization scope and endpoint/test/cost tradeoffs remain in the main results and conclusion.

- **4. Keep stronger claims conditional.**
  - **Forecasting:** freeze targets/features/horizons; compare against loss-history and norm baselines with temporal gaps, false alarms and fresh-seed confirmation. Current data support characterization without this project.
  - **Spike-specific modes/causal source:** require matched quiet windows/independent probes or targeted interventions. Current tracing supports participation.
  - **Superiority over representation-aware methods:** requires faithful AGC/Kosson/FOOF comparisons with matched tuning budgets. Current contribution is a specified control and bounded tests.
  - **Broad deployment:** stochastic probes, from-initialization use and larger-scale transfer remain future work; no generalization benefit is claimed.

- **5. Final manuscript checks — technical work complete; author/venue decisions remain.**
  - Warning-free compilation, all-page rendering/review, all 144 intervention cells, references, mathematical identities, captions, and abstract/conclusion consistency pass. [Audit and reproduction](work/readability_revision_20261114/OBSERVATIONS.md). Next action: author review of the complete draft before any transfer to the active manuscript.
  - Resolve the supplied [decision letter's revised-resubmission restriction](evidence/appendices/submission/decision-letter.pdf) and current official TNNLS format before submission. Keep the author's target; no contact or submission is authorized here.

- **Map completion:** **yes; the complete review manuscript and supplement are assembled and verified inside consolidation.** The matched intervention evidence and reporting gap is closed for the agreed single-initialization scope. Author review and submission-route/format checks remain; broader empirical claims retain their separate tests.
