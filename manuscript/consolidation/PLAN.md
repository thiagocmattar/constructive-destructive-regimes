# Paper anatomy and closure map

- **Central argument:** learning updates reinforce or reverse earlier changes. Their direction, amplification and accumulation connect optimization geometry to changes in the learned function; measuring those changes motivates selective control.
- **Keep the current paper:** title, Sections I–VI, III's theory spine and four GD/SGD figures. Develop **characterization → tracing → intervention** inside V.
- **Decision from this review:** the evidence supports this explanatory paper. Validated forecasting, a universal initiating layer and general optimizer superiority require additional evidence.
- **Reading contract:** this is the complete executive map. Links provide provenance, not additional required reading. Manuscript editing follows agreement on this plan.
- **Local assets:** [section-indexed evidence package](evidence/README.md); [display/table/algorithm map](evidence/ARTIFACTS.md).
- **Figure convention, 14 November:** Figures 1–6 use **Training step s = completed optimizer updates**. Representation changes are placed at their observed endpoint; lag/batch axes retain their own units. [Captions and exact clock conversions](evidence/FIGURE_NOTES.md). Intervention display revisions remain a later task.

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
  - **Known:** secant methods estimate scalar curvature; spectral stability studies measure Hessian eigenvalues.
  - **Gap addressed:** specify the curvature quantity encoded by norm/alignment, its approximation error and sampling contamination.
  - **Response/test:** exact smooth-GD path-average interpretation, pointwise error bound and explicit noise term; **1,188 directional-HVP pairs in a tanh MLP** provide an independent derivative reference. **III-A/B/C → IV Figures 1/4 and validation.**

- **Contribution 3 — distinguish reversal, amplification and loss consequences.**
  - **Known:** [Bai et al., §§4–5](https://arxiv.org/html/2506.04805v2#S4) relates Adam spikes to preconditioner dynamics.
  - **Gap addressed:** gradient reversal alone does not determine a finite-step loss change or identify what must be damped.
  - **Response/test:** explicit GD loss condition and optimizer-dependent response interpretation; exhaustive Adam characterization separates direction, growth, severity and timing. **III-E → IV checks → V-A.**

- **Contribution 4 — trace and control functional movement.**
  - **Known:** [Catapults](https://proceedings.mlr.press/v235/zhu24h.html) connects output/NTK dynamics and feature learning; representation-aware update methods already exist.
  - **Gap addressed:** locate which stages participate in an observed excursion and test whether selective movement control preserves progress.
  - **Response/test:** fixed-probe tracing algorithm, architecture-specific findings and the distributed q50 affine-displacement cap; selected training-window means improve **22.85%/16.22%/26.00%** across MLP/ResNet/Transformer while reversals persist. Place this result after LR, beta2 and clipping comparisons, each with its own unchanged baseline. **V-B/C.**

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
- **A/B — make cheap curvature estimation prominent:**
  - At state s, define r_s = ||g_s||/||g_(s−1)|| and **Q_s = eta*kappa_hat_s = 1 − rho_s*r_s**.
  - Under smooth full-batch GD, kappa_hat is **exactly the incoming-step average directional Hessian Rayleigh quotient**. Its starting-point error is at most L_H*eta*||g_(s−1)||/2 for a Lipschitz Hessian.
  - **O(p) reductions and one retained gradient; no extra backward pass or HVP.** This estimates a directional Hessian scalar; measure runtime/memory before claiming negligible overhead.
  - [BB, printed p.142/Eq. (6)](https://pages.cs.wisc.edu/~swright/726/handouts/barzilai-borwein.pdf#page=3) gives alpha_BB = ||delta_theta||²/(delta_thetaᵀ delta_g). Under plain GD, **kappa_hat = 1/alpha_BB**, when defined. [Tan, §2/Eq. (6)](https://papers.neurips.cc/paper_files/paper/2016/file/c86a7ee3d8ef0b551ed58e354a836f2b-Paper.pdf#page=3) explicitly gives the scalar-Hessian interpretation. Credit this identity; foreground our diagnostic, noise and validation contribution.
- **C — sampling:** retain noise and finite-population variance. [027](evidence/04-gd-sgd/observations/27-REPORT.md) reduces batch-32 median HVP discrepancy **.7467 → .0030** with offline full-gradient noise correction. These are dependent MLP measurements across states/seeds/batches, not 1,188 independent experiments. Less negative noisy cosine does not imply improved loss.
- **E — loss and indexing:**
  - State s means s completed updates. Q_s is available after backward, before s→s+1, but describes the **incoming** step.
  - Quadratic GD: **L_s−L_(s−1) = eta*||g_(s−1)||²*(Q_s−2)/2**. A strict rise requires Q_s>2, equivalently rho_s<−1/r_s; this implies reversal and growth, whose conjunction alone is weaker.
  - For nonlinear losses retain the approximation and residual. Outgoing prediction is a separate test; activation boundaries limit smooth expansions.
  - Compute Q directly from the inner product: previous zero norm makes Q undefined; current zero norm gives Q=1 while cosine is undefined. Never turn undefined cosine into a zero observation.

## IV. Empirical Characterization — preserve the four figures

- **Cases:** CIFAR-10 MLP/5,000 examples/half-squared error; CIFAR-10 ResNet-32/50,000 examples/cross-entropy; WikiText-2 Transformer/2,088,600 supervised tokens/token-weighted NLL.
- **Figure 1 — LR, curvature, loss:** [nine GD conditions over 3,000 updates](evidence/04-gd-sgd/figures/gd_loss_projected_change_spectral_clean.pdf).
  - **Central message:** existing gradient norms and alignment expose the directional-curvature signature of smooth and oscillatory GD.
  - **Evidence:** oscillatory conditions have late Q near two while loss continues to improve; smooth low-rate MLP/Transformer conditions can have Q near zero despite nonzero leading curvature. The two curvature quantities need not coincide.
  - **Role:** make III's inexpensive directional Hessian estimate concrete. The independent HVP comparison below tests its derivative interpretation; this spectral overlay supplies geometric context.
  - **Read precisely:** both curvature curves use the same centered 101-step smoother, with sparse spectra interpolated before smoothing. Spectra cover the full MLP objective and subsets for ResNet/Transformer. Smoothed crossings do not locate onset or establish a forecast.
- **Figure 2 — reversal with progress:** [norms, alignment and R](evidence/04-gd-sgd/figures/gd_reversal_norm_retention.pdf).
  - **Central message:** persistent local reversal can coexist with cumulative movement and optimization progress.
  - **Evidence:** across the three cases, high-rate reversal and small R coexist with lower endpoint loss and greater absolute displacement; read this figure alongside Figure 1 and the endpoint table.
  - **Role:** separate direction, magnitude and accumulation. R measures displacement relative to path length; a small R alone does not measure poor learning. The separate all-pair audit establishes positive total interaction in all nine cases despite negative adjacent interaction in seven; that result is not inferred from R alone.
- **Figure 3 — temporal organization:** [backward gradient alignment](evidence/04-gd-sgd/figures/gd_backward_compact.pdf), endpoint 2256, lags 1–256 with a 1–32 detail view.
  - **Central message:** reinforcing and opposing directions recur beyond adjacent steps; lag-one alignment cannot summarize the trajectory's temporal geometry.
  - **Evidence:** a small cosine at one separation can be followed by substantial alignment or opposition at larger lags. The early detail makes sign alternation visible.
  - **Role/boundary:** motivate the all-lag formulation and test lag-one dominance. Each point is one endpoint/lag pair from a recomputed continuation; this is neither an all-pair average nor evidence of monotone decay or repeated traversal of one minimum.
- **Figure 4 — sampling:** [546 SGD/GD continuations](evidence/04-gd-sgd/figures/sgd_batch_size_compact.pdf), three initializations, forks at initialization and after 2,000 GD updates.
  - **Central message:** sampled-gradient alignment depends jointly on batch size and the training state from which learning continues.
  - **Evidence:** the smallest batches weaken trained-state reversal in all three cases; at the initial MLP fork, the smallest batch gives negative alignment while full GD gives positive alignment. Sampling does more than uniformly attenuate an existing cosine.
  - **Role/boundary:** connect III's noise analysis to evolving trajectories. Changing the batch also changes the visited states; the frozen-state control below isolates sampling. Shading spans three initialization means, and equal updates entail unequal data exposure. Alignment alone does not rank loss progress.
- **Supporting figure — [frozen sampling calibration](evidence/04-gd-sgd/support/frozen_calibration.pdf):**
  - **Central message:** sampling alone changes the observed gradient geometry at fixed parameters.
  - **Evidence:** across saved MLP states, variance and noise projection match moment predictions within maximum relative discrepancies of **0.50%/0.74%**. The cosine approximation's maximum absolute error is **0.00745**, but its target misses 21/36 stochastic confidence intervals.
  - **Role/boundary:** isolate sampling noise and distinguish exact moment identities from the approximate cosine formula; this control contains no learning trajectory.
- **Supporting figure — [directional-HVP validation](evidence/04-gd-sgd/support/sgd_proxy_validation.pdf):**
  - **Central message:** accounting for realized sampling noise sharply improves agreement with independently measured directional curvature.
  - **Evidence:** batch-32 median discrepancy falls **.7467 → .0030**; all stochastic batches improve. Raw paired markers supply the derivative comparison, while smoothed curves provide context.
  - **Role/boundary:** full gradients make this an offline validation. The 1,188 dependent pairs in one MLP do not establish an online correction or a largest-eigenvalue estimate. [Source and reductions for both validation figures](evidence/04-gd-sgd/observations/27-001-controlled-minibatch-noise.md).
- **Separate finite-step check:** [024's residual audit](evidence/04-gd-sgd/observations/24-007-section-iv-critical-review.md) finds **1,325/2,999 loss-sign disagreements** for the high-LR Transformer. Directional-curvature validation and accuracy of the quadratic loss approximation are separate tests.
- **Closure:** existing evidence suffices. Label single-initialization GD versus replicated minibatch results; no additional GD training is needed.

## V. Raw-Gradient Diagnostics under Adam

- **Opening:** retain the Appendix A identity. For a fixed smooth objective and actual increment u, delta_g = integral H(theta+a*u)u da. Adam chooses u; the previous raw gradient supplies the projection. The same observable transfers as a **mixed Hessian response**, with different semantics from GD curvature or actual-update alignment.
- **Empirical scope:** controlled full-objective Adam trajectories for the three tasks; ResNet/Transformer representation probes cover subsets. These cases do not validate stochastic Adam deployment.

### A. Characterization — distill the timing/severity audit here

- **Figure 5 — Adam excursions:** [three-model high-LR overview](evidence/05-adam/a-characterization/figures/00-high-lr-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf).
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
- **Supporting figure — [severity, lead time and warning memory](evidence/05-adam/a-characterization/support/adam_forward_and_severity.pdf):**
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
- **Figure 6 — loss and functional motion:** [three-model loss and stage heatmaps](evidence/05-adam/b-tracing/figures/00-three-model-loss-representation-motion.pdf).
  - **Central message:** the observed oscillation has a distributed expression in representations: stages reverse and undergo movement bursts with architecture-specific participation and timing around the loss excursion.
  - **Complementary rows:** loss shows the objective change, cosine shows direction reversal, and RMS shows movement magnitude. Read the signs and bursts together; negative cosine alone does not identify excessive movement. The case findings above specify which stages participate, including the Transformer's positive embedding cosine while later stages reverse.
  - **Larger ResNet event:** [revised trace](evidence/05-adam/b-tracing/observations/63-007-resnet-major-peak.md) replaces the +31.9% peak at 293 with **0.1071 at 404 → 0.4407 at 415 (+311.5%)**, recovering at **439**. First-convolution displacement RMS peaks at 415; the other five endpoints peak at 416. First-convolution/final-stage alignment correlations with output are **.764/.995** in this window.
  - **Temporal evidence:** ResNet's first displayed negative output cosine is **391**, before the selected rise at **405** and +10% crossing at **406**; that earlier reversal is not a unique onset. Transformer reversal at **556** precedes its selected first increase at **558** and +10% crossing at **561**. MLP is already reversing at entry, so its onset is unobserved. Blue/black markers align the first displayed reversal/+10% crossing across all rows.
  - **Read precisely:** 5/6/6 measured endpoints; completed steps **451–501 / 390–450 / 553–601**. ResNet uses the same qualified native continuation and probe as the prior trace, not bitwise replay of the original Adam overview. RMS divides each stage by its own window maximum; full/probe losses expose subset coverage. These selected windows establish participation and timing, without identifying a unique initiating layer or validating a forecast.
- **Temporal-structure companion:** [spectra, stage overlap and signed output coefficients](evidence/05-adam/b-tracing/figures/01-three-model-temporal-structure.pdf). Fits now match Figure 6 at completed steps **451–501 / 390–450 / 553–601**. ResNet is freshly refitted over 61 changes after an exact replay of all 187 control losses from 264 through 450; all six stage cosine/RMS histories match Figure 6.
  - **Central message:** oscillatory training has a structured expression in representation space: much of the motion concentrates in leading modes, temporal structure is shared across depth, and the dominant output component repeatedly reverses.
  - **Energy — structure beyond gradients:** cumulative energy sums squared singular values of the first k modes as a fraction of the total. For output changes this measures squared movement; the raw-gradient comparison uses gradients themselves. The middle row gives each stage's top-one/top-ten fractions. Ten modes capture **99.5%/87.1%/90.8%** of output-change energy in MLP/ResNet/Transformer; this does not measure loss explained or information retained. ResNet raw-gradient top-ten energy is **94.7%**, versus **87.1%** for output changes in the paired update window.
  - **Overlap — shared timing across depth:** top-ten overlap measures how closely a stage's dominant temporal subspace matches the output's. The pattern generally becomes more output-like toward later stages; first learned-stage overlaps are **.75/.57/.46**. ResNet progresses through **.567/.630/.822/.985/.994/1.000** at the six displayed endpoints. This supports shared temporal structure across measured depth, without asserting uniformly high or universally monotonic overlap. Output self-overlap is one by construction.
  - **Signed coefficient — repeated reversal with changing amplitude:** alpha_1/sqrt(P) projects each output change onto one fixed fitted pattern and scales by the square root of the output-coordinate count. Repeated sign changes show back-and-forth motion along that pattern; its absolute value is the RMS magnitude of that component. Growing and shrinking swings add the amplitude dimension; Figure 6 supplies their temporal context relative to loss.
  - **Fresh ResNet coefficient:** the fixed leading output pattern contains **27.3%** of movement energy and changes sign **33/60** adjacent fitted pairs. This differs from **21/61** negative full-vector cosines: tiny component flips may carry little movement, and the cosine includes a predecessor outside the fit. Gradients are the undifferenced vectors used for the displayed updates, at states 389–449.
  - **Contribution/boundary:** leading-mode energy concentration plus signed reversals establishes structured functional oscillation. Saying “destructive interference concentrates in a few dimensions” would additionally require measuring negative cross-step contributions within those modes. Shared temporal overlap alone does not establish causal propagation through layers.
- **Supporting-record role:** previous all-stage plots and the detailed tracing report retain layer-level provenance behind these two selected views. The paired baseline/cap comparison remains in V-C.
- **Interpretation/cost:** SVD energy is time-permutation invariant and rank≤T automatically; window lengths/probes differ across architectures. The coefficient time series supplies the ordering that the spectrum alone lacks. Spike-specific concentration needs quiet-window contrasts. Rolling cosine/RMS costs O(P); full-window Gram costs O(T²P), eigendecomposition O(T³), and is retrospective.

### C. Intervention — test what the functional interpretation buys

- **Selected presentation — exactly four families, in this order:**

  1. **LR schedules / Figure 7:** [persistent half LR](evidence/05-adam/c-intervention/figures/01-persistent-half-lr-cifar10-mlp-resnet-wikitext2-transformer.pdf). Three architectures; reduce LR by half from the retained warm state, keeping each unchanged baseline.
  2. **Beta schedule / Figure 8:** [late beta2 reductions, all three models](evidence/05-adam/c-intervention/figures/15-beta2-reductions-cifar10-mlp-resnet-wikitext2-transformer.pdf). .999→.95/.9/.5 at states 512/256/544 for MLP/ResNet/Transformer, with moments/clock retained and constant LR. The new value enters both the recurrence and native bias correction; no reset or history-product correction.
  3. **Gradient clipping / Figure 9:** [all three models](evidence/05-adam/c-intervention/figures/16-gradient-clipping-cifar10-mlp-resnet-wikitext2-transformer.pdf). Global raw-gradient caps before Adam moments: MLP and Transformer thresholds .5/1, ResNet 1. Show raw and optimizer-input norms distinctly; ResNet .5 is explicitly missing.
  4. **Distributed displacement cap / Figure 10:** q50 is the **only representation-inspired intervention** presented. Retain the paired heatmaps, scalar diagnostics and MLP replication; other methods and quantiles are background evidence.
- **Conventional evidence — [complete paired outcomes](evidence/05-adam/c-intervention/tables/conventional-outcomes.md), one initialization per model:**
  - **Transformer:** beta2 reductions lower mean training loss 11.82–18.47%; clipping lowers it 17.74–19.02%. Both suppress peaks, yet terminal test loss worsens 24.64–33.12% and 11.30–14.84%, respectively.
  - **ResNet:** beta2 mean-loss changes are +28.46%/−3.53%/+6.28% for .95/.9/.5. Every reduction worsens the last-32 mean; .95 improves test loss. Existing clipping lowers the window mean but worsens the tail/test.
  - **MLP:** the retained beta2 and clipping settings all increase mean training loss. This architecture dependence rules out a blanket claim that a conventional intervention always helps or always fails.
  - **Coverage:** Runs289–291 add 10 verified continuations / 3,584 updates; new controls retain loss excursions. Keep the [ablation TODO](evidence/05-adam/c-intervention/TODO_ABLATIONS.md): seeds, shared forks/times, ResNet clip .5, intermediate held-out coverage and common functional probes remain distinct gaps.
- **Mechanistic bridge, already measured:** MLP loss satisfies **delta_L = <F−Y,delta_F>/N + ||delta_F||²/(2N)**. Over outgoing updates 448–1152, including the transition to 1153, all **377/371/379** baseline rises >1e−8 across three seeds have a negative first term outweighed by movement cost. This establishes finite-step functional overshoot in these trajectories. Cross-entropy uses its own readout, not this squared-loss identity.
- **Algorithm 2 — [local cap](evidence/05-adam/c-intervention/observations/65-010-local-displacement-cap-geometry.md):**
  - Current inputs H: n×d; weights W: m×d; bias b: m. Native stored Adam increments U,v induce **D = H Uᵀ + 1_n vᵀ**, n×m.
  - For unit j, s_j = RMS of column j across examples/positions. Scale the **entire weight row and its bias** by min(1,tau_l/s_j).
  - One tau_l per module: q50 pooled over units and eight untreated preceding updates. Current inputs change with training; budgets stay fixed. Zero score leaves the increment unchanged; zero budget disables that module's cap.
  - Adam moments/clock advance natively; the applied increment is changed. Verify the FP32 storage bound. Later gradients and moments consequently follow a different trajectory.
- **Geometry/coverage:** with bias-augmented inputs, s_j²=u_jᵀG u_j, G=H_augᵀH_aug/n. Radial rescaling uses input correlations without inverting G. It bounds the direct affine term; incoming-representation changes and their interaction remain unconstrained.
- **Actual targets:** MLP both hidden affines; ResNet all 31 convolutions; Transformer two QKV and four feed-forward affines. Output/classifier, normalization, embeddings and Transformer attention-output projections remain outside these caps.
- **Cost:** probe forward to capture inputs plus direct affine/convolution products and storage-rounding verification. The score requires no second complete candidate-network forward, Jacobian or SVD. Measure controller time/memory; current FP64 diagnostic implementation is not established as production-efficient.
- **q50 result to foreground:** mean loss changes **−22.85%/−16.22%/−26.00%** for MLP/ResNet/Transformer; two further MLP seeds give **−19.43%/−32.08%**. Reversals persist. MLP seed 0's last-32 mean is **+4.36%**, so retain the full trajectory and tail rather than only its favorable window mean.
- **Figure 10 — selected main base:** [standalone q50 heatmaps](evidence/05-adam/c-intervention/figures/07-distributed-q50-stage-heatmaps.pdf), preserving source Figure 07 page 2 exactly. Retain its three architecture columns and five rows: full training loss, baseline/cap consecutive-change cosine, baseline/cap movement RMS. This connects loss excursions, depth and time, and the intervention in one display: large movement bursts diminish while reversals persist.
- **Heatmap contract:** each row is one captured stage, reduced over all fixed-probe coordinates, not every individual module. Cosine uses a common [−1,1] scale. Both RMS maps divide by that stage's **same baseline maximum** over the displayed window; this is not normalization by activation magnitude, and colors do not compare absolute RMS across stages. No smoothing or new SVD. Windows: MLP 448–800, ResNet 264–576, Transformer 552–928; the MLP outcome table separately includes the extension to 1152. Preserve the architecture distinctions and make labels legible at final two-column width.
- **Outcome table / supporting diagnostics:** window mean, maximum, positive variation (sum of positive loss increments), final-32 mean, functional motion, held-out loss and cost. Keep [q50 scalar diagnostics](evidence/05-adam/c-intervention/figures/10-local-cap-distributed-manuscript.pdf) and [q50 MLP replication](evidence/05-adam/c-intervention/figures/11-local-cap-mlp-replication.pdf) as support. The selected heatmaps establish the paired cap comparison; ranking the four families still needs qualification. Terminal held-out loss worsens in all three q50 cases.
- **Retained supporting ablation:** [cap quantile by LR](evidence/05-adam/c-intervention/14-distributed-cap-quantile-ablation-mlp.pdf), kept directly in V-C. Five quantiles × three LRs × three initializations, with same-LR uncapped controls; plot mean training loss ± sample SD on the shared log scale. This characterizes sensitivity of the same cap method; q50 remains the main intervention. q50 at 1× has the lowest observed mean, while increasing nominal LR does not improve any quantile's own across-seed mean. Do not claim an optimal quantile or acceleration.
- **Background boundary:** entry caps, freezing, directional removal, output-wide budgets and retrospective norm controls are retained in source records for audit. They are outside the selected presentation; do not promote them into additional methods or mandatory experiments.
- **Comparison limits:** historical hardware/windows/budgets differ; half stored increments are not bitwise native half-LR. A late beta2 switch with retained moments does not refute Bai's lower-beta2 training prescription. Inactive clipping cannot support a superiority claim.

## VI. Conclusion and appendices

- **Conclusion:** interaction → curvature/noise → functional expression → selective control. Preserve the explanatory scope; distinguish training dynamics from generalization and computational efficiency.
- **Appendices:** proofs/clock/BB equivalence, HVP and finite-step residuals, complete Adam audit, probes/stochastic controls, q50 calibration/replication and held-out outcomes. Archived exploratory methods are not automatically appendix content.
- **Abstract last:** replace outdated “future interventions” with completed findings and their measured scope.

## Closure work — concrete order and stopping criteria

- **1. Assemble existing evidence — required, no new training.**
  - Rewrite I–III around the four contribution cards; retain IV's four figures.
  - Compress V-A as above; assemble V-B/C with two algorithms, Figures 5–10 and the intervention outcome table. V-C follows LR → beta2 → clipping → distributed q50. Defer secondary q50 diagnostics to appendices.
  - Standardize state/update notation, losses, sample coverage, seeds and source links. Preserve original evidence and manuscript structure.

- **2. Qualify intervention comparisons — required before direct ranking.**
  - Audit reusable branches against identical starting tensors, optimizer/RNG, objective, hardware, precision, probe and calibration. Otherwise repeat the unchanged baseline with its paired intervention.
  - Fixed seed-0 windows: **MLP 448→1152/LR .001; ResNet 264→576/.003; Transformer 552→928/.003**.
  - Configurations per architecture: unchanged Adam, native half-LR, representative late beta2=.95, global raw-gradient clipping and distributed q50 cap. Make the native beta2 bias-correction convention explicit; the beta figure retains all three tested reductions per model.
  - Clipping threshold: median untreated raw-gradient norm over the same eight-update prefix. Report clipping activity and applied-norm reduction; equal quantiles do not equal matched strength.
  - **Maximum selected comparison matrix: 15 continuations for seed 0; 45 across three initializations, before qualified reuse.** Use the same absolute start/window per seed; report absent excursions. Retrospective norm-control experiments are outside this matrix. Retain or recreate untreated prefixes/checkpoints as needed. The current conventional model-coverage extensions retain their original forks; they do not by themselves fill this matched matrix.
  - **Success criterion:** report all paired outcomes, including divergence, tail/test deterioration and inactive controls. A ranking requires replicated matched effects; otherwise state the case-dependent result. Do not retune on the reported windows.
  - Continue Analysis 065 with new run IDs for changed inputs. Prepare measured duration, live cost/storage limits and retrieval checks before any launch.

- **3. Complete method reporting — required.**
  - Report cap activation, budget/stored-increment checks, probe coverage and full held-out evaluation at start/start+32/end.
  - Measure baseline versus controller time and peak memory on matched hardware. Separate cheap scalar diagnostics, offline tracing and the intervention's cost.
  - Further comparative execution is needed only where existing records fail these contracts. Runs289–291 complete the requested conventional model coverage; the broader matched matrix remains planned.

- **4. Keep stronger claims conditional.**
  - **Forecasting:** freeze targets/features/horizons; compare against loss-history and norm baselines with temporal gaps, false alarms and fresh-seed confirmation. Current data support characterization without this project.
  - **Spike-specific modes/causal source:** require matched quiet windows/independent probes or targeted interventions. Current tracing supports participation.
  - **Superiority over representation-aware methods:** requires faithful AGC/Kosson/FOOF comparisons with matched tuning budgets. Current contribution is a specified control and bounded tests.
  - **Broad deployment:** stochastic probes, from-initialization use and larger-scale transfer remain future work; no generalization benefit is claimed.

- **5. Final manuscript checks — required.**
  - Compile/render, verify captions and references, align abstract/conclusion with measured endpoints, and package reproducibility records.
  - Resolve the supplied [decision letter's revised-resubmission restriction](evidence/appendices/submission/decision-letter.pdf) and current official TNNLS format before submission. Keep the author's target; no contact or submission is authorized here.

- **Map completion:** **yes, sufficient to proceed to paper closure.** Each contribution has its motivation, predecessor, precise claim, evidence, location and remaining test. The substantive empirical gap is qualified intervention comparison/reporting; the Adam timing audit is one completed characterization step.
