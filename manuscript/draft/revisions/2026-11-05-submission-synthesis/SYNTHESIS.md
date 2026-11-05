# Scientific synthesis for submission closure

5 November 2026. This synthesis consolidates existing definitions, derivations and experiments. It separates established observations from interpretation and proposed research. Sources and selection checks are in the [evidence inventory](reviews/evidence-inventory.md), [theory/novelty review](reviews/theory-novelty.md) and [source-pinned evidence selection](evidence_selection.json). No new training is required for the bounded argument below.

## 1. The scientific question and answer

**When successive gradients oppose one another, what cancels, over which horizon, and what does this imply for learning progress?**

The original constructive/destructive idea concerns reinforcement and cancellation of parameter motion. The accumulated evidence makes its meaning more precise: local opposition, cumulative displacement, objective decrease and optimizer direction are distinct quantities. Measuring them together reveals organized motion that a loss curve or adjacent cosine alone cannot describe.

The resulting contribution is a diagnostic framework supported by controlled observations and matched interventions. It locates temporal cancellation, clarifies how sampling and optimizer transformations alter its meaning, and tests whether changes to the visible symptoms improve the measured objective. It does not establish that destructive interaction is inherently undesirable.

## 2. Definitions and formulations

Labels used throughout: **Exact** means an identity under declared conditions; **Approximation** requires an explicit model or regularity condition; **Empirical** describes retained experiments; **Speculation** denotes an untested explanation or intervention.

| Symbol/object | Definition and interpretation |
| --- | --- |
| `theta_t`, `g_t` | Parameters after update `t`; raw gradient evaluated at `theta_(t-1)`, before clipping or optimizer transformation. |
| `rho_(u,t)`; `rho_t` | Cosine of gradients `g_u,g_t`; adjacent case `rho_(t-1,t)`. Positive/negative signs define constructive/destructive interaction for that pair. |
| `A_T`, `B_T`, `R_T` | `A=sum ||g_t||`, `B=||sum g_t||`, `R=B/A`. Under constant-rate plain SGD, `R` is displacement divided by path length. |
| `E_T`, `R_T^perp`, `Q_T` | `E=sum ||g_t||^2`; zero-cross-term reference `sqrt(E)/A`; `Q=B^2/E`. `Q<1` means negative total cross interaction, `Q>1` positive. |
| `h_j`, `C_T`, `D_T` | Pair sum `h_j=g_(2j-1)+g_(2j)`; `C=sum ||h_j||/A`; `D=B/sum ||h_j||`; hence `R=CD`. |
| `z_(t+1)` | Projected gradient response `1-g_t^T g_(t+1)/||g_t||^2`. It is known after the measured transition. |
| `m_t`, `d_t`, actual increment | First moment, adaptively scaled direction, and applied parameter change. Clipping, LR, decay and arithmetic distinguish these streams. |
| Excursion `a,p,c` | Pre-rise state, largest loss before recovery, and first return to the fixed pre-rise loss. Unrecovered events are censored. |

All Euclidean quantities refer to the declared trainable-parameter coordinates; they are not function-space invariants or amounts of retained knowledge. Zero-denominator quantities remain undefined. A batch-mean gradient cannot separately identify per-example energy and intra-batch interaction. [Measurement contracts](../../../../research/METRICS.md), [definitions](../../../../research/DEFINITIONS.md).

**Exact: temporal accounting.** Constant-rate plain SGD gives

\[
\Delta_T=-\eta\sum_tg_t,\qquad
\|\Delta_T\|^2=\eta^2E_T+2\eta^2\sum_{u<t}g_u^\top g_t.
\]

All lags remain present. The paired factorization locates immediate cancellation through `C` and subsequent turning through `D`. Small `R` alone does not imply negative total interaction: equal-length orthogonal steps already give `R=1/sqrt(T)`.

**Exact under smooth full GD:** `z/eta` is directional Hessian curvature averaged along the preceding step. It is also the classical secant quotient `(s^T y)/(s^T s)`, the reciprocal BB1 step where defined. **Approximation:** replacing path-average curvature by its starting value gives `z ≈ eta*kappa`; sampling adds an explicit noise term. The estimator is not a new mathematical primitive. [Barzilai–Borwein](https://doi.org/10.1093/imanum/8.1.141), [active derivation](../../sections/03-methods.tex).

**Approximation, exact on quadratics:** for full GD,

\[
\Delta\mathcal L\approx-\frac{\eta}{2}\|g_{old}\|^2(2-z).
\]

Reversal begins at `z>1`, whereas the quadratic loss-rise boundary is `z>2`. Thus reversal can coexist with descent. Nonlinear remainder and SGD sampling restrict the interpretation; classical Hessian identities need not hold across ReLU gradient jumps. [Loss derivation](../../sections/06-loss-appendix.tex).

**Exact under smooth full-objective Adam:** the response is a mixed projection `z/eta=g^T Hbar d/||g||^2`, not ordinary gradient-direction curvature. An analytical decomposition separates a frozen-time preconditioned proxy, a first-moment residual and finite-step curvature variation. Its neural calibration is unperformed; it is supporting theory, not an established spike mechanism. [Analysis057](../../../../analyses/057-2026-11-05-adam-response-memory/observations/001-response-memory.md).

## 3. Experiment families and their roles

| Family | Coverage and comparison | Role |
| --- | --- | --- |
| Pythia/MiniPile | Scratch 14M/31M/70M, three paired seed blocks; complete-pass geometry; matched schedule and coordinate-scaling branches | Replicated optimizer-aware geometry and intervention outcomes |
| Controlled GD | MLP/CIFAR-10, ResNet-32/CIFAR-10, Transformer/WikiText-2; three rates each, 3,000 updates, one initialization/model | Local cancellation, cumulative interaction and loss progress |
| SGD | 546 continuations of 200 updates; three initialization blocks, two starting stages, multiple batches and sampling repetitions | Sampling plus changing trajectories; separate frozen/noise controls |
| Full-objective Adam | Same three neural cases, three fixed rates/model, 1,024 updates; one initialization/model | Excursion anatomy and raw-gradient versus actual-step distinction |
| Adam interventions | Matched warm-state LR, clipping and decay branches; selected windows and mostly one seed | Consequences across mean, peak, late and endpoint loss |
| Rosenbrock/toy controls | Momentum/decay, rates, initial points, Gaussian noise; exact FP64 histories and 2-D Hessians | Intuition, instrumentation and counterexamples; not neural replication |
| Loss-arithmetic study | Thirty full-batch runs, ten paired initializations, native/fixed/FP64-loss arithmetic | Separate attained-state curvature question; outside the central paper |

The [analysis registry](../../../../analyses/INDEX.md) distinguishes executed studies, reanalyses and unexecuted plans. Replots and forks sharing parents are not new independent replications. In particular, [Rosenbrock curvature](../../../../analyses/016-2026-10-11-rosenbrock-hessian/README.md) shows lower loss with higher attained ordinary curvature in one deterministic setup; [loss arithmetic](../../../../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) shows lower attained curvature largely accompanying lower loss, with similar curvature at matched loss. Neither supplies a universal flatness or stability law.

## 4. Main empirical findings

### Local cancellation has longer-term structure

**Empirical:** all nine GD trajectories have positive total all-pair interaction, although seven have negative accumulated adjacent interaction. At the largest rates, `C` is 0.114/0.156/0.086 and `R` is 0.026/0.037/0.054 for MLP/ResNet/Transformer. These runs attain their sweep's lowest final training losses. Adjacent pairing removes substantial path length while leaving residual motion; it does not ensure loss decreases after every pair. Lag profiles show recurring odd/even organization beyond lag one. [Analysis056](../../../../analyses/056-2026-11-05-gd-paired-displacement/observations/001-paired-displacement.md).

Pythia provides a complementary replicated result: larger models have more frequent negative adjacent raw-gradient pairs and lower complete validation loss in every seed block. All nine cumulative raw interactions remain positive; lag-one truncation misses 97.90–98.57% of squared cumulative raw-gradient norm. These are raw-stream observations under AdamW, not actual-displacement identities or a scaling law. [Replicated conclusions](../../../../analyses/003-2026-10-04-raw-gradient-regimes/observations/003-replicated-conclusions.md).

### Sampling changes interpretation

**Empirical:** smaller batches weaken trained-state reversal in each of three initialization blocks across all three models. Nevertheless, Transformer batch eight has almost identical mean cosine from initialization and a trained state (about -0.26), while full-objective changes have opposite signs (-3.505 and +0.064). The same local signature therefore need not imply the same motion or progress. These continuations change both sampled gradients and later states; equal update counts also imply unequal data exposure. [Analysis037](../../../../analyses/037-2026-11-02-sgd-batch-extension/REPORT.md), [noise controls](../../../../analyses/027-2026-10-30-minibatch-noise/REPORT.md).

### Adam reversal is not an uphill update

**Empirical:** 806 excursions recover within the horizon; 129 exceed 10%. Of 127 defined pre-rise cosines, 117 are negative, and all 129 peak gradient norms exceed their pre-rise values. Yet all 794 outgoing updates within their rise intervals satisfy `g dot actual_increment < 0`. The intervals can contain individual downward steps; the claim concerns their total rise and each step's first-order direction. Negative raw cosine does not mean Adam points uphill relative to the current slope. [Analyses036/057](../../../../analyses/057-2026-11-05-adam-response-memory/observations/001-response-memory.md).

Ordinary low-rate trajectories also exhibit reversal and `z>2` without a qualifying near-future excursion. Adding cosine to log-norm severity fits worsens held-rate error in all seven folds. These observations delimit a diagnostic's specificity; they do not establish a prediction rule.

### Smoother symptoms need outcome-based evaluation

**Empirical, within-study comparisons:**

- Pythia held-LR branches show stronger cancellation but better complete validation loss than accelerated decay, which restores alignment earlier.
- Removing coordinate scaling at matched native adaptive-component norms worsens fixed-probe loss in all 18 branches; three-seed geometric penalties at fork225 are 26.84/57.90/106.49% for 14M/31M/70M.
- Halving Transformer LR at state1024 changes mean raw cosine from -0.299 to +0.589 while raising 64-update mean training loss 3.23%.
- MLP clipping at one raises whole-window mean loss 48.01% and maximum loss 3.82-fold despite improving its endpoint. ResNet clipping lowers mean/maximum loss 28.32/52.15% while raising last-32-update mean loss 78.03%.

These outcomes use different objectives and horizons and must not be pooled. A smaller peak, fewer events, lower endpoint and better average loss are distinct results. Selection, threshold, timing and warm-state dependence remain material. [Exact comparisons, methods and sources](evidence_selection.json).

## 5. Framing, evidence selection and implications

The literature already provides secant curvature, consecutive-gradient diagnostics, oscillatory descent and trajectory directionality. The contribution here is their disciplined integration with measured temporal cancellation and optimizer-aware empirical distinctions. The [novelty review](reviews/theory-novelty.md) maps the overlap and original rejection concerns; the formulation no longer assumes distant correlations vanish or equates parameter norms with stability.

**Retain centrally:** GD paired/all-lag accounting, three-case SGD evidence, compact Adam anatomy and actual-step alignment, replicated Pythia context, and representative matched interventions with clearly named outcomes.

**Retain as support:** exact identities, detailed measurement conventions, event thresholds/censoring, optimizer-memory derivation and negative severity-transfer result.

**Omit from the main argument, preserving records:** exhaustive severity regressions, predictive feature searches, repeated MLP schedule-development variants, toy sweeps, loss-arithmetic comparisons and unexecuted expansion plans. They answer adjacent questions or duplicate a point already established.

The practical implication is to diagnose the relevant vector stream and temporal scale before interpreting an intervention. This opens a route to better intervention hypotheses without claiming that an intervention has already been solved. Activation sparsity and orthogonalized updates remain **speculation**. Independent-seed Adam anatomy, calibrated HVP agreement, generalization guarantees and a validated controller remain outside the present contribution. The submission can close when its bounded claims, evidence, provenance and prose agree.
