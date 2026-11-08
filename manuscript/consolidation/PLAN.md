# Paper anatomy and closure map

- **Central argument:** learning updates reinforce or reverse earlier changes. Direction, amplification and accumulation explain training dynamics; tracing their functional expression makes targeted interventions possible.
- **Structure:** retain *Constructive and Destructive Training Regimes: A Framework for Interpreting SGD Stability*, Sections I–VI, existing theory subsections and four GD/SGD figures. Develop characterization → tracing → intervention inside V.
- **Reading contract:** this is the complete executive map. Links supply supporting evidence; no other consolidation document is required.

## I. Introduction — motivation, questions, contributions

- **Restore the current manuscript's motivation.**
  - Longer training horizons and constrained compute make disruptive excursions and hyperparameter sensitivity consequential. Retain the scaling/efficiency citations as motivation; these experiments measure dynamics, not large-scale compute savings.
  - Smaller updates can suppress excursions while slowing progress; oscillatory training can keep improving. **Which reversals are productive, which become excessive, and what should be controlled?**
  - Curvature explains geometric risk; representations expose changes in the learned function. Connect them using existing gradients, then localize and limit excessive functional movement.

- **1. Constructive/destructive interaction.**
  - **Known → question:** [EoS](https://arxiv.org/pdf/2103.00065) explains oscillatory progress; [Chee–Toulis](https://proceedings.mlr.press/v84/chee18a.html)/[Pesme](https://proceedings.mlr.press/v119/pesme20a.html) use gradient products for stationarity. How do local reversals combine into accumulated movement during learning?
  - **Contribution/evidence:** exact all-pair SGD displacement decomposition and normalized displacement R. All nine GD cases have positive total pair interaction; seven have negative adjacent interaction. **Home:** III opening/D → IV Figures 2–3.

- **2. Directional curvature without additional derivatives.**
  - **Known → question:** secants exploit gradient differences; spectral analyses require additional curvature measurements. What curvature quantity do existing norms/alignment recover, and how does sampling affect it?
  - **Contribution/evidence:** gradient-only directional Hessian estimation, exact smooth-GD path-average meaning, approximation bound and explicit noise term; **1,188 HVP comparisons** supply independent validation. **Home:** III-A/B/C → IV Figures 1/4 and support.

- **3. Reversal, amplification and loss excursions in time.**
  - **Known → question:** EoS and [Bai et al., §§4–5](https://arxiv.org/html/2506.04805v2#S4) explain oscillation and Adam's preconditioner mechanism. When do our raw-gradient measurements indicate upcoming increases, depending on magnitude and lead time?
  - **Contribution/evidence:** local GD loss relation plus exhaustive, timed Adam comparisons. Destructive alignment is a broad antecedent; amplification is more selective, with architecture-dependent timing. **Home:** III-E → IV validation → V-A.

- **4. Functional motion and intervention.**
  - **Known → question:** [Catapults](https://proceedings.mlr.press/v235/zhu24h.html) studies output/NTK dynamics; [Kosson, §6](https://arxiv.org/html/2410.23922v1#S6) connects update geometry and representation change. Which representations participate in excursions, and can selective control retain progress?
  - **Contribution/evidence:** fixed-probe tracing, temporal modes and per-unit affine-displacement caps. Selected MLP/ResNet/Transformer caps reduce window-mean training loss by **22.85%/16.22%/26.00%**, with MLP replication and norm-matched controls. **Home:** V-B/C.

- **Edit:** preserve these four components. Replace the broad claim that prior stability work is “fundamentally local” with the specific diagnostic and functional questions above.

## II. Stability Analysis of SGD — precise predecessor roles

- **Keep four subsections:** local curvature; minibatch stochasticity; curvature/noise/EoS; practical large-model stability.
- **Curvature/noise:** distinguish directional curvature, leading eigenvalues and sampling. Credit gradient-product diagnostics and [hypergradient adaptation](https://arxiv.org/abs/1703.04782); cosine and scalar secants are established ingredients.
- **Adam:** distinguish raw Hessian from **preconditioned** curvature in Bai. Our raw-gradient response does not independently prove preconditioner decoupling.
- **Representation control:** Kosson's §6/Appendix B.6 scales layers using a signal-to-noise model of **relative** response. Our cap directly limits per-unit **absolute** affine RMS using past budgets. Both fix local inputs; cite this closest intervention precedent.

## III. Constructive and Destructive Regimes — sharpen the existing theory

- **Opening:** retain exact all-pair displacement. Negative adjacent cosine labels a destructive *step*; persistence describes a regime. Cancellation alone does not establish erased knowledge.
- **A — alignment/curvature:** retain the boxed relation with plain-SGD, objective/noise and Taylor assumptions; distinguish reversal from amplification.
- **B — make inexpensive curvature estimation prominent.**
  - Define r = ||g_current||/||g_previous|| and Q = eta*kappa_hat = 1 − rho*r.
  - Under smooth full-batch GD, kappa_hat is **exactly the directional Hessian Rayleigh quotient averaged along the incoming step**. Starting-point error ≤ L_H*eta*||g_previous||/2 under a Lipschitz Hessian.
  - Cost: scalar reductions and one retained gradient; **O(p) work/storage, no extra backward pass or HVP**. Estimate a directional scalar, not the Hessian matrix or leading eigenvalue. Measure overhead before claiming negligible runtime cost.
  - **Exact prior art:** [BB, printed p.142, Eq. (6)](https://pages.cs.wisc.edu/~swright/726/handouts/barzilai-borwein.pdf#page=3) gives alpha_BB = ||delta_theta||²/(delta_thetaᵀ delta_g). Under plain GD, **kappa_hat = 1/alpha_BB** when that step size is defined. [Tan et al., §2/Eq. (6)](https://papers.neurips.cc/paper_files/paper/2016/file/c86a7ee3d8ef0b551ed58e354a836f2b-Paper.pdf#page=3) explicitly explains the scalar-Hessian interpretation. Claim our diagnostic interpretation, noise analysis and validation; credit the secant identity here.
- **C — sampling:** retain explicit noise and finite-population variance. [027](../../analyses/027-2026-10-30-minibatch-noise/REPORT.md): batch-32 median discrepancy **.7467 → .0030** after full-gradient noise correction. This is **offline validation**, not a free online SGD correction.
- **D — accumulation:** keep R, its example and longer lags. Low displacement/path-length ratio measures cancellation, not wasted compute or generalization.
- **E — loss criterion and clock:**
  - State s means s completed updates. Q_s uses gradients at s−1 and s, available after backward and before update s→s+1.
  - Quadratic full-batch GD gives L_s−L_(s−1) = eta*||g_previous||²*(Q_s−2)/2. Hence **Q_s>2 iff rho_s<−1/r_s** for a strict *incoming* rise. It implies reversal and growth; their conjunction alone is weaker. Equality gives zero quadratic change.
  - Nonlinear loss uses an approximation. Warning about L_(s+1)−L_s is a separate temporal test. Retain zero-denominator conventions and smoothness assumptions, including activation-boundary limitations.

## IV. Empirical Characterization — four existing figures, explicit tests

- **Cases:** CIFAR-10 MLP, 5,000 examples/half-squared error; CIFAR-10 ResNet-32, 50,000 examples/cross-entropy; WikiText-2 Transformer, 2,088,600 supervised tokens/token-weighted NLL.
- **Figure 1 — LR/curvature/loss:** retain [024's nine GD conditions](../draft/figures/analysis024/gd_loss_projected_change_spectral_clean.pdf), 3,000 updates. Distinguish proxy, directional HVP reference and spectral estimate; label full-MLP versus subset spectral coverage.
- **Figure 2 — reversal with progress:** retain [norms, alignment and R](../draft/figures/analysis024/gd_reversal_norm_retention.pdf); connect local reversal, positive all-pair accumulation and measured loss progress.
- **Figure 3 — temporal organization:** retain [035's backward lags](../draft/figures/analysis035/gd_backward_compact.pdf), endpoint 2256/lags 1–256; test the lag-one simplification.
- **Figure 4 — batch effects:** retain [037's 546 trajectories](../draft/figures/analysis037/sgd_batch_size_compact.pdf), three cases/three initializations. Frozen-state controls isolate sampling; moving trajectories also change geometry.
- **Supporting validation:** 027's HVP/error-versus-batch check and [024's finite-step loss check](../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/007-section-iv-critical-review.md). Of 26,991 GD transitions, high-LR Transformer has **1,325/2,999 sign disagreements**: the quadratic loss formula is not exact along neural trajectories.
- **Finish:** distinguish one-initialization GD characterization from replicated minibatch evidence. No new GD/HVP/lag/batch training is required.

## V. Raw-Gradient Diagnostics under Adam — characterize, trace, intervene

- **Opening:** preserve the observable and Appendix A identity. Adam determines the traversed direction; the preceding raw gradient determines the Hessian-response projection. The computation transfers; plain-GD thresholds do not automatically transfer.

### A. Characterization: exhaustive timing and severity

- **Qualitative entry:** retain [058's high-LR overview](../../analyses/058-2026-11-06-intervention-consolidation/figures/00-high-lr-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf). Peaks illustrate; the quantitative population contains every eligible update.
- **Completed audit:** nine Adam arms; **9,207** decisions, **2,233 increases/6,974 decreases**. No quiet-state/peak selection. Initial diagnostics are undefined; terminal states have no outgoing update.
- **Main table:** signals at s, outcome L_(s+1)−L_s; signal cells give **flagged increases / flagged decreases**.

| High-LR case | All increases / decreases | rho<0 | r>1 | Q≥2 |
|---|---:|---:|---:|---:|
| MLP, .001 | 454 / 569 | 404 / 412 | 261 / 285 | 172 / 136 |
| ResNet, .003 | 95 / 928 | 86 / 214 | 47 / 277 | 21 / 15 |
| Transformer, .003 | 152 / 871 | 145 / 376 | 99 / 322 | 81 / 66 |

- **Magnitude dependence:** compare rho<0, r>1, their conjunction and Q≥2 at minimum relative step increases **>0, .001%, .1%, 1%, 5%, 10%, 25%, 50%, 100%**.
  - At **≥10%**, negative cosine precedes **149/168, 12/12, 26/26** rises; growth **110/168, 11/12, 22/26**; their conjunction **105/168, 11/12, 22/26**; Q≥2 **82/168, 7/12, 22/26** (MLP/ResNet/Transformer).
  - Coverage strengthens with severity in ResNet/Transformer; MLP is nonmonotonic: all three ≥50% steps escape the current Q flag. Retain these cases and the small denominators.
  - rho<−1/r and Q>2 classify identically; no exact Q=2 occurs. They are two expressions of one condition.
- **Forward timing:** all horizons **1–16**; separate the increase at that future update, net change from current loss, and any rise above current loss within the window. Compare coverage, precision, false alarms and base frequency, including common states 1–1008 for each model.
  - Transformer ≥10% step rises: Q coverage **22/26 → 25/26 → 26/26** at leads 1→2→3. Lead-3 precision is **26/147**, versus base frequency **26/1,008**. The complete curve subsequently declines; three steps is exploratory.
  - Remembering a warning for 16 past states covers all **168/12/26** ≥10% target steps, while flagging **688/248/488** non-target steps. Here the common cohort is states 16–1023; each rise is counted once. Wider temporal coverage does not establish an accurate universal alarm.
  - Where loss is **not already rising**, negative cosine precedes a ≥10% rise within eight updates in **169/191, 29/33, 64/65** cases. Precision: **43.4%/13.6%/16.8%**; base frequency: **34.0%/3.6%/7.6%**.
- **Paper claim:** destructive alignment supplies a broad precursor; amplification supplies more selective, magnitude-dependent evidence; localization differs across models. **Pre-update association is demonstrated; added predictive value requires validation.**
- **Evidence:** [magnitude, lead-time and warning-memory curves](../../analyses/055-2026-11-05-adam-directional-amplification/figures/adam_forward_and_severity.pdf); [all nine arms/checks](../../analyses/055-2026-11-05-adam-directional-amplification/observations/004-timing-severity-audit.md). One seed/model, dependent states; no new training.

### B. Tracing: map functional oscillation through the network

- **Algorithm:** fixed inputs/stochastic context → activation increments → output/raw-gradient association → layerwise cosine/RMS → uncentered, unnormalized temporal Gram/SVD → signed coefficients. Refit each window; gradients themselves are not differenced.
- **Combined display:** loss/output motion, grouped layer heatmaps, energy spectrum and coefficients. Follow [065 Figure 07, page 2](../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/07-distributed-displacement-and-controls.pdf); include absolute amplitudes alongside within-layer normalization.
- **Existing [063 windows](../../analyses/063-2026-11-07-architecture-instability-tracing/observations/005-technical-report-event-window.md):** MLP 450–500/all 5,000 examples; ResNet 280–310/one intact 1,000-example normalization group; Transformer 552–600/64 sequences, 2,240 supervised positions.
- **Finding:** output rank95 **5/16/15**, first-mode energy **53.02%/24.03%/41.65%**: structured motion, not a universal single mode. Short-window MLP gradient/output correlation is **.642**, not the longer-window .776.
- **Interpretation:** localize participation/amplitude. Window-fitted SVD is retrospective; correlated layers do not establish a causal initiating layer. These windows differ from V-A's largest peaks. No new tracing run is required for this claim.

### C. Intervention: test whether functional interpretation helps

- **Author's order:** LR schedules → clipping → beta2 → local-displacement cap. Use [058](../../analyses/058-2026-11-06-intervention-consolidation/README.md), [059](../../analyses/059-2026-11-07-mlp-beta2-reduction/README.md) and ongoing [065](../../analyses/065-2026-11-07-mlp-frozen-lr-double/README.md).
- **Cap:** for each targeted affine unit, s=RMS(delta_W*x+delta_b) on current fixed local inputs; scale proposed increments by min(1,tau/s). Positive tau comes from untreated past updates; s=0 leaves the proposal unchanged. Document disabled zero-budget units; preserve Adam moments/clock.
- **Bound:** that unit's direct affine RMS on those inputs. Upstream changes and whole-network output remain unconstrained. The controller requires no SVD.
- **Mechanism:** full-probe MLP squared-error accounting separates beneficial residual correction from excessive squared output movement. Freezing can remove reversals yet worsen mean loss; caps can improve mean loss while reversals persist. Include the MLP norm-matched control.
- **Outcome table:** full-window mean, final-32 mean, positive loss variation, output motion and held-out loss; unchanged baseline throughout. Cap benefits are **training-window** improvements. Terminal held-out loss worsens in all three cases; the 54-arm MLP ablation does not support increasing LR above the same cap's native LR.
- **Comparators:** late beta2 switching with retained state differs from training throughout at lower beta2. Inactive clipping does not establish superiority over effective clipping. Resolve direct ranking with B below.

## VI. Conclusion and appendices

- **Conclusion:** interaction → curvature/noise → loss timing → functional motion → control. Separate demonstrated effects from forecasting, generalization and runtime-efficiency claims.
- **Appendices:** derivations/clock, BB equivalence, HVP/error checks, complete severity/horizon tables, tracing/probe contracts, calibration, controls and held-out outcomes.
- **Abstract:** update last; replace outdated “future interventions” with completed findings.

## Remaining empirical work — two bounded decisions

### A. Validate added predictive value if retaining a forecasting claim

- **Completed:** sign, severity, forward-horizon and causal warning-memory audits. An “every next-step increase” claim is rejected by existing data.
- **Next local test:** fixed L2-logistic comparisons: loss level/one- and four-step trend/16-step draw-up → add log gradient norm/log r → add rho/Q; include fixed threshold/memory baselines. Primary target: next-step ≥10% relative rise; secondary: ≥10% above current loss within eight updates.
- **Freeze before fitting:** features and warning memory. Use chronological 60/20/20 blocks, 16-update boundary gaps, training-only normalization and validation-only alarm thresholds. This is a backtest on already-inspected trajectories; fresh seeds provide confirmation.
- **Success:** improve average precision and recall at ≤5 false alarms per 100 negative decisions over the same loss/norm baseline; report each arm's support. Mark single-class test blocks unevaluable. Otherwise retain characterization; do not change targets/splits to obtain a positive result.
- **Confirmation:** two fresh initializations/architecture, 1,024 unchanged high-LR updates each; apply the frozen rule once. Reuse these six untreated trajectories for B where qualified. Otherwise retain exploratory temporal association.

### B. Make intervention comparisons fair and reproducible

- **Configurations:** unchanged Adam, native half-LR, global raw-gradient clipping before moments, distributed q50 cap; add beta2=.95 for MLP.
- **Improve clipping:** use the median untreated raw-gradient norm over the same eight-update pre-fork calibration prefix. Keep historical norm-1 results separately. Equal percentiles do not imply equal strength; report activation and applied-norm reduction.
- **Seed-0 forks:** MLP **448→1152, LR .001**; ResNet **264→576, LR .003**; Transformer **552→928, LR .003**. Match tensors, optimizer/RNG, objective, hardware, precision and V-B probes; freeze cap budgets.
- **Scope:** at most **13 seed-0 post-fork arms** before qualified reuse. Cross-seed ranking requires the same configurations on two additional initializations/model: **39 total arms**. Reuse qualified evidence or rerun the paired unchanged baseline.
- **Endpoints:** V-C's metrics, maximum loss, full held-out evaluation at fork/fork+32/endpoint, controller time/memory. Preserve divergences/inactive controls; no outcome-driven retuning.
- **Execution:** continue Analysis 065 with new run IDs. Prepare duration, cost/storage limits and verification before launch; this map specifies the work without executing it.

## Closure sequence and completion test

- **1 — Rewrite I–IV:** existing theory/evidence suffice; strengthen motivation/curvature and qualify the loss criterion.
- **2 — Assemble V-A/B:** use the completed population/temporal audit and retained tracing; run A for a validated forecasting claim.
- **3 — Complete V-C:** use B for replicated direct ranking; otherwise present controlled case studies within existing evidence.
- **4 — Finalize:** abstract/VI, equation timing, citations, figures, reproducibility, compilation and visual inspection. Resolve the supplied TNNLS revised-resubmission restriction before submission.
- **Map status:** sufficient to proceed to closure. Every contribution has a question, evidence, manuscript location and closing action. Manuscript and human originals remain unchanged.
