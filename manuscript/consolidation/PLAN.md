# Paper anatomy and closure map

- **Central argument:** learning updates reinforce or reverse earlier changes. Their direction, amplification and accumulation connect optimization geometry to changes in the learned function; measuring those changes motivates selective control.
- **Keep the current paper:** title, Sections I–VI, III's theory spine and four GD/SGD figures. Develop **characterization → tracing → intervention** inside V.
- **Decision from this review:** the evidence supports this explanatory paper. Validated forecasting, a universal initiating layer and general optimizer superiority require additional evidence.
- **Reading contract:** this is the complete executive map. Links provide provenance, not additional required reading. Manuscript editing follows agreement on this plan.
- **Local assets:** [section-indexed evidence package](evidence/README.md); [display/table/algorithm map](evidence/ARTIFACTS.md).

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
- **Figure 1 — LR, curvature, loss:** [024's nine GD conditions](evidence/04-gd-sgd/figures/gd_loss_projected_change_spectral_clean.pdf), 3,000 updates. Distinguish directional proxy from leading eigenvalue; label full-MLP versus subset spectral coverage.
- **Figure 2 — reversal with progress:** [norms, alignment and R](evidence/04-gd-sgd/figures/gd_reversal_norm_retention.pdf); relate adjacent reversal to positive all-pair accumulation and actual loss progress.
- **Figure 3 — temporal organization:** [035's backward lags](evidence/04-gd-sgd/figures/gd_backward_compact.pdf), endpoint 2256/lags 1–256. Test, rather than assume, lag-one dominance.
- **Figure 4 — sampling:** [037's 546 trajectories](evidence/04-gd-sgd/figures/sgd_batch_size_compact.pdf), three initializations. Equal updates entail different data exposure; separate frozen-state sampling controls from evolving trajectories.
- **Validation inset/appendix:** 027's directional-HVP comparison and [024's finite-step residual](evidence/04-gd-sgd/observations/24-007-section-iv-critical-review.md). High-LR Transformer has **1,325/2,999 loss-sign disagreements** with the quadratic approximation; an eigenvalue overlay cannot validate that formula.
- **Closure:** existing evidence suffices. Label single-initialization GD versus replicated minibatch results; no additional GD training is needed.

## V. Raw-Gradient Diagnostics under Adam

- **Opening:** retain the Appendix A identity. For a fixed smooth objective and actual increment u, delta_g = integral H(theta+a*u)u da. Adam chooses u; the previous raw gradient supplies the projection. The same observable transfers as a **mixed Hessian response**, with different semantics from GD curvature or actual-update alignment.
- **Empirical scope:** controlled full-objective Adam trajectories for the three tasks; ResNet/Transformer representation probes cover subsets. These cases do not validate stochastic Adam deployment.

### A. Characterization — distill the timing/severity audit here

- **Figure 5:** retain [058's high-LR overview](evidence/05-adam/a-characterization/figures/00-high-lr-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf); add a compact population summary. Peaks illustrate the pattern; [055's exhaustive audit](evidence/05-adam/a-characterization/observations/55-004-timing-severity-audit.md) checks **9,207 decisions across nine arms**.
- **Main finding:** reversal is a broad antecedent; amplification is more selective and differently timed across models. At one-step lead, Q≥2 covers **172/454, 21/95, 81/152** increases and also precedes decreases.
- **Severity comparison:** counts below are signals available before an individual **≥10% relative loss-increase step**, not independent spikes.

| Signal | MLP: 168 rises | ResNet: 12 | Transformer: 26 |
|---|---:|---:|---:|
| rho<0 | 149 | 12 | 26 |
| r>1 | 110 | 11 | 22 |
| Both | 105 | 11 | 22 |
| Q≥2 | 82 | 7 | 22 |

- **Timing:** Transformer Q coverage becomes **22/26 → 25/26 → 26/26** at leads 1→2→3; lead-3 precision is **26/147**, versus base frequency **26/1,008**. MLP's three ≥50% rises escape the current Q flag.
- **Supporting display:** complete [severity/horizon curves](evidence/05-adam/a-characterization/support/adam_forward_and_severity.pdf), decreases/false alarms and warning memory in the appendix. Keep step increase, cumulative rise and recovered excursion distinct; retain censored events.
- **Claim:** descriptive pre-update association, one seed/model. The angle/norm condition equals Q>2 algebraically; it is not a second validating signal. Forecasting is optional further work.

### B. Tracing — turn the visual pattern into a reproducible anatomical result

- **Algorithm 1:** fixed samples and stochastic context → consecutive activation changes → layerwise cosine/RMS → temporal Gram/SVD → signed coefficients and output association.
- **Dimensions:** D_l is T×P_l; each row flattens all matched examples/tokens/features of one activation change. Do not average samples, center or normalize the primary rows. Raw gradients form their own undifferenced stream.
- **Three case findings from [063](evidence/05-adam/b-tracing/observations/63-005-technical-report-event-window.md):**

| Case / outgoing window / probe | Anatomical finding | Output rank95 / fitted rows |
|---|---|---:|
| MLP / 450–500 / all 5,000 examples | First projection reverses; [tanh attenuates transmission](evidence/05-adam/b-tracing/observations/062-development-summary.md); all five learned stages reverse throughout. | 5 / 51 |
| ResNet / 280–310 / intact 1,000-image BN group | Output-like reversal strengthens through residual depth; first/final-stage alignment-series correlation with output is .578/.991. | 16 / 31 |
| Transformer / 552–600 / 64 sequences, 2,240 positions | At update 558: embedding cosine +.390, attention output −.672, network output −.920. | 15 / 49 |

- **Rule out “only a few large steps”:** [retained-Gram controls](evidence/05-adam/b-tracing/observations/63-006-spectral-controls.md) give top-mode energy **53.0/24.0/41.7%**, versus **7.8/10.5/16.7%** with identical step norms and orthogonal directions. Unit-row rank95 is **5/18/29**: amplitude weighting matters, especially in Transformer.
- **Connect to loss:** use raw MLP outputs and cross-entropy log probabilities. Correct-label readouts reproduce probe loss by their negative mean; their alignment histories correlate **.943/.946** with full-output histories in ResNet/Transformer. The MLP exact decomposition below provides a stronger loss connection.
- **Figure 6:** three model columns; full/probe loss, stage-by-time cosine/RMS heatmaps, spectrum and signed coefficients over the recorded tracing windows. Use the layout of the [selected q50 heatmaps](evidence/05-adam/c-intervention/figures/07-distributed-q50-stage-heatmaps.pdf), extracted from 065 Figure 07 page 2; keep the paired baseline/cap comparison in V-C. Heatmaps and temporal SVD answer different questions.
- **Interpretation/cost:** directional structure and repeated reversals are supported. SVD alone is time-permutation invariant; rank≤T automatically. No unique causal origin or spike-specific concentration follows without interventions/quiet-window contrasts. Rolling cosine/RMS costs O(P); full-window Gram costs O(T²P), eigendecomposition O(T³), and is retrospective.

### C. Intervention — test what the functional interpretation buys

- **Selected presentation — exactly four families, in this order:**

  1. **LR schedules / Figure 7:** [persistent half LR](evidence/05-adam/c-intervention/figures/01-persistent-half-lr-cifar10-mlp-resnet-wikitext2-transformer.pdf). Three architectures; reduce LR by half from the retained warm state, keeping each unchanged baseline.
  2. **Beta schedule / Figure 8:** [late beta2 reductions](evidence/05-adam/c-intervention/figures/01-beta2-reductions-cifar10-mlp.pdf). MLP only; .999→.95/.9/.5 at state 512 with moments/clock retained and constant LR.
  3. **Gradient clipping / Figure 9:** [MLP and ResNet clipping](evidence/05-adam/c-intervention/figures/05-gradient-clipping-cifar10-mlp-resnet.pdf). Global raw-gradient caps before Adam moments: MLP thresholds .5/1, ResNet 1; show raw and clipped norms distinctly.
  4. **Distributed displacement cap / Figure 10:** q50 is the **only representation-inspired intervention** presented. Retain the paired heatmaps, scalar diagnostics and MLP replication; other methods and quantiles are background evidence.
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
- **Background boundary:** entry caps, other quantiles, freezing, directional removal, output-wide budgets and retrospective norm controls are retained in source records for audit. They are outside the selected presentation; do not promote them into additional methods or mandatory experiments.
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
  - Configurations per architecture: unchanged Adam, native half-LR, global raw-gradient clipping and distributed q50 cap. Add representative late beta2=.95 for MLP, with its native bias-correction convention explicit; the existing beta figure retains all three tested reductions.
  - Clipping threshold: median untreated raw-gradient norm over the same eight-update prefix. Report clipping activity and applied-norm reduction; equal quantiles do not equal matched strength.
  - **Maximum selected comparison matrix: 13 continuations for seed 0; 39 across three initializations, before qualified reuse.** Use the same absolute start/window per seed; report absent excursions. Retrospective norm-control experiments are outside this matrix. Retain or recreate untreated prefixes/checkpoints as needed.
  - **Success criterion:** report all paired outcomes, including divergence, tail/test deterioration and inactive controls. A ranking requires replicated matched effects; otherwise state the case-dependent result. Do not retune on the reported windows.
  - Continue Analysis 065 with new run IDs for changed inputs. Prepare measured duration, live cost/storage limits and retrieval checks before any launch.

- **3. Complete method reporting — required.**
  - Report cap activation, budget/stored-increment checks, probe coverage and full held-out evaluation at start/start+32/end.
  - Measure baseline versus controller time and peak memory on matched hardware. Separate cheap scalar diagnostics, offline tracing and the intervention's cost.
  - New comparative execution is needed only where existing records fail these contracts. This review launches none.

- **4. Keep stronger claims conditional.**
  - **Forecasting:** freeze targets/features/horizons; compare against loss-history and norm baselines with temporal gaps, false alarms and fresh-seed confirmation. Current data support characterization without this project.
  - **Spike-specific modes/causal source:** require matched quiet windows/independent probes or targeted interventions. Current tracing supports participation.
  - **Superiority over representation-aware methods:** requires faithful AGC/Kosson/FOOF comparisons with matched tuning budgets. Current contribution is a specified control and bounded tests.
  - **Broad deployment:** stochastic probes, from-initialization use and larger-scale transfer remain future work; no generalization benefit is claimed.

- **5. Final manuscript checks — required.**
  - Compile/render, verify captions and references, align abstract/conclusion with measured endpoints, and package reproducibility records.
  - Resolve the supplied [decision letter's revised-resubmission restriction](evidence/appendices/submission/decision-letter.pdf) and current official TNNLS format before submission. Keep the author's target; no contact or submission is authorized here.

- **Map completion:** **yes, sufficient to proceed to paper closure.** Each contribution has its motivation, predecessor, precise claim, evidence, location and remaining test. The substantive empirical gap is qualified intervention comparison/reporting; the Adam timing audit is one completed characterization step.
