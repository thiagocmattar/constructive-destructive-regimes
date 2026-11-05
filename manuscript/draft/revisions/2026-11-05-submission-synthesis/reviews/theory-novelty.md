# Independent theory and novelty review

Date: 5 November 2026. Role: theory/novelty reviewer; initial manuscript before the submission-synthesis rewrite. This report evaluates scientific scope, mathematical interpretation and overlap with primary literature. No new training or manuscript edits were performed. It is an internal simulated review, not a prediction of an editorial decision.

## 1. Recommendation and central argument

The existing evidence supports a diagnostic paper about **what temporal gradient interference measures, how it changes across temporal and optimizer scales, and why reducing its visible symptoms need not improve optimization**. It does not support a new general theory of stability, a new curvature estimator, a universal spike mechanism, or a validated intervention rule.

The strongest central question is: **When successive updates oppose one another, what is being cancelled, over which horizon, and what does that imply for objective progress?** This preserves the original constructive/destructive idea while addressing the rejection directly. Its answer has three distinct parts:

1. Adjacent opposition is a local geometric fact. Exact path accounting shows that strong adjacent cancellation can coexist with positive total interaction over all gradient pairs and continued loss reduction.
2. The interpretation depends on the optimizer and sampling process. In plain GD the projected response has a classical secant-curvature meaning; in SGD it mixes changing states with sampling; in Adam raw-gradient reversal does not identify an uphill applied update.
3. A diagnosis must retain the measured objective as its outcome. Existing matched LR/clipping interventions can show why making norms, oscillations, or alignment look smoother is not itself success.

This is a constructive result, rather than a collection of warnings: it provides a consistent way to locate cancellation and avoid interventions aimed at the wrong vector or temporal scale. Describe the method as a diagnostic accounting or measurement framework. Do not advertise the elementary identities as difficult new optimization theory.

The strongest original-looking component is the **integrated empirical interpretation**, including the paired-update accounting and optimizer-aware counterexamples. I found no basis in this focused review for an exhaustive priority claim. A TNNLS/PAMI reviewer can still consider the contribution incremental; no wording can guarantee acceptance. The response should be clear evidence and precise distinction from close work, not additional unconnected experiments.

## 2. Contribution audit

| Component | Assessment | Defensible role |
| --- | --- | --- |
| Squared norm of a sum equals individual energies plus pair products | Elementary identity; same criticism as the prior rejection if sold as a theorem | Accounting foundation that preserves all lags |
| Consecutive-gradient cosine | Established diagnostic ingredient | Operational definition of local constructive/destructive interaction |
| Directional curvature from consecutive GD gradients | Exactly a classical secant quotient, inverse BB1 when defined | Interpretation of measured response, with explicit noise and smoothness scope |
| Reversal threshold one versus quadratic loss threshold two | Classical quadratic/line-search geometry expressed in observed quantities | Explain why reversal is not failure |
| Ratio of displacement to path length | Standard straightness-type geometry | Quantify motion; never call it retained learning or optimization efficiency |
| Zero-cross-term reference and paired factorization | Correct elementary refinements; not a deep theorem | Locate cancellation and prevent a false inference from a small path ratio |
| All-nine positive total interactions despite seven negative adjacent interactions | Specific, useful retained-data observation | Directly defeats the lag-one cumulative interpretation |
| SGD state/batch comparisons | Useful controlled evidence with three initialization blocks | Show observed cosine is sampling- and state-dependent |
| Adam rise steps remain locally descending | Strong empirical distinction in the recorded sample | Separate successive raw gradients from the applied direction |
| Adam memory/curvature discrepancy identity | Correct algebra under stated assumptions; neural calibration absent | Optional appendix connection, not the main novelty or a new experiment prerequisite |
| Existing intervention results | Useful if sources and matched comparisons are retained | Demonstrate that symptom changes and objective improvements need separate assessment |

## 3. Closest primary literature and required positioning

### Secant curvature: Barzilai and Borwein

Let `s = theta_(t-1) - theta_(t-2) = -eta g_(t-1)` and `y = g_t - g_(t-1)` for full GD. Then

\[
\frac{s^\top y}{s^\top s}
=\frac{1-g_{t-1}^\top g_t/\|g_{t-1}\|^2}{\eta}
=\widehat\kappa_t.
\]

The BB1 step is `(s^T s)/(s^T y)`; the paper's curvature statistic is its reciprocal wherever that reciprocal is defined. The quotient itself can be zero or negative, so do not imply a positive permissible step always exists. This equivalence is direct algebra using the classical definition. Restore the existing `barzilai1988two` citation, and describe the innovation as interpretation and measurement rather than invention of Hessian-free curvature. [Barzilai and Borwein, *Two-Point Step Size Gradient Methods*, IMA Journal of Numerical Analysis 8(1):141–148, 1988](https://doi.org/10.1093/imanum/8.1.141); [primary paper mirror](https://pages.cs.wisc.edu/~swright/726/handouts/barzilai-borwein.pdf).

### Consecutive inner products: convergence diagnostics and hypergradients

Chee and Toulis use the Pflug family of diagnostics to detect the transition to a stationary region during constant-rate SGD. Their object is a convergence/stationarity test, not the same task as descriptive path accounting. [Chee and Toulis, AISTATS 2018, PMLR 84:1476–1485](https://proceedings.mlr.press/v84/chee18a.html).

Pesme, Dieuleveut and Flammarion analyze the consecutive-inner-product test and show that its running average can fail as an efficient stationarity diagnostic even on quadratics. An expected sign does not make a reliable finite-sample test. The present manuscript should explicitly inherit this distinction rather than reintroduce a sign-based convergence claim. [Pesme et al., ICML 2020, PMLR 119:7641–7651](https://proceedings.mlr.press/v119/pesme20a.html).

Adjacent gradient products also arise in hypergradient learning-rate adaptation, including variants for momentum and Adam. Retaining one previous gradient and using a dot product is therefore not new. The present study does not claim a competitive adaptive optimizer. [Baydin et al., *Online Learning Rate Adaptation with Hypergradient Descent*, ICLR 2018](https://arxiv.org/abs/1703.04782).

### Trajectory directionality and oscillatory descent

Singh et al. study pairwise parameter-direction maps, angles between consecutive updates, distance from initialization and path behavior across vision/language tasks. They distinguish macro-scale directionality from micro-scale oscillations and analyze momentum/weight-decay effects. The manuscript cannot claim the first distinction between local reversal and global progress. Its useful specific addition is a transparent decomposition of adjacent cancellation versus subsequent accumulation, retaining gradient magnitudes, together with explicit loss outcomes and optimizer-stream semantics. [Singh et al., *The Directionality of Optimization Trajectories in Neural Networks*, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37771cc0be272368102a37f202bb88d8-Abstract-Conference.html).

The paper already cites the EoS, self-stabilization and central-flow literature. Keep that acknowledgment adjacent to the GD finding. Do not state that the present algebra explains a mechanism that those works establish: the paired factors measure motion but do not identify its causal source.

### Adam loss excursions

Bai et al. provide a second-moment/preconditioner mechanism, a momentum-adjusted directional quantity and experiments including stochastic Transformers. Their published Eq. (7) gives the same frozen-time scaling used in the manuscript's comparison. Their baseline is a directional HVP computation, not a full Hessian spectrum. The present observations cannot replace or outperform that mechanism simply by plotting a similar raw-gradient response. The defensible addition is the exact distinction between the response along the actual Adam step and the frozen-time proxy, plus recorded cross-alignments and intervention outcomes. [Bai et al., *Adaptive Preconditioners Trigger Loss Spikes in Adam*, ICML 2026, PMLR 306:5566–5601](https://proceedings.mlr.press/v306/bai26o.html).

### Temporal interference versus simultaneous sample/task conflict

Gradient diversity concerns gradients concurrently evaluated at a shared state and its effect on mini-batch/distributed optimization. Its vector scope differs from consecutive gradients at different states. [Yin et al., AISTATS 2018, PMLR 84:1998–2007](https://proceedings.mlr.press/v84/yin18a.html).

PCGrad projects conflicting task gradients to reduce task interference. The present temporal diagnostic does not establish task conflict, representation erasure or a benefit from orthogonalizing successive steps. [Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html). The author's sparsity/orthogonalization ideas may motivate future research, but remain untested ideas and need not appear as proposed methods here.

## 4. Mathematical audit and wording corrections

**Core algebra.** The current displacement, zero-interaction reference, paired factorization, finite-population covariance and Adam residual recurrences are consistent under their declared nonzero-denominator and objective assumptions. The noise expectation requires fresh gradients conditionally unbiased given the past; it is not a statement about arbitrary shuffled-epoch batches.

**Smoothness.** The classical path-average Hessian identities require the specified smoothness along the complete step. ReLU models can cross gradient discontinuities at activation boundaries. They should not be presented as exact Hessian averages for every recorded ResNet/Transformer step. The directly measured secant/projection remains well defined when its endpoint gradients and denominator exist.

**Loss statement.** The endpoint-gradient loss formula is a trapezoidal approximation, exact on quadratics. A sign prediction is reliable only when the predicted change dominates approximation error. Replace “jointly produce loss increases” with “determine the loss sign in a quadratic model” or “give a local explanation under the stated approximation.” A near-zero approximate sign is particularly fragile. The correct positive contribution of destructive alignment is not a causal attribution of a neural-network spike.

**Time indexing.** The response formed from `g_t` and `g_(t+1)` is known after traversing the corresponding step. It describes that observed transition. This is appropriate for diagnosis and should not acquire advance-warning language through phrases such as “predicts the next spike.” Event-peak norm amplification is explicitly retrospective.

**Adam bridge.** With `a=(1-beta1)/(1+beta1)` and `r_t=m_t-a g_t`, direct substitution gives `r_t=beta1 r_(t-1)+beta1 a(g_t+g_(t-1))`. The bias-corrected residual and discrepancy bound then follow as stated. The decomposition subtracts a chosen proportional reference; it establishes a comparison of quantities, not an identified mechanism of spike creation. A small unweighted residual is not enough if the preconditioner magnifies it. Keep the published proxy as a *directional proxy*: the Euclidean quotient of the generally nonsymmetric product `P H` is not automatically an extremal Hessian eigenvalue.

**Geometric scope.** Euclidean parameter angles and path length depend on parameterization and parameter scope. They describe the chosen trainable-coordinate system. They are not invariant measures of function-space change or retained knowledge. “Destructive” must consistently mean cancellation of a component of motion at the specified lag, not detrimental learning.

**Replication scope.** Multiple events in one trajectory are not independent initializations. One full-batch initialization per model supports conditional demonstrations and counterexamples, not a population claim about how often a universal mechanism occurs. The three-block SGD/Pythia cohorts should receive their actual stronger scope where included, without transferring it to Adam full-batch events.

## 5. Direct response to the previous rejection

| Prior concern | Current resolution | Remaining writing obligation |
| --- | --- | --- |
| Undefined stability and parameter norm as its surrogate | Reframe around measured temporal interference and displacement from initialization | Remove stability from the headline claim/title unless a precise claimed stability property exists |
| Elementary norm expansion sold as analysis | Present identities as the bookkeeping needed for empirical diagnosis | Lead contribution list with the diagnostic question and findings |
| Beyond-lag-one correlations assumed negligible | Exact all-lag identity, lag measurements, and positive all-pair/negative-adjacent observation | State explicitly that no lag truncation is assumed |
| Norm-control literature omitted | Acknowledge that probabilistic bounded-iterate/generalization analysis asks a different question | Do not imply the decomposition supplies a high-probability bound |
| Covariance called impossible online | Distinguish exact dataset covariance from stochastic approximations | Say the selected diagnostics use existing batch gradients, not that competing estimates cannot be obtained |
| Negative “batch distance” expression | Inactive legacy quantity is absent from compiled contribution | Do not revive it without a separate estimator/positivity audit |

The two reviewer-cited papers are real primary research: [Lei, *Towards Better Generalization Bounds of Stochastic Optimization for Nonconvex Learning*, author manuscript](https://leiyw.github.io/paper/TPAMI2025.pdf), and [Li and Liu, *High Probability Guarantees for Nonconvex Stochastic Gradient Descent with Heavy Tails*, ICML 2022](https://proceedings.mlr.press/v162/li22q/li22q.pdf). Their probabilistic guarantees should not be equated with empirical parameter-path accounting. A short positioning sentence is enough; reproducing their proof program would change this paper's scope.

## 6. Evidence order and stopping rule

Recommended argument order:

1. Define constructive/destructive **temporal** interaction and distinguish vector scope and horizon.
2. Provide compact exact path accounting and the classical secant/loss interpretation.
3. Demonstrate local cancellation together with cumulative motion and objective progress in GD; use SGD to expose the sampling interpretation.
4. Separate raw-gradient response from Adam's applied direction, using the rise-step result as a prominent observation.
5. Use the strongest already-completed Pythia and clipping/LR comparisons to show why diagnosed symptoms require outcome-based evaluation.
6. End with what the framework enables: interpreting training trajectories and formulating targeted intervention hypotheses. Do not turn untested interventions into a contribution.

Move the detailed Adam memory/HVP bridge to an appendix and retain a brief conceptual reference. Remove main-text severity-regression figures if they interrupt this argument; retain the negative transfer finding in a short boundary statement or appendix. Preserve their artifacts and provenance.

The 5 November proposed experiment plan asks a legitimate *different* question: whether the raw response can be calibrated against a frozen-time HVP quantity and whether memory/scaling changes explain events. It is not required for the bounded diagnostic framing above. No new runs are necessary merely to remove unsupported estimator-advantage, mechanism, prediction or scale claims.

The submission stopping rule should be: every main claim has a retained source and correct scope; the closest literature is acknowledged; sections answer the central question; mathematics and compiled presentation pass review. Do not continue until simulated reviewers promise acceptance. Use adversarial review to expose contradictions and unsupported claims, then state residual novelty and replication limitations honestly.

## 7. Source inspection record

Read active `sections/01-introduction.tex`, `02-related-works.tex`, `03-methods.tex`, `04-gd-experiments.tex`, `05-sgd-noise-experiments.tex`, `06-adam-diagnostics.tex`, `06-loss-appendix.tex`, `07-gd-diagnostics.tex`, `05-conclusion.tex`, and the active entry point. Read `research/INDEX.md`, `DEFINITIONS.md`, `METHODS.md`, `METRICS.md`, `MANUSCRIPT.md`, `REVIEW_CONTEXT.md`, manuscript orientation, original reviewer PDFs, and the 5 November Adam experiment plan. Original rejection email PDF has no extractable text; its preserved textual transcription and both reviewer PDFs provided the substantive concerns. Primary literature was checked online on 5 November 2026; Singh's full published paper and the previously downloaded published Bai paper were inspected locally as well. This is a focused novelty review, not an exhaustive systematic literature search.
