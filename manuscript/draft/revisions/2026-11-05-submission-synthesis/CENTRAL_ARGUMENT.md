# Central argument for the submission revision

Date: 5 November 2026. This document fixes the scientific framing for the current revision. It draws on the retained research record, the original rejection, independent advisor reviews, and the revised introduction and methods. It selects and organizes existing evidence; it does not authorize or require new experiments.

## 1. Thesis and research question

**Temporal gradient interaction becomes informative when local cancellation, cumulative parameter motion, and objective progress are measured together. Their relationship depends on the temporal horizon, sampled objective, and optimizer transformation.**

The central question is:

> When successive updates oppose one another, where does cancellation occur, how does the residual motion accumulate, and what does that geometry establish about objective progress?

The paper answers this question through exact accounting, qualified local interpretations, and controlled observations. It shows how to identify immediate cancellation, distinguish it from total interaction along the path, and relate both to measured loss changes. Under adaptive optimization, it additionally identifies which vector stream is being diagnosed.

The argument has a positive practical purpose: it gives researchers an organized way to inspect training and formulate intervention hypotheses. If a learning-rate reduction changes a raw-gradient cosine, the framework says what changed geometrically and which additional measurement is needed to judge the outcome. That is useful even when the intervention does not improve loss.

### Terms that must retain one meaning

| Term | Meaning in this paper |
| --- | --- |
| Constructive/destructive interaction | A positive/negative inner product between specified vectors at a specified temporal separation |
| Constructive/destructive adjacent regime | A summary of adjacent interactions over an explicitly stated observation window |
| Cancellation | Reduction in the norm of a vector sum relative to the sum of its members' lengths |
| Cumulative interaction | The sum of all cross-step products over the stated path, including longer lags |
| Objective progress | Measured reduction of a specified loss over a specified horizon |
| Stability | Used only for established, explicitly specified dynamical notions or cited prior work; no new general stability criterion is claimed |

Destructive interaction is a geometric observation. It does not itself measure forgotten representations, wasted computation, harmful learning, or failure to converge. These distinctions are central results and definitions, rather than disclaimers added after a stronger claim.

## 2. Motivation, gap, and credit

Training can repeatedly reverse direction while continuing to improve. Conversely, a smoother trajectory can attain worse loss at the same update or token budget. Norms and losses alone do not say how successive changes combine; an adjacent cosine alone does not describe accumulation over the complete path. The motivation is to make these available observations mutually interpretable.

The literature already establishes oscillatory descent, gradient-product diagnostics, directional secants, and differences between macroscopic trajectory direction and microscopic motion. The paper should state this before presenting its contributions. The gap is therefore precise: **an explicit accounting that connects the temporal scale of cancellation to cumulative motion and measured loss, while preserving the different meanings of raw gradients and applied updates across GD, SGD, and Adam.** This is the chosen diagnostic problem, not a claim that previous work ignores trajectories.

| Established work | Credit and boundary for this paper |
| --- | --- |
| Barzilai and Borwein, *Two-Point Step Size Gradient Methods*, IMA Journal of Numerical Analysis 8(1):141--148, 1988 | The full-GD quantity `kappa_hat = s^T y / s^T s` is the reciprocal first BB step where that reciprocal exists. The curvature statistic is classical. [Original publication](https://doi.org/10.1093/imanum/8.1.141). |
| Chee and Toulis, *Convergence Diagnostics for Stochastic Gradient Descent with Constant Learning Rate*, AISTATS 2018 | Consecutive-gradient products already support stationarity diagnostics. The present paper uses such observations for finite-path interpretation. [Primary source](https://proceedings.mlr.press/v84/chee18a.html). |
| Pesme, Dieuleveut and Flammarion, *On Convergence-Diagnostic based Step Sizes for Stochastic Gradient Descent*, ICML 2020 | A gradient-product sign or expectation does not automatically yield a reliable finite-sample convergence test. [Primary source](https://proceedings.mlr.press/v119/pesme20a.html). |
| Baydin et al., *Online Learning Rate Adaptation with Hypergradient Descent*, ICLR 2018 | Adjacent-gradient products and an additional retained gradient are established ingredients for learning-rate adaptation. This paper supplies no competing optimizer. [Primary source](https://arxiv.org/abs/1703.04782). |
| Mishkin et al., *Directional Smoothness and Gradient Methods: Convergence and Adaptivity*, NeurIPS 2024 | Directional, path-dependent smoothness is established. The current account measures a signed realized response and does not claim their type of convergence guarantee. [Primary source](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1ac83203e88eb6cf6b30642f0239b932-Abstract-Conference.html). |
| Singh et al., *The Directionality of Optimization Trajectories in Neural Networks*, ICLR 2025 | Macro/micro directionality and useful oscillatory motion are established. The specific question here is how much path length is canceled inside adjacent pairs, how their residuals accumulate, and how these quantities compare with measured loss. [Primary source](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37771cc0be272368102a37f202bb88d8-Abstract-Conference.html). |
| Bai et al., *Adaptive Preconditioners Trigger Loss Spikes in Adam*, ICML 2026, PMLR 306:5566--5601 | Adam's second-moment spike mechanism and a one-HVP directional proxy are prior work. The present raw-response/memory identities qualify a comparison; they do not demonstrate a replacement mechanism or empirical proxy superiority. [Primary source](https://proceedings.mlr.press/v306/bai26o.html). |

The same-state sample/task-gradient literature concerns a different object from gradients at successive parameter states. Therefore temporal reversal does not establish conflicting examples, conflicting tasks, or representation erasure. Likewise, empirical displacement accounting differs from high-probability optimization and generalization guarantees.

This positioning is a focused overlap assessment. It supports narrow, explicit contribution language rather than an exhaustive priority claim.

## 3. Three contributions and their novelty level

### Contribution 1: An account of cancellation across temporal scales

The paper retains every cross-step interaction in the displacement identity, introduces a zero-cross-term reference for interpreting the path ratio, and uses the exact factorization `R_T = C_T D_T` to separate cancellation within adjacent pairs from accumulation of their residuals. This answers a concrete question that either an adjacent cosine or a normalized displacement alone leaves unresolved.

**Novelty level:** an elementary diagnostic construction and its organized empirical use. The norm expansion and factorization are not difficult new optimization theory. Presenting their value through the information they reveal is stronger than presenting algebraic simplicity as a theorem-level advance. No exhaustive claim that the exact factorization has never appeared elsewhere is needed.

### Contribution 2: A consistent interpretation across update rules

The paper connects classical directional secant geometry to consecutive alignment and gradient magnitude, states the local quadratic loss relation, and makes sampling and finite-step qualifications explicit. Under Adam it distinguishes the raw gradient, stored moment, adaptive direction, and actual increment. The memory discrepancy identities specify why a raw response need not equal a preconditioned-curvature proxy.

**Novelty level:** synthesis and qualified extension. Secant curvature, gradient products, quadratic reversal thresholds, and optimizer transformations are established. Their joint interpretation supplies the diagnostic framework. The Adam discrepancy is an analytical connection whose empirical neural calibration remains absent; it cannot carry a claim of measured fidelity or computational superiority.

### Contribution 3: Controlled evidence for the consequences of those distinctions

The evidence shows that strong adjacent cancellation can coexist with constructive total interaction and substantial loss reduction, that observed SGD alignment depends on state and sampling, that Adam raw-gradient reversal does not identify an uphill first-order update, and that a more favorable-looking diagnostic can accompany worse matched-budget loss. Existing Pythia and intervention studies give these conclusions practical context.

**Novelty level:** the specific quantitative characterization and integrated empirical argument. Useful oscillation itself is known. The contribution rests on the exact measured distinctions, source-qualified controlled comparisons, and what they imply for interpreting interventions. Its ultimate significance relative to nearby work remains a journal-review judgment.

## 4. Claim-to-evidence hierarchy

The manuscript should distinguish exact identities, conditional demonstrations, replicated comparisons, and exploratory observations. They are complementary evidence classes, not interchangeable independent replications.

| Central claim | Evidence and role | Limit that travels with the claim |
| --- | --- | --- |
| Adjacent opposition does not determine cumulative interaction | [Analysis024](../../../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/004-completed-calibration.md) and [Analysis056](../../../../analyses/056-2026-11-05-gd-paired-displacement/observations/001-paired-displacement.md): positive final all-pair interaction in all nine paths despite negative adjacent interaction in seven | One controlled initialization per model; no universal frequency claim |
| Pairing locates cancellation that a nearly saturated negative cosine cannot quantify | Analysis056: pair factors, zero-cross-term reference, both pairing offsets, and observed loss changes | Geometric accounting; neither factor ensures local descent |
| Longer lags cannot be discarded | [Analysis035](../../../../analyses/035-2026-11-02-eos-backward-anchors/REPORT.md) and Analysis056: structured backward lag profiles at selected endpoints | Finite endpoint coverage and separately recomputed continuations, not exhaustive temporal or asymptotic evidence |
| Sampling changes the observed diagnostic and the path | [Analysis037](../../../../analyses/037-2026-11-02-sgd-batch-extension/REPORT.md): moving continuations across three initialization blocks | Equal updates imply unequal data exposure; this does not isolate measurement noise at two frozen endpoints |
| Raw-gradient reversal is distinct from Adam's applied descent direction | [Analysis036](../../../../analyses/036-2026-11-02-controlled-adam-diagnostics/README.md) and [Analysis057](../../../../analyses/057-2026-11-05-adam-response-memory/observations/001-response-memory.md): all 794 rise-phase applied steps are descending to first order | Recovered-event sample from one initialization per model; finite-step loss can rise despite negative local linear contribution |
| The event-product association is not established incremental directional information | Analysis057: identical-event component comparisons and all seven held-out-rate log-norm-plus-cosine fits worsen | Descriptive associations, dependent events, and rate transfer; no prospective prediction conclusion |
| More positive alignment need not improve a measured outcome | [Analysis004](../../../../analyses/004-2026-10-04-lr-schedule-intervention/README.md) via [Analysis028's observation](../../../../analyses/028-2026-10-30-adam-diagnostics/observations/001-adam-extension.md): held-rate Pythia branches achieve lower endpoint validation loss despite stronger raw reversal | One matched seed per size; schedule changes also affect cumulative step and decay exposure |
| Adaptive direction matters beyond its magnitude | [Analysis025](../../../../analyses/025-2026-10-17-pythia-descent-mechanism/REPORT.md): all 18 norm-controlled removal-of-scaling branches worsen fixed-probe loss | Three seed/data-order blocks, two forks, one recipe; the matched norm is the native-parent adaptive component, not the complete rounded increment |
| Intervention value depends on model, horizon, and outcome | [Analysis043](../../../../analyses/043-2026-11-03-spark-fixed-step-interventions/FINDINGS.md), [Analysis051](../../../../analyses/051-2026-11-04-mlp-global-clipping/FINDINGS.md), and [Analysis054](../../../../analyses/054-2026-11-05-resnet-clip1/FINDINGS.md): matched LR/clipping responses, including disagreement between mean, peak and endpoint outcomes | Selected states and short windows; no universal ranking of clipping, schedules or decay |

The selected reductions and source contracts are recorded in [evidence_selection.json](evidence_selection.json). Effects from different objectives and horizons should remain separate. Their common contribution is the interpretation they test, not a pooled treatment-effect estimate.

### Main-text priority

The GD pairing/reference result is the empirical center. The SGD evidence establishes sampling scope. The Adam actual-step finding establishes optimizer scope. The retained intervention studies explain why these distinctions matter when researchers act on diagnostics. Detailed event-density, severity-regression, lag-atlas, and historical selector results belong in supporting records or appendices when they do not advance this sequence.

## 5. Section roles and reader progression

| Section | Question it answers | Required reader takeaway |
| --- | --- | --- |
| Introduction | Why do the available training signals need a joint interpretation? | The three-level diagnostic question and the three contributions |
| Related work and diagnostic scope | What is already known, and what specific question remains here? | The paper builds on established secants, gradient products and trajectory theory |
| Constructive and destructive regimes | What is measured, and how are the quantities related? | Exact all-lag accounting, pair factorization, vector/horizon conventions, qualified loss interpretation |
| GD and SGD | What happens when local reversal is measured alongside accumulation and loss? | Strong local cancellation can coexist with constructive total interaction and progress; sampling complicates the observation |
| Adam | What changes when gradients are transformed before application? | Raw reversal and an uphill applied direction are different events; measured finite-step outcomes remain essential |
| Implications and existing interventions | What does this mean for changing training? | A diagnostic change is a measured effect, while improved loss requires its own matched comparison and horizon |
| Conclusion | What can a researcher now diagnose more clearly? | Locate cancellation, identify its vector/time scale, and evaluate objective consequences |
| Appendices | Which derivations and measurement details support the concise account? | Reproducible, qualified support without a second competing central story |

Each figure should answer one of these questions. Each section should close by explaining what its result contributes to the next question. Avoid repeating the complete caveat list in every paragraph; place each material qualification with the claim it limits.

## 6. Response to the original rejection

| Original concern | Response in the revised argument | What remains to be judged |
| --- | --- | --- |
| Stability is undefined and parameter norm is not linked to it | Diagnose displacement and vector interaction; remove general stability from the title/headline contribution | Whether the diagnostic contribution is sufficiently important |
| The norm expansion is standard | Treat it as the accounting foundation; foreground specific measurements and controlled findings | Mathematical novelty remains modest |
| Nonadjacent correlations are assumed negligible | Retain all lags exactly and demonstrate their material role | Finite empirical coverage, explicitly stated |
| Existing norm-control work is omitted | Distinguish realized-path identities from probabilistic optimization/generalization guarantees and cite the reviewer-named work | No new bounded-iterate or generalization theorem is supplied |
| Exact covariance is described as impossible online | Describe the information unavailable from an aggregated gradient; acknowledge approximation routes | This paper does not benchmark every possible covariance estimator |
| A proposed batch-distance expression may be negative | The disputed legacy quantity is outside the compiled contribution | Do not revive it without a separate audit |

The original paper's scientific interest is preserved: sequential updates can reinforce or partially undo parameter motion, and this geometry can guide future training research. The revised evidence changes what can responsibly be concluded. It does not support calling all cancellation harmful or claiming erased knowledge. Correcting that inference strengthens the original project rather than abandoning it.

### Exact bibliography details for the review-named works

```bibtex
@inproceedings{li2022high,
  author    = {Li, Shaojie and Liu, Yong},
  title     = {High Probability Guarantees for Nonconvex Stochastic Gradient Descent with Heavy Tails},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {162},
  pages     = {12931--12963},
  year      = {2022},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v162/li22q.html}
}

@article{lei2026generalization,
  author  = {Lei, Yunwen},
  title   = {Towards Better Generalization Bounds of Stochastic Optimization for Nonconvex Learning},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {48},
  number  = {3},
  pages   = {2104--2117},
  year    = {2026},
  doi     = {10.1109/TPAMI.2025.3621591}
}
```

Li and Liu's metadata comes from the [official PMLR publication](https://proceedings.mlr.press/v162/li22q.html). Lei's final issue metadata is listed on the [author's publication page](https://leiyw.github.io/); the [institutional record](https://hub.hku.hk/handle/10722/366758) and [author manuscript](https://leiyw.github.io/paper/TPAMI2025.pdf) identify the same work. The original review's 2025 citation corresponds to early-publication metadata. It is not a separate article.

## 7. Disposition of the new experiment plan

The [November 5 Adam experiment plan](../2026-11-05-human-input-adam/EXPERIMENT_PLAN.md) is a future research program. It would test calibration against a one-HVP proxy, memory/scaling effects, independent event replication, stochastic monitoring, and scale/overhead claims. Those are legitimate questions, but they are not all submission prerequisites for the chosen diagnostic contribution.

For this revision:

- Preserve the plan and its unexecuted status.
- Do not claim neural calibration, computational superiority, broad spike mechanism, independent full-batch replication, or controller efficacy that its future experiments would be needed to establish.
- Present the memory/proxy relationship as an analytical qualification and future question.
- Use the already completed SGD, Pythia, and intervention evidence for the scope it actually supports.
- Keep sparsity and orthogonal-step ideas as untested motivation for subsequent research; they are not methods proposed or evaluated in this paper.

This decision closes the present argument without spending to support claims that the paper has chosen not to make.

## 8. Stopping rule and unresolved risks

The revision is ready for author submission review when every main claim maps to retained evidence or a checked derivation; close literature is credited; vector/objective/horizon conventions remain consistent; prose and figures continuously support the thesis; references, numerical statements and build checks pass; and adversarial review exposes no known unsupported inference or internal contradiction.

Simulated reviews are used to find specific defects and assess whether revisions answer them. They do not produce evidence of likely acceptance, and review cycles should not continue merely until all simulated reviewers use positive language.

The following risks remain even after careful revision:

1. **Significance and novelty.** A referee may consider an integrated diagnostic account of elementary identities and known phenomena too incremental. The response is a precise comparison and substantive empirical findings, not stronger novelty adjectives.
2. **Replication coverage.** The main full-batch GD/Adam sweeps use one initialization per model. Three-block SGD and Pythia studies broaden specific claims but cannot be substituted for missing independent full-batch event replication.
3. **Coordinate dependence.** Euclidean angles and path length describe the selected parameterization; they do not establish function-space change or retained information.
4. **Finite objective and time coverage.** Small controlled tasks and short matched continuations support the reported horizons and outcomes. They do not establish asymptotic behavior or broad large-model training laws.
5. **Practical measurement cost.** The minimal gradient statistic avoids additional gradient evaluations/HVPs but retains an additional parameter-sized vector and reductions. No measured negligible-overhead or superior-cost claim is established.
6. **Intervention scope.** Existing controls demonstrate diagnostic/outcome distinctions and treatment-specific responses. They do not validate a diagnostic-driven policy or settle whether future sparsity or orthogonalization methods help.

These are the honest boundaries of the current contribution. Closing means delivering the strongest coherent paper that the existing evidence supports, with these boundaries visible and the remaining scientific choices left to the authors.
