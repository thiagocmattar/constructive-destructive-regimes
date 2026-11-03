# Author feedback, mathematics and evidence review

The author's clarification governs this revision: improve the methodology of
writing, not the experimental methodology. The supplied annotated PDF is
preserved in the preceding revision folder. Its 12 comments and original
metadata are recorded in [feedback.json](feedback.json).

This document records the initial full-paper pass (`f05098a`). The author's
subsequent appendix-shortening request is evaluated in
[APPENDIX_REVIEW.md](APPENDIX_REVIEW.md), which supersedes the initial appendix
layout and duplicate-derivation decisions. Current build and visual acceptance
records describe the shortened 14-page PDF.

## Responses to the PDF comments

| Comment | Resolution |
| --- | --- |
| C01 | The Adam opening starts with the diagnostic question and references the established cosine and projected-change equations. The moment equations explain how the raw input influences the adaptive step. A local expansion explicitly connects projected raw-gradient change to curvature acting on that displacement. |
| C02 | Replaced “interspersed” with “interrupted by transient spikes.” |
| C03 | Removed the final-loss negation. The overview now states the positive finding: the highest tested rates achieve the lowest final losses, motivating attenuation of excursions within productive training. |
| C04 | Removed the temporal-filter digression. State the largest recovered relative-spike selection rule and the three selected peaks. |
| C05 | Replaced the vague timing qualification with the role of the growing first moment in Adam's temporal average. |
| C06 | Made the Transformer timing explicit: diagnostic crossing at 788 with +0.40% loss, +10% two updates later, loss maximum six updates later; raw/momentum norm maxima at 792/793. |
| C07 | Describe constructive recovery segments in the selected ResNet/Transformer events and predominantly destructive recovery with shrinking oscillations in MLP. Momentum alignment is named separately. |
| C08 | Explain a fixed-baseline excursion intuitively, name pre-rise/peak/recovery states, then give the recursive definition, ratios and durations. Empty-search and positive-baseline cases are explicit. |
| C09 | State the purpose of the >10% filter affirmatively: focus anatomy on pronounced departures from local training loss. |
| C10 | Replace “illustrations” with “selected cases,” “events” or “micro-plots,” as appropriate. |
| C11 | Discuss amplitude growth, slow recovery, predominant destructive starts and universal endpoint norm amplification. Move procedural detail to the caption/appendix. |
| C12 | Regenerate the table across the full text width with 13 centered columns, larger base spacing, scientific LR notation, dataset/model line breaks and the requested rounding. Define every column, event anchor, vector stream and aggregation in the caption. |

## Full-paper writing decisions

- The title retains “Constructive and Destructive Training Regimes” and broadens
  its subtitle from SGD to neural-network stability, matching the Adam extension.
- The abstract and introduction state practical value, contributions and the
  principal quantitative evidence directly.
- Background now progresses through curvature, sampling, prior temporal
  diagnostics and practical loss-spike monitoring. The former one-dimensional
  break-even equation is replaced by a concise description of its stated model.
- The regime definition is explicit and shared across all sections: sign of
  adjacent raw-gradient cosine. Positive/negative moment cosines describe
  momentum alignment rather than silently redefining the regime.
- Each main result explains what its evidence adds: productive destructive
  descent; cumulative cancellation; longer-lag structure; sampling effects;
  early Adam signals; repeated spike anatomy; joint magnitude association.
- Essential scope is concentrated in the local mathematical assumptions,
  the closing Adam paragraph and the measurement appendix.
- “Early warning” refers to an observed lead relative to the pronounced rise
  or loss maximum. The eventual-peak scatter remains a retrospective descriptor.
- The introduction, conclusion and appendix use the same distinctions. No
  gradient-maximization or alignment-maximization objective is proposed.

## Mathematical audit

All active display equations and surrounding interpretations were read.
The numerical checks supplement this analytical review.

| Object | Check and resolution |
| --- | --- |
| State clock | Parameters theta_t follow t completed updates; g_t is evaluated at theta_(t-1). Event state s uses gradient/moment index s+1. |
| Quadratic stability | Strict contraction applies to a fixed positive-definite Hessian. The 1 and 2 thresholds distinguish sign alternation and amplification. Neutral/zero-curvature modes are outside that stated assumption. |
| Sampling covariance | Use the exact factor (N-b)/(b(N-1)) for uniform subsets without replacement, matching the covariance normalization 1/N. Checked by exhaustive subset enumeration. |
| Displacement and pairwise interaction | Exact full sum over all pairs, including the factor two and eta squared. Direct vector sums agree with explicit pairwise accounting. |
| Regime definition | Positive/negative adjacent raw cosine defines constructive/destructive; zero is orthogonal and zero norms give undefined cosines. |
| Gradient evolution | Taylor remainder retained before the first-order approximation; noise difference is xi_t minus xi_(t-1). |
| Directional curvature | The Rayleigh quotient uses the Hessian at theta_(t-2) and the preceding gradient. The directional minimum interpretation requires positive curvature. |
| Projected change | Exact identity is 1-rho_t*norm(g_t)/norm(g_(t-1)) = -g_(t-1)^T(g_t-g_(t-1))/norm(g_(t-1))^2 for the defined pair. |
| Noise interaction | Fresh conditional sampling gives the negative previous-noise energy in the unnormalized expectation. This does not assert the same sign for an expectation of a ratio. |
| Cosine attenuation | Independent estimates at fixed states and concentrating norms are stated before the approximation. Evolving-state coupling remains explicit. |
| Retention | R is defined for positive path length; triangle inequality gives [0,1]. Orthogonal and opposing examples pass. O(p) streaming work/storage replaces explicit O(T^2) pair enumeration for aggregate accounting. |
| Loss connection | Trapezoidal gradient integration is exact on quadratics; local Lipschitz Hessian gives a third-order loss remainder. Both endpoint noise terms have the required plus sign. |
| Destructive loss increase | Under the stated noiseless approximation, rho < -norm(previous)/norm(current) requires reversal and sufficient norm growth. The directional 1-to-2 interval permits continued descent. |
| Adam response | Substituting theta_(t-1)-theta_(t-2) = -eta*d_(t-1) yields the bilinear curvature response g_(t-1)^T H_(t-1) d_(t-1), rather than the GD Rayleigh quotient. Verified on 320 adaptive quadratic steps. |
| Adam moments | Unclipped raw input, zero initial states, bias corrections and coordinatewise epsilon placement match the actual recipe. |
| Spike definition | Recursive first rise after the preceding recovery; fixed baseline; earliest maximum; first return; search ends if the next rise/return is absent. All actual pre-rise losses are positive. |
| Event timing | Rise p-a, recovery c-p. Before diagnostics are measured at a; recovery diagnostics average arrival states p+1 through c. Norm columns are endpoint ratios. |
| Scatter | Signed pre-rise cosine times rise endpoint norm ratio, with log10 relative peak loss as response. Seven per-rate OLS fits and in-sample R-squared values independently rechecked. |
| Appendix derivations | Corrected the one-step substitution to reference the curvature-form equation first, then the alignment form. Conditional first-order noise cancellation and the two-step starting Hessian are retained. |

The retained mathematical suite has 186 cases. The new suite adds 328 cases:
320 Adam projected-response steps, five finite-population covariance checks and
three all-pair displacement checks. These are mathematical validation cases,
not new empirical training experiments.

## Evidence audit

- Re-ran the prior Adam raw-history auditor into ignored scratch destinations,
  preserving previous revisions and current source artifacts.
- Independently re-segmented all 806 recovered events and rechecked every anchor,
  pre-rise diagnostic and recovery mean in the 129-event sample.
- Recomputed each displayed table summary from event rows and checked it against
  the frozen summaries. Rounding is applied only at display time.
- Rechecked the selected largest spikes, their timings, all reported counts,
  highest-rate endpoint claims, and the seven regression scores.
- Rechecked the retained GD/SGD evidence: 57 curves, 134,560 plotted values,
  45 displayed scalars and nine endpoint rows.
- All four approved Adam figures and all other active figures remain byte-identical.
- All 12 feedback comments are preserved; previous manuscript snapshots remain
  unchanged. The previous raw-audit script writes only into the new scratch tree.

## Primary literature and metadata

Sources were checked for the specific retained claim or bibliographic correction.
The contribution is a joint diagnostic account and its empirical evidence;
successive gradient products, cosine and Adam's averaging are credited.

| Source | Use in this revision |
| --- | --- |
| [Chee and Toulis, AISTATS 2018](https://proceedings.mlr.press/v84/chee18a.html) | Prior convergence-phase diagnostics using successive gradients. Added citation. |
| [Pesme et al., ICML 2020](https://proceedings.mlr.press/v119/pesme20a.html) | Prior analysis of gradient-product diagnostics and signal/variation limitations. Added citation. |
| [Baydin et al., ICLR 2018](https://arxiv.org/html/1703.04782v3) | Hypergradient LR adaptation, including the consecutive-gradient product for GD. Added citation. |
| [Xing et al., 2018](https://arxiv.org/abs/1802.08770) | Prior trajectory/valley-crossing observations; reused the existing entry and protected SGD capitalization. |
| [Jastrzebski et al., ICLR 2020](https://arxiv.org/pdf/2002.09572) | Break-even model and early hyperparameter influence; corrected the seven-author list. |
| [Arora et al., ICML 2022](https://proceedings.mlr.press/v162/arora22a.html) | Retained EoS context; checked venue/pages and scope. |
| [Bertsekas author errata](https://www.mit.edu/~dimitrib/nlperrata.pdf) and [MIT course](https://ocw.mit.edu/courses/15-084j-nonlinear-programming-spring-2004/pages/readings/) | Corrected a book-review record to Nonlinear Programming, 2nd edition, Athena Scientific, 1999. Legacy citation key is retained. |
| [Mandt et al., JMLR 2017](https://jmlr.org/papers/v18/17-214.html) | Added article number 134. |
| [Three Factors Influencing Minima in SGD](https://arxiv.org/abs/1711.04623) | Corrected author order and the unsupported ICLR-conference entry to the actual arXiv record; legacy key retained. |
| [PaLM Section 5.1](https://arxiv.org/html/2204.02311v5) | Approximately 20 spikes and checkpoint/data-skipping recovery. |
| [OPT Section 2.5](https://arxiv.org/html/2205.01068v4) | Activation growth, LR/restart recovery and tighter clipping. |
| [OLMo 2 Section 3.2](https://arxiv.org/html/2501.00656v3) | Rolling loss and gradient-norm spike scores. |
| [Llama 3 Section 3.4](https://arxiv.org/html/2407.21783v3) | Warmup, staged batches and largely stable pretraining. |

The existing Pascanu and Zhang clipping references retain the primary-source
verification in the preceding revision. The discussion presents clipping,
learning-rate adjustment and data mixing as motivated future interventions.
