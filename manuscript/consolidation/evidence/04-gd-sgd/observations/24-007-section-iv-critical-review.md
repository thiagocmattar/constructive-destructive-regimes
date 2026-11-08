> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [007-section-iv-critical-review.md](../../../../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/007-section-iv-critical-review.md).

# Observation007: second critical review of manuscript Section IV

Date: 2026-10-30. User request: critically review the new empirical section.
Reviewed snapshot: Section IV on pages 6-9 of the compiled draft following
commit `7eb8ddf`. The draft and publication figures are unchanged by this review.
The source and compiled-PDF hashes are in `../section_iv_review_checks.json`.

## Assessment

The section is a credible descriptive study with clear figures, genuinely paired
learning-rate comparisons within each model case, and unusually transparent
coverage and selection disclosures. It separates local reversal, path-relative
displacement and training progress. It is not yet a compelling empirical test of
the local curvature/loss approximation or a stability diagnostic. The principal
revision should sharpen the positive contribution and use the retained temporal
interaction evidence, rather than add more qualifications to every paragraph.

The strongest defensible message is: in these matched full-GD trajectories,
successive updates can strongly oppose one another while the trajectory makes
net progress; local reversal, whole-path interaction, displacement relative to
path length, and loss reduction are distinct observables.

## 1. Major: make the local-versus-cumulative result explicit

Location: IV-C, especially its first two paragraphs; Fig. 2 and Table I.

The text correctly says that low R does not imply smaller absolute displacement
and that adjacent cosine does not determine the all-lag sum. The actual retained
data demonstrate both points more strongly than the text currently conveys:

- The complete-prefix all-pair interaction is positive at update 3000 in all
  nine conditions, including the seven with negative cumulative lag-one
  interaction and late negative mean cosine.
- Absolute displacement computed from the gradient sum increases with LR within
  every model case, while R decreases. For example, the MLP displacement is
  approximately 0.951, 3.139 and 4.763; its path length is approximately 1.198,
  64.361 and 185.542. R falls because path length grows faster than net motion.

Here the recorded interaction uses the factor-two convention:

`C_T = 2 sum_{u<t} g_u^T g_t = B_T^2 - E_T`, where
`E_T = sum_t ||g_t||^2`, `B_T = ||sum_t g_t||`, and
`C_T = C_lag1 + C_beyond_lag1`.

The all-pair sum being positive does not imply that every pair is positive or
that this sum stays positive at every prefix. It establishes that the final
complete trajectory cannot be called destructive on the basis of a negative
all-pair sum. Local reversal remains real. Even mutually orthogonal equal-norm
updates give `R_T = 1/sqrt(T)` with zero all-pair interaction: low R alone is not
a sign test for destructive cross terms.

Recommended revision: present the measured lag-one/all-pair distinction as a
central result, specify the horizon whenever saying "destructive," and add a
compact decomposition or endpoint summary using the already retained quantities.
The existing positive statement about R as path-length retention can remain.

## 2. Major: distinguish illustration from validation of Section III-D

Location: IV-B's claim that the observations support Section III-D; IV-D's
projected-change interpretation and closing paragraph.

Continued long-timescale loss reduction with negative cosine illustrates the
conceptual separation. It does not quantify the accuracy of the one-step
quadratic loss approximation. A second pass over the existing scalar logs checks
that distinction directly, with the manuscript's indexing:

`predicted_delta_L[t-1] = -(eta/2) ||g[t-1]||^2 (2-q[t])`,
`q[t] = eta*kappa_hat[t]`, versus
`actual_delta_L[t-1] = L(theta[t-1]) - L(theta[t-2])`.

There are 2999 paired transitions per condition, t=2..3000. The last update has
no subsequent recorded gradient and is excluded. No smoothed value enters this
check. Opposite signs occur as follows:

| Model | LR | Opposite-sign transitions / 2999 |
| --- | ---: | ---: |
| MLP | 0.004 | 0 |
| MLP | 0.02 | 145 |
| MLP | 0.04 | 281 |
| ResNet-32 | 0.025 | 595 |
| ResNet-32 | 0.05 | 694 |
| ResNet-32 | 0.1 | 814 |
| Transformer | 0.05 | 0 |
| Transformer | 0.1 | 542 |
| Transformer | 0.2 | 1325 |

The last count is 44.2%. Excluding transitions where either actual or predicted
change has magnitude <=1e-5 times the larger neighboring loss leaves 1239/2837,
or 43.7%. The JSON also includes 1e-7 and 1e-6 sensitivity thresholds and the
median absolute discrepancy normalized by `eta*||g_previous||^2`.

These are post-hoc descriptive checks, not independent Bernoulli trials or
predictive accuracies: q itself requires the subsequent gradient. The thresholds
are sensitivity choices, not calibrated floating-point error bounds. Opposite
signs near a nearly flat quadratic prediction can accompany a small residual
relative to the first-order term. This audit does not isolate finite-step error,
nonsmoothness or numerical effects and does not refute the exact identities.
All nine final losses are below their state-2000 losses, so the reported longer
timescale coexistence remains supported.

Recommended revision: explicitly call the current plots an illustration, and
add a focused observed-versus-approximated loss-change check with residual sizes
and difficult regimes visible. A calibration of pointwise directional curvature
would separately require same-objective HVP measurements at matched states.
No new training is needed for the scalar loss-change audit.

## 3. Major: clarify what is added beyond Edge of Stability

Location: opening paragraph and IV-D.

[Cohen et al.](https://arxiv.org/abs/2103.00065) already report short-timescale
nonmonotonic loss alongside longer-timescale decrease during full-GD training.
That coexistence alone should not carry the novelty claim. This is a focused
comparison to the cited source, not an exhaustive novelty search.

The distinctive contribution to emphasize here is the operational separation
between adjacent reversal and all-lag accumulation, measured from gradients,
and its interpretation alongside objective progress. The new all-pair check
above gives that argument concrete evidence. State the empirical question and
answer before the long measurement protocol.

For nonzero gradients, `q_t>1 iff rho_t<0` is exact by definition. q additionally
uses the consecutive norm ratio, so it is not identical to rho in magnitude;
their sign thresholds nevertheless cannot independently corroborate a regime.
The current disclosure of this fact is good and should remain.

## 4. Moderate: the spectral column currently invites an unanswerable comparison

Location: IV-A spectral paragraph, Table I last column, IV-D second paragraph.

The text discloses three distinct mismatches: full objective versus subset,
directional finite-step change versus largest pointwise eigenvalue, and late
mean versus final endpoint. These are scientifically correct qualifications,
but together they leave little inferential role for the side-by-side numbers.
In particular, q near two and eta*lambda_max above two neither validate nor
invalidate q as a pointwise curvature estimate.

Recommended revision: retain the full-objective MLP spectral comparison as
context, and move the larger-model subset endpoints to a clearly exploratory
appendix unless a specific question is answered. If a temporal comparison is
added, align the states: a Hessian at state s corresponds to the preceding-step
local expansion for q at gradient index s+2. The final spectrum at state 3000
has no matching gradient pair in the present capture. Residual-checked
eigenpairs verify numerical eigenpair quality, not subset representativeness.

## 5. Moderate: focus the prose and preserve the appropriate inference scope

Location: IV-A, captions, numerical repetition in IV-B/C and conclusion.

One frozen initialization per model and calibration-selected rates are
appropriate for the stated illustrative purpose. They cannot estimate how often
the observed patterns occur across initializations. The number of time points
does not fix this, and uncertainty bands over time would not substitute for
replication. More seeds are a later requirement if the paper makes robust
cross-initialization claims, rather than a prerequisite for every statement in
this section. Full-GD observations do not test the moving-SGD noise relation.

Keep the main methods focused on objective, architecture, fixed BN grouping,
matched initialization, rates, horizon and reduction definitions. Move token
exclusions, physical accumulation, detailed precision settings and solver
history to a reproducibility appendix. Preserve the vocabulary construction
disclosure there and add an actual reference/revision for the unnamed tutorial.
Do not silently turn this training-dynamics evidence into held-out evaluation.
Clarify that the MLP loss averages over examples and sums the ten output
coordinates; "mean half-squared error" can otherwise imply an additional 1/10.

Table I already carries all endpoints: use one illustrative numeric contrast
in the prose and develop its meaning instead of repeating every R value.
Consolidate repeated single-seed and non-predictive caveats without deleting the
figure-specific smoothing and spectral coverage disclosures. "Complementary
parameterizations" would be clearer as "algebraically linked diagnostics."

## 6. Boundary issue: terminology in Sections III and V can undermine IV

Section III currently calls squared displacement a measure of the magnitude
of learned information. Section IV correctly limits its claim to geometric
motion, and the new endpoint check shows why R cannot stand in for either
absolute displacement or information. The information interpretation requires
separate justification; this experiment does not provide it.

Section V returns to stronger stability, near-minimum and noise-absorption
interpretations. Even a careful Section IV will not shield those statements
from review. Align the surrounding terminology before submission, while keeping
this review's requested scope centered on IV.

## Presentation and recommended revision order

Pages 6-9 were freshly rendered and visually inspected. The figures are legible;
labels no longer overlap curves, and loss precedes the geometry comparisons.
The second figure is the strongest argumentative figure. The third is useful
only with a clear explanation of the norm-ratio information and the limits of
the curvature interpretation. Raw alignment in Appendix D preserves transients.
Avoid the current page-7/page-9 split of "Consequently" across the full figure
page when revising float placement; this is an editorial issue, not lost content.

Recommended order, without changing the scientific design:

1. State the contribution as local reversal versus cumulative geometry and loss.
2. Add the retained all-pair/absolute-displacement findings to IV-C.
3. Quantify the local loss approximation instead of suggesting validation by
   coexistence alone; keep failures visible and the audit retrospective.
4. Give the spectral comparison a narrower, explicit role.
5. Compress methods/repeated caveats and align the Section III/IV boundary.

## Sources, verification and limits

Generating script: `../10_review_section_iv.py`; results and source hashes:
`../section_iv_review_checks.json`. The script verifies each metrics file against
the accepted `results.json`, then checks all 27000 prefix identities for energy,
path length, B/A and the factor-two temporal decomposition, plus the q identity
for all 26991 defined pairs. It uses retained scalars, not recovered vector
histories. Absolute displacement values quoted above use `eta*B`, subject to
the existing FP32 parameter-update qualification.

This is additional local postprocessing of the same nine seed-0 trajectories,
with no new training, figures, manuscript edits, cloud resources or spending.
The prior figure acceptance and mechanical manuscript checks remain valid for
their original purpose; they did not establish the scientific approximation's
accuracy. This review adds that distinction and a reproducible scalar audit.
