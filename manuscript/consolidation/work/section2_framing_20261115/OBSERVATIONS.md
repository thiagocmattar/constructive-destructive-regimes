# Section II revision: related work and conceptual foundations

15 November 2026. Applied the author-approved surgical plan following the
[contribution framing](../../CONTRIBUTION_FRAMING.md) and the completed
[introduction revision](../introduction_framing_20261115/OBSERVATIONS.md).

## Changes

The [revised Section II](../../draft/sections/02-related-work.tex) has three
subsections supporting the four contributions:

- **II-A: Oscillatory Dynamics, Stability, and Sampling Noise.** Combines the
  original curvature, sampling and EoS material. Retains the update, Jacobian,
  quadratic stability condition, noise decomposition and gradient covariance
  equations in their original order, with identical mathematical text and labels.
  Connects stochastic stability and path-length analysis to the distinction
  between adjacent opposition and accumulated interactions.
- **II-B: Training-Time Diagnostics of Curvature and Loss Increases.** Positions
  spectral/directional monitoring, convergence diagnostics, learning-rate
  adaptation and secant methods by purpose. Explicitly acknowledges the BB1
  reciprocal and distinguishes realized loss balance from gradient-difference
  step averages. Exact completed-step characterization remains separate from
  the Adam outgoing-step warning experiment.
- **II-C: Representation Dynamics and Update Control.** Replaces the extended
  pretraining-recipe survey with comparisons to representation similarity,
  catapults/feature learning, representation-change correction and unit-wise
  clipping. Keeps one brief practical-control sentence. Ends with functional
  characterization, matched intervention tests and the transition to Section III.

Only the introduction's roadmap sentence changes. Three bibliography entries
are appended; existing entries are preserved. The draft README records this
revision. The main PDF remains 22 pages. Abstract, Section III onward, figures,
tables, supplementary PDF and supplied originals are unchanged at source level.
Page flow and citation numbering necessarily change in the compiled PDF.

## Primary-source checks

Checked on 15 November 2026; the bibliography pins preprint versions rather than
implying a peer-reviewed venue.

| Source | Support used in this revision |
| --- | --- |
| [Andreyev and Beneventano, v5](https://arxiv.org/abs/2412.20553v5), [Section 5](https://arxiv.org/html/2412.20553) | Batch-dependent directional curvature; oscillations under sampling are not sufficient evidence of instability. Added `andreyev2025stochastic`. |
| [Gupta, Balakrishnan and Ramdas, JMLR 2021](https://jmlr.org/papers/v22/19-979.html) | Path-length analysis characterizes optimization trajectories beyond convergence rates. Added `gupta2021path`. |
| [Lee and Lee, v2, Section 6.2](https://arxiv.org/html/2608.18415v2#S6.SS2) | A completed-step loss-balance quantity uses gradient, applied direction and observed loss change without HVPs. Added `lee2026road`; no claim that it measures the same step-average quantity. |
| [Bai et al., Sections 4 and Appendix C](https://arxiv.org/html/2506.04805v2) | Adaptive-preconditioner/gradient decoupling, directional monitoring and its additional derivative cost. Reuses the existing entry. |
| [Pesme et al., ICML 2020](https://proceedings.mlr.press/v119/pesme20a.html) | Limitations of the classical consecutive-gradient convergence test even for noisy quadratic objectives. |
| [SVCCA](https://arxiv.org/abs/1706.05806), [CKA](https://proceedings.mlr.press/v97/kornblith19a.html) | Representation comparisons across network sites/states; the present measurement instead follows consecutive changes. No assertion that representation dynamics was previously unstudied. |
| [Kosson et al., Sections 6 and B.6](https://arxiv.org/html/2410.23922), [Brock et al.](https://proceedings.mlr.press/v139/brock21a.html) | Prior representation-change correction and adaptive clipping; motivates positioning the cap as an intervention instrument. |

The existing discussion record supplies the checked Central Flows architectural
gap and the BB reciprocal derivation. Those claims are retained, with existing
citations. The present revision does not claim causal localization, universal
prediction, curvature reduction from noise alone, or novelty for the secant
formula or representation-based control as broad ideas.

## Verification and reproduction

[Build record](build.json) and [verification](verification.json) record hashes,
scope, preserved equations and labels, bibliography additions, resolved
references and length. The build passes without warnings, unresolved references,
or overfull/underfull boxes. Pages 2-4 were inspected individually and all 22
pages were inspected in contact sheets for downstream reflow and floats.

```powershell
python manuscript/consolidation/work/section2_framing_20261115/01_build.py
python manuscript/consolidation/work/section2_framing_20261115/02_verify.py
```

Ignored `scratch/` retains the baseline sources and PDF, scope manifests,
build logs and page renders. Verification of unchanged inputs requires those
local manifests. The revised PDF is tracked and committed with its sources.
Pre-existing unrelated edits are preserved and excluded from the commit.
No experiment, remote execution, or cloud spending was performed.
