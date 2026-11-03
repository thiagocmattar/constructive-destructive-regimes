# Curvature estimation as the central contribution

The author's follow-up restores the core practical argument: use gradients
already computed by training to estimate directional curvature online, explain
local instability within destructive regimes, and interpret stabilization
heuristics. This pass updates the same revision and deliverable; the shortened
appendices and all experimental inputs are retained.

## Comparison with the original paper

Read the original submitted manuscript's introduction and conclusion in
`manuscript/2026-10-03 review/_IEEE__Constructive_and_Destructive_Training_Regimes__A_Framework_for_Interpreting_SGD_Stability (5).pdf`
(12 pages; SHA-256
`6687a3903ed05c0b1629c7065f935d9e940fa4d10a6d659512cb5f2566699eea`).
Also read the corresponding source sections in `manuscript/latex-zip.zip`,
whose SHA-256 matches the preserved inventory:
`185580d6a356ea5ad27794ab3c54b3983e3ee4d616dfd9dad0d72a15f0d1819c`.

The original motivation joins inexpensive online diagnostics, temporal gradient
interaction, and explanations of optimization heuristics. The revision restores
that connection and makes the curvature estimator explicit. The current
controlled evidence supports the regime and spike account; the original
momentum/weight-decay intervention conclusions are not substituted for it.

## Scientific statements and placement

- **Formal curvature content.** Under smooth full-batch GD, the estimate equals
  the directional Hessian curvature averaged along the completed step. The
  appendix derives this identity and the starting-point error bound
  `L_H * eta * ||g_(t-1)|| / 2`. On quadratics the estimate is exact at the
  starting point as well.
- **Sampling.** The same plain-SGD identity includes the explicitly defined
  endpoint-noise projection. It is not silently equated to noiseless curvature.
- **Adaptive steps.** Adam exposes a bilinear projected curvature response
  along its adaptive displacement. The Rayleigh quotient applies to a step
  along the raw gradient. This distinction is made positively in the Adam
  opening and supported by the shared path-integral argument.
- **Regime and instability.** Adjacent raw cosine defines the two regimes.
  In the local GD model, loss increase requires destructive alignment plus
  sufficient norm amplification; destructive descent remains possible. The
  Adam population retains the actual 117/127 negative pre-rise count.
- **Online cost and clock.** One additional gradient vector in memory, scalar
  norms and linear reductions suffice. No disk file, extra backward pass or
  Hessian--vector product is required by the diagnostic. It describes the
  completed step when the next ordinary gradient becomes available, before
  the next parameter update. This temporal interpretation is compatible with
  the observed lead before later loss peaks. Runtime overhead is not benchmarked.
- **Heuristics.** The discussion relates step size, clipping, momentum and
  data mixing to the measured components, then motivates future matched
  interventions. Clipping caps the optimizer input, not the already-computed
  raw gradient.

Revised the abstract, contributions, related-work positioning, estimator
interpretation and cost paragraph, GD discussion, Adam opening/discussion,
conclusion, and concise mathematical appendix. Reviewed every affected
paragraph and the curvature figure caption for consistency.

Curvature from successive gradients also underlies
[Barzilai and Borwein's two-point step sizes](https://doi.org/10.1093/imanum/8.1.141).
The citation credits that secant tradition; the contribution is the joint
curvature/regime interpretation and the controlled spike evidence.

## Verification

The mathematical verifier adds 180 smooth nonquadratic cases to the retained
328 supplemental checks: GD path averages and the Lipschitz error bound,
SGD endpoint-noise correction, and Adam's integrated bilinear response.
Independent three-node quadrature integrates analytic Hessians of quartic
objectives; GD secant curvature is cross-checked separately. These checks
validate the stated mathematics and do not run new empirical experiments.
The 186 earlier mathematical cases and frozen GD/SGD/Adam evidence checks
also pass. The PDF remains 14 pages, with eight figures, two main tables,
and no appendix floats. The current verification and visual acceptance files
pin the delivered PDF and source hashes.
