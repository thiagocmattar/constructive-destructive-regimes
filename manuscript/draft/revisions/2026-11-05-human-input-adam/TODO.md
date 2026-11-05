# Adam revision TODO

Status at this revision: supported manuscript changes and existing-data work are
complete. New training remains proposed in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).
The checklist separates completed algebra from unperformed empirical calibration.

## Incorporated and checked

- [x] Preserve input.md and the supplied 05-adam-diagnostics.tex unchanged.
  Integrate into the actual active file, sections/06-adam-diagnostics.tex.
- [x] Organize the section around response, loss-defined events, amplification,
  recovery, severity comparisons and the analytical optimizer-state connection.
- [x] Introduce z as projected gradient response, retaining numerical equivalence
  with eta*kappa_hat in existing figures/table. Explain exact geometric meanings
  of z>1 and z>2 without claiming an Adam loss-rise threshold.
- [x] Verify Bai et al.'s published citation, momentum/stochastic scope and
  one-HVP baseline. Explain the proposed contribution without claiming priority
  for momentum or replacing an explicit full Hessian.
- [x] Derive the path-average mixed projection, first-moment/finite-step
  discrepancy, adjacent-sum memory recursion, finite-time alternation expression
  and Lipschitz bound. Use r_t^(m) to avoid collision with the existing Taylor
  remainder r_t. State smooth-path assumptions and the nonsmooth limitation.
- [x] Numerically check these identities on rotated quadratics, exact alternation,
  arbitrary gradient histories and smooth nonquadratic functions. These are
  mathematical verification cases, not new empirical neural-network findings.
- [x] Add 5/10/20% event counts, >10% occupancy, absolute increases, rise/recovery
  durations and loss-excess sums from all retained histories. Keep six all-amplitude
  censored events in accounting; explicitly describe the two pronounced ones.
- [x] Report per-trajectory pre-rise reversal denominators and MLP concentration
  (84 of 127 defined pronounced events in its middle/high-rate arms).
- [x] Add retained ordinary-trajectory controls: low-rate ResNet/Transformer have
  reversal/z>2 without a >=10% future rise over the fixed eight-update horizon.
  State that this is not the requested duration-matched event comparison.
- [x] Compare rho-only, linear/log norm-only, additive, interaction and product-only
  severity fits; retain all coefficients and held-out-rate results. Show that
  in-sample alignment gains do not transfer uniformly; no held-out-seed claim.
- [x] Reuse the already recorded current gradient/moment and gradient/direction
  alignments. Check actual-update dot products independently. Report all 794
  rise-phase steps as locally descending; no need to rerun for these scalars.
- [x] Regenerate the severity scatter with fit lines restricted to observed
  support, add the ablation figure and incidence table, preserve prior assets.
- [x] Update abstract, introduction, related work, conclusion and measurement
  appendix to reflect the revised claim and empirical limits.
- [x] Compile and visually inspect the complete current PDF, preserve a revision
  copy, check evidence/source hashes, and commit only this task's changes.

## Still needed for the stronger scientific claims

| Priority | Addition | What is missing | Planned source / acceptance criterion |
| --- | --- | --- | --- |
| A1 | Empirical one-HVP calibration | Neural H*g and decomposed corrections were not logged in Run215; final states cannot recover their past | Stage A: fixed grid plus loss-selected phase anchors, same-objective and indexing checks, signed/absolute residuals |
| A2 | Finite-step isolation | No endpoint-gradient trials at fixed d and state | alpha=1,1/2,1/4; conditional 1/8; preserve P/m/d and report nonsmooth/numerical limitations |
| B1 | Independent initialization replication | One initialization per existing architecture, dependent events, varied original hardware | Three seed blocks x nine conditions; report all runs and leave out entire seed blocks |
| B2 | Ordinary windows matched to event duration | Analysis055 uses a fixed eight-update future, not event-length matching | Fixed duration/stage/loss calipers, no outcome-dependent relaxation; publish matched/unmatched support |
| B3 | Population phase profiles across seeds | Current selected examples and single-seed event summaries do not establish architecture-level recovery modes | Dense raw/cross-alignment phase records, event and run weighting explicit, all seed blocks shown |
| C1 | Same-state optimizer-state effects | Existing LR/clipping/decay work does not implement the proposed moment/preconditioner factorial | Four direction choices, natural/matched norms, seven unique one-step branches, exact common state |
| C2 | Short controlled continuations | One-step effects do not establish the subsequent excursion response | 32 updates, pre-rise and ordinary anchors, matched data/randomness; mean/final loss and excess area |
| D1 | Stochastic Adam interpretation | Current Adam event evidence is full objective | Two observation streams, fixed monitoring objective, dense loss plus occasional adjacent same-objective gradients |
| D2 | Clipping/AdamW/BN realism | Changes to optimizer input, decay and batch-dependent objectives alter the derivation | Separate explicitly designed arms; log raw gradient, transformed input and actual increment; no automatic extrapolation |
| E1 | Measured computational advantage | No new on/off/HVP timing or memory benchmark | Identical-checkpoint windows, warmup, time/memory/probe cost and fidelity against one HVP |
| E2 | Larger-model scientific finding | Existing small models do not establish reach beyond the comparison paper | Conditional 1B study with both total and non-embedding counts and reproducible phase findings; single-GPU scope explicit |

The smallest useful next step is A1/A2. B/C follow only after the pilot supports
reliable measurement. D/E are conditional extensions with separate budgets.
No placeholder result, unfinished experiment or anticipated outcome is presented
as a finding in the paper. Calibration/negative outcomes may narrow the claims.

## Existing work that informs, but does not replace, these tests

- Analysis055: complete trajectories and restricted context matching; limited
  support, residual imbalance, one seed and a different outcome definition.
- Analysis038: selected high-rate rise steps already showed local descent;
  Analysis057 extends this check to every pronounced recovered event.
- Analyses040-043: matched LR/clipping/decay interventions; useful feasibility and
  timing evidence, but not the proposed first-moment/preconditioner factorial.
- Run222: Spark timings and same-device replay; its predecessor's cross-GPU
  gradient qualification failure remains a reason to use fresh matched controls.
