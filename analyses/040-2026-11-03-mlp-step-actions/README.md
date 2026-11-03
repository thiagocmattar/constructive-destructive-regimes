# Analysis040: moment-preserving action implementation contracts

Preparation only; H028 reused, no new hypothesis. [Run218](../../runs/218-2026-11-03-mlp-adam-step-actions/README.md)
owns the new source/config/tests and branch identity records. The future matched
counterfactual is A0 from the identical common parent; no neural comparison has
been executed. The definitions follow the frozen Analysis038 protocol and the
coordinator's16a9f02 execution plan, without editing either.

- A0 retains its native Adam parameter-application path. A1/A2 weights match
  native half/quarter-LR Adam from inherited state; all five implemented arms
  preserve native FP32 m/v/age under identical supplied gradients.
- Modified arms stage the current ordinary moments exactly once at zero LR,
  then apply a scalar through the verified native FP32 parameter expression.
  Same-step tests inspect moment age and direction before any outgoing motion.
- A4's fixed median uses only64 prior ideal complete proposal norms and the
  original base LR, with explicit prior-state clocks. Raw/actual norm streams
  are adversarially changed in tests without changing the selected definition.
  Missing, zero, nonfinite or misindexed threshold histories are unusable.
- A4 bounds the ideal proposal; an FP32 rounding example exceeds that ideal
  bound in the actual applied increment. The distinct outgoing audit records
  the excess and the difference from the chosen ideal proposal.
- A8's formula, interior/floor/cap cases, strict alignment tolerance, signed
  geometry, zero/missing/nonfinite history and nonfinite derived quantities
  have explicit eligibility/fallback tests. The rule never reads the current
  outgoing increment and supplies no monotone-loss guarantee.
- Decision fields and outgoing audits have separate availability clocks.
  Branch retention and cumulative rounding use the fork origin and per-step
  chosen alphas; an inherited constant-LR cumulative formula is not reused.
- Every arm branches through the exact parent scientific identity and file
  hash, preserving moments/age/RNG/history. A generated stochastic-oracle
  continuation reproduces after save/resume for all five arms. Wrong arm,
  parent, implementation, action transformation, file hash, origin or cap is
  rejected. Immutable branch/fork bytes survive rolling recovery and failures.
- Final verification:104 contract checks pass,20 source/config/reference entries
  match, and the accepted Run217/Run215 inputs are unchanged. H028 remains
  untested; these are engineering contracts, with no neural mechanism,
  performance, stability or generalization conclusion.

Implemented: A0/A1/A2/A4/A8. Explicitly unsupported: A3/A5/A6/A7. No production
MLP, neural smoke/calibration, remote execution, spending or manuscript change.
Ready for coordinator inspection and the next bounded assignment. CUDA numerical
application, timing and real-device replay remain unqualified. Next IDs are
Run219 / Analysis041 / H029, subject to a fresh registry check.
