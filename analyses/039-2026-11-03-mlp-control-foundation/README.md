# Analysis039: MLP control foundation contract checks

Implementation/preparation only; no completed neural experiment. H028 is reused;
no new hypothesis allocated. [Run217](../../runs/217-2026-11-03-mlp-adam-control-foundation/README.md)
owns the executable A0 driver, tests, config and input identities. The study's
counterfactual will be contemporary unchanged Adam from the exact same complete
fork; no treatment has been implemented or compared in this milestone.

- Ordinary Adam capture is bitwise equivalent to capture-disabled Adam across
  12 generated-vector updates, for parameters, optimizer moments/ages and gradients.
- Pre-update products use only current/previous raw gradients and the previous
  actual increment; outgoing products have a separate post-update clock.
  Completed-step q and kappa are scalar observations, not implemented actions.
- Null/reason handling covers absent history and zero denominators. Zero
  numerators remain defined when the denominator is nonzero; signed slopes are
  preserved. Nonfinite incoming vectors are rejected without observer mutation.
- Actual increments subtract FP32 endpoints in FP64. A large-coordinate test
  observes an absorbed FP32 step, keeping it distinct from the ideal Adam proposal.
- Full checkpoint roundtrip restores model, optimizer ages/moments, observer
  history and RNG; a generated stochastic-oracle suffix is bitwise reproducible.
  Initialized CUDA RNG is checked with a mock, with no GPU workload.
- Fork and endpoint snapshots precede current-gradient evaluation. Selected
  dense audits preserve g_current, g_previous, y and s_previous. Rolling recovery
  replacement, overwrite rejection and simulated disk failure preserve all forks.
- Identity/normalization/position/age/history corruption is rejected; preparation
  and objective failures leave explicit status/tracebacks and last good recovery.
  Graceful stop preserves a boundary without consuming another gradient.
- Final focused/reused unit checks pass; 12 inherited file identities match
  Run215, which remains unchanged. These checks establish implementation
  contracts only: H028 remains untested, with no treatment effect, historical
  replay, speed, stability, generalization or mechanism conclusion.

The coordinator can review the foundation and assign the next bounded milestone.
No intervention arm, neural smoke/calibration, training, remote command,
provisioning, spending or manuscript edit occurred. Next IDs are Run218 /
Analysis040 / H029, subject to a fresh registry check.
