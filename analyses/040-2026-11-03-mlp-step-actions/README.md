> Empirical pilot completed after explicit approval:8 complete128-update arms,
> exact historical689/703/729 event and bitwise contemporary control. A1/A4 meet
> the training screen. [Compact report](REPORT.md), [figure](figures/local_pilot.pdf)
> and [numeric table](comparison.csv).64.06s worker, zero cloud cost; single seed.
> Preparation notes below describe the earlier implementation-only milestones.

# Analysis040: Adam intervention implementation contracts

Mutable preparation under Run218; H028 reused, no neural experiment or new
hypothesis/run ID. The future counterfactual is unchanged A0 from the exact same
common parent. The accepted scalar-action implementation is4cd84ac0; this
increment adds A3/A5/A6 and leaves A7 explicitly unsupported. No manuscript edit.

- A0/A1/A2/A4/A8 retain their accepted contracts. All eight implemented arms
  reproduce a generated stochastic-oracle continuation after branch save/resume,
  retaining the same parent and complete optimizer/observer/RNG state.
- A3's cap uses64 strictly preceding raw complete-gradient norms. Native Adam
  fed the independently calculated clipped vector matches its weights/moments;
  an adversarial two-contribution example distinguishes global from per-chunk
  clipping. Raw diagnostics remain raw and the actual FP32 input is separate.
- Zero A3 input is defined without division; moments still decay/age. Missing,
  zero-median, nonfinite or misindexed prior thresholds are explicitly unusable.
  Fixed caps and input clipping counts/geometries survive recovery.
- A5 updates ordinary stored moments from raw g but applies a raw numerator.
  Independent calculations distinguish its reference adaptive direction from
  the applied direction. Zero raw g can yield no applied motion despite nonzero
  inherited moments. Invalid m_hat-based applied-rounding fields are omitted.
- A6 retains the common parent's v and age, persists beta2=.99 and uses the
  stable1-.999^f*.99^k correction throughout application, direction capture,
  rounding and resume. Independent constant/mixed-gradient checks cover the
  first and multiple changed steps; using1-.99^(f+k) gives a different result.
- A6's validator checks its declared history/recipe separately; the unchanged
  A0 validator rejects the switched optimizer. Wrong arm, numerator, cap/history,
  stream clock, source/parent, normalization/position or optimizer recipe is
  rejected. The authenticated parent is always loaded before treatment setup.
- Decision inputs and outgoing audits have separate clocks. Checkpoints retain
  raw/input/applied histories, all moment ages and method counters. Staged-moment
  overflow fails before parameter motion and preserves last good recovery.
- Final150 generated tensor/mock/contracts pass;21 input/source/reference hashes
  match, including14 unchanged foundation entries. These are engineering checks.
  H028 remains untested; no neural treatment effect, mechanism, speed, stability
  or generalization conclusion is supported by this implementation milestone.

[Run218](../../runs/218-2026-11-03-mlp-adam-step-actions/README.md) contains the
source/config/manifest/tests/verification. No real MLP, neural smoke/calibration,
remote execution or spending. A7 and real-device execution/qualification remain
outside this milestone. Next Run219 / Analysis041 / H029 remains unchanged.
