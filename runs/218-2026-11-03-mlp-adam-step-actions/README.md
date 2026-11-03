> Completed approved local pilot: all8 x128 branches,657-update prefix,
> historical event and contemporary replay qualified. Worker64.06s, zero cloud
> spending. A1/A4 pass the selected-fork training screen; A5/A8 worsen excursions.
> See [empirical report](../../analyses/040-2026-11-03-mlp-step-actions/REPORT.md).
> Original preparation/implementation notes below remain as provenance. Frozen
> config flags do not supersede the later explicit attempt authorization.

# Run218: Adam intervention implementation preparation

Mutable, unexecuted preparation; H028 / Analysis040,3 November2026. Implemented:
A0/A1/A2/A3/A4/A5/A6/A8. A7 Armijo remains explicitly unsupported. No new run or
hypothesis is allocated for this increment. Accepted foundations are commits
9af24113 (Run217) and4cd84ac0 (Run218 scalar actions); their source snapshots
remain in Git. All frozen Run215/Run217 scientific inputs remain unchanged.
No real MLP, neural smoke/calibration, remote execution, spending or manuscript
edit occurred. Engineering checks do not establish treatment benefit.

## Matched scientific inputs

The parent retains the original3072-200-200-10 tanh CIFAR-10 MLP, same prepared
5,000 images and normalization, physical1,000-image chunks, half squared error
summed over outputs/averaged over images, exact seed0 weights, Adam LR0.001,
betas(0.9,0.999), epsilon1e-8, no decay/schedule/pre-existing clipping, FP32
parameters/gradients/moments and FP64 products; TF32 off, foreach/fused false.
The planned fork remains after657 complete updates before consuming g_657,
with up to128 branch updates. No physical partition or objective was changed.

The accepted A0 source is imported only after its pinned hash matches. Parent
loading still uses its exact source/config/input/runtime and optimizer-age
validators. A0 keeps native Adam.step(); the accepted A1/A2/A4/A8 paths remain.

## Implemented methods

- A1/A2 apply half/quarter base LR, preserving ordinary moments and age.
- A3 freezes C_g from64 strictly preceding RAW complete-gradient norms
  [f-64,f-1] (593..656 at the planned fork). One FP32 global transform
  h=g*min(1,C_g/norm(g)) occurs after full accumulation and before either
  native moment recurrence. A zero g gives zero h with factor1. A zero median
  makes the threshold unusable and raises. Native Adam consumes h exactly once;
  original p.grad references and retained raw g are restored/preserved. The
  independently logged h is the actual FP32 optimizer input, not the ideal
  real-valued scaling. No per-chunk clipping or extra objective pass occurs.
- A4 freezes the median of64 preceding IDEAL complete Adam proposal norms,
  not raw or actual-step norms. It scales the ordinary proposal at fixed cap;
  rounded FP32 motion can exceed the ideal cap and is audited separately.
- A5 updates ordinary m/v once from raw g. Only the applied numerator changes:
  d_applied=g/(sqrt(v_hat)+epsilon), while the reference adaptive direction
  still uses m_hat. Reference/applied direction streams and norms stay distinct.
  The inherited m_hat-based applied-rounding fields are omitted; the outgoing
  audit compares the actual increment with -base_lr*d_applied. A zero raw input
  can therefore apply zero motion while the inherited moments continue decaying.
- A6 authenticates the ordinary parent first, preserves v_f and global age f,
  then persists beta2=0.99 in the treatment optimizer. After k new gradients,
  v=.99*v+.01*g^2 and the bias denominator is1-.999^f*.99^k. It is evaluated
  stably using -expm1(f*log1p(.999-1)+k*log1p(.99-1)). The first moment keeps
  beta1=.9 and global age f+k. Native zero-LR staging advances m/v/age once;
  explicit parameter application uses the HISTORY-CORRECT denominator, never
  native1-.99^(f+k). Direction streams, ideal proposals, audits and resumed
  suffixes all use the same correction. The nominal group beta2 stays .99;
  it is not masked as .999 to pass an A0 check.
- A8 retains its prior-actual-step transported-secant rule, strict1e-8 alignment
  tolerance,0.05*base floor, base-LR cap and explicit base-LR fallbacks. It uses
  current raw/staged direction and prior history, never the outgoing increment.
  No method here establishes monotone loss or resolved next-direction curvature.

A3/A5/A6 live in run-local `moment_treatments.py`. The new branches stage native
FP32 moments once at zero LR, then apply the explicit FP32 parameter expression.
The native single-tensor Adam source hash is pinned. The ordinary A0 validators
remain unchanged. A6 has a separate explicit optimizer/history and checkpoint
validator that requires the switched beta, exact global/changed ages, correction,
precision, data/normalization, observer and RNG state; it cannot masquerade as A0.

## Availability and diagnostics

The inherited pre-update observer always receives the RAW complete gradient.
A3/A5/A6 log actual optimizer-input geometry separately, with clipping factor,
input count and fraction. Their last complete FP32 input and FP64 applied
direction are checkpointed. Input and applied-direction geometries start at the
fork; inherited raw/moment/reference/actual histories retain the common prefix.
A5's adaptive stream is explicitly the ordinary reference from its actual stored
moments; its applied stream is the raw-numerator direction. A6's adaptive and
applied streams both use the actual historical beta2 correction.

`actions.jsonl` contains input transforms, numerator/bias choices, direction age,
ideal norms, alpha and availability fields. Outgoing increment norms and rounding
residuals live separately in post-update `action_audit` in `metrics.jsonl`.
The old constant-LR cumulative audit is removed from branch rows; branch totals
accumulate the actual per-step chosen ideal proposal and actual increment from
the same fork origin. Undefined zero-path retention remains null with a reason.
All eight methods preserve the existing observation clock and raw history.

## Authenticated forks and recovery

Every branch requires a parent file SHA-256 and an independently constructed
expected A0 scientific identity. Its new identity pins that parent/age, exact
input transform, fixed raw/proposal threshold values, numerator or beta2 history,
and current implementation/config/input identities. A3's raw threshold, A4's
proposal threshold, A6 fork/changed age/log-product/correction, treatment stream
prefixes/counters and last input/direction persist through resume.

Branch start/end snapshots remain immutable and precede their current gradient;
rolling recovery remains separate. Resume needs the exact arm/parent/transform/
implementation, fixed cap/origin and file hash. A6 uses branch-specific restore
rather than relaxing the ordinary loader. Unsupported A7, wrong arm/numerator,
cap/history/stream age, altered normalization/position or optimizer recipe is
rejected. Nonfinite/negative new-treatment moments fail before applying motion
and cannot replace the last good recovery. SIGINT/termination stop at boundaries.

## Checks and remaining readiness

150 unit/contracts pass:46 new treatment checks,68 updated Run218 action checks,
25 accepted Run217 contracts and11 original diagnostic unit checks. Tests use
only generated tensors/algebraic oracles and unique system pytest temp paths.
They include independent NumPy/native calculations, full-vector clipping versus
per-chunk alternatives, A5 reference/applied separation, A6 constant and mixed
histories, first/multiple changed steps, all-arm stochastic-oracle save/resume,
corruption rejection, availability clocks and overflow recovery preservation.
The21-file manifest pins14 unchanged accepted foundation entries, its manifest,
the current driver/helper/config and three read-only method-definition documents.
Exact checks are in verification.json and Analysis040.

Safe inspection: `python runs/218-2026-11-03-mlp-adam-step-actions/01_actions.py --help`.
Actual invocation still requires --arm, --parent, --parent-sha256 and a new
--output; resume additionally requires --resume and --resume-sha256 with the same
parent/arm. No training launch is authorized by this preparation. A7, real-device
qualification, neural smoke/timing/replay and empirical comparisons remain later
assignments. No speed, optimization benefit, stability or generalization claim.
