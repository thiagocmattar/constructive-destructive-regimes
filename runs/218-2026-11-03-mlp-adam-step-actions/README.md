# Run218: moment-preserving Adam step actions

Implementation/preparation only, 3 November 2026; H028 / Analysis040. Implements
A0, A1, A2, A4 and A8. A3/A5/A6/A7 raise explicit unsupported-arm errors.
No production MLP, neural smoke/calibration, remote command, rental, spending or
manuscript edit occurred. The accepted Run217 foundation and all Run215 inputs
remain unchanged. One planned run owns this implementation, not one per arm.

## Fixed scientific contract and application

The parent is the original complete-objective CIFAR-10 tanh MLP: unchanged
3072-200-200-10, same prepared5,000 images, physical1,000 chunks, half squared
error summed over outputs/averaged over images, frozen seed0 initialization,
Adam LR0.001, betas(0.9,0.999), eps1e-8, no decay/clipping/schedule, FP32
parameters/gradients/moments, FP64 products, TF32 off, foreach/fused false.
The planned fork remains after657 updates before consuming g_657; each branch
has at most128 new updates. These are implementation requirements, not a launch.

`01_actions.py` imports the accepted Run217 driver only after checking its pinned
SHA-256. It never modifies that file, its source/config/manifest or strict A0
checks. A0 calls ordinary `Adam.step()` directly. Every modified arm calls one
native Adam step with LR0 to stage the current FP32 m/v and age, verifies no
parameter motion, then applies the selected scalar using the pinned native
single-tensor FP32 `addcdiv_` parameter expression. There is no second moment
recurrence. The local native Adam source hash is pinned and verified at CLI
preparation; native equivalence is tested independently under common gradients.

- A1/A2 apply fixed alpha=0.0005/0.00025. Weights match native Adam at those LRs,
  including an inherited optimizer age. Moments retain the original recurrences.
- A4 fixes C_d once from the median of the64 strictly preceding ideal complete
  proposals `norm(-0.001*d)` at gradient states[f-64,f-1], where d is reconstructed
  in FP64 from actual FP32 moments. At f657 these are states593..656. It applies
  alpha=0.001*min(1,C_d/norm(-0.001*d)). A zero proposal stays zero; a zero prior
  median makes this fork's threshold unusable and raises, rather than inventing
  a replacement. Neither raw-gradient norms nor rounded increments set C_d.
- A8 uses g_current, the current ordinary staged d, the preceding raw gradient
  and previous ACTUAL increment s. With y=g_current-g_previous, it forms
  k_hat=(s dot y)/(s dot s) and alpha_hat=(g dot d)/(k_hat*(d dot d)). It requires
  finite inputs/reductions/proposal, nonzero s/d, g dot d>0 and
  s dot y>1e-8*norm(s)*norm(y). If valid, alpha=min(0.001,max(0.00005,alpha_hat));
  otherwise the selector returns base LR with an exact reason. The alignment
  tolerance is a development choice, not certified curvature resolution.
  Nonfinite raw training gradients/parameters independently fail the execution
  contract rather than being silently applied.

The nominal optimizer-group LR is restored to0.001 at every complete boundary.
The branch identity and chosen-alpha records explicitly declare the applied
transformation. This permits the unchanged strict A0 state/age validator to
check ordinary moments; it does not treat a branch as an A0 resume.

## Availability, diagnostics and recovery

The inherited observer captures g_current/g_previous/s_previous before moment
staging. Modified-arm directions use age t+1 after consuming g_t once, before
parameter application. `choose_action` has no outgoing-increment argument.
A0's alpha is fixed a priori; its direction is captured after its native step
and explicitly labeled as such. `actions.jsonl` contains decision/ideal-proposal
fields only. Outgoing increment norms, rounding residuals and any A4 actual-cap
excess live in post-update `action_audit` under `metrics.jsonl` and the retained
observer history. A cap on the ideal proposal does not imply the same exact
bound on the rounded FP32 increment; the tests exhibit that distinction.

Inherited stream histories remain complete. The foundation's constant-LR
cumulative rounding formula is removed from new branch rows because alpha can
vary. Each branch separately retains fork-origin parameters, accumulated ideal
and actual increments, path length, displacement/retention from the fork,
rounding residual, and A8 fallback/floor/upper-cap/clamp fractions. Zero path
length gives null retention plus its reason. Current-step rounding audits use
the chosen alpha rather than the nominal base LR.

`branch_from_parent` requires an expected parent file hash AND the independently
validated exact scientific identity from Run217 preparation. The ordinary A0
loader checks source/config/input/runtime, normalization/order, moments/ages,
observer and RNG state. The declared fork age is checked; no optimizer reset is
allowed. The new identity includes the parent SHA-256/size/scientific identity,
fork age, exact arm transformation (including A4's64 values and fixed cap), and
new implementation/config/input identities. The output separately records the
parent path for provenance without requiring a fixed physical file location.

Branch snapshots are marked as branches. Loading one requires its own hash,
expected full branch identity, validated original parent origin and fixed cap;
changing arm, parent, transformation or implementation is rejected. The accepted
A0 CLI cannot silently resume a branch because its expected identity differs.
Checkpoints preserve model/moments/ages, RNG, data normalization/order, diagnostic
history and branch counters/accumulations. Branch start and endpoint snapshots
are immutable and saved before that boundary's gradient. Rolling `recovery.pt`
is separate, initially/every32 branch updates/at graceful stop/endpoint. A failed
step leaves the last good recovery and explicit exception/traceback status.
Resume uses a new output directory; original parent and fork bytes are retained.

## Verification and remaining work

`test_actions.py` uses generated vectors and an algebraic gradient oracle.
Together with the accepted contracts,104 checks pass (68 new +36 existing).
The20-entry manifest pins all14 accepted foundation entries, its manifest, the
new driver/config and the three read-only method-definition documents. See
`verification.json` and [Analysis040](../../analyses/040-2026-11-03-mlp-step-actions/README.md).

Safe CLI inspection: `python runs/218-2026-11-03-mlp-adam-step-actions/01_actions.py --help`.
Actual invocation requires --arm, --parent, --parent-sha256, a new --output,
optionally --updates (max128), and --device. Resume additionally requires both
--resume and --resume-sha256, retaining the same parent and arm. Neural execution
still needs a later coordinator assignment and run-plan confirmation. Real-device
application/replay qualification, timing, held-out evaluations, moment-changing
arms and empirical treatment comparisons remain outside this milestone. No
monotone-loss, stability, generalization or treatment-benefit claim is established.
