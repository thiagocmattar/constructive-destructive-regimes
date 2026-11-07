# Run236: inside the MLP Adam loss peaks

H041 / Analysis061. The user authorizes diagnostic replay and iteration on the
owned Spark to explain weights, biases, activations and the final linear layer
at the last pre512 peak and recovery. State means completed optimizer updates.
The literal last local maximum is503 (pre-rise502, return505). It lies inside
the fixed-anchor excursion492/493/508. Inspect these and the nearby larger
465/474/485 excursion;512/522/529 is a secondary continued-control comparison.
Selection uses only retained losses, before inspecting internal measurements.

Replay the unchanged Run223 prefix from the pinned seed0 initialization to535.
Same3072-200-200-10 tanh MLP,656810 parameters, fixed5000-image full MSE,
physical1000 batches, Adam(.9,.999),LR.001,epsilon1e-8, no decay/clipping,
FP32 weights/gradients/optimizer, FP64 diagnostic reductions, deterministic
kernels and TF32 off. Every loss and raw-gradient norm must exactly match
retained Run223; at512 compare full model, optimizer, gradient history and RNG.
No new training design, stochastic minibatches or held-out data selection.

At every state448..535 measure each weight/bias tensor's norm, RMS, maximum,
raw gradient and Adam/actual-step geometry. Observe both hidden preactivations,
tanh activations, RMS/mean/spread, saturation |h|>.95/.99 and mean tanh slope.
Measure output/target residual distributions, accuracy, mean-residual and
centered MSE, common-class error, classwise output means and error tails.
Reductions cover all5000 examples and all coordinates; no subsampling.
Zero-denominator ratios/cosines are null. This uses no per-sample gradients.

Retain parameter/optimizer/RNG/gradient snapshots and full pre/post-activation
arrays at each event's pre-rise, peak-1, peak and recovery, plus512 and535.
At these snapshots measure exact singular values and output-head norm bounds.
Compare rise and recovery with all64 hybrids of the six weight/bias groups:
evaluate the same full objective, exact six-player Shapley allocation of loss
change, and individual replacement/reversion tests. These are static parameter
recombinations, not retrained/frozen-layer interventions; interactions and
reference dependence prevent unique mechanistic causal attribution.

Decompose output differences exactly into old-head times hidden-feature change,
head-weight change on old features, their interaction, and output-bias change.
Allocate quadratic loss change by linear residual plus symmetric cross-term
sharing, checking sum identities. Inspect hidden/output change coherence,
feature sign changes and saturation transitions. Evaluate loss along the actual
incoming peak step with scales0..1.5 to distinguish stale/misaligned direction
from finite-step overshoot; include exact endpoints and original gradient dot
actual update. No Hessian or curvature theorem is inferred from cosine alone.

Expected compute under5 minutes, qualified by earlier ~100 full-gradient steps/s
but allowing substantial SVD/hybrid overhead. Hard1200s worker limit, detached
1230s timeout and30s kill grace. Persist progress, manifest, source identities,
rolling32-step recovery and all selected snapshots; retain evidence locally and
on Spark after size/SHA verification, expected below1.5GB including archives.
Stop on replay mismatch/nonfinite evidence. Zero cloud cost. Monitor boundedly
and preserve unrelated processes. One selected seed/window: descriptive and
static counterfactual evidence, no independent-seed or universal claim.

## Completed and verified

Completed7 November2026 at13:19:22UTC. All535 unchanged Adam updates replayed;
all536 boundary losses/raw norms match retained Run223 exactly. Full state512
model/optimizer/RNG/gradient equality passes. Observer preserves RNG/gradients.
Recorded88 internal boundaries and15 paired parameter/activation snapshots;
512 six-group hybrid evaluations and100 actual-step scale evaluations complete.

Worker12.895s, replay/observer phase11.906s, peak CUDA allocation351,147,520 bytes.
All42 pinned inputs and42 retrieved files plus archive pass size/SHA checks.
Independent CPU checks validate full-example MSE and output identities and
Shapley via all720 parameter-group permutations. Post hoc Analysis061 probes
add feature geometry and32 norm-restored/magnitude-only evaluations from saved
states; no new training updates or changes to this run's executed inputs.

Loss peaks occur with small parameter-norm changes and nearly unchanged hidden
activation RMS. Static probes place most of the loss change in hidden features;
incoming Adam steps overshoot lower-loss portions of their measured paths.
See [Analysis061](../../analyses/061-2026-11-07-mlp-adam-spike-internals/observations/001-spike-internals.md)
for the full answer, distinctions between allocation and causation, recovery,
tanh/readout interpretation, figures and uncertainty limits.

All task processes exited. Run folder retains887,767,040 physical bytes on
Spark including archive; analysis probes retain102,400 more. Local raw files
465,460,440 logical bytes plus422,095,295-byte archive. Zero cloud resources
or cost. No unrelated process changes, manuscript edits or external publication.
