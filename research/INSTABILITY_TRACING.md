# Reusing the representation-oscillation tracing workflow

Use this guide when asked to trace a loss spike, instability or oscillation
backward through another architecture or training setup. The goal is descriptive
localization and candidate mechanisms. A causal intervention is a separate task.
Use existing session authorization; this guide adds no approval round or default
cloud spending. It is an optional diagnostic, not a baseline requirement.

Read the [consolidated scientific section](../analyses/063-2026-11-07-architecture-instability-tracing/observations/004-consolidated-instability-tracing.md)
and [formal definitions](../analyses/063-2026-11-07-architecture-instability-tracing/METHOD.md).
Reuse their conventions unless the requested question requires an explicit change.

## 1. Establish the contract before capture

Record the checkpoint/run identity, dataset/model revision, initialization,
optimizer and LR, precision, original training batches, update indexing and
continuation/replay qualification. Define a loss excursion using a declared loss
series and baseline/recovery rule. Include pre-excursion and recovery states;
label retrospective event selection and censored recovery. Negative cosine is
not itself the loss-event detector.

Record the computation graph boundaries: input, projections/embeddings,
pre/post nonlinearities and normalization, both residual branches and their sum,
pooling, output projection and declared output space. Retain repeated module-call
identifiers. First-pass Transformer attention boundaries do not expose internal
query/key/value or attention-weight mechanisms. Avoid calling partial coverage
an examination of every internal operation.

For displayed outgoing updates a,...,b, capture states a-1,...,b+1. Use cadence
one around the event: subsampling can hide or cancel alternating motion. A stride
greater than one measures a different lag and must be labeled accordingly.
Do not promise reconstruction from a final checkpoint when intermediate states
and the original training/recovery history are missing.

## 2. Decide probes explicitly

**Prefer the full fixed example set when its measured cost fits. Otherwise choose
a fixed, structure-preserving subset and document why it fits.** Probe size is
not a universal constant; do not carry 1,000 images or 64 sequences into a new
architecture without calibration. The representation probe changes measurement
coverage, not the native training objective or update batch.

| Decision | Required treatment |
|---|---|
| Selection | Record split, IDs, order, size, selection rule and seed. Prefer a predeclared representative sample; a convenient prefix needs an explicit representativeness limitation. Never select examples because their later trace supports the hypothesis. |
| Batch-dependent operations | Preserve complete original normalization groups and context, including synchronized BatchNorm groups where applicable. A smaller batch can change the function being measured. |
| Language/structured data | Fix sequences, context, token positions, masks and loss weights. Keep coordinate correspondence across states. Record supervised tokens separately from sequence count. |
| Augmentation/dropout | Freeze diagnostic inputs and stochastic choices across states, while preserving the original stochastic training process. Save/restore RNG and buffers, or observe an equivalent isolated model copy. Repeat the probe at identical weights to check the noise floor. |
| Model mode | Declare training-mode or inference-mode measurement. Switching to eval mode is a different diagnostic function, especially with normalization; never do it silently. |
| Output/loss space | Use the task's declared output space. For cross-entropy, log-probabilities remove common-logit shifts; a correct-label log-probability vector checks the loss readout. Do not impose this on squared-error outputs. |

Store fixed inputs/targets or immutable identities sufficient to reconstruct
them. No per-example averaging before SVD. Dynamic routing or variable shapes
require explicit correspondence/masks, not silent zero-filled coordinates.

Log native training loss with its actual coverage, and probe loss at every
capture. If native training is minibatched, its loss is not a full-objective
evaluation. Preserve training unchanged; a fixed/full reference loss or fixed-
probe gradient is an explicitly costed additional diagnostic. Native adjacent
minibatch gradients include sampling changes, while fixed-probe representation
motion isolates model changes under the probe contract. Raw gradients, clipped
gradients and actual updates remain separate streams.

Check probe/full-reference loss timing where available. Agreement is useful but
does not prove population-wide activation representativeness. For stronger
claims, use a predeclared independent second probe or expanded coverage; document
the additional cost and any disagreement. Current ResNet/Transformer conclusions
come from one probe each, not this stronger replication.

## 3. Budget the actual computation

Let L be the number of captured representation streams, T the displayed updates,
P_ell the scalar coordinates per stream, and S=sum(P_ell). If gradient SVD is
included, add its parameter count as another stream. Representation capture
normally needs an extra probe forward at each of T+2 states, plus device-to-host
copies, storage and model snapshots. Hooks can sometimes reuse an identical
native forward; qualify this before assuming its cost is zero.

| Operation | Leading cost for dense exact measurements |
|---|---|
| Adjacent cosine, activation/displacement RMS | O(S) arithmetic per state; O(S) rolling storage, in addition to model execution |
| Retain FP32 activations for the whole window | Approximately 4(T+2)S bytes, excluding file overhead, checkpoints and optional duplicates |
| Exact temporal Grams | O(T^2 S) arithmetic; O(L T^2) retained numbers |
| Full temporal eigendecomposition | O(L T^3) arithmetic |
| Optional explicit k spatial patterns | Additional O(T S k) arithmetic and up to O(S k) output storage; temporal coefficients alone do not require these patterns |

Use coordinate blocks and FP64 accumulation. A block width B requires roughly
O(TB+T^2) working storage per stream, rather than a full FP64 T-by-P array.
The T+1 displacement rows include the predecessor needed for the first cosine;
fit the SVD to the final T rows. Model parameters, activations, autograd and
checkpoint buffers have additional memory costs. Raw-gradient capture on a large
model can dominate storage; keeping only scalar gradient norms/cosines does not
preserve a gradient SVD. Declare which coverage is feasible before launch.

**Two valid execution choices:** retain a rolling buffer for inexpensive adjacent
metrics, or retain/reconstruct a dense event window for the fixed SVD. A later
event trigger cannot recover its prehistory without a buffer or replayable
checkpoints. Choose this retention policy in advance. The window-fitted SVD and
window-maximum scaling are retrospective; future online use requires a separately
specified past-only basis/scale and prospective validation.

Calibrate full-size training updates, probe forwards/copies, representative Gram
blocks, eigendecomposition, checkpoint writes and transfers separately. Estimate
the total using [COMPUTE.md](COMPUTE.md), with storage and retrieval margin. Peak
GPU memory alone can greatly understate host/disk requirements. If cost is too
high, explicitly choose fewer stages, a shorter window, a smaller valid probe or
a declared approximation; do not silently pool samples, skip coordinates or
replace exact Grams with sketches. These choices change the measurement contract.

### Historical calibration, not a new throughput promise

[Run 243](../runs/243-2026-11-07-architecture-instability-tracing/README.md) on an
RTX 4090 with PyTorch 2.11.0+cu130 measured:

| Case | Captured states | Captured representation streams | Whole worker, capture + reduction | Peak allocated CUDA | Derived raw-cache bytes |
|---|---:|---:|---:|---:|---:|
| ResNet, 1,000 images | 33 | 131 | 1,494.88 s | 3.53 GB | 161.81 GB |
| Transformer, 64 sequences | 51 | 20 | 753.83 s | 1.75 GB | 37.65 GB |

Each also records raw gradients; the auxiliary correct-label stream was reduced
after capture. Cache sizes include the raw gradient files. The worker times
include native continuation and diagnostics and are not isolated diagnostic
overhead; some CPU/GPU work overlapped. For scale, one Transformer vocabulary
representation is 2,240 x 33,278 = 74,542,720 coordinates: about 298 MB per FP32
state before retaining its history. The MLP companion reused existing Grams and
required no new capture; this does not make fresh MLP tracing cost-free.

## 4. Execute, verify and hand off

### Keep fixed basis and changing motion distinct

Fit one SVD per layer over the declared window: its spatial patterns are fixed
by construction, while their coefficients change at each update. Sign changes
establish reversal along a fixed component. Relative coefficients can also
rotate the motion inside the fitted span; high top-k energy does not establish
a stationary instantaneous direction or local dominant subspace. Cross-layer
temporal overlap is not a subspace-rotation test. A requested rotation test needs
explicit local windows and principal angles/subspace overlap, with mode signs
and near-degeneracy handled; it has not been performed in Analysis 063.

When shortening a window, refit the Gram submatrix and all SVD summaries.
Cropping existing coefficients retains the old basis and answers a different
question. Keep the predecessor displacement for the first cosine, but exclude
that predecessor from the fitted rows. The [five-page technical report](../analyses/063-2026-11-07-architecture-instability-tracing/report/instability-tracing-report.pdf)
illustrates this with MLP 450-500, while preserving the earlier 448-800 analysis.

1. Qualify observer/plain-forward parity and unchanged training gradients,
   RNG/buffers. Clone values before in-place operations. Verify fixed inputs.
2. Continue training unchanged with detached logging and recovery state when
   needed. Record actual losses, completed updates, throughput and memory.
3. Compute outgoing differences, rho, coordinate RMS, exact Grams, fixed-window
   SVD energies/coefficients and temporal overlap. Raw-gradient SVD uses g_t.
4. Verify diagonal norms/adjacent products independently, PSD/trace/reconstruction,
   zero-denominator handling, source identities and replay agreement. Do not
   represent smoke/calibration output as scientific evidence.
5. Report loss and gradient/output alignment; all-stage cosine and relative-RMS
   heatmaps; spectral coverage/temporal overlap; selected absolute RMS and signed
   coefficient traces. Include dimensions, masks, probe coverage, event timing,
   complete spectra/rank95/rank99, captions and generating scripts.
6. Retain dense evidence or enough qualified checkpoints, data and state to
   reproduce it. Grams alone preserve the reported temporal geometry but cannot
   recover arbitrary spatial mechanisms. Keep an inventory of discarded derived
   caches, verify downloaded files, then close only task-owned compute.
7. Write the observation and candidate boundaries, update research records and
   commit completed work. Keep causal interventions separately specified.

Reuse [Run 243 capture/reduction code](../runs/243-2026-11-07-architecture-instability-tracing/README.md)
and [Analysis 063 report code](../analyses/063-2026-11-07-architecture-instability-tracing/README.md)
as qualified examples, not a universal module-hook adapter. Add only the
architecture-specific capture/mask logic required by the new setup. Never infer
a unique cause, generalization, or a validated alarm from negative cosine, a
large norm ratio, or top-ten concentration alone.
