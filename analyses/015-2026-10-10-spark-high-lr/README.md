# Analysis 015: higher learning rates on DGX Spark

Status: launched at2026-10-11 00:48:48 UTC on Spark; calibration then two full passes. [Launch verification](launch-verification.json) pins the detached queue and source packet. The user requested
LR 0.02 and 0.04 on the owned DGX Spark and automatic results while away.
Runs 053/054 are new matched Pythia-14M passes; completed Spark Run023 at
LR 0.01 is the reference. H014 is exploratory, with no directional prediction.

Only peak learning rate varies. Match the frozen Run002 initial tensors,
seed1234 data order, pinned MiniPile tokens and tokenizer, architecture,
FP16 compute / FP32 parameters and moments, AdamW betas (0.9,0.95), epsilon
1e-8, weight decay0.1 with bias/LayerNorm exclusions, clipping1.0, sequence2048,
effective batch1024 and microbatch4. The existing 1% pre-step warmup starts
at zero and reaches the constant plateau at update9. Weight decay is not
an LR schedule; changing LR also changes its parameter displacement.

Each pass covers all728,375 sequences, 1,491,711,416 input tokens and
1,490,983,041 supervised targets in712 accepted updates. Final update712
has311 sequences; retain and mark it. Initial/final validation both include
all339 sequences and693,329 supervised targets, including the short tail.
Training loss is supervised-token-mean cross-entropy within each update.
Both seeds are1234; this is one matched realization, not seed replication.

## Measurements and figures

Capture every accepted whole-model raw gradient g after accumulation/unscaling
and before clipping; stored uncorrected FP32 Adam first moment m after the
accepted update from clipped gradients; and bias-corrected/preconditioned
Adam direction d from the actual FP32 moments, excluding LR and decay.
For each stream x, record norm, adjacent cosine, A=sum||x||, B=||sum x||,
and B/A. Prefix accumulation/reductions are FP64; previous g/m are FP32 and
previous d is FP64. First cosine and zero denominators are null with reasons.
Overflow retries repeat the same batch and do not advance measurement clocks.
These are vector-stream statistics, not realized parameter displacements.

Deliver two PDFs: a compact loss/norm figure and a full training-dynamics
figure with loss first, per-LR raw/momentum norms, adjacent raw/momentum/
adaptive cosines, raw/momentum A and B, and all three B/A curves.
Loss is unsmoothed. Raw norms use a trailing10-update arithmetic mean in norm
units before log display, with faint originals and fill to unsmoothed momentum.
Raw cosine uses a trailing50-update arithmetic mean; momentum/adaptive cosine
uses25; faint originals remain visible. Prefix windows use available finite
values and preserve the undefined first cosine. Cumulative geometry is unsmoothed.
All712 accepted updates remain in completed plots and data; no added floor.
The fill is a magnitude gap, not an uncertainty band or causal noise estimate.

## Execution and automatic completion

Single GB10, sequential runs; expect roughly10 hours total from the matched
completed Spark timings (about25.3 seconds/update), with10-12 hours as a
planning allowance. No cloud rental or provider billing. Each full pass has
an8-hour process-group timeout; calibration is separately reset and bounded.
Retain initial, midpoint, every100-update and final model/optimizer/scaler/RNG/
data-position/three-stream recovery states. Allow20GB additional task evidence
on each machine; shared input files are reused read-only.

Freeze scientific source/config hashes before launch. A detached Spark queue
performs separate3-update mechanics calibrations, independent verification,
full training, final reconstruction/coverage verification, byte inventories,
and automatic remote plot generation. Nonfinite loss or exhausted overflow
retries are recorded failures; do not reduce LR, alter precision, restart
scientific inputs or mask failed conditions. Continue to the other authorized LR.

A hidden local controller polls every60 seconds, records step/tokens, loss,
throughput, GPU temperature/power, memory, ETC and warning states, retrieves
terminal outputs and all checkpoints, and checks every byte count/SHA-256.
It retries transport failures with a bounded deadline, preserves Spark data,
and retrieves the remotely generated figures. Remote training and plotting
survive desktop disconnection; local retrieval resumes when its controller
can reach Spark. No machine reconfiguration, termination or cloud provisioning.

Interpret outcomes descriptively. Loss deterioration or numerical failure is
observable; adjacent signs/norm growth alone do not define stability or
generalization. Clipping and EMA affect raw/momentum magnitudes, and larger
LR also scales decay. One initialization/order/seed and one pass do not yield
a universal LR threshold. All figures need local sources, reductions, captions,
coverage, verification and declared review status.
