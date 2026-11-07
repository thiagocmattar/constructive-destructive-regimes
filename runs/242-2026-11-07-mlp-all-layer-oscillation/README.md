# Run242: complete forward-path oscillation, including input and first linear

User-authorized H042 / Analysis062 extension. Replay unchanged Adam0..801 and
capture first-linear preactivations Z1, first-layer weights W1 (200x3072) and
biases b1 (200) at every state447..801. Save one flattened input matrix X
(5000x3072), and verify exact equality of every input batch to the first linear
against that fixed reference, preserving sample order at all355 states. Pixels
and Flatten are separate from the learned first linear representation. If input
changes are exactly zero, report zero change energy/rank and undefined cosine
and normalized SVD fractions, never a fabricated zero cosine or meaningful
zero-matrix singular direction. No input gradients are measured.

Verify every raw gradient/H2 against Run240, H1/Z2 against Run241, and output
against Run239; native boundaries and full512/800/801 checkpoints must match.
Same seed0,5000 fixed training examples/order, full mean half squared error,
physical1000,3072-200-200-10 tanh MLP, Adam LR.001/betas(.9,.999)/eps1e-8, no
clipping/decay, FP32 native and TF32off. Observer preserves native gradients/RNG.
Raw gradients are not differenced; the user correction remains in force.

Analysis window t448..800, outgoing differences X_(t+1)-X_t. Complete path:
fixed input/Flatten -> Z1 -> first tanh H1 -> second linear Z2 -> second tanh H2
-> final output F. Reuse existing histories/spectra downstream by hash; fit new
uncentered FP64 SVDs to Z1 changes and actual first-layer parameter increments
concat(deltaW1,delta b1). Actual increments are distinct from raw gradients,
moments, and gradient differences. Preserve all examples/coordinates, fixed
mode signs, complete spectra and ten patterns. Compare temporal alignments and
temporal SVD subspaces, not arbitrary cross-layer neuron coordinates.

For fixed X, deltaZ1=X deltaW1^T+delta b1. Separate weight/bias contributions,
then use the observed first-tanh secant to account for deltaH1, including
forward-roundoff residuals. Equal preactivations use the derivative limit.
Record tanh directional gain, activation norms, actual increment norms and
cosines; no causal intervention or unique optimizer-instability cause claimed.

Expected replay below3min based on preceding18.4s replay;600s worker cap,
detached630s timeout plus30s TERM grace. Recovery/flush every64 updates,
persistent logs/events/manifest, full final801 and byte/SHA256 inventory.
Approx2.4GB raw / below5GB raw-plus-archive; transfer and local reductions a few
additional minutes. Owned Spark, zero cloud cost. Verify all download hashes
and task worker exit; retain evidence and preserve unrelated workloads. Input,
capture and secant tests before conclusions. One seed/window, post hoc bases.

## Completion

Completed801 updates in17.615s,344,022,528bytes peak GPU allocation. All355
input matrices match exactly, including sample order; all downstream arrays
and raw gradients match prior captures. Native boundaries/full512/800/801
checks pass.49 source and14 downloaded artifact identities verified. Worker
exited;4,592,317,389bytes retained including archive/source records, zero cloud
cost. Analysis062 observation012 and Figures11/12 complete the forward trace.

Input-change energy/rank are zero; its motion cosine and normalized singular
fractions are undefined. First-linear alignment correlates0.99101 with outputs;
top10 modes retain95.57% energy. First tanh retains32.16--36.82% of displacement
norm while preserving reversal timing. About18% of sample-neuron coordinates
with secant>0.5 carry about88% of surviving first-hidden change energy. First
weight motion dominates the learned input projection; no unique initiating
optimizer mechanism inferred. Independent reductions and rendered figures pass.
