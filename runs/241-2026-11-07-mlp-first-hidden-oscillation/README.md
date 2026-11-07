# Run241: trace oscillation into the first hidden representation

User-authorized H042 / Analysis062 extension. The two hidden layers both have
200 activation columns and5000 fixed sample rows. Replay the same unchanged
Adam control0..801. Capture first-tanh activations H1, second-linear outputs Z2
(before the second tanh), and second-layer200x200 weights/200 biases at every
state447..801. Verify every full raw gradient and H2 array against Run240 and
every output against Run239. Native reference boundaries through800 and full
model/optimizer/RNG/current/previous gradients512/800/801 must match. Observer
preserves native gradients and RNG. No intervention or new seed.

Same pinned5000-image training objective, order, initialization,3072-200-200-10
tanh MLP, physical1000, half squared error, Adam LR.001/betas(.9,.999)/eps1e-8,
no clipping/decay, native FP32 and TF32off. Persist all355 H1/Z2 arrays and
second-layer states; reuse already retained H2/output/gradient histories. Raw
gradients are never differenced (previous user correction remains in force).

Analysis rows t448..800 are outgoing changes X_(t+1)-X_t. Compare H1, Z2, H2
and outputs with global consecutive-change cosines and fixed uncentered FP64
SVDs; preserve every sample/coordinate. Do not compare H1 and H2 vectors by a
direct cosine just because their widths match: columns are different neurons.
Compare their alignment time courses and temporal SVD subspaces instead.

Decompose deltaZ2 exactly into deltaH1 W2_t^T, H1_t deltaW2^T, interaction,
and broadcast delta b2. Propagate all terms through the observed elementwise
tanh secant slope to account for deltaH2, retaining FP32 forward-roundoff.
For exactly equal preactivations, use the tanh derivative as the zero-change
secant limit. This is trajectory accounting, not causal intervention. Measure
linear/tanh directional gains, activation norms and singular energy coverage.
A square200-to200 layer must not be assumed to have the10/190 visible-null
split of the final200-to10 output layer. Zero norm gives undefined/null.

Expected replay below3min using prior13.8s replay as a qualified reference;
600s worker cap, detached630s timeout with30s grace. Recovery/flush every64,
persistent progress, manifests and byte/SHA256 inventory. Approx2.9GB raw and
below6GB raw-plus-archive retained; transfer/analysis several additional minutes.
Owned Spark, no cloud charge. Verify all download hashes and task worker exit;
preserve unrelated workloads and all previously executed inputs. New capture
tests before launch; SVD/projector and decomposition checks before reporting.
One seed/window, future-informed descriptive bases, no unique cause claimed.

## Completion

Completed801 updates in18.354s,270,178,304bytes peak GPU allocation. All355
gradients/H2 arrays match Run240 and all355 outputs match Run239; native
reference and full512/800/801 checks pass.47 inputs and12 artifact hashes
verified. Worker exited;5,456,855,500bytes retained including archive and source
records on owned Spark, zero cloud cost. See completion/retrieval records.

Analysis062 observation011 and Figures09/10 report first-hidden alignment
correlation0.99590 with second-hidden alignment and0.99155 with outputs. Top10
H1 modes retain95.39% change energy and share almost the same temporal subspace
as H2. The second tanh retains86.84--89.30% of displacement norm. First-hidden
motion contributes82.70% by signed projection to H2 motion; W2 changes17.36%.
These are trajectory identities, not a unique initiating-cause result. All
new SVD and finite-transmission checks pass; figures rendered and reviewed.
