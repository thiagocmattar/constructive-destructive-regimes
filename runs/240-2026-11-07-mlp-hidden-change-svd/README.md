# Run240: gradient, output, and hidden changes in the same Adam control

User-authorized H042 / Analysis062 extension. Replay the exact unchanged Run223
control through800 and the same extra update to801 as Run239. Capture raw full
mean gradients, raw outputs, final hidden activations after the second tanh,
and output-layer weights/biases at every state447..801. No intervention or new
seed. All5000 examples, same ordering, physical1000, half squared error,
3072-200-200-10 tanh MLP, all656810 weights/biases, Adam LR.001, betas(.9,.999),
eps1e-8, no clipping/decay, FP32 native training with TF32off.

Each requested change is outgoing: X_(t+1)-X_t, for t448..800. Gradient changes
mean differences of raw full-objective gradients, not raw gradients themselves,
momentum, or applied increments. Hidden means the final200 activations before
the output head. Persist all355 raw arrays so differences, alternate reductions,
and independent checks remain possible. FP64 subtraction/products in analysis.
No centering, per-row normalization, sample averaging or state subsampling.

Matched comparison: independently fit fixed uncentered temporal SVDs to
353x656810 gradient changes,353x50000 output changes,353x1000000 hidden changes.
Signs fixed once per basis. Compare spectra, energy coverage, leading signed
coefficients, and global consecutive-change cosines; axes across spaces need
not correspond. Fixed output-head row space at state448 separates visible
hidden motion from its orthogonal nullspace (rank at most10 in200 dimensions).
Measure both motion energy and activation energy there, plus consecutive
cosines and their SVDs. Quantify reference-head drift. Exact head-change
decomposition and projection onto the existing output v1 test which terms
supply the leading output oscillation; interaction retained, no causal claim.
Zero norms give undefined/null, never epsilon-normalized cosines.

Qualification: all801 reference loss/gradient-norm pairs and800 adjacent
statistics; full model/optimizer/RNG/current/previous gradients512/800;
all355 raw output arrays and full801 state against Run239. Observer must
preserve native gradients and RNG. Pin source/cache/reference hashes.
Recovery every64 updates, raw arrays memory-mapped and flushed at recovery,
final full801 state, progress events, manifest and byte/SHA256 inventory.

Owned Spark; zero cloud cost. Previous replay12.3s, additional capture/I/O cost
unmeasured: expected replay below3min,600s worker cap, detached630s timeout plus
30s TERM grace. Approximately2.45GB raw arrays; raw plus compressed archive
expected below5GB, retain both after verified download. Local reductions and
transfer expected a few additional minutes; no cloud resource provisioning.
Read-only bounded monitoring, preserve unrelated workloads. Capture tests
before launch; numerical SVD/projector tests and raw-data verification before
scientific conclusions. One seed/window; descriptive basis fitted post hoc.

## User correction and completion

After launch the user clarified that gradient alignment, not gradient changes,
was intended. Acquisition already retained raw gradients, so executed inputs
and the captured trajectory remain unchanged. The analysis fits raw g_t and
uses cos(g_t,g_(t-1)); it does not difference gradients. The corrected contract
is in Analysis062 observation009; results and limits are in observation010.

Completed801 updates in13.766s,245,492,224bytes peak GPU allocation. All801
native reference loss/gradient-norm pairs and800 adjacent statistics match;
full512/800/801 model/optimizer/RNG/current/previous gradients match. All355
output arrays exactly match Run239. All45 input and13 downloaded artifact
identities verified. Process exited;4,718,201,884bytes retained on owned Spark
including raw histories/archive and source records, no cloud charges.

Analysis062 compares raw-gradient/output-change/hidden-change SVDs and the
fixed448-head visible/null split, with two verified figures. Hidden motion
concentrates68.14% of its energy in the10-dimensional visible subspace, but the
null component also reverses. The first10 modes retain99.04/98.16/97.40% in
the three main spaces. Existing-head transmission of hidden changes dominates
output motion; this is descriptive accounting, not a unique causal explanation.
