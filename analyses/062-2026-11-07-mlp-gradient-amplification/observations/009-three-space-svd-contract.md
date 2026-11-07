# Three-space SVD: corrected measurement contract

The user clarified after capture launch that "gradient changes" meant gradient
alignment, not differencing the gradients. Accordingly the gradient SVD uses
**raw gradient vectors g_t**, and its alignment is cos(g_t,g_(t-1)). No gradient
differences are analyzed. Run240 captures raw gradients and thus needs no rerun
or change to executed scientific inputs. Its original launch record is retained
unchanged; this document supersedes the planned gradient reduction there.

All fitted rows correspond to t448..800 (353 rows). At every t, compare:

| Space | Row vector | Coordinates | Adjacent cosine at t |
|---|---|---:|---|
| Gradient | raw full-mean g_t | 656810 weights and biases | cos(g_t,g_(t-1)) |
| Output | flatten(F_(t+1)-F_t) | 5000 examples x10 outputs | cos(outgoing,incoming) |
| Hidden | flatten(H_(t+1)-H_t) | 5000 examples x200 hidden units | cos(outgoing,incoming) |

H is after the second tanh, before the final linear layer. The same fixed5000
training examples and unchanged Adam trajectory apply throughout. Source
states447..801 supply all edges; no validation subsampling or sample averaging.
Fixed independent uncentered SVDs use FP64 subtraction and reductions from
native FP32 arrays, no per-row normalization, and no smoothing. Each pattern's
sign is fixed by its largest-absolute coordinate. Retain ten patterns and the
complete singular spectrum. Rank fractions refer to squared vector energy;
gradient energy and output/hidden change energy have different units. Similar
mode numbers across spaces do not establish a shared direction. A coefficient
sign flip shows gradient reversal in the gradient space, and displacement
reversal in output/hidden spaces. Do not call g_t an actual Adam displacement.

Use the output head W448 (10x200) as a fixed reference. Its orthonormal row basis
Q defines P=Q^T Q. Split each hidden change into deltaH P and deltaH(I-P).
Visible coordinates deltaH Q^T preserve visible norms/inner products exactly;
null vectors remain in the original200 hidden coordinates. Fit additional SVDs
to these two components. Compare the visible share of activation energy to
the visible share of change energy, consecutive cosines, and absolute RMS
norms. This tests concentration without equating a nearly constant activation
norm to a stationary representation. Quantify W_t drift and reference-subspace
leakage; visibility is defined relative to W448, not automatically every W_t.

For output attribution, evaluate exact terms deltaH W_t^T, H_t deltaW^T,
deltaH deltaW^T, and broadcast delta b. Project them onto the existing fixed
output v1. Sum must reproduce its coefficient within FP32 forward-roundoff;
report this residual explicitly. This is an observed-trajectory decomposition,
not a causal intervention or fraction-of-loss explanation.

Zero denominators produce undefined/null, no added epsilon. Singular eigenvalue
roundoff is bounded relative to total energy before truncating tiny negatives.
All actual fitted streams must have positive total energy. Validate small known
SVD/projector cases first; verify direct projection, orthonormality, energy and
selected reconstruction residuals against raw arrays. Verify raw gradient
alignments and output spectra against previous independent records. Retain raw
histories, patterns, coefficient tables, source hashes, plots, and observations.
Single seed and window, post hoc fitted bases; no IID inference or unique cause.
