# Run238: norm-based intra-batch coherence through state800

H042 / Analysis062 extension authorized by the user's five-row figure request.
Replay the unchanged seed0 CIFAR10 MLP Adam control0..800 and measure every
state448..800, including both endpoints. All801 loss/raw-gradient-norm pairs
and all defined adjacent-gradient statistics must match Run223. Full states512
and800 must match model, optimizer, RNG and previous/current gradients.

The new requested quantity is C=||sum_i g_i||/sum_i||g_i||, equivalently
||mean_i g_i||/mean_i||g_i||. It differs from Run237's squared-energy coherence
C_energy=||mean_i g_i||^2/mean_i||g_i||^2. Compute each example's full squared
gradient norm by summing its linear-layer weight and bias contributions before
taking the square root. Never average layer norms in place of the whole-model
norm. Zero mean individual norm gives null; preserve old definitions/artifacts.

All5000 fixed training examples, all656810 parameters, same physical1000,
3072-200-200-10 tanh MLP, full half-squared-error objective, Adam(.9,.999),
LR.001, eps1e-8, no clipping or weight decay. No initialization/data/optimizer
change. Native FP32 vectors, FP64 norm reductions, TF32 disabled. Pin all parent
MLP inputs and reference states by SHA256. Observer preserves RNG and gradients.
Retain per-example gradient energies for all353 measured states, enabling
independent reduction of both coherences. Final/recovery states retain optimizer,
RNG, data-state index and adjacent-gradient history. Small double-precision
autograd tests verify per-example norms and distinguish the two ratios.

Figure uses incoming loss change L_t-L_(t-1), rho_t=cos(g_t,g_(t-1)), and
eta*kappa_hat=1-rho_t*||g_t||/||g_(t-1)||. The last expression retains existing
notation but is projected gradient change under Adam, not a curvature estimate
or stability guarantee. Record state447 history so all rows start at448. No
smoothing/subsampling, no causal claim from aligned peaks, one seed/window.

Owned Spark, zero cloud cost; expected under3 minutes based on the preceding
10.458s535-update replay with heavier diagnostics, but new overhead is not yet
measured. Worker300s cap, detached timeout330s with30s kill grace. Progress every64
updates, recovery every64. Expected retained output plus archive below100MB;
verify sizes/hashes on retrieval and confirm task workers exited. Preserve
unrelated workloads and prior executed scientific inputs.

## Completion

Completed800 unchanged Adam updates in10.113s on Spark. All801 reference
loss/gradient-norm pairs and800 adjacent statistics pass; full512/800 states
match. Captured353 full-example coherence states and retained all1,765,000
per-example energies. Peak GPU allocation286,736,896bytes. All10 artifact files
verified on retrieval:38,126,836 raw bytes plus35,354,243 archive bytes.
Worker exited, zero cloud cost; remote retained size is in `completion.json`.

Analysis062's `04_control_coherence_figure.py` verifies all43 input hashes,
independently reduces all per-example energies, checks the88-state overlap with
Run237 and all1,765 plotted values, and generates the reviewed single-column
[five-row PDF](../../analyses/062-2026-11-07-mlp-gradient-amplification/figures/03-control-coherence-448-800.pdf).
Executed scientific source/config inputs remain frozen. No manuscript changes.
