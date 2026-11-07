# Run237: what amplifies gradients during MLP Adam loss spikes?

H042 / Analysis062. Authorized follow-up to Run236: test the user's small-RMS
normalization hypothesis and search deeper for quantities that spike with loss.
This MLP contains only Flatten, Linear, Tanh, Linear, Tanh, Linear. Earlier RMS
curves were measurements, not normalization operations. Input preprocessing
uses a fixed cached normalization over50,000 training images; no activation-
dependent normalization is introduced. Audit the actual module list and fixed
input normalization scales. Adam's second-moment denominator is a separate
potential amplification mechanism to measure, not conflate with RMSNorm.

Replay the same seed0 original Adam control from0 through535, with diagnostics
every state448..535. Same pinned5,000-image full MSE,3072-200-200-10 MLP,
physical1,000, Adam(.9,.999), LR.001,epsilon1e-8, no clipping or decay, FP32
model/optimizer/analytic diagnostic vectors and FP64 reductions. TF32 off and
deterministic kernels. All536 loss/raw-gradient norm pairs must exactly match
Run223; all15 retained Run236 anchors must match model, optimizer, RNG and
gradient histories. Diagnostics preserve native parameter gradients and RNG.

Measure unit-loss backpropagation: residual r, dh2=r W3, dz2=dh2*(1-h2^2),
dh1=dz2 W2, dz1=dh1*(1-h1^2), and input adjoint. Record norms/quantiles,
per-neuron energy, backward gain ratios and hidden per-example RMS minima.
Compute actual per-example parameter-gradient energy using the outer-product
identity ||dW_i||_F^2=||dz_i||^2||input_i||^2, including biases, then compare
the squared batch-mean gradient with that energy. This genuinely measures
per-example gradient energy; it is not inferred from batch-mean gradients.
Define coherence=||mean_i g_i||^2 / mean_i||g_i||^2, with zero denominator null.
Verify analytic aggregate gradients against native full-objective gradients.

Measure mean squared Frobenius output Jacobian with respect to parameters and
inputs through exact layer-chain formulas, without materializing the enormous
full parameter Jacobian. This is a trace/sensitivity statistic, not a largest
eigenvalue. All examples and coordinates included. Measure per-example loss,
classwise loss and tail concentration. Retain full residual arrays for all88
states and backpropagation arrays at15 anchors for independent audit.

For every outgoing update448..534 record actual parameter displacement,
moment norms, bias-corrected Adam denominators and inverse gains, lower-tail
coordinate contributions to update energy, epsilon proximity, and pointwise
denominator changes. Compare the direction using new momentum/old denominator
and old momentum/new denominator. These are static direction probes, not
counterfactual training. Check the beta2-implied lower bound on a one-step
denominator decrease; a small absolute denominator alone is not an explosion.

Propagate first and second directional network derivatives along the actual
parameter increment. Record g.dot(delta), Gauss-Newton term0.5 E||J delta||^2,
residual-curvature term0.5 E[r.dot(f'')], and next-boundary actual loss change.
These distinguish output-change energy from sensitivity changes and verify
the Hessian directional decomposition. They are local predictions, not proof
of full-step Taylor accuracy. Save directional outputs and native checkpoints.

Small double-precision tests compare formulas with autograd, per-example
gradients, full Jacobians and directional Hessians. Empirical analysis remains
exploratory, one seed and overlapping selected events, with no universal or
unique causal claim. Expected under3 minutes but additional diagnostic cost
is not yet measured; hard1,200s worker cap with detached1,230s timeout and30s
kill grace. Persist progress/manifest/32-step recovery and audit arrays; retain
below0.6GB expected including archive after SHA/size verification. Owned Spark,
zero cloud cost; preserve unrelated processes and all prior executed inputs.

## Completion

Completed unchanged control replay on Spark in10.458s, peak allocated GPU
memory356,956,672bytes. All536 reference loss/norm pairs and15 exact state anchors
pass. Retained88 backward states,87 Adam/directional update measurements,15
backpropagation snapshots and full residual/directional outputs. Downloaded and
verified28 files (203,378,373bytes) plus187,694,812-byte archive. Worker exited;
391,237,632bytes retained in the remote run, no cloud cost. See `completion.json`.

[Analysis062](../../analyses/062-2026-11-07-mlp-gradient-amplification/README.md)
adds static crossed-Jacobian probes and independent CPU verification. The MLP
has no dynamic normalization. Evidence favors growing, reversing output changes
and less gradient cancellation, with larger Adam increments along existing
sensitive directions. No sudden Adam denominator collapse; tanh attenuates
backpropagation. Two verified figures and full observation retain the caveats.
Executed scientific source/config files remain unchanged.
