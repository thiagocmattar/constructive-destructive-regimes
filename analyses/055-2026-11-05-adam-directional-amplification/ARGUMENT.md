# Argument to develop for the Adam section

## Central question

Does gradient amplification during a destructive regime provide advance warning
of a subsequent pronounced loss excursion, beyond magnitude and recent loss?
The present step establishes descriptive coverage and counterexamples. A claim
of incremental forecasting utility requires the deferred comparisons and tests.

## Paper sequence and evidence roles

1. Explain the observable: Q=1-rho*r combines the incoming raw-gradient direction
   and amplification along the path taken by Adam. Both are available after
   computing the current loss/gradient and before selecting the next update.
2. Show repeated development and recovery using the author's winning
   [Analysis036 figure](../036-2026-11-02-controlled-adam-diagnostics/figures/adam_manuscript_spikes_multiple_lognorm.pdf).
3. Show complete trajectories, ordinary periods and low-rate controls. Ask if
   similar magnitude growth has different associations at different directions.
4. State what survives those controls, and defer calibrated warning probabilities
   and intervention utility until separately tested. Avoid making the severity
   regression with future peak norms the centerpiece of an online-warning claim.

An alarm at +0.4% current loss can be useful if it precedes much larger increases.
For the inspected Transformer case, Q crosses two at state 788 (loss +0.4% above
L_787), before +10% at 790 and the peak794. At788 a monitor can act before update 789;
it does not undo the displacement that reached788. Early escalation detection,
lead to first increase, lead to +10% and lead to peak are distinct quantities.

## Mathematical scope

For nonzero gradients Q=1-rho*r is an exact identity. Under smooth fixed-objective
GD, Q/eta equals gradient-direction Hessian curvature averaged along the preceding
step. Under Adam, its exact smooth-path counterpart instead involves

```
Q/eta = integral_0^1 g_previous^T H(theta_previous-u*eta*d_previous)
                         d_previous / ||g_previous||^2 du.
```

This is a mixed raw-gradient/adaptive-direction response, not a Hessian eigenvalue
or the same quantity as an Adam preconditioned-Hessian Rayleigh quotient.
The endpoint/trapezoidal GD approximation gives
DeltaL approximately -eta*||g_previous||^2*(1+rho*r)/2, exactly for quadratics.
Thus rho*r<-1 is the completed-step GD quadratic loss-increase boundary. It
motivates an interpretable score; it is not an Adam necessary condition or a
guarantee about future updates. Raw-gradient growth need not enlarge an Adam step.
The current paper already contains the formal identities in its curvature appendix.

## Build on prior work, without equating quantities

[Bai et al., ICML 2026](https://proceedings.mlr.press/v306/bai26o.html),
[full paper](https://arxiv.org/html/2506.04805), identify adaptive-preconditioner
decoupling and monitor gradient-directional curvature with HVPs. Their discussion
identifies high-dimensional mechanism attribution, stochastic prediction timing
and scaling substantially beyond200M as limitations. Our potential complement is
an interpretable training-gradient-only monitor with tested warning usefulness.
It does not solve their causal attribution problem or automatically approximate
their eigenvalue predictor. A direct cost/utility comparison remains backlog.

[Mishkin et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1ac83203e88eb6cf6b30642f0239b932-Abstract-Conference.html)
already study path-dependent directional smoothness. Secant curvature itself has
classical antecedents, including Barzilai-Borwein (1988), cited in the manuscript.
[SPAM](https://arxiv.org/abs/2501.06842) already studies gradient spikes and
spike-aware clipping/momentum reset. Neither gradient monitoring nor clipping
alone should be claimed as new. The contribution must rest on the formal
interpretation, conditional evidence and any demonstrated incremental utility.
These primary sources were checked on2026-11-05; this is a focused overlap map,
not an exhaustive literature review.

## Batch size: supported observation and open mechanism

Analysis037 shows that smaller batches attenuate strongly negative sampled
alignment at the selected trained SGD states in all three cases. In the noiseless
GD surrogate, rho=-0.9 requires r>1.11 for loss increase, while rho=-0.3 requires
r>3.33. This motivates the hypothesis that weakening coherent reversal can
interrupt amplification. It does not show that noise reduces the norm ratio:
sampling changes both norms and the path, and adds terms to the deterministic
loss relation. Sampled alignment attenuation may reflect measurement noise rather
than reduced underlying full-gradient curvature. In the existing trained-state
Transformer case, smaller-batch alignment is less negative but objective progress
can be worse. Therefore no demonstrated Adam spike stabilization is claimed.

## Scalability: feasible design, still a demonstration to do

The minimal observer needs one retained gradient, O(P) vector reductions and no
additional gradient/HVP evaluation. A retained FP32 gradient uses 4P bytes (1.2 GB
at 300M parameters,4 GB at 1B, decimal units). Distributed implementations can retain
local shards and reduce scalar products; memory and communication are not free.
Current Pythia evidence reaches 70M. A >200M example plus measured capture overhead
could establish feasibility; a quiet successful run alone would not establish
spike-prediction utility. This is backlog, with no model/budget/launch selected.
