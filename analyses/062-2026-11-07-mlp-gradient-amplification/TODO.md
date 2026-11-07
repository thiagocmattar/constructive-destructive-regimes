# Deferred causal tests

User decision, 7 November2026: keep these pending while applying representation
tracing to ResNet/CIFAR-10 and Transformer/WikiText-2. No architecture or optimizer
intervention is authorized by the tracing protocol itself.

- [ ] First-layer-only Adam LR0.5x/0.25x, matched continued/global/later-layer controls.
- [ ] Temporarily suppress first-layer updates while continuing moment clocks.
- [ ] Project actual first-layer increments onto input covariance eigenspaces.
- [ ] Consider covariance-aware damping or function-preserving regularized whitening,
      with explicit optimizer-state controls.

See [observation013](observations/013-input-distributions-and-intervention-options.md)
for definitions, proposed endpoints, confounders and implementation requirements.
The earliest oscillating forward representation is not yet a proven causal origin.
