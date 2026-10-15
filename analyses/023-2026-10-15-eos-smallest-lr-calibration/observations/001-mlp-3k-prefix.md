# Verified smallest-LR MLP prefix

Source: completed immutable Run059, CIFAR-10 first5k, tanh200-200 MLP, plain
full-GD LR0.004, seed0. Analysis019's verified inventory and independent
mathematical/state/forward checks cover the owning20k trajectory. Analysis023
uses only its first3k updates and states0..3000; reuse is not a fresh seed/run.

Reduction: `05_reduce.py --mlp-only`. Used-row coverage3,000 updates,2,999
adjacent pairs,31 residual-checked spectral states. Every used q, cumulative
A/energy, B-squared/all-pair, lag-one sum and B/A identity is checked; initial
tensor/source/config/transfer hashes are retained in `mlp_prefix_results.json`.
The state3000 loss comes from the next row's preupdate full loss, not the
20k endpoint. No smoothing or cross-seed aggregation is applied.

| Quantity | Value |
| --- | ---: |
| Initial loss | 0.5967808664 |
| Loss at state3000 | 0.3365076333 |
| Mean adjacent cosine, final1000 pairs | 0.9999999572 |
| Mean q, final1000 pairs | 0.0000437457 |
| B/A after3000 gradients | 0.7941435956 |
| Maximum sampled eta*lambda_max | 0.3168271484 |
| First negative adjacent pair | Not observed within3000 |

This low-LR trajectory supplies a positively aligned calibration baseline;
it does not demonstrate a reversal in3k or establish an asymptotic outcome.
One seed supplies no ensemble uncertainty. The gradient-derived q and cosine
are algebraically connected and are not independent confirmation. No figure,
manuscript change, new optimizer execution or cloud spending is involved.
