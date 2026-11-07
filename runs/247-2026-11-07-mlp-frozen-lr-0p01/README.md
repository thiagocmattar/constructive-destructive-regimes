# Run 247: I4, frozen first layer with Adam LR 0.01 from 500

Continue H045 in the same Analysis 064/065. I4 freezes first-linear weights and
biases from state 448 and sets Adam LR to 0.01 at 500 (half I3, 10x unchanged
baseline). First changed update: 500 -> 501; end state 801. Branch from the same
qualified frozen state 500 as I2/I3, not from either intervention's final state.
The primary comparison is unchanged Adam; earlier interventions are supplementary.

Match seed 0, all 5000 examples/order, mean half squared error, physical batch
1000, tanh MLP 3072-200-200-10, FP32/TF32 off, Adam betas (0.9,0.999), epsilon
1e-8, no clipping/decay, and native moments/clocks without reset. Native Adam
steps all parameters, then restores the first weight/bias; diagnostic gradients
remain available, and first-layer actual increments are exactly zero.

Reuse the frozen-parent reconstruction and require exact boundaries and raw
gradients 448-501 plus all five representations 499-501. Restore 500 and change
only LR. First-step moments/clocks must match and later increments scale 10x
within FP32 rounding (atol 1e-6). Retain dense gradients/representations 499-551,
rolling full-data diagnostics through 800, actual increments, and full fork,
qualification, recovery and final checkpoints. Same windows: 500-550 and 500-800,
with unchanged 448-499 context. No baseline retraining or new analysis folder.

Owned Spark; no cloud resources or charges. Run 246 measured 7.96 s; expect under
2 minutes for execution including qualification and I/O, plus retrieval. Worker
limit 600 s; detached timeout 630 s with 30 s TERM grace. Retain below 3 GB remotely
including archive, with verified local copies. Check all 67 source identities,
monitor persisted progress/logs, stop on nonfinite loss/gradients, and verify all
retrieved sizes/SHA256 plus task exit. Preserve unrelated work. One seed/fork;
training behavior only, no generalization or universal stability threshold claim.

## Completed and verified

Completed state 801 in 8.075 s; 301 intervention updates,
peak allocated GPU memory 456.7 MB. All 67 source identities and 22 retrieved
files pass verification. Exact parent qualification, LR-only initial change,
unchanged moments/clocks, first-layer constancy and finite final state pass.

Loss maximum 500-800: 12.5043435 at 505; loss800: 0.0349163665.
Stopping assessment: **peak_present**, 58 upward transitions above 1e-8.
All full fork500 tensors/optimizer/RNG/gradients match the earlier I2 fork exactly.
See [the shared observation](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/003-lr-halving.md).

Worker exited; 1.952 GB retained in the task directory on owned Spark,
with verified local copies and zero cloud cost. No further halving after the first
passing rate,0.000625, and the user's confirmation to stop.
