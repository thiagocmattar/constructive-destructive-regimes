# Run 250: I7, first layer frozen with LR 0.00125

Same ongoing MLP Analysis064/065, H045. Complete intervention: freeze first-linear
weights/biases from448 and set Adam LR0.00125 at500, half the previous tested rate.
Branch from the same qualified frozen parent500, first changed update500->501,
end801. Compare primarily with unchanged Adam; previous rates are supplementary.
All5000 examples/order, seed0, tanh3072-200-200-10, mean half squared error,
physical1000, FP32/TF32off, Adam(.9,.999), eps1e-8, no clipping/decay are matched.
Native moments/clocks continue; zero first-layer applied increments with raw
gradients retained. Exact reconstruction/qualification448-501; full states and
all five representation streams, dense499-551, rolling500-800, retained as before.

Adaptive no-peak rule, fixed before this run: all losses500-800 finite, no adjacent
increase >1e-8, and no loss above loss500+1e-8. If failed, halve again from the same
fork; retain unsuccessful runs. This is a conservative finite-window criterion,
not a universal stability or generalization claim. See analysis halving_plan.json.

Owned Spark, zero cloud cost; measured prior run8.08s. Qualified ETC under2min
for execution plus retrieval;600s worker limit,630s detached timeout+30s grace.
Below3GB retained per run including archive. Verify67 source identities, native
moment/increment parity and freeze constancy; persisted progress, finite checks,
full recovery states, SHA256/size-verified retrieval and task exit required.
Preserve unrelated work. Same diagnostic definitions as Run247 and analysis065.

## Completed and verified

Completed state 801 in 7.846 s; 301 intervention updates,
peak allocated GPU memory 456.7 MB. All 67 source identities and 22 retrieved
files pass verification. Exact parent qualification, LR-only initial change,
unchanged moments/clocks, first-layer constancy and finite final state pass.

Loss maximum 500-800: 0.0283370516 at 501; loss800: 0.0269098519.
Stopping assessment: **peak_present**, 1 upward transitions above 1e-8.
All full fork500 tensors/optimizer/RNG/gradients match the earlier I2 fork exactly.
See [the shared observation](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/003-lr-halving.md).

Worker exited; 2.021 GB retained in the task directory on owned Spark,
with verified local copies and zero cloud cost. No further halving after the first
passing rate,0.000625, and the user's confirmation to stop.
