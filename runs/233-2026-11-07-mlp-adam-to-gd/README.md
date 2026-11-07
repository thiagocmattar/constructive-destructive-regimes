# Run233: switch warm MLP from Adam to GD

H040 / Analysis060. Authorized7 November2026 after the beta2 study: plain
full-batch GD at LR0.02 on owned Spark. Load the original Run223 state512,
not a beta2-treated endpoint. Same weights, data/order, loss, physical1000
partitions and RNG;5000-image full MSE,seed0. Run288 updates to800.
Discard Adam moments from the update rule; no momentum, adaptive scaling,
clipping or weight decay. FP32 model/gradients, FP64 whole-model diagnostics.

Counterfactual: retained same-Spark Adam(.9,.999),LR.001. Exact fork-gradient
and two-update Adam tensor/moment/history/RNG replay must pass. GD first2
updates must equal a1+1 resumed path, including model/optimizer/gradient/RNG.
This jointly changes optimizer and learning rate, so no optimizer-only causal
claim. No new full Adam control, prefix or beta treatment is run.

Record full loss/raw norm/cosine/projected change at512..800; outgoing
gradient/actual-step cosine, dot product, increment norm and deviation from
-0.02*g at512..799. Raw history includes the preceding Adam gradient; the
first adjacent pair crosses the optimizer boundary. Zero denominators yield
null. Projected-change row retains the same formula as other figures, not a
universal stability test. Full10000-image test loss at800 is secondary.

Save first-step audit513, rolling recovery every32, final800, completed-state
counter, SGD optimizer/RNG/raw-gradient history. Detached worker with600-second
limit; terminal artifacts and logs survive SSH loss. Expected seconds of compute
and several minutes total preparation/retrieval; allow60 seconds for added checks.
New evidence expected below0.2GB, retained locally and on Spark after size/SHA
verification. Zero cloud cost; leave all unrelated Spark processes/files alone.

Primary outcome: mean513..800, last32 mean, endpoint/max loss and >10% recovered/
censored excursions against unchanged Adam. One selected seed/window; no
generalization or convergence inference. Analysis060 owns figures/observations.

## Completed

Completed7 November at12:52:25UTC:288 GD updates,2.78s treatment and3.73s worker
including6 qualification updates. Exact retained-Adam replay and GD1+1 resume
pass. Independent CPU checks confirm no optimizer momentum/Adam state, the
first-step GD formula and complete objective coverage. All39 artifacts and
archive SHA/size verified. FP32 rounding is explicitly bounded in the actual
increment geometry audit; recorded evidence and executed inputs are unchanged.

No training loss increases or>10% excursions in this window. Mean loss+2.30%
and tail mean+55.74% versus Adam; max loss-59.92%, endpoint-30.82%. The lower
test endpoint is a single-seed secondary result. [Analysis060](../../analyses/060-2026-11-07-mlp-adam-to-gd/README.md)
contains the report, figure and limits. Worker exited;147,365,888 bytes retained
on owned Spark including archive. Zero cloud resources/cost; no unrelated changes.
