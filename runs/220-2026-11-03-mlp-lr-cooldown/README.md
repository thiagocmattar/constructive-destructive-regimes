# Run220: eight-update cooldown and cosine recovery

Completed H028 / Analysis042; user approved `Go on` after the explicit8-update
hold and8-update cosine-recovery proposal. [Frozen config](config.json) and
[26-input manifest](input_manifest.json) define the comparison. Five matched
367-update continuations, state657 to1024, use the same seed0 CIFAR-10 full-objective
MLP,5,000 images/physical1,000, tanh3072-200-200-10, half squared error, Adam
LR0.001/betas(0.9,0.999)/epsilon1e-8, no decay/clipping, FP32 state and FP64 products.
See the original Run217 recipe and authenticated Run218 parent; no prefix retraining.

Available $\eta\widehat\kappa_t\geq1.8$ triggers absolute half LR0.0005. Arms:
unchanged, always-half, one-step,8-update hold, and the same hold plus8-update
cosine recovery. Every trigger restarts the hold including the current update;
no compounding. Undefined finite values preserve active schedules or use base
when inactive; nonfinite values fail. The cosine ramp is0.0005+
0.00025(1-cos(pi*k/8)), k=1..8. Scientific inputs were frozen before execution.

01_cooldown.py owns the scientific rules/native Adam continuation.20 focused
checks pass; [focused_checks.json](focused_checks.json). Attempt001-local stops
on an overly strict replay comparison of UTC metadata, after60 qualification
updates. Its failure/traceback and all evidence remain. 02_continue_after_audit.py
excludes only observer-history pre-observation UTC fields; altered numerical
values still fail its synthetic audit. It reuses the completed qualification,
finishes the48 remaining qualification updates and executes all five main arms
in attempt002-audit-continuation, with unchanged scientific identity.

Both new policies24 direct versus12+12 resume match weights/moments/RNG/policy
and full numerical observer. All three original policies reproduce every368
Run219 boundary and their endpoint saved state exactly. All new starts match
weights, moments, age, RNG and previous raw/actual vectors. Complete scalar
histories and immutable start/end plus rolling recovery checkpoints are retained.

[Report](../../analyses/042-2026-11-03-mlp-lr-cooldown/REPORT.md): cosine recovery
improves mean/final training loss2.26%/4.30% over always-half here, but repeated
triggers keep it below base LR365/367 updates. Abrupt cooldown remains inferior
to always-half. One development seed/state; no generalization/timing claim.

1,835 main plus108 qualification updates;1,957 complete gradient calls,9.785M
image evaluations plus50k endpoint tests. Active workers57.75s, attempt span155.32s,
under600s; owned5070Ti only, zero cloud cost. Raw evidence2,487,747,183 bytes.
No additional variants, full implementation suite rerun, manuscript edit or push.
