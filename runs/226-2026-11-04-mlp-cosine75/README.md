# Run226: MLP cosine32 decay to 0.75x

H033 / Analysis047. User-authorized Spark follow-up. One new branch uses the exact Run223 warm state512 and a32-step cosine reduction from base LR0.001 to0.00075 at544, then holds that LR through the last applied update799. Endpoint800 is observed without an update. No LR rise or optimizer/moment reset.

Reuse the completed Run223 same-Spark unchanged baseline and cosine32-to0.5x branch. Match seed0, original MLP/data/order, full5000-image CIFAR-10 training objective with physical1000 partitions, native Adam clocks/parameters, FP32 vectors and FP64 diagnostics. Require exact warm512 gradient and two-update tensor/moment/history/RNG replay; states512 and513 must match baseline because the outgoing update512 still uses base LR. Record every raw gradient diagnostic and applied LR, save complete state544 and800, and retain recovery. Full10000-image test evaluation remains secondary.

Compute plan:288 treatment updates plus2 qualification updates, no full control/prefix/half-floor rerun. Recent identical MLP branches took2-3seconds, with several minutes for staging/retrieval/reporting; timeout1800seconds. Spark only, zero cloud rental; preserve unrelated PID458832 and existing Run223 cloud work.

Analysis compares each intervention only against unchanged LR, in separate columns (0.75x then0.5x), using log loss, raw rho, raw gradient norm, eta kappa hat and applied/base LR. Solid raw curves, independent panel ranges, original448..800 display. Compare mean513..800, last32, endpoint, and >10% loss excursion amplitudes/recovery/censoring. One selected seed/window, no general stability or generalization claim. Freeze executed source/config/input identities; no manuscript edit or push.

## Completed and verified

Completed 2026-11-04 at 11:51:21 UTC on NVIDIA GB10. Worker4.29seconds including qualification, treatment3.53seconds. All18 transferred artifacts and the archive were size/SHA-verified; all288 applied LRs, checkpoint544/800 clocks/current gradients and raw projected-gradient identities passed. Exact warm512 gradient/two-update replay and common boundary512/513 checks passed. [Analysis047](../../analyses/047-2026-11-04-mlp-cosine-floors/README.md) provides the requested two baseline-only paired columns. No cloud rental, manuscript edit or push.
