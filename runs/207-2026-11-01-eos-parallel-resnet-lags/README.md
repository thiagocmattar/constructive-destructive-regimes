# Run207: completed eos parallel resnet lags

Six later ResNet profiles on three reused 4090 workers; both assigned baselines reach 256 before extensions.

[Analysis033](../../analyses/033-2026-11-01-eos-long-lags/REPORT.md) owns
the complete metric/method contract, figures, observations and closeout.
This run supplies 3 canonical final profiles; earlier prefixes
and unused valid partial continuations remain raw evidence. The joint cohort
contains 27 profiles,25 primary, all through 256 and up to 2048.

Executed config, worker, input manifest and shared scientific runner remain
frozen. Original data/order, seed, full objective, LRs and plain SGD match the
parent runs. ResNet BN groups remain fixed. Exact CUDA/NumPy FP64 products
of FP32 full gradients require no compression. Hardware/rounding can perturb
future trajectories; no bitwise, stability or generalization claim is made.

Ignored artifacts retain scalar rows, sparse full vectors and terminal
model/optimizer/RNG recovery states; the final gradient is unapplied.
All retrieved sizes/SHA-256 pass. All 20 task Pods and attached
storage were deleted, with no network volume created. Joint compute estimate
USD 29.48 is provisional within the approved USD 30/four-hour
envelope. See the analysis for failures, activation/source records and final QA.
