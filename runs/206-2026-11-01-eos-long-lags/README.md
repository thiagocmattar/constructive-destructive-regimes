# Run206: completed streaming exact long-lag continuations

Initial full-objective continuations from all 27 Run205 starts for
[Analysis033](../../analyses/033-2026-11-01-eos-long-lags/REPORT.md).
The joint cohort reaches 256; the largest measured lag is 2048.
There are 36,096 exact anchored cosine values, 25 primary profiles
and two previously unqualified Run062 reconstructions retained separately.
Run206 supplies 20 canonical final profiles. Run207 owns
the six later ResNet starts; Run208 owns four qualified checkpoint handoffs.
The source model/optimizer/RNG states and independently verified lag 0..10
histories come from Run205; no replay from initialization was needed.

`config.json`, `01_worker.py`, `input_manifest.json` and copied source protocol
metadata remain frozen. The common Transformer compute chunk selected by the
declared calibration was 512; all nine starts
passed numerical qualification. ResNet's fixed BN groups were unchanged.
Exact GPU/CPU FP64 streaming products avoid compression and growing history
memory. Raw evidence under ignored `artifacts/attempt-*/workerN/` retains
per-lag products, gradients at 11/final pair, and model/optimizer/RNG end states.
The final sampled gradient is not applied. Full reference histories remain in
Run205. `test_streaming.py` is a synthetic mechanics check, not paper evidence.

All 674 files / 2.847 GB were locally
size/SHA-256 verified before termination. All 20 task Pods and their storage
were deleted; no network volume was created. Allocation-to-last-deletion time
was 197.2 minutes and compute estimate USD 29.48
is provisional pending posted billing/storage, within the approved USD 30 cap.
The analysis owns figures, numerical/visual reviews, observations and closeout.
No unrelated resource, original scientific input or manuscript asset was edited.
