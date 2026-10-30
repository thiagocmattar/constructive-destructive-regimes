# Run185 Mini batch noise mechanics and calibration

H023 / Analysis027. Short synthetic checks and complete-objective, abbreviated
GPU calibration of the approved Run186 implementation. This is mechanics and
timing evidence, not paper training evidence. Initial local FP64 enumeration,
MLP per-example energy, moving-noise identity, finite-difference HVP and geometry
recovery checks pass. The Runpod RTX 4090 calibration also passes, including byte-identical scalar replay after recovery. Its conservative full-design projection is 4,729.96 seconds (78.8 minutes); this is an estimate, not a measured full-run duration. See `verification.json`.

Uses the same pinned CIFAR inputs and MLP tensors as Run186. The calibration
records its shortened configuration independently, including complete5k
reference gradients, frozen sampling, moving updates, HVPs, state writes and
an exact interrupted-recovery comparison. Spark is not accessed.

Completed 2026-10-30. The subsequent full production took 1,371.56 seconds;
the conservative calibration estimate was not reported as observed runtime.
All 90 calibration inventory files (274.0 MB) are locally size/SHA-256 verified.
The task Runpod pod and volume are deleted. See Analysis027 closeout.
