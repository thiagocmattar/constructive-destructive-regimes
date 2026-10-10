# Run 020: DGX Spark readiness and timing calibration

Status: completed and retrieved/verified; calibration only. No full scientific experiment is authorized
or launched by this record. The user requested SSH connectivity, experiment
readiness, and full-run timing estimates for Pythia-14M/31M/70M.

Interpretation of full run: the previous one-pass MiniPile recipe, 1,491,711,416
input tokens, 728,375 sequences including the tail, and 712 AdamW updates at
effective batch 1,024 and sequence length 2,048. Each estimate is for one seed.
The future scientific question and any new diagnostics remain undecided.

Calibration varies model size on the user's idle NVIDIA GB10 Spark, using the
frozen Run002/003/004 architectures and recipes, seed 1234, microbatch 4,
FP16 dynamic scaling, FP32 parameters/moments, Flash SDPA, and the existing
per-update raw/adaptive FP64 geometry. Three complete updates per size are
measured; the first is warmup, leaving two timing samples. Calibration uses the
first three updates of the verified seed-1234 source permutation. Full validation
uses all 693,668 input tokens including its tail. Neither a shorter sequence nor
a smaller effective batch is substituted. This is mechanics evidence only.

Attempt01 stopped before its first update because Run002's older config lacks
the explicit architecture guard required by the shared scale entry point.
Attempt02 injects the same existing 14M guard in memory; scientific settings and
input bytes are unchanged. Both immutable transfer packets and the failed
manifest preserve the execution history. Fourteen relevant local geometry,
adaptive-direction, and scale-contract checks passed.

An isolated user-owned environment is prepared under
`~/experiments/constructive-destructive-regimes/spark-calibration-2026-10-08`.
No system Python, Docker permissions, or administrator configuration is changed.
The SSH account lacks passwordless sudo and direct Docker access. PyTorch 2.11.0
CUDA 13.0 is used because the host is aarch64/GB10; the historical cloud runtime
used CUDA 12.8. Cross-hardware replay is not expected to be bitwise identical.

`00_prepare.py` verifies the full source cache and packages a small matched
training prefix, full validation, frozen recipe copies, and current shared code.
`01_calibrate.py` reuses the existing optimizer boundary and measurement code,
records complete update/validation/checkpoint timings, memory, loss, finite
gradient acceptance, environment, and source identity. A detached shell wrapper
has a 15-minute execution timeout and persistent events/manifest.

Timing estimates must include qualification limits: two post-warmup samples,
cold-cache/setup/transfer overhead, sustained thermals, checkpoint IO, and any
future diagnostic or recipe changes. No paper claim or hypothesis is added.

Results: approximately 5.0/5.7/7.6 hours per full 14M/31M/70M pass, respectively;
peak allocated CUDA memory 5.45/6.37/8.52 GiB. All nine calibration updates were
accepted without overflow retries. All 14 retrieved files passed size/SHA checks.
The GPU is idle. Approximately 9.48 GB remains in the task-owned Spark directory,
including its reusable environment and 4.60 GB of calibration checkpoints.
See [the report and planning limits](REPORT.md). Fresh ARM initializations have
different hashes from historical cloud states; matched experiments must load a
common checkpoint when exact starting tensors matter. No cloud resource exists.
