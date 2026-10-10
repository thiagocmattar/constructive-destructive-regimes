# DGX Spark readiness and full-pass estimates

The connection to `tcml-spark@spark-2277.local` works. The user's NVIDIA GB10
Spark successfully executed all three requested Pythia sizes with the existing
FP16 AdamW training recipe and raw/adaptive FP64 diagnostics. No full scientific
run was launched. The calibration process finished, and the GPU returned to idle.

## Measured estimates

Each full run means **one seed, one complete MiniPile pass**: 1,491,711,416 input
tokens, 728,375 packed sequences, and 712 optimizer updates. Sequence length is
2,048, effective batch is 1,024 sequences, and microbatch is 4. The final update
has 311 sequences including the short tail.

| Model | Post-warmup seconds/update | Input tokens/second | Estimated full pass | Practical allowance | Peak allocated GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pythia-14M | 25.14 | 83,431 | 4 h 58 min | 5–6 h | 5.45 GiB |
| Pythia-31M | 28.68 | 73,127 | 5 h 40 min | 5.7–7 h | 6.37 GiB |
| Pythia-70M | 38.27 | 54,804 | 7 h 34 min | 7.6–9 h | 8.52 GiB |

Running all three sequentially projects to **18 h 12 min**, with approximately
**20–22 hours** a reasonable scheduling allowance. Additional seeds or passes
roughly multiply training time. These are single-GPU sequential estimates.

The point estimate is the mean of updates 2 and 3 multiplied by
`711 + 311/1024`, plus two measured complete validation passes and three times
one measured full recovery-checkpoint save. Validation took 2.8–4.3 seconds per
pass, and individual recovery saves took 0.9–6.1 seconds. The three-save estimate
is conservative for the smaller initial optimizer state. Diagnostic time is
already included in each update and is not added again. Treating the final
partial update as proportional is an approximation; its short-tail shape adds
minor unmeasured overhead.

Practical ranges are judgmental allowances, not confidence intervals. Only two
post-warmup updates per model were sampled, over several minutes rather than a
multi-hour thermal soak. Source-cache preparation, full-data transfer, artifact
retrieval/hash verification, and future diagnostic changes are excluded from the
point estimates. The isolated software environment is already installed. Only
the calibration prefix and full validation are on the Spark; the verified 5.97 GB
full training cache must still be staged for a future pass. Recovery artifacts
are approximately 1.7/3.7/8.5 GB per three-checkpoint 14M/31M/70M pass under the
conservative accounting used here.

## Hardware and runtime

The SSH audit found aarch64 Ubuntu, NVIDIA GB10, driver 580.178.04, CUDA 13.0,
approximately 121.7 GiB total unified memory and 119 GiB available while idle,
and 3.5 TiB free disk. Measured GPU temperatures during the samples were
50–63°C. `nvidia-smi` does not report global dedicated memory usage on this
unified-memory platform; the table uses PyTorch's allocated-memory accounting.

The SSH user lacks direct Docker permission and passwordless sudo. A dedicated
user-owned venv avoids both requirements. Runtime: PyTorch 2.11.0+cu130,
Transformers 5.12.1, NumPy 2.5.0, Python 3.12.3. The environment reports GB10
compute capability 12.1. System configuration was not changed. Official
[PyTorch CUDA 13 wheels](https://download.pytorch.org/whl/cu130/torch/) and
[NVIDIA Spark documentation](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)
were consulted for ARM/CUDA compatibility and memory reporting.

Fresh ARM initializations do **not** match the historical tensor SHA-256 values,
despite using the same seed and SmallInit/WangInit recipe. For an experiment that
requires identical initialization across hardware or conditions, load a common
frozen checkpoint. This calibration establishes execution readiness and timing,
not bitwise replay, scientific replication, or new paper evidence.

## Verification and retained state

Fourteen relevant local geometry/adaptive/scale checks passed. Both calibration
packets were checked by size and SHA-256 after transfer. All nine accepted
updates had finite loss and gradients and zero overflow retries; whole-model
raw norms agree with the independent clipping reduction to maximum relative
error 1.004e-7. Both validation passes per model cover all 693,668 input tokens
and 693,329 targets, including the tail. All 14 retrieved evidence files passed
individual size/SHA-256 checks; see [the acceptance receipt](acceptance-receipt.json).

Attempt01 failed before any update because the older 14M config lacked a shared
entry-point architecture guard. Attempt02 added the existing identical guard
without changing scientific inputs. The failed manifest and both packet
identities are preserved.

The task-owned Spark directory is
`~/experiments/constructive-destructive-regimes/spark-calibration-2026-10-08`.
It retains approximately **9.48 GB / 8.83 GiB**, including **4.60 GB** of
calibration checkpoints, the reusable venv, small input prefix, and logs. No
cloud resources or cloud storage were created; there is no cloud GPU charge.
Only ordinary local electricity applies, which was not measured as a cost.

Sources: [successful summary](artifacts/attempt-02/summary.json),
[hardware audit](connection-audit.txt), [input identity](input_identity.json),
[executed source inventory](source_inventory-attempt-02.json), and
[generating script](01_calibrate.py).
