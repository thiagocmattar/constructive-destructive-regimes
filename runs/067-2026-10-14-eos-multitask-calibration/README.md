# Run067: EOS task preparation and owned-Spark calibration

Status: complete; all three owned-Spark calibrations retrieved and hash verified.
See [timings and spectra](calibration_results.json), [mathematical checks](mathematical_verification.json)
and [retrieval receipt](retrieval_verification.json).

This run owns input preparation, source parity, mathematical checks and measured
runtime/memory calibration for [Analysis020](../../analyses/020-2026-10-14-eos-multitask/README.md).
Calibration is mechanics evidence only. Production runs use distinct IDs and
immutable configs. Prepared datasets and reference source downloads stay in
ignored cache; all original CIFAR inputs and previous runs are preserved.

The unchanged MLP baseline has656,810 parameters and5,000 images. ResNet-32
has464,250 parameters and all50,000 CIFAR images in fixed1,000-image ghost
groups. The historical tutorial Transformer has13,828,478 parameters,
vocabulary33,278 and2,088,600 supervised training tokens over59,680 fixed
35-token sequence units. All three initializations are frozen in the input cache.

Warm complete-gradient timings: MLP0.0127s, ResNet12.7446s, Transformer30.7291s.
Initial largest-algebraic-eigenvalue solves took0.28/156.84/165.98s, with
2,500 images / five ghost groups /2,500 sequence units respectively. These are
explicit diagnostic subsets; production MLP uses all5,000 images. Relative
residuals0.000258/0.00000806/0.001660 are below0.005. Peak allocations
0.25/5.89/2.75GiB fit on the owned GB10. Early-state measurements qualify ETC;
later eigenvector clustering/solver iterations remain unmeasured.

Cold preparation prerequisites, using pinned revisions:

```powershell
python runs/067-2026-10-14-eos-multitask-calibration/06_fetch_references.py
python runs/067-2026-10-14-eos-multitask-calibration/00_prepare.py
python runs/067-2026-10-14-eos-multitask-calibration/01_verify.py
```

An existing prepared cache is immutable and will not be overwritten. The first
detached SSH launch's acknowledgement timed out; read-only status confirmed its
queue was running. It was not relaunched. The confirmed receipt is inlaunch.json.
Production's separate launcher closes all inherited SSH descriptors explicitly.
No rental, sudo, Docker or shared-runtime modification was required.
