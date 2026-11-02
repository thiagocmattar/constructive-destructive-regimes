# Analysis036: controlled Adam diagnostics from initialization

The user confirmed the same frozen initial weights as Analysis024. This extends
Analysis029's Adam diagnostics to full-objective CIFAR-10/MLP,
CIFAR-10/ResNet-32 and WikiText-2/Transformer training. There is one fixed
initialization per case, no seed sweep and no Hessian/HVP measurements. Every
trajectory uses one constant LR and fresh Adam moments.

The user requested calibration first and a reduced search informed by
Analysis034's continuation evidence. [FAST_CALIBRATION.md](FAST_CALIBRATION.md)
specifies nine arms of 128 updates: MLP rates 1e-4/3e-4/1e-3, and
ResNet/Transformer rates 3e-4/1e-3/3e-3. This replaces the unexecuted 21-arm
proposal. The main 1,024-update horizon remains provisional and is not launched
by this calibration controller.

Calibration completed on November 2, 2026: all nine arms /1,152 updates are verified. MLP uses the owned RTX5070Ti
laptop across all its rates; ResNet and Transformer use matched RTX5090-class
Runpod workers across their rates. All nine arms preserve the same scientific
config and pinned inputs. See [REPORT.md](REPORT.md), [results](fast_calibration_results.json) and
[execution receipt](execution_receipt.json). All task pods/disks are deleted;
USD2.26 estimated Runpod cost is provisional. Training-only
calibration skips validation/test passes. Labels describe this short horizon;
they do not establish stability or convergence.

[PROTOCOL.md](PROTOCOL.md) defines the comparison, measurement semantics,
interpretation limits and later study. [Run213](../../runs/213-2026-11-02-eos-adam-from-init-calibration/README.md)
owns scientific code, config and immutable input manifest. Analysis035/Run212 is
separate work. Raw-gradient projected change q remains a gradient-change
diagnostic under Adam; it does not identify Hessian curvature or a stability
threshold.

Eleven focused diagnostics tests and a two-update full-objective MLP mechanics
check pass. [verification.json](verification.json) records preparation checks;
the smoke is excluded from LR evidence. The input verifier hashes data, initial
weights and scientific sources. Executed scientific inputs must not be rebuilt
or changed in place.

The earlier [parallel Runpod estimate](RUNPOD_ESTIMATE.md) is retained as the
historical estimate for the original screening plus main-study proposal. The
reduced calibration has its own USD20 / 90-minute maximum, inside the earlier
USD50 suggested study envelope, with independent guard and verified retrieval.

The two reviewed PDFs are [calibration](figures/fast_calibration.pdf) and
[stream comparison](figures/fast_calibration_streams.pdf). Eight loss labels are
smooth-progress and one is unclassified; none meets the declared sustained
loss-oscillation label. ResNet/Transformer show raw-gradient reversal despite
coherent adaptive directions. MLP remains smooth across the tested grid.
The [observation](observations/001-fast-calibration.md) records figure sources,
reduction, coverage and interpretation limits.
