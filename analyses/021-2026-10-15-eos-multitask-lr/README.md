# Analysis 021: full-batch GD learning rates across the EOS tasks

Status: launched on 2026-10-15, waiting for the preceding Spark queue. The
requested figure needs additional trajectories; it
cannot be obtained by multiplying an existing trajectory by another learning
rate. See [the protocol](PROTOCOL.md) for the frozen comparison and coverage.

The figure has three columns (CIFAR-10/MLP, CIFAR-10/ResNet-32,
WikiText-2/Transformer) and three rows (unsmoothed training loss,
eta times the largest Hessian eigenvalue, and eta times estimated directional
curvature). Each column overlays three learning rates and three seeds. These
are three task/model cases over two datasets.

The new runs use the owned Spark, sequentially after the existing Analysis 020
queue. No previous run is interrupted and no cloud rental is used. After
completion, the controller retrieves and hashes the evidence, generates the
figure and inserts it immediately after Table 1 (`tab:sign`) in the restatement
report. Scientific validation is automatic; visual review remains explicit.

There are 27 conditions: three learning rates per case and three seeds per rate.
Eleven matching prefixes are reused; Runs 104-119 supply sixteen new conditions.
New compute is estimated at 38.73 hours (54.22-hour planning allowance).
At 10:00 UTC on October 15, the combined estimate including the preceding queue
was about 56.5 hours (79-hour allowance). See [execution status](REPORT.md).
The publication figure and updated report have **not yet been generated**.

Scripts: `00_prepare.py`, `01_queue.py`, `02_launch.py`, `03_collect.py`,
`04_plot.py`, `05_manuscript.py`, and `06_controller.py`. Executed inputs and
source snapshots are immutable. The original Analysis 020 cohort is unchanged.
