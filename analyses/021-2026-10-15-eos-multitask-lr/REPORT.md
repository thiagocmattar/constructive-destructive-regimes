# Execution and verification status

The user requested a 3x3 full-batch GD figure with multiple learning rates in
each task column, placed after Table 1 (`tab:sign`) in the restatement report.
The current Analysis 020 cohort has only one learning rate per case. Additional
trajectories are necessary; no rescaled trajectory is used as another LR run.

| Case | Learning rates | Seeds | Updates per condition |
| --- | --- | --- | --- |
| CIFAR-10 / tanh MLP | 0.004, 0.01, 0.02 | 0,1,2 | 500 |
| CIFAR-10 / ResNet-32 | 0.1, 0.2, 0.4 | 0,1,2 | 500 |
| WikiText-2 / Transformer | 0.05, 0.1, 0.2 | 0,1,2 | 500 |

Eleven existing conditions are reused, including the first 500 updates of
Runs 059/061/062. Sixteen new frozen conditions are Runs 104-119. The learning
rate is the only varied training factor within each case/seed. Initial tensors,
objective, physical accumulation, data order, precision, diagnostics and fixed
horizon match Analysis 020. The scientific contracts and interpretation limits
are in [PROTOCOL.md](PROTOCOL.md). Three cases use two datasets.

## Live execution

The isolated detached queue launched on owned Spark at 09:53 UTC on 2026-10-15,
PID458832. Its source packet SHA-256 matches the uploaded packet; see
`launch.json`. It is waiting for the unchanged Analysis 020 queue to terminate,
without allocating a CUDA context or interrupting the current Transformer run.
No cloud rental or additional paid storage is used.

At 09:59:50 UTC the preceding Run 075 had 81/500 updates, full-objective loss
7.75161, observed 0.03087 updates/s and peak allocation 3.063 GB. Its queue had
six new runs complete; this sweep had zero new updates. The combined refreshed
ETC was 56.46 hours, with a 79.05-hour planning allowance. This projects to
October 17 afternoon in Sao Paulo, with allowance into October 18. It is an
estimate rather than a guarantee, especially for eigensolver work at new rates.

The extra compute estimate is 38.73 hours; the allowance is 54.22 hours. Its hard
execution limit is 72 hours after the predecessor finishes, with a separate
72-hour waiting bound. Additional retained raw evidence is bounded at 25 GB on
each machine; disk checks found 227 GB local and 3.7 TB remote free. No machine or
unrelated process is stopped. The local controller reads step/loss/throughput,
memory, ETA and stale-progress warnings every five minutes; state is in ignored
`local-controller.json` and `monitoring.jsonl`. Both remote training and logs
survive SSH disconnection. Local plotting resumes when the workstation is awake.

## Checks completed

The unchanged training source passed the existing mathematical/objective-parity
and full-shape smoke tests. New checks verify all 27 conditions, three seeds at
every rate, all frozen source/config/packet hashes, and that only LR/run metadata
differs from each matching baseline. All 45 retained files of Runs 059/061/062
were rehashed. Their 1,497 actual adjacent pairs independently reproduce
`1-rho*norm_current/norm_previous`; a separate quadratic calculation verifies
the sign and learning-rate factor. See `verification.json` and `preflight.json`.

An explicitly marked scratch layout repeats completed MLP data to test panel
spacing only; it is not a multitask empirical figure. Its current render was
reviewed. A scratch report compiled in two passes without overfull boxes or
undefined references; the new figure follows the Table 1 page. The table, figure
and following page were visually checked. The insertion preserves surrounding
source bytes and is idempotent. These layout checks provide no new scientific
evidence. The active manuscript and its PDF remain unchanged while data are
pending.

## Automatic completion

The controller will retrieve and check every retained byte/hash, then generate
`figures/eos_multitask_learning_rates.pdf`. Rows are unsmoothed full-objective
loss, eta times the measured largest Hessian eigenvalue, and unsmoothed
eta*kappa_hat. Each LR has a thick three-seed mean and faint individual traces.
Spectral markers occur at states 0,100,...,500. The spectral objectives and
finite-step directional-curvature caveat are explicit in the caption.

After all conditions pass, `05_manuscript.py` inserts the figure directly after
`tab:sign` in the then-current source, compiles the report, checks the rendered
page order, and attaches the new evidence hashes to its manifest. It preserves
later user edits. Scientific or postprocessing failures stop publication and
retain logs; rates/seeds are not silently dropped. Both final figure and report
remain awaiting explicit visual review. They do not exist yet as completed
deliverables, and no empirical pattern from these new rates is claimed here.

For a later review, inspect `local-controller.json`, then the current standalone
and report renders. If they pass, run `04_plot.py --accept-reviewed` and
`05_manuscript.py --accept-reviewed`, review the task diff and commit the completed
figure/report evidence locally. Do not push without authorization.
