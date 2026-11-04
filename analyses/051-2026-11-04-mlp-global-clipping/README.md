# Analysis051: MLP global clipping response

Completed Run228 / H035. [Findings](FINDINGS.md), [vector figure](figures/mlp_global_clipping.pdf), [metrics and provenance](results.json), [exact plotted values](plotted_values.csv).

Global full-gradient L2 clipping1.0 before Adam moments, constant LR0.001, exact state512..800, versus Run223 fresh same-Spark unchanged control. The five solid unsmoothed rows show log loss, raw rho, raw gradient norm, eta kappa hat, and applied/base LR. Shared prefix448..512 is shaded; fork512 is marked; no hardware transition. Curves share each metric's axis only within this matched comparison. All diagnostic gradients remain raw; clipping applies only to Adam input.

Clipping is active43.4% of updates, but mean response loss rises48.01% and maximum loss3.82x. Earlier spike reduction does not prevent later instability. Single selected seed/window; no universal or mechanistic claim.

Reproduce with `python analyses/051-2026-11-04-mlp-global-clipping/01_report.py` after collecting Run228 and retaining Run223 artifacts. Verify checkpoints with Run228/03_verify.py. No training in the report.
