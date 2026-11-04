# Analysis052: MLP clipping thresholds

Completed Run229 / H036. [Findings](FINDINGS.md), [vector figure](figures/mlp_clipping_thresholds.pdf), [results](results.json), [plotted values](plotted_values.csv).

Global norm clipping0.5 before Adam, unchanged LR0.001, seed0 and exact warm state512..800. Compare with Run228 clipping1.0 and fresh same-Spark Run223 unchanged control. The figure places each threshold beside the other, each against unchanged LR, with independent panel ranges. Five rows: log training loss, raw rho, raw and recorded clipped gradient norm, raw eta kappa hat, applied/base LR. Solid curves, no smoothing; shared prefix448..512 shaded, fork512 marked, no hardware transition. Green Adam-input curve is recorded only for states512..799; no outgoing update at800.

Threshold0.5 lowers mean loss31.03% and maximum77.81% versus1.0, but mean loss remains2.08% above unchanged and final32-state mean24.48% above. Peak suppression improves without eliminating oscillations. Single selected seed/window.

Reproduce using `python analyses/052-2026-11-04-mlp-clipping-thresholds/01_report.py` with verified Run223/228/229 artifacts. No new training in report. Run229/03_verify.py independently checks exact replay and final optimizer clocks.
