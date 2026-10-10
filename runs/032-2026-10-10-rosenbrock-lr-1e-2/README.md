# Run032: Rosenbrock, momentum without decay, LR 1e-2

Completed and verified, 2026-10-10. Exactly 2,000 local CPU updates in attempt
`001-local`; final objective 0.00014134833913, point (1.000349215, 0.999510164).
No cloud resources, billing or transfers. This is one of three matched LR
conditions authorized by the user's refinement; [Analysis011](../../analyses/011-2026-10-10-rosenbrock-lr/README.md)
owns the single comparison figure and H010 method.

f(x,y)=(1-x)^2+100(y-x^2)^2, start (-1.2,1), Adam with betas (0.9,0.95),
epsilon 1e-8 outside sqrt, constant LR 1e-2, weight decay 0,
zero initial moments, no clipping, exact gradients and float64. Only LR varies.
No data, sampling, batch or experiment seed. [Config](config.json) is frozen.

[Observation001](observations/001-trajectory-geometry.md) records results,
coverage and limitations. [Verification](verification.json) and the compact
[manifest](artifacts/attempts/001-local/manifest.json), [summary](artifacts/attempts/001-local/summary.json),
[final state](artifacts/attempts/001-local/final_state.json) and
[inventory](artifacts/attempts/001-local/inventory.json) preserve provenance.
The exact raw history remains ignored locally, 944,736 bytes.
Raw gradients, stored first moments, adaptive directions and actual displacements
are captured at every transition. Post-update loss/gradient norms cover all positions.

`01_run.py` calls [the shared implementation](../../src/controlled/rosenbrock.py),
promoted without numerical changes from frozen Run028. Do not overwrite an
executed attempt or change its source/config. Analysis011 `01_verify.py` replays
in memory; `02_plot.py` recreates the PDF; `03_record.py` records these reductions.
No run jointly sustains the declared loss/gradient stationarity thresholds.
