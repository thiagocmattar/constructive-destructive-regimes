# Does clipping add anything to cosine decay?

[Vector figure](figures/mlp_cosine32_clip1.pdf) · [Measurements](results.json) · [Plot values](plotted_values.csv)

- The combined branch used the exact Run223 MLP checkpoint at step 512 and passed a two-update replay. Each of its 288 LRs matched the cosine-only trace, including the floor of 0.0005 at step 544.
- The largest gradient presented for clipping under cosine decay had norm 0.92675. The cap of 1.0 stayed inactive over all 288 updates. At constant base LR, the same cap activated 125 times.
- Every recorded scientific boundary value and every update metric matched cosine decay alone exactly. CPU read-back confirmed exact agreement of final parameters, Adam state, current and previous gradients, RNG state and clock.
- Mean training loss over states 513–800 was 0.01533 for either cosine arm, 0.02080 for baseline and 0.03079 for clipping alone. This gives a 26.3% reduction for the combined policy versus baseline.
- The baseline had 16 excursions above 10%, clipping alone had 13, and either cosine arm had one at step 520. The remaining window after reaching half LR was free of excursions above this threshold.
- Peak training loss was 0.02689 for either cosine arm, compared with 0.05506 for baseline and 0.21016 for clipping alone. The final 32-state mean was 0.01092 for either cosine arm versus 0.01346 for baseline.
- H037 has a definite result for this seed and window. Adding clipping at 1.0 to this schedule had zero incremental effect. In the right column the recorded clipped norm overlaps the raw norm from the fork. Shaded states precede the intervention.

The figure uses unsmoothed per-state records and independent panel ranges. Each column compares one intervention against the same Spark baseline. The shared prefix starts at 448, training loss is logarithmic, and the recorded optimizer input covers states 512–799. State 800 is the terminal observation. The retained Analysis036 detector counts excursions relative to their preceding troughs. No sampling intervals are estimated from this single selected window.

Regenerate with `01_report.py` using the local Spark payloads of Runs 223, 228 and 230. The script verifies three archive hashes and 90 file hashes and sizes. The vector render was visually inspected.
