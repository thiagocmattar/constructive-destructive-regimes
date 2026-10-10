# Observation002: exact all-pair interaction and lag-one adequacy

Question: does the manuscript's lag-one truncation represent the temporal
cross-step term within these same training windows?

## Sources, reduction and coverage

- [14M Run017](../../../runs/017-2026-10-04-pythia14m-temporal-sketch-seed1235/README.md), `attempt-01/full-pass`: raw sketches, update events, exact pairs/window aggregates, metadata and frozen config. Archive SHA-256: `960f9275c39d642931d93d6d7e0573b78a8d76377c3a004d1f23e05a9383925d`.
- [31M Run018](../../../runs/018-2026-10-04-pythia31m-temporal-sketch-seed1235/README.md), `attempt-02/full-pass`: raw sketches, update events, exact pairs/window aggregates, metadata and frozen config. Archive SHA-256: `63da75f9fcb986668a2b782bf2f4719e6dc11f3cdee869e166a5725786f51850`.
- [70M Run019](../../../runs/019-2026-10-04-pythia70m-temporal-sketch-seed1235/README.md), `attempt-02/full-pass`: raw sketches, update events, exact pairs/window aggregates, metadata and frozen config. Archive SHA-256: `fc18b71737c67981827c14c4ae5f67c40095513f607de4f9fc5221834f663b7b`.

The three 128-update windows, seed and raw-gradient scope match Observation001.
The recorder independently sums raw gradients in FP64 inside each window and
records E=sum||g||², B²=||sum g||², J=(B²−E)/2, exact I₁ and J−I₁. Thus the total
and the beyond-lag-one residual do not depend on the projection. Individual
I_k are estimated from projected dots; they are norm-weighted sums, rather than
the equally weighted cosine averages. The last partial update remains included.
The reduction verifies E and I₁ against the update stream, the exact aggregate
identity, and every stored projected calibration dot against the sketch history.

## Results

| Size | Window | E | Exact I1 | Exact J | Signed I1/J | Relative B² error using lag1 | Projected J relative error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14M | early | 43.233 | 27.7425 | 803.755 | 3.45% | 94.02% | -0.74% |
| 14M | middle | 34.9944 | -10.6946 | 169.095 | -6.32% | 96.35% | 0.22% |
| 14M | late | 15.7826 | 4.11554 | 212.188 | 1.94% | 94.54% | 0.32% |
| 31M | early | 48.3639 | 24.6024 | 333.67 | 7.37% | 86.37% | 0.52% |
| 31M | middle | 31.5427 | -10.0077 | 159.631 | -6.27% | 96.71% | 0.34% |
| 31M | late | 12.666 | 2.44642 | 186.567 | 1.31% | 95.45% | 0.31% |
| 70M | early | 66.5142 | 24.9544 | 199.606 | 12.50% | 75.00% | -0.71% |
| 70M | middle | 30.0204 | -9.92271 | 127.208 | -7.80% | 96.42% | 0.42% |
| 70M | late | 12.2229 | -0.786907 | 139.85 | -0.56% | 96.35% | 0.40% |

The signed share can be negative when lag1 cancels but the all-pair total is
constructive; it is not a fraction constrained to[0,1]. The squared-sum error
is |B²−(E+2I₁)|/B². Projected-total relative error is (estimated J−J)/|J|.
Exact numeric values and all127 lag contributions are in `../results.json`.

The same exact aggregate check is also available for each full712-update
trajectory from the retained baseline streaming sum, raw norms and adjacent dots:

| Size | Full-pass E | Exact I1 | Exact J | Signed I1/J | Relative B² error using lag1 |
| --- | --- | --- | --- | --- | --- |
| 14M | 162.659 | 5.0207 | 5959.67 | 0.08% | 98.57% |
| 31M | 157.353 | -1.45412 | 4713.27 | -0.03% | 98.39% |
| 70M | 168.609 | -2.68484 | 3878.67 | -0.07% | 97.94% |

This full-pass table retains update712 and does not use projected dot products.
It covers the same three trajectories, not three additional training replications.

## Figures and captions

- `../figures/cumulative_lag_interaction.pdf`: Twice the cumulative estimated
  lag interaction through k, divided by exact raw-gradient energy E, for each
  model and phase. This primary plot covers k1–64 and uses the four-map mean.
  Different lag contributions have128−k pairs and different gradient norms;
  the curve is not a cosine decay curve or an AdamW displacement.
- `../figures/cumulative_lag_interaction_all_lags.pdf`: The same cumulative
  quantity through all127 lags. Shading gives the four-map range; diamonds at127
  give the independently measured exact2J/E endpoint. Coverage shrinks to one
  pair at the final lag. Endpoint agreement calibrates the aggregate but alone
  does not validate each individual lag profile.

Generating script: `../03_analyze.py`; sources/hashes: `../results.json`;
PDF integrity and visual QA: `../artifact_validation.json`.

## Interpretation and limits

These exact raw-gradient aggregates test the lag-one approximation directly.
The SGD displacement identity requires the corresponding plain-SGD update rule
and step-size assumptions; these AdamW trajectories do not establish it for
actual parameter updates. A positive all-pair sum can coexist with negative
adjacent alignment. Neither sign alone establishes stability or learning quality.
This is a first, single-seed measurement test in three windows, not a general law
over model scale or training schedules. See `../REPORT.md` for the narrow claim.
