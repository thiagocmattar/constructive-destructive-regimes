# Analysis042: eight-update cooldown and cosine recovery

[Report](REPORT.md), [comparison](comparison.csv), and [paired three-column figure](figures/paired_three_column.pdf).
Each column compares unchanged Adam with one intervention: half learning rate,
eight-update cooldown, or cooldown plus cosine recovery. All trajectories use
solid lines. Five rows show loss, raw adjacent-gradient rho, gradient norm,
projected gradient change, and applied LR. Every panel has an independent Y
range; both curves within that panel use the same units and scale.
[Paired fixed zoom](figures/paired_three_column_zoom.pdf) preserves states673..745.

Five matched367-update continuations from state657. Cosine recovery improves
mean/final training loss2.26%/4.30% over always-half in this one development case,
but retriggers prevent a return to full LR;365/367 updates stay below base.
The abrupt hold still has excursions. No incremental timing/generalization claim.

02_paired_comparison.py generates the current paired views without training.
[Observation002](observations/002-paired-comparison.md), [paired values](paired_comparison_values.csv),
and [paired checks](paired_comparison_verification.json) record definitions,
source hashes, all plotted values and independent axes. 01_report.py regenerates
the original outcome tables and earlier overlay figures; [Observation001](observations/001-cooldown.md),
[original checks](verification.json), and [raw inventory](raw_inventory.csv) remain.
The earlier [four-arm overview](figures/focal_four_arm.pdf) and complete five-arm
evidence are preserved. No new training or changes to the measured outcomes.
