# Analysis042: eight-update cooldown and cosine recovery

[Report](REPORT.md), [comparison](comparison.csv), and [clean four-arm figure](figures/focal_four_arm.pdf).
The author requested a view without the orange one-step curve; all four metric
rows, including raw gradient norms, remain. [Clean fixed zoom](figures/focal_four_arm_zoom.pdf)
preserves states673..745. The complete five-arm evidence is retained separately.

Five matched367-update continuations from state657. Cosine recovery improves
mean/final training loss2.26%/4.30% over always-half in this one development case,
but retriggers prevent a return to full LR;365/367 updates stay below base.
The abrupt hold still has excursions. No incremental timing/generalization claim.

01_report.py regenerates all CSVs/PDFs without training. [Observation](observations/001-cooldown.md)
records sources, reductions, visual checks and limits. [Verification](verification.json)
and [raw inventory](raw_inventory.csv) retain exact evidence. Run220 owns the
frozen scientific code and both attempts; the UTC-only audit correction is explicit.
