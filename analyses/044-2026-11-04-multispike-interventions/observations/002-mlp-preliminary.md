# MLP preliminary: persistent reductions across repeated spikes

Run223 MLP is completed; ResNet/Transformer remain in progress. Seed0, CIFAR-10 full5000-image objective, Adam baseLR0.001. Fork512, end800, displayed448..800.

Sources: runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker/{prefix,control,fixed_half,cosine32_half}/boundaries.jsonl, terminal checkpoint/replay evidence and verified transfer inventory. Generator:02_mlp_preview.py. Numeric reduction and source hashes:mlp_preliminary.json; every plotted scalar:mlp_preliminary_values.csv. Figure:figures/mlp_preliminary_paired.pdf.

- Fresh same-Spark control is required: historical MLP trace does not replay exactly on GB10. No historical bitwise replay is claimed. All branches share the exact reconstructed state512 weights/moments and observer history.
- The unchanged continuation has16 excursions above10% of their own fixed pre-rise losses, including one unrecovered at800. Prominent peaks522/683/716/799 lie inside the unchanged fixed view.
- Immediate halfLR has zero >10% excursions,25.04% lower mean training loss and17.14% lower last32-state mean versus the matched control. Zero counted excursions does not assert zero numerical fluctuation.
- Cosine32 to half has one16.63% early excursion (peak520, recovered523), then no further >10% excursion through800. Mean training loss is26.30% lower and last32-state mean18.88% lower than control.
- Cosine has1.67% lower mean loss and2.03% lower endpoint loss than immediate half. Immediate half suppresses the first transient more strongly. This is a small single-seed policy tradeoff, not proof of optimality or an isolated smoothing mechanism.
- Endpoint loss reductions64.54%/65.26% are exaggerated relative to average gains because control is inside a new, censored excursion. The mean and last32-state mean are reported together with endpoint results.
- Raw gradient norms and projected-change oscillations attenuate; raw rho approaches+1 on both treatment trajectories. These are diagnostic observations, not an Adam curvature/stability guarantee or generalization evidence.

All56 transferred files rechecked by size/SHA;7056 plotted source scalars verified exactly. Five rows, two paired columns, raw solid curves, independent Y ranges; logarithmic training loss. Gray preperiod and vertical line identify the intervention boundary; all observations are from Spark, with no hardware splice. LR closure at800 draws the final interval only and is not another update. The PDF was rendered and visually reviewed, including exposed control peak labels and legends moved off the large spike.
