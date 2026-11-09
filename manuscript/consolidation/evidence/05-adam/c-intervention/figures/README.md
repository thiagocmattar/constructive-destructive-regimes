# Selected intervention figures

Present these four families in this order. Every comparison retains unchanged Adam.

| Order / planned figure | Selected PDF | Scope |
|---|---|---|
| 1 / Figure 7: LR schedules | [Persistent half LR](01-persistent-half-lr-cifar10-mlp-resnet-wikitext2-transformer.pdf) | MLP, ResNet and Transformer; independent warm-state comparisons |
| 2 / Figure 8: beta schedule | [Beta2 reductions](15-beta2-reductions-cifar10-mlp-resnet-wikitext2-transformer.pdf) | All three models; .999 to .95/.9/.5 at states 512/256/544, retained moments/clock and native bias correction |
| 3 / Figure 9: gradient clipping | [Global raw-gradient clipping](16-gradient-clipping-cifar10-mlp-resnet-wikitext2-transformer.pdf) | MLP/Transformer .5/1, ResNet 1; before Adam moments; dashed curves show optimizer-input norms |
| 4 / Figure 10: distributed displacement cap q50 | [Stage heatmaps](07-distributed-q50-stage-heatmaps.pdf) | Main display; exact extraction of Analysis 065 Figure 07 page 2 |
| q50 support | [Scalar diagnostics](10-local-cap-distributed-manuscript.pdf) | All three architectures; only unchanged Adam and distributed q50 |
| q50 support | [MLP replication](11-local-cap-mlp-replication.pdf) | Three initializations; only unchanged Adam and distributed q50 |

- **Only representation-inspired intervention:** distributed displacement cap q50. Other quantiles, entry caps and alternative functional interventions are excluded here.
- **Supporting ablation:** the [quantile/LR figure](../14-distributed-cap-quantile-ablation-mlp.pdf) is retained directly in c-intervention to track sensitivity of the same cap method.
- **Heatmaps:** both arms share each captured stage's baseline RMS maximum; cosine uses the same scale. MLP shows 448–800, ResNet 264–576, Transformer 552–928. The MLP diagnostic/replication PDFs extend to 1152.
- **Comparison scope:** these are existing paired studies with different forks/windows; presenting them together does not establish a matched ranking.
- **Remaining experiments:** [ablation TODO](../TODO_ABLATIONS.md). ResNet clip .5 is marked as missing in the figure; every displayed treatment has its own unchanged control.
- **Provenance:** [section records](../README.md), [manifest](../../../copy_manifest.json), [background archive](../background/README.md). Original analysis PDFs remain intact.
