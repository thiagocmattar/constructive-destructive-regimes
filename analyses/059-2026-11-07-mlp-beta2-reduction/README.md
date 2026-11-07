# Analysis059: MLP beta2 intervention

Run232 / H039. Three beta2 reductions (.95,.9,.5) from the unchanged Run223
warm state512; constant LR.001, beta1=.9, original moments/clocks retained.
Compare through800 against the qualified same-Spark beta2=.999 control.
Native Adam bias correction changes with beta2; no moment rescaling or reset.

Primary reductions: mean513..800, last32 mean, final/max training loss and
recovered/censored >10% excursions, separately for each treatment. Secondary
complete test loss. Single seed and selected window; no mechanism, stability,
generalization or independent-replication claim. Protocol and capture contract:
[Run232](../../runs/232-2026-11-07-mlp-beta2-reduction/README.md).

Completed: all864 treatment updates verified; exact retained-control and observer
replay pass. Mean training loss increases16.78%/18.16%/127.02% for beta2=.95/.9/.5.
One censored excursion dominates the0.5 response; endpoint test loss is lower
for all three, without replicated generalization evidence.

- [Findings, caption and verification](observations/001-beta2-reduction.md)
- [Three-column comparison](figures/01-beta2-reductions-cifar10-mlp.pdf)
- [Numerical results and provenance](results.json)

Generate/check with `python analyses/059-2026-11-07-mlp-beta2-reduction/01_report.py`.
All10,584 plotted scalars and the full-size render pass. Worker12.15s, owned
Spark, zero cloud cost. Run232 retains audit/recovery/final checkpoints and logs.
