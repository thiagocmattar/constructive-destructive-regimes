# Observation001: real-data frozen-gradient sampling

Sources: immutable Run068 config/input manifest, complete original metrics and
batch indices in its verified Spark attempt, byte/SHA256 retrieval receipt;
Analysis020 frozen_results.json and frozen_figure_manifest.json.
Generating script:06_plot_frozen.py; original experiment:Run06801_experiment.py.

Reduction: each128-draw sequence contributes its final complete-prefix B/A,
127-pair mean cosine and mean projected change. Average32 independent sequences
within each seed, then average three initializations. Horizontal positions are
the RMS sampled-gradient error divided by the complete-gradient norm. All plots
use observed reductions without smoothing. Shading is the range across seed
means; it is not a confidence interval. Conditional noise-energy Monte Carlo SE
is retained in each original condition summary.

Coverage:36 conditions,3 architectures,2 real datasets,3 initializations,
32 sequences each,128 draws,0 optimizer updates. Full-objective references cover
5k CIFAR images forMLP,50k forResNet, and all prepared WikiText-2 training tokens.
ResNet batches contain whole fixed 1k ghost groups. Their gradients are averaged
inFP64 from frozenFP32 unit gradients; no BN group context changes. All parameter
hashes remain unchanged. Each adjacent noise-corrected change is zero up to the
recorded roundoff; full batch has cosine/B/A one.

Caption: Real-data gradient-noise control at frozen initial parameters. Columns
compare CIFAR-10/MLP, CIFAR-10/ResNet-32 andWikiText-2/Transformer; rows show
adjacent cosine,128-draw normalized accumulation and gradient-only projected
change with its zero noise-corrected reference. Measured noise RMS is normalized
by the complete-gradient signal. Seed means and ranges are shown.

Interpretation: the seed-mean ordering is recorded quantitatively inREPORT.md.
Noise does not universally make cosine zero:
the initial ResNet signal remains strong relative to its minimum1k-group noise.
The projected gradient-change observable is noise contaminated even without
parameter motion. This tests a conditional mechanism; it is not evidence about
moving SGD trajectories, EOS entry, loss decrease or generalization.

| Case | Smallest batch | Relative noise RMS | Mean adjacent cosine | Mean B/A |
| --- | ---: | ---: | ---: | ---: |
| mlp | 32 images | 1.232 | 0.398 | 0.639 |
| resnet | 1000 images | 0.403 | 0.863 | 0.930 |
| transformer | 8 sequences | 1.071 | 0.467 | 0.687 |
