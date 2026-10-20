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
consecutive cosine rho_t, normalized gradient accumulation R_128 and the
gradient-only estimate eta*kappa_hat_t with its zero noise-corrected reference.
Measured noise RMS is normalized by the complete-gradient signal. Seed means and ranges are shown.

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

## Manuscript notation update, October20

The regenerated figure follows `manuscript/draft/sections/03-methods.tex`.
The exact TeX snapshot hash appears in `frozen_figure_manifest.json`.

| Previous figure label | Current notation | Meaning in this control |
| --- | --- | --- |
| Adjacent cosine rho | Consecutive cosine rho_t | Mean over127 adjacent pairs per sequence |
| Normalized accumulation B/A | Normalized accumulation R_128 | norm(sum g_t)/sum norm(g_t), evaluated after128 draws |
| Projected change q | Gradient-only estimate eta*kappa_hat_t | 1-rho_t*norm(g_t)/norm(g_previous) |
| Noise-corrected change=0 | eta*kappa_hat_t+epsilon_t=0 | Exact frozen-full-gradient cancellation, up to roundoff |
| Gradient of F | Gradient of mathcal L(theta_0) | Complete-objective reference at the fixed initialization |

The horizontal coordinate is the measured RMS of xi_t=g_t-gradient L(theta_0),
divided by the full-gradient norm. Its expectation is estimated over32x128 draws
within each seed; the heavy curve then averages the three seed coordinates.
The covariance formula is not substituted for the observed noise. The draft's
population covariance uses1/N, whereas the original protocol's sample covariance
uses1/(N-1); both finite-population formulas describe the same variance after
that normalization is accounted for. No variance convention or scalar changes.

Mini-batch sizes are32/128/512 images plus full5k for the MLP;
1/5/20 fixed1k-image ghost groups plus all50 groups for ResNet; and
8/32/128 sequences plus all59,680 prepared sequences for the Transformer.
Each draw samples units without replacement inside its batch; successive draws
use fresh permutations. Batch sizes share permutation prefixes within each seed.
Three initializations x four batch conditions x three models give36 conditions,
1,152 sequences and147,456 recorded gradients. Full-batch repetitions are
identical deterministic gradients.

The manuscript calls R_T normalized displacement along constant-step plain SGD.
Here theta stays equal to theta_0, so the plotted R_128 is the gradient-sum
accumulation form of that statistic, not actual parameter displacement. The
label retains this qualification. The bottom-row expression is evaluated
algebraically on the frozen gradient draws; no learning rate or optimizer step
is applied. Since the full gradient is unchanged, eta*kappa_hat_t=-epsilon_t
exactly up to roundoff. The zero corrected line measures zero full-gradient
change along this control; it does not measure or imply zero Hessian curvature.

All27 original seed curves (108 points), all nine mean curves and every seed
range are unchanged. Full-batch markers and axis scaling are preserved. Only
notation, spacing and the explicit frozen-state footer change. The source
Run068 results and `frozen_results.json` are byte-identical. PDF text, exact
artist/source arrays and the current nine-panel Poppler render pass review.
No new gradients, training, noise draws, manuscript edits or remote execution.
