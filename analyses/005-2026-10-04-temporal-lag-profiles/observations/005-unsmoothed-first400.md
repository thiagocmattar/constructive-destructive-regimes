# Observation005: unsmoothed overlays over updates 1-400

The user requested a separate 2x2 figure with all variables unsmoothed and the
horizontal range restricted to updates 1-400. The [new figure](../figures/training_size_overlay_raw_1_400.pdf)
overlays Pythia-14M/31M/70M from Runs017/018/019, seed 1235, using solid blue,
orange and green curves. The previous full-horizon overlay and its data, source
script and review manifest are preserved byte for byte.

Sources and metric definitions are those of [observation003](003-training-overview.md).
The generator verifies the complete source CSV hash and the sizes/SHA-256 of
its underlying run sources. It selects exactly updates 1..400 per model, with
no averaging, interpolation, thinning or normalization. All 1,200 selected updates
use full 1,024-sequence batches. Training inputs, data order, optimizer/LR recipe
and diagnostic boundary remain matched across sizes as previously recorded;
architecture and initialized tensors vary. No training or new seed is added.

Loss and raw-gradient norms have 400 values per model. Adjacent rho and
q=eta*kappa_hat=1-rho_t*norm(g_t)/norm(g_(t-1)) have 399 defined values per model;
their first point stays undefined. Raw norms refer to the unique trainable
parameters after accumulation/unscaling, before clipping and AdamW. The norm
axis remains logarithmic; the other axes are linear and include all finite values.
All panels use horizontal limits 1..400. Dotted references mark q=1 and rho=0.

**Caption.** Unsmoothed MiniPile training dynamics at Pythia-14M/31M/70M, seed
1235, updates 1-400. Panels show minibatch training loss, the dimensionless
gradient-change proxy eta*kappa_hat, raw adjacent-gradient cosine rho and raw
gradient L2 norm. Every recorded value in this interval is retained. Model sizes
are encoded by solid blue/orange/green lines. The first adjacent pair is undefined;
only the norm panel uses log Y. These one-seed descriptive trajectories provide
no seed uncertainty, causal size mechanism or curvature/stability conclusion.
AdamW and changing minibatches retain the interpretation limits of observation004.

Reproduce with `14_plot_raw_first400.py`. The
[plot data](../training_size_overlay_raw_1_400_data.csv) contain all selected
values; missing first pairs are blank. The
[manifest](../training_size_overlay_raw_1_400_manifest.json) records source/output
hashes, 1,197 raw identity checks, all 4,794 finite plotted-array checks, CSV
roundtrip, preserved-file hashes and visual acceptance. Render with Poppler to
`scratch/training_size_overlay_raw_1_400`, inspect the page, then run the generator
with `--accept-reviewed`. No independent scientific finding is added by this
display change, and no new run/analysis ID is used.
