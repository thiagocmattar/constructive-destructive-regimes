# Spike figure: gradient and momentum

Output: [spike figure](../figures/adam_manuscript_spikes.pdf).
Generator: [01_plot_spikes.py](../01_plot_spikes.py).

Sources: archived `paper_evidence/figure_manifest.json`,
`full_unsmoothed_manifest.json`, and `peak_temporal_results.json`; Run215
`results.json` and the three selected `metrics.jsonl` histories. The script checks
the histories against their archived SHA-256 hashes and compares every plotted
value with the archived arrays. Archived files remain unchanged.

Coverage: MLP LR 0.001, states 693–729; ResNet-32 LR 0.003, states 412–481;
Transformer LR 0.003, states 784–852. All 18 curves are unsmoothed. The previous
15 curves retain their exact values. Three momentum-cosine curves are added.

Caption: Training-loss peaks and associated projected raw-gradient change,
consecutive cosine similarity, and relative vector norms. Blue solid curves
denote raw gradients and orange dashed curves denote stored Adam first moments
in both the cosine and norm rows; the shared legend is below the plots.
Norms are divided by their value at the first displayed state. Light dotted
vertical lines mark episode onset; dark dashed lines mark the selected peak.
Loss and vector diagnostics use parameter state t; moment diagnostics incorporate
g_t before update t+1. Consecutive momentum cosines compare these successive
stored moments; positive bias correction does not change their cosine.

Scope: three selected examples from one initialization per model; descriptive
views without uncertainty intervals or causal inference. No new training.

Verification: 18 unsmoothed curves / 1,056 points pass exact source-array checks;
all three added cosine curves match the recorded moment stream. The shared
legend clears the x labels, PDF text stays inside the page, and the Poppler
render was visually reviewed for legibility and spacing.
