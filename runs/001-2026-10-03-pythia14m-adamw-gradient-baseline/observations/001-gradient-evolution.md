# Observation 001: full-pass raw-gradient evolution

Question: how do adjacent-gradient alignment, accumulated gradient norms, and their ratio evolve during one dense AdamW pass?

Source: `../artifacts/attempt-04/full-pass/events.jsonl` and `summary.json`, accepted by `03_verify.py` and the transfer inventory. SHA-256 identities and all 712 scalar rows are in `../figures/figure_data.json`. Original reduction/plots: `../04_analyze.py`; revised gradient display: `../06_replot_gradient_evolution.py`, with displayed rows and hashes in `../figures/gradient_evolution_plot_data.json`. Whole model, FP64 cumulative sums/reductions, raw unscaled gradients captured before clipping; one update includes accumulated microbatches. Measurements and the statistics below retain all updates. At the user's request, the revised gradient-evolution PDF omits step 712 from all four panels; it displays only the 711 full-sized batches. Prefix cumulative values through step 711 are unchanged. First adjacent cosine is undefined and plotted as a gap. The trailing median uses up to 25 defined update records and is descriptive smoothing, not uncertainty.

Coverage: 712 updates, 728,375 packed sequences including the short tail, 1,491,711,416 input tokens and 1,490,983,041 supervised targets. Seed 1234; one trajectory. The last update contains 311 sequences rather than 1,024 and is omitted from the revised gradient-evolution display. The original PDF with all 712 updates is preserved as `../figures/gradient_evolution_all_updates.pdf`. Optimization-context and temporal-interaction figures retain their original complete coverage.

Adjacent cosines: median -0.043422, mean 0.007969, range [-0.807262, 0.997750]; 378/711 defined pairs are negative.

The trajectory begins with strong positive adjacent alignment, becomes mostly
negative in the middle, and returns to mostly positive alignment late. In updates
179–356, 164/178 pairs are negative and the median is −0.34676. In updates
535–712, only 6/178 pairs are negative and the median is +0.34467. These phases
co-occur with the scheduled learning-rate decline; this single trajectory does
not isolate a causal effect of the schedule.

The cumulative ratio reaches its minimum 0.303528 at update 435 (minimum over
all saved prefix ratios), then recovers to 0.334295. The final smaller update has
raw-gradient norm 1.603219 and adjacent cosine −0.080197; its batch/data differ
from preceding full updates, so it should not represent the late phase alone.

At the final update, the cumulative sum of norms is 321.798123; the norm of the cumulative vector sum is 107.575517; their ratio is 0.334295. This ratio describes how much of total raw-gradient vector length remains in the cumulative resultant. It includes all temporal interactions among the recorded vectors and does not truncate at lag one. A small ratio can arise from orthogonality as well as opposing directions.

| Updates | Median adjacent cosine | Negative pairs | Median gradient norm | Ratio at phase end |
| --- | ---: | ---: | ---: | ---: |
| 1–178 | -0.02033 | 93/177 | 0.41826 | 0.48213 |
| 179–356 | -0.34676 | 164/178 | 0.51655 | 0.31776 |
| 357–534 | -0.09672 | 115/178 | 0.42224 | 0.30988 |
| 535–712 | 0.34467 | 6/178 | 0.31532 | 0.33430 |

Optimization context: full-validation loss moves from 11.032736 to 5.182962, using all 500 validation documents, including the short packed tail. Training losses use changing batches and are not fixed-probe losses. The first AdamW update has zero learning rate under the reference pre-step schedule; its gradient is still measured and included.

Figure caption (`gradient_evolution.pdf`): Raw-gradient geometry during a full MiniPile pass with randomly initialized Pythia-14M and AdamW. Displayed coverage: updates 1–711, each containing 1,024 sequences; final partial update 712 (311 sequences) omitted from all four panels at the user's request. Top left: cosine between consecutive accumulated, unscaled gradients (displayed updates and trailing 25-update median). Top right: cumulative sum of gradient norms and norm of the cumulative gradient vector through each displayed step. Bottom left: their ratio. Bottom right: individual gradient norms. One seed; gradients captured before clipping and AdamW transformation. Undefined first cosine is omitted. No uncertainty band: this is one descriptive trajectory. The original `gradient_evolution_all_updates.pdf` includes all 712 updates and marks the smaller final update. This display choice does not establish that batch size alone caused the final norm increase.

Figure caption (`optimization_context.pdf`): Token-weighted training update loss, initial/final full-validation loss, and reference learning-rate schedule for the same trajectory. Validation points use a fixed split; training batches change across updates.

Interpretation limits: raw-gradient alignment under AdamW does not reconstruct optimizer directions or parameter displacement. Neither alignment sign nor norm trend establishes stability, generalization, or causality. This run does not isolate per-sample energy from intra-batch alignment, validate the manuscript's lag-one approximation, or provide uncertainty across seeds. No manuscript text is changed by this observation.
