# Observation 002: simplified comparison layouts

Question: distinguish comparison across learning rates from comparison between vector streams within one learning rate. This layout revision preserves the results and coverage in [the initial two-run observation](../../interim-021-022/observations/001-two-run-comparison.md).

`metrics_by_learning_rate.pdf` is 3x2 in row-major order: training loss, raw cosine, momentum cosine, adaptive Adam cosine, raw individual norms and momentum individual norms. Every panel overlays all selected LRs; color and line style identify LR. Light loss/cosine traces are unsmoothed; bold loss is a trailing 25-update token-weighted mean; bold cosines are trailing 25-update medians. Norms are unsmoothed on linear axes; all three cosine panels share [-1.05, 1.05].

`geometry_within_learning_rate.pdf` has one column per LR. Top panels compare raw/momentum A and B (solid A, dashed B); bottom panels compare raw/momentum/adaptive B/A. Color identifies stream. Both rows share scales across LR columns; no smoothing is applied.

Sources: completed verified Run021/022 event logs, plus Run023 when explicitly selected after verified retrieval. Display: full-sized updates 1-711. All 712 updates including the partial final batch remain in `../simplified_figure_data.json`, with source/generator hashes. Undefined first cosines are gaps. One matched seed, identical initial tensors/order, LR-scaled AdamW decay; no new causal, stability or convergence claim. Previous numerical findings are unchanged.

Generating script: `../../08_plot_simplified.py`. Future extension adds the third LR to every metric panel and adds a third geometry column, using `--runs 021 022 023 --output complete-three-lr`.
