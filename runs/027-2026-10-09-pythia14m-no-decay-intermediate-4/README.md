# Run 027: intermediate warmup-only LR 0.00464158883361

Status: completed full pass,712 updates, all input/target tokens and full validation;
remote scientific verification passed and completed metric files retrieved/hashed.
Final validation loss: 4.318091; zero overflow retries.
Recovery-state retrieval and cloud cleanup remain in progress.
H008 exploratory comparison. One full MiniPile pass, 712 updates, seed1234;
frozen initialization, token bytes, order, batch and AdamW recipe match Run021.
Only peak LR varies within the new four-run cohort. Warmup-only, no LR decay.
Every accepted update records raw gradient g, actual uncorrected Adam moment m,
and adaptive direction d: adjacent cosine, norm, A, B, B/A and norm ratios.
Full initial/final token-weighted validation includes the tail. Recovery states
are retained initially, every100 updates, at356, and finally, with all streams,
optimizer/scaler/RNG/order position. Original Runs021-023 remain frozen.
Hardware/runtime differences from Spark must be disclosed in pooled comparisons.
See [Analysis009](../../analyses/009-2026-10-09-intermediate-lr/README.md) for execution and limits.
