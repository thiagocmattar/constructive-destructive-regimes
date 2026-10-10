# Run019: 70M temporal raw-gradient sketches

Completed: one full MiniPile pass, 712 updates and 1,491,711,416 input tokens;
seed 1235. The original initialization SHA, pinned token cache and paired order
were retained. The executed recipe/config and sources are frozen.

Accepted evidence: `artifacts/attempt-02/full-pass/`; local archive and every
inventoried artifact passed SHA-256/size verification and the scientific checks.
Raw sketch history covers all 712 updates, with four fixed 16,384-bucket maps,
exact raw norms, adjacent pairs, sampled longer-lag pairs and exact window sums.
Initial, update356 and final recovery checkpoints are retained locally.

This is a measurement replay, not a bitwise copy of its earlier reference or an
independent training seed. Actual worker: Secure RTX4090, USD0.74/GPU-hour.
Infrastructure retry details and the frozen-input inventory are in `prelaunch/`.
Resource removal and provisional cost are documented by Analysis005 closeout.

[Analysis005](../../analyses/005-2026-10-04-temporal-lag-profiles/REPORT.md) owns
the cross-run conclusions, figures, calibration and limitations. Final full
token-weighted validation loss: 4.009942; all 500 validation
documents and 693,668 input tokens were covered, including short tails.
