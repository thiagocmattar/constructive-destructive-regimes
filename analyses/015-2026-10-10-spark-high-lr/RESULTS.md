# Higher learning rates on DGX Spark: recorded outcomes

Automatic terminal reduction generated 2026-10-11T10:52:16.930663+00:00. No replacement
training settings or lower-LR retries were introduced. Failed/missing conditions
remain explicitly labeled; prefix endpoints are not full-pass results.

| Run | LR | Status | Updates | Final validation loss | Last100 available full-update loss | Raw negative pairs | Momentum negative pairs |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 023 | 0.01 | completed_reference | 712 | 6.63649 | 6.30906 | 0.684951 | 0 |
| 053 | 0.02 | completed | 712 | 6.63438 | 6.50126 | 0.639944 | 0 |
| 054 | 0.04 | completed | 712 | 5.19195 | 5.47645 | 0.625879 | 0 |

Training loss in the figures is unsmoothed. The table's loss is token weighted
over the last up-to100 available full-sized updates, excluding partial update712.
Negative-pair fractions use all defined recorded adjacent cosines, including
the partial endpoint when present. They are descriptive fractions, not regimes
or convergence thresholds. Full-precision outcomes: [summary.csv](summary.csv).

[Compact loss and norms](figures/loss_and_norms_by_learning_rate.pdf).
[Full training dynamics](figures/training_dynamics.pdf).
[Sources and exact plotted values](comparison_data.json).
[Automated figure verification](figure_verification.json).
[Caption, reductions and limitations](observations/001-high-learning-rates.md).

Remote full-pass verification reconstructs all three streaming buffers from
the final recovery and checks actual stored momentum against optimizer exp_avg,
complete training/validation coverage and schedule. Local download receipts
separately verify byte sizes and SHA-256 before declaring retrieval complete.
Render/text/bounds checks are automatic; final human/model visual review remains
pending unless separately recorded. One seed1234, matched initialization/order,
one finite pass and LR-scaled weight decay provide no universal stability,
generalization or causal alignment claim.
