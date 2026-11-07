# Analysis060: warm Adam-to-GD switch

Runs233--235 / H040. Compare GD learning rates0.02,0.03,0.04 against retained
Adam at0.001 from the identical Run223 MLP state512. Full5000-image CIFAR-10
MSE, seed0, physical1000 batches, FP32 vectors and FP64 diagnostic reductions.
Each GD branch independently reloads the untreated parent. Comparisons among
GD rates isolate LR within this design; comparisons to Adam change optimizer
and LR jointly. Exact fork/control replay and GD resume checks pass.

- LR0.02: completed through800; mean/tail loss+2.30%/+55.74% versus Adam;
  no loss increases or >10% excursions. Retained Run233 evidence unchanged.
- LR0.03: completed through800; mean/tail loss+111.42%/+191.15%; maximum
  loss0.302456 and one unrecovered >10% excursion.
- LR0.04: completed through800; mean/tail loss+394.45%/+232.35%; maximum
  loss1.61503 and one long, unrecovered >10% excursion.
- LR0.08:35 updates to547, where full-objective loss overflows to infinity.
  Last finite loss6.0501e34 at546. No endpoint or full-window averages reported.

The regenerated [five-row, three-column PDF](figures/01-adam-to-gd-cifar10-mlp.pdf)
shows loss, projected gradient change, adjacent raw-gradient cosine, raw-gradient
norm and applied LR for0.02/0.03/0.04. All10,584 plotted scalars match raw
evidence, including all3528 original scalars. Loss/norm use log scales and each
row shares y ranges across columns. The0.08 divergence is retained in the
historical results and archived figure, with no post-failure extrapolation.

- [Current findings, caption and verification](observations/003-intermediate-gd-rate.md)
- [Prior0.04/0.08 findings](observations/002-gd-learning-rates.md)
- [Original single-rate observation](observations/001-adam-to-gd.md)
- [Original figure snapshot](archive/2026-11-07-single-rate/README.md)
- [Results and source hashes](results.json)
- [Run234 verification](../../runs/234-2026-11-07-mlp-gd-learning-rates/verification_final.json)
- [Run235 verification](../../runs/235-2026-11-07-mlp-gd-lr0p03/verification_final.json)

Reproduce with `python analyses/060-2026-11-07-mlp-adam-to-gd/01_report.py`.
Run234 worker4.16s and Run235 worker3.84s;611 new scientific updates plus16
qualification updates, with three extra recorded-update replays to audit the
overflow. All107 new artifacts pass size/SHA checks. Workers exited; about408MB
retained on owned Spark including archives; zero cloud cost. One seed/window,
no uncertainty bands or general convergence/generalization claim. LR0.03 was
chosen after viewing the earlier results. Manuscript unchanged.
