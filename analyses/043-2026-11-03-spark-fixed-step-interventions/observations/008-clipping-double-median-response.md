# Observation008: inactive double-median clipping across three cases

Verified snapshot 20261104T062940Z contains 25/30 completed primary branches
and no failures. Global clipping at twice the historical median is complete for
all three cases. Decoupled decay tests remain unfinished.

Source: Run222 artifacts/snapshots/20261104T062940Z/payload, selected by
retrieval.json. All 193 captured files pass byte-size/SHA256 checks.
Archive SHA256:
32cb0fbf4e8d8e336176ec860493adb4d589e3ecf96b299ba758ad87e3d93b4b.
Historical states 960..1023 and threshold medians use pinned Run221 histories.
Generating script: ../01_report.py. Numeric reductions: ../comparison.csv and
../results.json; aligned source values: ../aligned_values.csv.

Figure: ../figures/clip_median_x2.pdf; SHA256
04e0092220d236770c187b9252ce0d16fcadcb21fcc158329d750433a7b3035f.
Caption: unchanged Adam versus global raw-gradient norm clipping after warm
state 1024 at twice the fixed median from historical states 960..1023. LR stays
at its original highest tested value. Plotted rho and gradient norms use raw
task gradients. Columns identify MLP/CIFAR-10, ResNet-32/CIFAR-10 and
Transformer/WikiText-2. Rows show loss, raw rho, raw gradient norm,
$\eta\widehat\kappa_t$ and applied LR/base. Solid unsmoothed curves retain both
complete trajectories with independent Y ranges and a marked historical-to-Spark
hardware boundary at intervention onset.

Coverage: 64 historical observations, 65 Spark boundaries and 64 applied updates
per case. All 2,910 plotted scalars match source arrays; 20,370 pass across the
seven available figures. Independent LR, norm/cosine identity, clip-factor and
identical-start checks pass. The new PDF was rendered and visually inspected:
legible labels, raw rho, gradient norms in row three, independent ranges, no
clipped curves or collisions. The six earlier PDFs remain byte-identical.

Clipping activates on 0/64 updates in every case. All 65 boundary values for
loss, rho, raw gradient norm, projected change, parameter norm and LR match the
fresh unchanged control exactly; all 64 actual increment norms also match.
Mean/final training loss changes are therefore 0%/0% for all three cases.
Overlapping curves in this figure reflect the verified inactive condition.
This threshold did not test active clipping on this selected response window;
it provides no evidence that clipping in general is ineffective.

One selected seed and warm state, a 64-update horizon and unequal intervention
activity limit scope. No long-term convergence or held-out benefit is established.
Run221's ResNet cross-GPU gradient gate failure remains retained; exact
within-Spark replay does not establish historical bitwise replay. Frozen science
is unchanged. Full checkpoint retrieval and final findings remain pending.
No manuscript change, cloud expense or push.
