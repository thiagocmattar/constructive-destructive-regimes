# Observation010: primary grid complete; decoupled decay 0.1 response

All 30 primary branches completed without failure, with the last finishing at
2026-11-04 08:03:14 UTC. Verified snapshot 20261104T083012Z includes the complete
primary grid and ongoing optional cosine durations. Full checkpoint collection
and the final suite interpretation are still pending.

Source: Run222 artifacts/snapshots/20261104T083012Z/payload, selected by
retrieval.json. All 228 captured files pass byte-size/SHA256 verification.
Archive SHA256:
3cf0d610cf97da32e1d383c7e7179e32a716d9b03c731c4ba11b5262358140cf.
Pinned Run221 histories supply states 960..1023. Generating script:
../01_report.py. Reductions: ../comparison.csv and ../results.json;
aligned values: ../aligned_values.csv.

Main figure: ../figures/adamw_0p1.pdf; SHA256
3e397a3cc57595886bd1fc7ae9671501e87aac3e6dbb18db80b37a894ceaaa97.
Caption: unchanged Adam versus decoupled decay 0.1 after the same warm state
1024, with the original highest tested LR. Decay applies before the native Adam
task-gradient update to parameters with at least two dimensions; biases and
normalization vectors are excluded. Loss and gradient diagnostics remain
unregularized task quantities. The three columns identify MLP/CIFAR-10,
ResNet-32/CIFAR-10 and Transformer/WikiText-2. Rows show loss, raw rho, raw
gradient norm, $\eta\widehat\kappa_t$ and applied LR/base. Solid unsmoothed curves
retain complete trajectories with independent panel ranges. Intervention onset
marks the historical-to-Spark hardware handover for all arms.

Companion: ../figures/adamw_0p1_increments.pdf; SHA256
0a36a0bac5dae62373c18272ed2996020241a2a9a7346c5a14b81a0b302a895c.
Caption: post-intervention parameter norm, actual parameter increment norm and
decay increment norm against the same matched control. Three model columns,
solid raw observations and independent Y ranges are retained. No historical
parameter norms are fabricated; this companion is explicitly post-only.

Coverage: the main figure uses 64 historical observations, 65 Spark boundaries
and 64 applied updates per case, with 2,910 checked plotted scalars. Its companion
adds 1,158; 28,506 pass across the 11 available PDFs. Independent LR, clip-factor,
norm/cosine identity and identical-start checks pass. Both new PDFs were rendered
and visually inspected for legible legends/ticks, complete curves, independent
ranges and no collisions. Raw rho and third-row norms are present in the main
response. The nine previously reviewed PDFs remain byte-identical.

Mean/final unregularized training loss changes versus the fresh Spark unchanged
control are +1.902%/+14.659% (MLP), +6.354%/+12.872% (ResNet), and
+0.656%/+1.159% (Transformer). Mean covers states 1025..1088; final is 1088.
All three final parameter norms are smaller than control, while both mean and
final task loss are higher. The tenfold stronger decay does not improve these
short-horizon training-loss comparisons. Actual and decay increments are measured
separately and are distinct from raw gradient norms.

One selected seed and warm state, a 64-update horizon and the hardware handover
limit scope. No uncertainty over seeds, long-term convergence or held-out gain
is established. Run221's ResNet cross-GPU gradient gate failure remains preserved;
Run222 exact within-Spark replay does not establish historical bitwise replay.
Frozen science and optional admission rules are unchanged. Final retrieval and
findings remain pending. No manuscript change, cloud expense or push.
