# Observation009: complete decoupled decay 0.01 response

Verified snapshot 20261104T072951Z contains 29/30 completed primary branches
and no failures. Decoupled decay 0.01 is complete across all three cases;
the final primary condition and optional schedules remain unfinished.

Source: Run222 artifacts/snapshots/20261104T072951Z/payload, selected by
retrieval.json. All 213 captured files pass byte-size/SHA256 verification.
Archive SHA256:
8f378934b380293de0b62eb55395c0b5ec46a3b8522cc7e3a4e440bfc1a7e879.
Historical states 960..1023 use pinned Run221 parent histories. Generating script:
../01_report.py. Reductions: ../comparison.csv and ../results.json;
aligned values: ../aligned_values.csv.

Main figure: ../figures/adamw_0p01.pdf; SHA256
eac4dd06e5f235f2d286a94971d39b76ad2b7795b963551758ba1a553767318d.
Caption: unchanged Adam versus decoupled decay 0.01 after the same warm state
1024, at the original highest tested LR. Decay applies only to parameters with
at least two dimensions, before the native task-gradient Adam update; biases
and normalization vectors are excluded. The plotted loss is unregularized task
loss and diagnostics use raw task gradients. Columns identify MLP/CIFAR-10,
ResNet-32/CIFAR-10 and Transformer/WikiText-2. Rows show loss, raw rho, raw
gradient norm, $\eta\widehat\kappa_t$ and applied LR/base. Solid unsmoothed curves
retain both trajectories within independent Y ranges. Intervention onset marks
the historical-to-Spark hardware boundary for all arms.

Companion: ../figures/adamw_0p01_increments.pdf; SHA256
7b5d7f215b2797e3d610852517c30d68d6acd035b56e8a5ce49ea4e543941922.
Caption: post-intervention parameter norm, actual parameter increment norm and
decay increment norm against the same matched control, in three model columns.
The companion is post-only because historical parameter norms were not retained.
It uses solid raw observations and independent panel ranges.

Coverage: the main figure uses 64 historical observations, 65 Spark boundaries
and 64 applied updates per case, with 2,910 source-aligned plotted scalars. The
companion adds 1,158 checked scalars; 24,438 pass across nine available PDFs.
Independent LR, clip-factor, norm/cosine identity and identical-start checks pass.
Both new vector PDFs were rendered and visually inspected for legible labels,
complete curves, independent ranges and no collisions. Raw rho and third-row
norms are present in the main response. The seven earlier PDFs remain identical.

Mean/final unregularized training loss changes versus the fresh Spark unchanged
control are +0.182%/+1.869% (MLP), +0.615%/+1.228% (ResNet), and
+0.025%/+0.015% (Transformer). Mean covers states 1025..1088; final is 1088.
All three final parameter norms are smaller than their controls. Thus reduced
parameter norm here does not imply lower attained task loss. The decay increment
and actual increment are recorded separately; neither is the raw gradient norm.

One selected seed and warm state, a 64-update horizon and the hardware handover
limit scope. No uncertainty over seeds, long-term convergence or held-out gain
is established. Run221's ResNet cross-GPU gradient gate failure remains retained;
exact within-Spark replay does not establish historical bitwise replay. Frozen
science is unchanged. Full checkpoint retrieval and final findings remain pending.
No manuscript change, cloud expense or push.
