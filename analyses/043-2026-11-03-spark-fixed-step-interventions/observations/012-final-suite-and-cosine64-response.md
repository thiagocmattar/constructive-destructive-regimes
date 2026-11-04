# Observation012: terminal suite verification and final cosine duration

Final snapshot 20261104T100016Z contains all 36 completed branches with no failure
or skipped optional condition. Sources: Run222 retrieval.json and the selected
artifacts/snapshots/20261104T100016Z/payload. All 347 files pass byte-size/SHA256
checks and cover all 337 worker-inventory files. Final archive SHA256:
b24f34a121e658c5c8da03ca0d051819a1c6ee0c8b97322bed2043b15d22c0d1.
Generating script: ../01_report.py. Independent terminal verifier:
../02_verify_final.py; results: ../verification.json. Numeric reductions:
../comparison.csv and ../results.json; source-aligned values: ../aligned_values.csv.
The eight consolidated findings and limits are in ../FINDINGS.md.

Final figure: ../figures/cosine64_floor0p25.pdf; SHA256
8394e76285105460eec8a7c039590ab22a39b1bc0afa25edd24b381cb6fd0c19.
Caption: unchanged Adam versus cosine LR decay with duration 64 and floor 0.25
from exact warm state 1024, for MLP/CIFAR-10, ResNet-32/CIFAR-10 and
Transformer/WikiText-2. Applied updates use k=0..63; the exact floor at k=64
would belong to the next, unexecuted update. The plotted terminal LR closes the
last applied interval. No extra update or floor-hold period is represented.
Rows show loss, raw rho, raw gradient norm, $\eta\widehat\kappa_t$ and actual
LR/base, with solid unsmoothed curves and independent panel ranges retaining
both trajectories. Relative zero marks intervention and the hardware handover
to Spark for all arms. Shared historical states 960..1023 are pinned context.

Coverage: 64 historical observations, 65 Spark boundaries and 64 applied updates
per case; 2,910 source-aligned scalars in the final figure. The full 13-PDF set
checks 34,326 plotted scalars. The new vector PDF was rendered and visually
inspected for legible labels, raw rho, norms in row three, independent ranges
and no clipped curves/collisions. The earlier 12 PDFs remain byte-identical.
Each earlier PDF has its source/caption/coverage record in observations002..011;
decay companions are explicitly post-only, without invented historical norms.

Mean/final training-loss changes against fresh unchanged Spark controls are
-44.907%/-39.218% (MLP), +2.022%/+7.046% (ResNet), and +1.133%/+3.477%
(Transformer). Mean covers states 1025..1088; final is 1088. Mean applied LR is
0.630859375x base, so duration comparisons change LR exposure as well as shape.
These are single-seed/state conditional responses, without across-seed uncertainty,
long-horizon convergence or a held-out benefit claim.

The verifier independently validates all frozen inputs, stored exact within-Spark
replay, every checkpoint clock/finite state and four inactive clipping endpoints
against their controls. Both optional admissions satisfy the frozen time rule.
Run221's ResNet relative-gradient gate failure 0.0087187473 remains retained;
no historical bitwise replay or Adam curvature/stability threshold is claimed.
The task worker has exited after finishing at 06:52 BRT; unrelated PID458832 stays
suspended. All raw checkpoints remain retained. No manuscript edit, cloud expense
or push. Eight final findings are consolidated against the fresh Spark controls.
