# Observation004: complete 32-step cosine response to half LR

At verified snapshot 20261104T025955Z, 14/30 primary branches are complete with
no failed branch. The 32-step cosine to 0.5x LR has completed across all three
cases. The broader suite is still running; this is a partial response record.

Source: Run222 scalar snapshot under
artifacts/snapshots/20261104T025955Z/payload, selected by retrieval.json.
Transfer archive SHA256:
4660482477658666236959385df383d357c709941fd71a21d30a80f837716b09.
All 138 captured files pass byte-size/SHA256 checks. Historical 960..1023 values
come from the pinned Run221 parent histories. Generating script: ../01_report.py;
source-aligned table ../aligned_values.csv; reductions ../comparison.csv and
../results.json. Frozen scientific inputs and admission rules are unchanged.

Figure: ../figures/cosine32_floor0p5.pdf; SHA256
176c2ff6c7238a95448bc1af1e0778cd1ac889905954310ac400aff3c213d80b.
Each column compares unchanged Adam with a cosine LR reduction starting from
the same warm state 1024, reaching 0.5x at update 32 and holding thereafter.
Columns identify MLP/CIFAR-10, ResNet-32/CIFAR-10 and Transformer/WikiText-2.
Rows show loss, raw rho, raw gradient norm, $\eta\widehat\kappa_t$ and applied
LR/base. Lines are solid and unsmoothed; Y ranges are independent and retain
both curves. The marked boundary separates the shared historical before period
from fresh Spark continuations for every arm.

Coverage: 64 historical before observations and 65 Spark boundaries per case,
plus 64 applied updates. All 2,910 plotted scalars match the source arrays;
across the three available response figures 8,730 scalars pass these checks.
Independent LR, norm/cosine identity, clip-factor and identical starting-boundary
checks pass. Terminal LR only closes the last applied interval. The new vector
PDF renders as one legible page with raw rho, gradient norms in the third row,
independent ranges and no clipped curves or label collisions. The previously
reviewed fixed-half and fixed-quarter PDFs remain byte-identical.

Mean/final unregularized training loss changes against the fresh unchanged
Spark continuation are -45.876%/-39.306% (MLP), +2.641%/+7.078% (ResNet), and
+1.582%/+3.449% (Transformer). Mean covers 1025..1088; final is 1088. Lower
training loss and smoother raw diagnostics are distinct observations. These
results do not establish a final ranking while other cosine, clipping and decay
conditions are unfinished; LR exposure also differs between schedules.

Run221's ResNet cross-GPU gradient gate failure remains retained. Run222 exact
within-Spark replay and identical treatment starts support the matched control
comparison; they do not establish historical bitwise replay. One selected seed
and warm state, a 64-update horizon and the hardware handover limit scope.
Full checkpoint retrieval and final 5-10 factual findings remain pending.
No manuscript change, cloud expense or push.
