# Observation005: complete 32-step cosine response to quarter LR

Verified snapshot 20261104T035947Z contains 17/30 completed primary branches
and no failures. The 32-step cosine to 0.25x LR is complete for all three cases;
other treatments remain unfinished, so no final method ranking is claimed.

Source: Run222 artifacts/snapshots/20261104T035947Z/payload, selected by
retrieval.json. All 153 captured files pass byte-size/SHA256 verification.
Archive SHA256:
b9ceba05e2c1892a4f34fdab9d8c80443d37d78220af2d38563f79ce622a2dbe.
Historical values at states 960..1023 come from the pinned Run221 parent histories.
Generating script: ../01_report.py. Numeric reductions: ../comparison.csv and
../results.json; source-aligned values: ../aligned_values.csv.

Figure: ../figures/cosine32_floor0p25.pdf; SHA256
ebd45948178a8f9b896a4f26a2d074ac5362d05d12ea566c54b2e073c5c01b65.
Caption: unchanged Adam versus cosine LR reduction from the same warm state
1024, reaching 0.25x after 32 updates and holding thereafter. Columns identify
MLP/CIFAR-10, ResNet-32/CIFAR-10 and Transformer/WikiText-2. Rows show loss,
raw rho, raw gradient norm, $\eta\widehat\kappa_t$ and applied LR/base.
Solid unsmoothed curves use independent panel ranges and retain both complete
trajectories. Relative update zero marks the historical-to-Spark hardware boundary
for all arms, as well as intervention onset.

Coverage: 64 historical before observations, 65 Spark boundaries and 64 applied
updates per case. All 2,910 scalars in this figure match source arrays; 11,640
scalars pass across the four available figures. Independent LR, norm/cosine
identity, clip-factor and identical-start checks pass. The new PDF was rendered
and visually inspected: legible legends/ticks, gradient norms in the third row,
raw rho, independent ranges, no clipped curves or label collisions. Earlier
three reviewed PDFs remain byte-identical. No smoothing or extra terminal step
is introduced.

Mean/final unregularized training loss changes against the fresh Spark unchanged
control are -45.722%/-37.338% (MLP), +4.031%/+10.911% (ResNet), and
+2.687%/+6.129% (Transformer). Mean covers states 1025..1088; final is 1088.
These are short responses conditional on one selected seed and warm state.
Smoother raw diagnostics do not imply lower attained loss; schedule LR exposure
also differs. No uncertainty over seeds, longer-horizon conclusion or held-out
benefit is established.

Run221's ResNet cross-GPU gradient gate failure is preserved. Run222 passes exact
within-Spark replay and treatment-start matching, without claiming historical
bitwise replay. Frozen science is unchanged. Full checkpoint retrieval and final
5-10 factual findings remain pending. No manuscript change, cloud expense or push.
