# Observation011: complete optional eight-step cosine response

Verified snapshot 20261104T090007Z contains all 30 completed primary branches,
all three eight-step cosine branches and the first 64-step cosine branch, without
failure. Both optional durations were admitted by the frozen worker time rule;
the 64-step duration remains unfinished across the three cases.

Source: Run222 artifacts/snapshots/20261104T090007Z/payload, selected by
retrieval.json. All 238 captured files pass byte-size/SHA256 verification.
Archive SHA256:
a528ba5569b58dcdbb33a7fc037080cf23775ab6b0edd01ac2e56a000560c599.
Pinned Run221 histories supply states 960..1023. Generating script:
../01_report.py. Reductions: ../comparison.csv and ../results.json;
aligned values: ../aligned_values.csv.

Figure: ../figures/cosine8_floor0p25.pdf; SHA256
5a2416b07499a056f2e82e6facb91d71607f2bffdcedd21595137b92ae752a5f.
Caption: unchanged Adam versus an eight-step cosine LR reduction from the same
warm state 1024, reaching 0.25x at update eight and holding thereafter. Columns
identify MLP/CIFAR-10, ResNet-32/CIFAR-10 and Transformer/WikiText-2. Rows show
loss, raw rho, raw gradient norm, $\eta\widehat\kappa_t$ and applied LR/base.
Solid unsmoothed observations retain both complete trajectories within independent
panel ranges. Relative update zero marks intervention onset and the historical-to-
Spark hardware handover for all arms.

Coverage: 64 historical observations, 65 Spark boundaries and 64 applied updates
per case. All 2,910 plotted scalars match source arrays; 31,416 pass across the
12 available PDFs. Independent LR, clip-factor, norm/cosine identity and identical-
start checks pass. The new vector PDF was rendered and visually inspected for
legible labels, raw rho, norms in row three, independent ranges and no clipped
curves or collisions. All 11 earlier reviewed PDFs remain byte-identical.

Mean/final unregularized training loss changes versus the fresh Spark unchanged
control are -46.666%/-36.025% (MLP), +6.362%/+13.944% (ResNet), and
+4.862%/+8.338% (Transformer). Mean covers states 1025..1088; final is 1088.
The mean applied LR is 0.302734375x base, compared with 0.443359375x for the
32-step cosine at the same floor. Duration therefore changes both transition
shape and LR exposure; this comparison alone does not isolate a smoothing effect.

The one selected seed and warm state, 64-update horizon and hardware handover
limit scope. No uncertainty over seeds, long-term convergence or held-out gain
is established. Run221's ResNet cross-GPU gradient gate failure remains retained;
exact within-Spark replay does not establish historical bitwise replay. Frozen
science and admission rules are unchanged. Final retrieval and findings remain
pending. No manuscript change, cloud expense or push.
