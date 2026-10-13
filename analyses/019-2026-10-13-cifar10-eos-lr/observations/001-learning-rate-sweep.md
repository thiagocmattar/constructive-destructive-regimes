# Observation001: CIFAR-10 full-batch learning-rate sweep

Sources: frozen Runs059-066 `artifacts/attempt-01-spark/production/{metrics.jsonl, curvature.jsonl, manifest.json, summary.json, final.pt}`, the [cohort/config identity](../cohort.json), [retrieval inventories](../retrieval.json), [independent reductions](../verification.json) and [results](../results.json). The [report](../REPORT.md) contains the complete numerical tables.

Primary external references: [original Edge of Stability](https://arxiv.org/abs/2103.00065), [Adaptive Gradient Methods at the Edge of Stability](https://arxiv.org/abs/2207.14484), and their [released implementation](https://github.com/locuslab/edge-of-stability), pinned by Run058. We reproduce the small5k tanh squared-loss setup; the frozen default seed 0 initialization, LA selection and fixed 20k horizon are explicit choices, not a claim of pixel-identical paper figures.

Reduction and generator: [04_reduce_and_verify.py](../04_reduce_and_verify.py) audits all rows, eigenpairs, identities and final states; [05_figures.py](../05_figures.py) produces publication PDFs with source and plotted array hashes. Complete100-pair trailing means are right-aligned at the last update; the first is at update101 because the first cosine/kappa has no previous vector. Norm means use 10 complete samples and first appear at update10. Undefined first values are not replaced by zero.

Coverage: one matched seed 0, first 5,000 official training examples, all 10,000 test examples each update, all 20,000 updates in each of eight runs; ordinary spectra at states0..20,000 every 100, and Adam preconditioned spectra at 100..20,000 every 100. All 656,810 trainable coordinates contribute to the geometry; no clipping/decay/noise/schedule. FP32 optimizer state and FP64 vector prefixes are distinct. Initial weights/data hashes and runtime match throughout.

Uncertainty and caveats: no independent seed replication, no confidence bands, no optimized LR grid, finite horizon only. Spectra pass relative residual<= 0.005 but remain iterative estimates on100-step sampled states. Hessian largest algebraic eigenvalues are not the largest absolute eigenvalues. Curvature at the labeled state and the pairwise gradient proxy refer to different operational quantities. The Adam 38/eta line is a local frozen-preconditioner reference, not a global nonlinear threshold. No per-image or missing dense long-lag gradient decomposition is inferred.

## cifar10_eos_comparison.pdf

Caption: Matched GD/Adam learning-rate comparison: unsmoothed loss, full train/test accuracy,100-pair raw/update or mhat cosine, ordinary top 2 eigenvalues, optimizer-specific spectral reference, raw B/A, and 100-pair eta-kappa. No shared LR values or universal stability inference.

Source and reduction: full histories above; actual plot arrays and PDF SHA256 are recorded in [figure_manifest.json](../figure_manifest.json). Training loss is unsmoothed, norms/cumulative lengths are explicitly logarithmic, cosines/B/A use linear Y, and all lines share the complete fixed horizon. Color encodes learning rate within each method; styles identify vector or eigenvalue.

## cifar10_gd_geometry.pdf

Caption: GD full-horizon scalar geometry with unsmoothed loss,100-pair cosine means, log vector norms and cumulative A/B, linear B/A. Adam separates stored m, bias-corrected mhat, adaptive d and actual displacement; GD has no momentum.

Source and reduction: full histories above; actual plot arrays and PDF SHA256 are recorded in [figure_manifest.json](../figure_manifest.json). Training loss is unsmoothed, norms/cumulative lengths are explicitly logarithmic, cosines/B/A use linear Y, and all lines share the complete fixed horizon. Color encodes learning rate within each method; styles identify vector or eigenvalue.

## cifar10_adam_geometry.pdf

Caption: ADAM full-horizon scalar geometry with unsmoothed loss,100-pair cosine means, log vector norms and cumulative A/B, linear B/A. Adam separates stored m, bias-corrected mhat, adaptive d and actual displacement; GD has no momentum.

Source and reduction: full histories above; actual plot arrays and PDF SHA256 are recorded in [figure_manifest.json](../figure_manifest.json). Training loss is unsmoothed, norms/cumulative lengths are explicitly logarithmic, cosines/B/A use linear Y, and all lines share the complete fixed horizon. Color encodes learning rate within each method; styles identify vector or eigenvalue.

## loss_and_norms_by_learning_rate.pdf

Caption: Unsmoothed loss followed by one panel per run, common log norm Y limits, faint original raw-gradient norms,10-update raw trailing means, and fill to unsmoothed mhat on Adam or actual-update norm/LR on GD.

Source and reduction: full histories above; actual plot arrays and PDF SHA256 are recorded in [figure_manifest.json](../figure_manifest.json). Training loss is unsmoothed, norms/cumulative lengths are explicitly logarithmic, cosines/B/A use linear Y, and all lines share the complete fixed horizon. Color encodes learning rate within each method; styles identify vector or eigenvalue.

Numerical findings are descriptive and preserved in the report. Visual QA and PDF text/page checks are recorded separately in `visual_verification.json`; all figures must pass before closeout.
