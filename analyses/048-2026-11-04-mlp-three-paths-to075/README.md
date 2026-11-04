# Analysis048: three paths to 0.75x LR

Completed 2026-11-04. User-requested figure using existing Runs223/224/225/226; no training, new run, seed or hypothesis is introduced. [Figure](figures/mlp_three_paths_to075.pdf), [observation](observations/001-three-paths.md), [source-verified values](plotted_values.csv), [results](results.json).

Three columns, each comparing one intervention only with the same unchanged baseline:

1. Run226: cosine32 from base0.001 at512 to0.00075 at544, then held.
2. Run224 cosine branch: cosine32 to0.0005 at544, hold to650, abrupt increase to0.00075, then held.
3. Run225: cosine32 to0.0005 at544, hold to650, cosine70 increase to0.00075 at720, then held.

Common model, full5000-image CIFAR-10 objective, physical1000 partitions, seed0, exact warm512, Adam, precision and Spark. Runs224/225 also have exact common state650. The fresh Run223 same-Spark unchanged control is reused without rerunning it; this is not a claim of historical cross-GPU bitwise replay.

Figure rows: log training loss, raw rho, raw gradient norm, normalized projected gradient change, applied/base LR. Display448..800; all curves solid and unsmoothed with independent panel scales. Gray is unchanged; blue/magenta/green are the three interventions. Gray shading marks shared pre-fork context; black vertical512 marks the intervention fork. Colored650 markers identify the rises; the final column additionally shades650..720 and marks720. All data are on Spark, so there is no hardware boundary in the plotted interval. Endpoint800 is observed and unapplied; the final LR segment closes the last applied interval.

Run01_report.py to reproduce. It rechecks119 source artifact files and10584 plotted values, writes results.json and plotted_values.csv, saves one vector PDF, and renders a PNG under ignored scratch/. The PDF contains no raster image objects and passed visual inspection. No executed input or manuscript file changed. Existing H031/H032/H033 findings apply; next run remains227, next analysis049, next hypothesisH034.
