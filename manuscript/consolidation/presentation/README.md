# Constructive and Destructive Training Regimes

[Project the slides](presentation.pdf) · [Read the timed speaker notes](speaker-notes.pdf)

The 23-slide talk is paced for 45 minutes of presentation, with discussion afterward. The notes pair each slide with its time window, a slide thumbnail, and prompts for explaining the figures. The last notes page provides technical answers about curvature, retention, and the singular-value analysis.

| Elapsed time | Slides | Question |
|---|---|---|
| 0–9 min | 1–5 | How can steps reverse while loss falls? |
| 9–19 min | 6–10 | What changes with learning rate and batch size? |
| 19–23 min | 11–12 | When does a raw-gradient warning anticipate an Adam loss increase? |
| 23–35 min | 13–18 | How much of each change survives in predictions and hidden representations? |
| 35–45 min | 19–23 | What happens when Adam's largest hidden-unit response changes are capped? |

## Edit and build

Edit [talk.tex](talk.tex) for both the projected content and the speaking notes. Each slide has five arguments in order: clock interval, title, source credit, slide content, and presenter notes. [presentation.tex](presentation.tex) defines the Beamer layout. [speaker-notes.tex](speaker-notes.tex) formats the notes and includes the slide thumbnails.

From this folder, compile in the following order. The vector plots are included, so LaTeX alone is sufficient to rebuild both PDFs.

~~~sh
latexmk -pdf -interaction=nonstopmode -halt-on-error presentation.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error speaker-notes.tex
~~~

## Evidence behind the lecture plots

[make_figures.py](make_figures.py) redraws selected paper measurements with larger labels. Run it with Python, NumPy and Matplotlib only when changing the figures. The one-weight loss and the two-input activation example are teaching examples. The reported network results use the manuscript measurements.

| Lecture material | Source and reduction |
|---|---|
| Slide 4, quadratic | Exact iterations \(w_t=(1-\eta)^t\) from \(w_0=1\); the loss is half the squared weight. |
| Slides 7–9, deterministic GD | [Retained Figure 1–3 arrays](../work/figure_revision_20261114/plotted_arrays.npz). MLP runs 059 and 122 for loss, run 122 for step curvature, sharpness, cosine and norm. Curvature series use the paper's centered 101-step display means; cosine and norm use trailing 50-observation means. The 12 lag values are individual observations from the fixed endpoint at update 2,256. |
| Slide 10, mini-batches | [Seed-0 batch comparison](../../../analyses/037-2026-11-02-sgd-batch-extension/observations/002-seed0-rate-01.md). The Transformer panel uses the means in seed0_cells.csv, learning rate 0.1, 200-update forks, eight sampling repetitions and a full-GD reference for each starting state. |
| Slides 11–12, Adam | The ResNet overview and raw-gradient cosine come from the [retained Figure 5 series](../work/figure_revision_20261114/plotted_arrays.npz). The warning slide uses the [current diagnostic table](../draft/tables/diagnostic-0p1pct.tex). |
| Slides 14–15, functional motion | [Matched ResNet trace](../work/resnet_peak_revision_20261114/figure_manifest.json). The cosine and normalized RMS arrays are shown intact across all six sites, at completed updates 390–450. The selected native continuation peaks at 415. The separately recorded overview on slide 11 has a local peak at 422. The retention table uses the full changes on the paper's fixed reference inputs. |
| Slides 17–18, SVD | [ResNet-window-matched decomposition](../work/resnet_temporal_revision_20261114/plotted_arrays.npz) for all three models. The plot shows the signed leading output coefficient divided by the square root of the number of coordinates, one fixed direction per model and window. |
| Slides 19–23, intervention | [Matched outcomes](../evidence/05-adam/c-intervention/data/matched-results.json), [paper table](../draft/tables/matched-main-outcomes.tex), and [intervention section](../draft/sections/06-interventions.tex). The lecture's trade-off plot selects unchanged Adam, q25 and the tenfold learning-rate reduction for ResNet. Each axis is a ratio to unchanged Adam from full-precision results. |

The network measurements describe the tested trajectories and selected windows. The batch repetitions vary sampling at one initialization. They do not provide independent training seeds for the full-batch comparisons. The closing warning-triggered intervention is a proposed experiment.
