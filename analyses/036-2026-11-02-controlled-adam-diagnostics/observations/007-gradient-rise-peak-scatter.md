# Rise gradient norm ratio versus peak loss ratio

Figure: [adam_gradient_rise_peak_scatter.pdf](../figures/adam_gradient_rise_peak_scatter.pdf).
Generator: [10_plot_gradient_rise_peak_scatter.py](../10_plot_gradient_rise_peak_scatter.py).
Source hashes, exact event membership, coordinates and checks:
[manifest](../gradient_rise_peak_scatter_manifest.json).

Source and coverage: the current [anatomy sample](../spike_anatomy/README.md),
129 events with peak increase strictly greater than 10% and recovery observed
by update 1024. All have defined endpoint gradient norm ratios. Counts in
ascending LR order are MLP [8,47,39], ResNet-32 [0,10,7], Transformer [0,6,12].

Caption: Raw-gradient norm growth from the pre-rise state to the loss peak versus
peak loss relative to the pre-rise baseline, under controlled full-batch Adam.
Each point represents one event; columns show CIFAR-10/MLP, CIFAR-10/ResNet-32
and WikiText-2/Transformer. Colors identify the constant LRs within each model.
The x-coordinate is ||g_(p+1)|| / ||g_(a+1)||, where a is the last state before
the first loss increase and p is the event's loss-peak state. This is a ratio of
endpoint norms, not a maximum gradient norm or an average of stepwise ratios.
The y-coordinate is loss_peak / loss_before = 1 + peak_change_percent / 100.
The shared x-axis is linear (0..32); the shared y-axis is logarithmic (1..11),
where 1 means no loss increase, 2 means +100%, and 10 means +900%.
No smoothing, averaging, jitter or fit. The 7.16 x 2.55 inch layout, typography
and LR palette follow the existing Analysis024-style figures.

Observation: larger loss peaks occur with larger endpoint gradient norm ratios
in several groups, with visible dispersion and model/LR differences. The scatter
is descriptive; no monotonic relation, fitted association or causal claim is imposed.

Limits and uncertainty: both axes use the loss-peak state and the fixed pre-rise
baseline, so this is retrospective event anatomy, not a pre-onset predictor.
The gradient norm maximum need not coincide with the loss peak. The sample is
conditional on >10% amplitude and observed recovery; it omits small excursions
and non-event states. Events from the same single-initialization trajectory can
be dependent. No confidence bands or cross-seed inference. Existing Run215
histories only; no new training, cloud use, interventions or manuscript edits.

Verification: source hash matches the anatomy manifest; event IDs are unique,
all 129 events pass the selection, and both norm/loss ratios agree with their
endpoint values. Scatter artist coordinates equal source-derived coordinates
exactly. Every point is inside the common axes. PDF contains one vector page,
embedded fonts and text inside the page. Poppler rendering visually reviewed:
all panels, points, legends and axis labels readable without overlap or clipping.
