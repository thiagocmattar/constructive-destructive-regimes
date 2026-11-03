# First-rise projected gradient change versus peak increase

Figure: [adam_first_rise_eta_peak_scatter.pdf](../figures/adam_first_rise_eta_peak_scatter.pdf).
Generator: [09_plot_first_rise_peak_scatter.py](../09_plot_first_rise_peak_scatter.py).
Exact event membership, coordinates, source hashes and checks:
[manifest](../first_rise_peak_scatter_manifest.json).

Comparison: the [pre-rise scatter](005-prerise-eta-peak-scatter.md) samples state a;
this companion samples state a+1, immediately after the first strict loss increase
above the fixed pre-rise baseline. This is not the first crossing of +10%.
The loss-defined event boundaries, >10% amplitude filter, observed-recovery
criterion, LR colors and logarithmic loss-ratio y-axis are unchanged.

Caption: Raw-gradient projected change immediately after the first loss increase
versus event peak magnitude under full-batch Adam. Each point is one of the
129 recovered >10% events; columns show CIFAR-10/MLP, CIFAR-10/ResNet-32 and
WikiText-2/Transformer. The x-coordinate at state a+1 (paper index a+2) is
1 - <g_(a+2),g_(a+1)> / ||g_(a+1)||^2, comparing gradients at states a and a+1.
It describes the response to the update that first increases loss. The y-coordinate
is peak loss / pre-rise loss = 1 + peak percentage increase / 100, on a shared
logarithmic axis. One is no increase and two is +100%. The shared linear x-axis
is 0..3.6 to include all values. No smoothing, averaging, fitting or jitter.
Typography and the 7.16 x 2.55 inch layout follow Analysis024.

Coverage: all 129 anatomy events have a defined first-rise diagnostic; ascending-LR
counts are MLP [8,47,39], ResNet [0,10,7], Transformer [0,6,12]. The pre-rise view
omits two events beginning at initialization and has 127 points. The additional
two points must be accounted for in any numerical comparison between views.
The original pre-rise figure and its inputs remain unchanged.

Observation and limits: first-rise values for the selected Transformer events
are all above two. This describes these selected events, not an established
prediction rule. In 13/129 events the first elevated state already is the loss
maximum, leaving no lead time to the peak. For the remaining events the
measurement precedes the peak, but the rise has already started. Prediction of
subsequent amplification would require comparison with other first-loss-increase
states, including small excursions; this plot conditions on a future >10% peak
and recovery. Loss at the diagnostic time is also part of the eventual peak
outcome. No claim of improved predictive performance is made. Events within
single-initialization trajectories can be dependent. Raw projected gradient
change along an Adam trajectory is not a direct directional-curvature estimate.
No new training, causal intervention, cloud execution or manuscript edit.

Verification: anatomy and state-table SHA-256 hashes checked against their source
manifests. Join is one-to-one by event and onset state; onset is exactly a+1,
its loss matches the event's first elevated loss, and the projected-change
identity holds using onset cosine/norm and pre-rise norm. Percentage/ratio
identity and exact artist coordinates checked. All points lie inside the axes;
PDF is one vector page with embedded fonts and text inside the page. Poppler
render visually reviewed: all panels, points, labels and legends readable,
without clipping or overlaps.
