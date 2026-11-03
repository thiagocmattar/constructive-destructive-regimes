# Pre-rise projected gradient change versus peak increase

Figure: [adam_prerise_eta_peak_scatter.pdf](../figures/adam_prerise_eta_peak_scatter.pdf).
Generator: [08_plot_prerise_peak_scatter.py](../08_plot_prerise_peak_scatter.py).
Source membership, coordinates, hashes and checks: [manifest](../prerise_peak_scatter_manifest.json).

Source: the current [spike-anatomy sample](../spike_anatomy/README.md), restricted
to peak increase strictly greater than 10% and recovery observed by update 1024.
There are 129 events; two initialization events have undefined pre-rise projected
change and are omitted from this plot, leaving 127 points. Plotted counts by
ascending LR are MLP [8,46,38], ResNet-32 [0,10,7], Transformer [0,6,12].
The manifest records both omitted event IDs. Empty LR groups retain legend entries.

Caption: Pre-rise raw-gradient projected change versus subsequent peak training-loss
increase under full-batch Adam. Columns show CIFAR-10/MLP, CIFAR-10/ResNet-32
and WikiText-2/Transformer. Each point is one recovered >10% event; colors identify
the three constant learning rates within each model, as in the overview and density
figures. The x-coordinate is one unsmoothed diagnostic at the last pre-rise parameter
state a (paper index a+1), using gradients at states a-1 and a. The y-coordinate is
100 times (peak loss / fixed pre-rise loss minus one), expressed in percent.
Both axes are linear and shared. No fit, averaging, smoothing, jitter or uncertainty
band is applied. Typography and 7.16 x 2.55 inch layout follow Analysis024.

Observation: the plotted points span overlapping pre-rise diagnostic values across
spike magnitudes. The largest ResNet increase is visible without clipping; the
common y-axis consequently compresses smaller increases in the other panels.
This is an exploratory view, without a correlation or predictive-performance claim.

Coverage and limits: only three models, three constant LRs per model and one frozen
initialization per model, from the existing 1024-update Run215 trajectories. Events
within a trajectory can be dependent. Selection requires a future >10% increase
and observed recovery; non-event states are absent. The scatter therefore describes
conditional event anatomy, rather than establishing prospective prediction or a
causal mechanism. Under Adam, the raw-gradient diagnostic is normalized projected
gradient change along its trajectory, not a direct directional-curvature estimate.
No new training, cloud resources, interventions or manuscript changes.

Verification: source SHA-256 matches the anatomy manifest; event selection,
undefined initialization cases and peak percentage identity checked. Artist
coordinates equal source values exactly; every point lies within the common axes.
PDF contains one vector page with embedded fonts and text inside the page.
Poppler rendering visually reviewed: all panels, points, legends and labels are
readable, with no clipping or overlaps.
