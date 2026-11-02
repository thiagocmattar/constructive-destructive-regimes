# Descriptive statistics by model and learning rate

Figure: [3 x 3 panel](../figures/adam_spike_descriptive_statistics.pdf).
Values: [summary CSV](../spike_descriptive_statistics.csv).
Generator: [04_plot_spike_statistics.py](../04_plot_spike_statistics.py).
Provenance and checks: [manifest](../spike_descriptive_statistics_manifest.json).

Source: the verified `spike_dataset/loss_spikes.csv` and its extraction manifest.
The existing event definition and every event row are unchanged. No new training.

Caption: Descriptive loss-excursion statistics for the controlled full-batch Adam
trajectories. Rows are CIFAR-10/MLP, CIFAR-10/ResNet-32 and
WikiText-2/Transformer. Columns are the lower, middle and higher tested learning
rates within each model; exact rates are printed in every panel. Peak rise is
100 times (peak loss / fixed pre-rise loss minus one). Rise, recovery and total
durations count optimizer updates. Each panel reports mean, sample standard
deviation (denominator N-1), minimum, P25, P75 and maximum. Quartiles use linear
interpolation. Counts of recovered and censored events are shown per panel.

Coverage: all 806 recovered excursions across the nine Run215 arms. The six
right-censored excursions are excluded from all four metrics, keeping a shared
cohort and avoiding provisional peaks or unobserved durations. Statistics are
therefore conditional on recovery within the observed 1,024-update horizon.
All candidate magnitudes are retained without a minimum amplitude or startup
exclusion. Inputs are unsmoothed. Display rounding does not change CSV values.

Uncertainty and limits: standard deviations describe variation among observed
events, not uncertainty across independent initializations or standard errors.
Each arm has one initialization, and events from an arm may be dependent.
No causal or significance claim is made. Numeric summaries are independently
checked using Python's statistics module; font, cell and PDF bounds are checked
before a Poppler render is visually reviewed. Times New Roman/STIX typography
and the 7.16-inch page width follow the Analysis024 figure convention.
