# Observation 003: raw alignment, momentum norm separation and norm fluctuations

User intuition: when raw-gradient adjacent cosine becomes negative, raw and momentum norms separate and the raw norm becomes noisy. This observation examines that intuition descriptively along the two completed matched trajectories.

## Figure and coverage

[The 2x2 figure](../figures/loss_alignment_norms.pdf) overlays both LRs in (a) training loss and (b) raw adjacent cosine; (c) and (d) show raw and momentum individual norms at LR 0.001 and 0.0001. Dots in the cosine panel and vertical lines in the norm panels mark the first negative unsmoothed raw pair, at updates 54 and 385. Gray regions mark all subsequent time, including later positive pairs. The shading no longer uses a smoothed sign; the trailing median remains only a curve display aid. Bottom panels share a linear y-range; raw/momentum norm curves are unsmoothed. Faint loss/cosine traces show individual updates; bold loss is a trailing token-weighted mean and bold cosine a trailing median, both with window 25.

The main display contains full-sized updates 1-711; all 712 accepted updates and the partial final batch are retained in [data and source/generator hashes](../loss_alignment_norms_data.json). Source event hashes match the previous verified analysis. Both runs cover the complete identical token stream/order and initial tensors; LR also scales AdamW decay. No new training or replay is performed.

## Operational meaning and findings

Here 'separation' means the ratio of stored-momentum norm to raw-gradient norm, ||m||/||g||. 'Jaggedness' is measured as median absolute consecutive changes of norms and log norms. These are scalar trajectory fluctuations, not estimates of stochastic gradient-noise variance at a fixed parameter state. The first event is selected by rho_t < 0 on the original unsmoothed record, without a persistence requirement. The following table examines the preceding, event and next update, avoiding a smoothed crossing as a proxy for onset.

| LR | Update | Position relative to first negative pair | Raw adjacent cosine | Raw norm | Momentum norm | m/g norm ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0.001 | 53 | preceding update | 0.084070 | 0.600460 | 0.376319 | 0.626718 |
| 0.001 | 54 | first negative pair | -0.518844 | 0.486766 | 0.358275 | 0.736031 |
| 0.001 | 55 | next update | 0.971299 | 0.400765 | 0.346003 | 0.863356 |
| 0.0001 | 384 | preceding update | 0.048503 | 0.669628 | 0.408211 | 0.609609 |
| 0.0001 | 385 | first negative pair | -0.194492 | 0.648419 | 0.406913 | 0.627547 |
| 0.0001 | 386 | next update | 0.267766 | 0.466749 | 0.404517 | 0.866669 |

At the preceding updates 53 and 384, raw adjacent cosine is still positive (0.084 and 0.049), yet momentum norm is already only 62.7% and 61.0% of raw norm. At the first negative pair itself, m/g increases to 73.6% and 62.8%, rather than suddenly dropping. Thus the first negative pair does not identify the exact start of norm separation. Loss of directional coherence is already visible before the sign crossing. A strictly defined onset of 'disconnection' would require an additional declared criterion, not the first cosine sign alone.

First negative individual raw pairs occur at updates 54 and 385; trailing 25-update median sign crossings occur much later, at 112 and 496. The first pair is a direct event reference and the shaded region marks time after it, not a permanently negative regime.

## Local windows around the first event

These three consecutive 25-update windows describe timing without claiming a causal change point. The raw-norm fluctuation at the window's first update compares with the immediately preceding update. Windows are temporally dependent, differ in training age and gradient scale, and have no independent-seed uncertainty.

| LR | Window | Updates | Negative raw pairs | Median m/g norm ratio | Median absolute raw norm change | Median absolute momentum norm change | Median absolute raw log-norm change | Median absolute momentum log-norm change |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 25 updates before first negative | 29-53 | 0 | 1.164509 | 0.018445 | 0.013163 | 0.039873 | 0.024518 |
| 0.001 | 25 updates from first negative | 54-78 | 11 | 0.709131 | 0.106955 | 0.007967 | 0.256902 | 0.033944 |
| 0.001 | following 25 updates | 79-103 | 7 | 0.615318 | 0.054248 | 0.002353 | 0.179065 | 0.015617 |
| 0.0001 | 25 updates before first negative | 360-384 | 0 | 0.848390 | 0.030476 | 0.001062 | 0.064206 | 0.002539 |
| 0.0001 | 25 updates from first negative | 385-409 | 3 | 0.716590 | 0.076676 | 0.001054 | 0.146593 | 0.002634 |
| 0.0001 | following 25 updates | 410-434 | 5 | 0.725684 | 0.053506 | 0.001334 | 0.125722 | 0.003514 |

Compared with the preceding 25-update window, the first 25 updates from the event have a lower median m/g ratio (1.165 to 0.709 at LR 0.001; 0.848 to 0.717 at LR 0.0001). Their raw absolute norm fluctuations increase about 5.8-fold and 2.5-fold, respectively, while momentum fluctuations fall or remain similar. This is consistent with raw norms becoming more jagged near the first observed negative pair. The per-update table nevertheless shows that the norm gap is already present before that pair.

## Interpretation limits

Raw g is pre-clipping; stored m is formed from globally clipped gradients and is uncorrected for initialization bias. Clipping, EMA history and changing gradient magnitudes all affect the norm gap. At initialization, m starts at zero, so early norm differences also include its buildup. In exact arithmetic, m_t = beta*m_(t-1) + (1-beta)*g_t_clipped, so its squared norm includes a cross term between previous momentum and the current clipped gradient. Adjacent raw cosine alone does not determine the momentum norm. The figure is consistent with cancellation/smoothing contributing to norm separation, but it does not isolate their causal role. One matched seed, temporally dependent observations and two LR-scaled trajectories provide no cross-seed uncertainty, stability or convergence guarantee.

Generating script: `../../09_plot_loss_alignment_norms.py`; verified event-loading/smoothing helper: `../../08_plot_simplified.py`. Each source event SHA-256, all raw records, display mask and group reductions are in the figure data. Prior analyses and executed scientific inputs remain unchanged.
