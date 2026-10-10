# Observation004: implications for the surgical manuscript proposal

Date: 2026-10-09. User-requested interpretation of the seven-rate figures alongside the [2026-10-07 proposal](../../../../manuscript/2026-10-07-surgical-changes-proposal/README.md). This is a retrospective interpretation and local aggregate audit, not a new training experiment or a manuscript edit.

## Sources, reduction and coverage

The four current PDFs were visually reviewed, including both pages of the within-LR view. Their sources and plotting reductions remain in Observations001-003. The numerical source is `../comparison_data.json`, SHA256 a02a1ba7ace7a7011063767e6e7fb1c25548be0a12be06b6dad6826e5e7b2739. The proposal's evidence map, surgical changes, mathematical core, experiment plan and positioning document were read together.

Generator for the additional numerical audit: `../../12_audit_interference.py`; output: [interference_audit.json](../interference_audit.json), including source/generator hashes, full-prefix results for each stream, and endpoints with/without the final partial update. For each vector stream, compute Q=sum of squared individual norms and J=(B^2-Q)/2, the sum of all unordered cross-step inner products. Adjacent norm-weighted interaction I1 is reconstructed from consecutive norms and recorded cosine; J-I1 is the total over separations of at least two updates. This is the exact aggregate identity evaluated from recorded scalar reductions, subject to floating-point precision. It does not reconstruct individual lag profiles.

All712 accepted updates enter the main audit, including the zero-LR first update; the711 endpoint is the full-sized-batch sensitivity. The script checks the source hash, update/batch coverage, finite inputs, reconstructed A against the stored terminal A, and cumulative squared norms against an independently summed endpoint. These are geometric sums over updates, without token reweighting. Loss comparisons use the complete token-weighted final validation; late cosine summaries use the existing updates612-711 median, not a transition threshold. All conditions use one seed1234; original three use Spark and new four use RTX4090. Peak LR also scales AdamW decay.

## Interpretation of the figures

The results strengthen the proposal's distinction among local alignment, cumulative interaction and loss. Negative raw adjacent cosine appears earlier as LR increases: first negative pairs occur at updates385,168,97,54,30,18,12 in increasing-LR order. Six runs continue to improve loss after negative raw pairs appear. The best tested endpoint is LR0.00464159, validation4.318091; LR0.01 ends at6.636489 with worsening late training loss. Negative-pair onset describes a trajectory change but cannot itself identify convergence or a harmful LR.

Momentum-only measurements now directly address part of the proposal's open component comparison. At the six lower rates, m reduces terminal A by35.64-75.24% relative to raw g while B decreases by0.98-5.80%. All seven conditions have higher momentum B/A and lower B than raw gradients. At LR0.01 the reductions are95.46% for A and33.91% for B. Thus the higher ratio is driven primarily by attenuation of A, with no observed terminal increase of B. This supports the separate lower-A/higher-ratio predictions and rejects a universal higher-B prediction for these stored moments.

The proposal already gives the relevant EMA boundary identity. For the actual clipped-gradient input c, zero initial moment and beta1=0.9, real arithmetic gives sum(m)=sum(c)-9*m_T. The comparison to raw g also includes clipping; no causal momentum-only attribution follows. Frequent clipping matters especially at LR0.01 (517/712 updates). Bias-corrected m-hat and a separate clipped-gradient geometry stream remain missing from the requested comparison. Increasing the geometric ratio does not quantify a training benefit or a fraction of useful learning saved.

First-moment and adaptive adjacent cosines are also not monotone performance criteria. Within the hardware-matched new runs, LR0.00215443 has late momentum cosine0.956273 and validation4.410370; LR0.00464159 has lower momentum cosine0.852282 and better validation4.318091. Adaptive late cosine is nearly identical at the best tested LR and at0.01 (0.853524 versus0.854019), despite sharply different loss. Adaptive final B/A is also lower at the best tested rate than at0.00215443 (0.436078 versus0.591829). Adjacent momentum alignment is a measure of direction changes in an overlapping EMA sequence, not a direct measurement of negative longer-lag interaction.

The high-LR failure has a joint signature: worsening late loss, large raw norms (late median5.565900, versus0.820563/1.227853 at0.00215443/0.00464159), frequent clipping, very low raw B/A, and falling momentum coherence. The raw/momentum norm separation alone appears at successful rates too. This is compatible with amplified short-timescale variation, but these moving-parameter batch gradients do not identify stochastic gradient-noise variance or a particular curvature mechanism.

## Additional all-pair result

| Peak LR | Raw B/A at712 | Raw J at712 | Raw J at711 |
| --- | ---: | ---: | ---: |
| 0.0001 | 0.594543 | 46049.411 | 46010.000 |
| 0.000215443469 | 0.304452 | 14478.636 | 14476.840 |
| 0.000464158883 | 0.204918 | 6987.343 | 6985.808 |
| 0.001 | 0.249074 | 7636.962 | 7627.443 |
| 0.002154434690 | 0.334955 | 10501.796 | 10486.181 |
| 0.004641588834 | 0.239815 | 8418.531 | 8393.802 |
| 0.01 | 0.044843 | -538.290 | -580.995 |

Only LR0.01 has negative whole-prefix raw J, and the conclusion survives exclusion of the partial final update. Its positive-to-negative prefix transition first occurs at update674. For this run, final Q=8699.180329 and B^2/Q=0.876243; I1=-1035.528496 while the sum over lags2+ remains positive497.238123. Negative total interaction therefore does not imply every longer lag is negative. Momentum and adaptive J remain positive in all seven conditions, including the failing run.

This adds a useful full-horizon contrast to the existing local-negative/global-positive examples. It does not establish a predictive threshold: the transition is late, after the high-LR trailing25 loss has attained its minimum at update520, and this is one retrospectively examined failure condition. Prefix J includes the entire early history and mixes magnitudes with directions. Values and signs must retain their vector and horizon labels. No rolling-window J or dense lag profile is available from these scalar prefix metrics alone.

## Implications for the proposal

The characterization claim is stronger: raw local cancellation, all-pair accumulation, momentum filtering, preconditioning and performance are distinct. The direct m measurement resolves the narrow stored-moment question without requiring the proposed sketch reconstruction. The lower-A/similar-B pattern agrees with the elementary filtering identity and should be presented as measured optimizer behavior, not as a new theorem. The wider g/c/m/m-hat comparison and realized parameter displacement remain incomplete.

A suitable bounded empirical statement is: across the measured warmup-only learning rates, successful loss reduction coexists with negative raw adjacent alignment and substantial cancellation relative to aligned accumulation; momentum reduces cumulative norm accumulation much more than the resultant norm. The largest tested LR exhibits a different whole-prefix raw interaction sign alongside deteriorating loss, while transformed directions retain positive aggregate interaction.

The practical direction remains conditional. The next discriminating design is the proposal's matched-checkpoint LR branching, comparing initial changes in recent geometry against subsequent independent loss outcomes, with common optimizer state/data and matched hardware. It should test whether geometry adds information beyond LR, training phase and trailing loss. Replication across seeds and separate clipping/bias-correction/actual-update capture would strengthen the component interpretation. Neither maximizing cosine nor maximizing prefix B/A is justified as an LR policy by the present sweep.

No draft, proposal, executed input, source evidence or PDF was edited. No additional compute was launched.
