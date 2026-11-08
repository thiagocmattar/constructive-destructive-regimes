> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [006-spectral-controls.md](../../../../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/006-spectral-controls.md).

# Spectral concentration: magnitude and directional controls

- **Question:** could the event-window concentration come entirely from a few large updates, without repeated directional structure?
- **Answer:** directional overlap contributes beyond the step-magnitude profile in all three selected windows. Amplitude weighting contributes substantially in the Transformer. Temporal alternation requires the existing cosine and signed-coefficient histories; the spectrum alone cannot establish oscillation.
- **Scope:** local reduction of the original retained output Grams; no new training, captures, probes, windows, hypotheses or runs.

## Reduction and results

- Let the rows of D be flattened consecutive output changes; K = D D^T. Use exactly the windows/probes of [observation 005](63-005-technical-report-event-window.md): MLP 450–500, ResNet 280–310, Transformer 552–600. The incoming predecessor supplies the original cosine but is excluded from every spectral fit.
- **Observed:** uncentered, unnormalized K, retaining magnitude and direction.
- **Same-energy orthogonal reference:** diag(K). Each step keeps its original squared norm, while every cross-step inner product becomes zero. This is a deterministic counterfactual reference, not a randomized null or a quiet-window comparison.
- **Unit-row directional reference:** diag(K)^(-1/2) K diag(K)^(-1/2). Every nonzero change has unit norm, removing amplitude weighting. No zero rows occur in these three fits.
- Rank95 is the number of modes explaining 95% of squared displacement energy. Its ceiling is the number of fitted time rows T.

| Output / fitted rows | Observed top-one / rank95 | Same-energy orthogonal top-one / rank95 | Unit-row top-one / rank95 |
|---|---:|---:|---:|
| MLP raw outputs / 51 | 53.02% / 5 | 7.85% / 42 | 36.21% / 5 |
| ResNet log probabilities / 31 | 24.03% / 16 | 10.55% / 27 | 18.03% / 18 |
| Transformer log probabilities / 49 | 41.65% / 15 | 16.72% / 20 | 16.77% / 29 |

## Interpretation and uncertainty

- The observed spectrum is more concentrated than the same-norm orthogonal reference. The original concentration cannot be attributed solely to unequal step magnitudes.
- Equalizing magnitudes preserves strong concentration in the MLP; it increases Transformer rank95 from 15 to 29. Report both direction and amplitude rather than claiming a universal single oscillatory mode or uniform intrinsic dimension.
- Permuting time rows leaves singular values unchanged. The measured leading-coefficient sign changes and adjacent-change cosines supply the separate evidence of reversals.
- These are descriptive single-seed selected-window results. Different probes, output spaces and T prevent controlled architecture rankings. There are no sampling confidence intervals, independent-probe replications or quiet-window contrasts. The controls do not establish spike-specific concentration, a causal initiating layer, stationary axes, or a forecasting signal.

## Reproduction and checks

- Generator: [10_spectral_controls.py](../../../../../../analyses/063-2026-11-07-architecture-instability-tracing/10_spectral_controls.py), run locally with Python/NumPy.
- Results, complete spectra and source SHA-256 hashes: [spectral_controls.json](../data/spectral_controls.json).
- Sources: Analysis 062 MLP retained Grams and Run 243 ResNet/Transformer retained Grams, pinned against the existing three result manifests.
- Checks: original top-one/rank95 reproduction; positive semidefiniteness and trace; agreement with a separate SVD of each Gram; exact diagonal-reference spectrum; normalized unit diagonal; time-permutation invariance.
- No new figure is generated. The table supplies the adversarial control for V-B of the [single consolidation plan](../../../../PLAN.md).
