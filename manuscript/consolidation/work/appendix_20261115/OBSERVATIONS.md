# Compact reviewer appendix, 2026-11-15

- Deliverable: `../../draft/manuscript.pdf`, 22 pages; appendix begins on page 17, figures on pages 20-21, references on page 22.
- Scope: presentation and consolidation of existing evidence. No new training, scientific settings, or cloud spending. Active manuscript and evidence originals preserved.
- Review method: the local human/review-skill checks each retained item against a claim or likely reviewer objection. Main-text references now lead to the embedded appendix; the unchanged 27-page supplementary PDF remains an extended research record.

## Minimum retained evidence

| Element | Reviewer question answered | Coverage / source |
|---|---|---|
| Appendix A: short identities and error bounds | What is exact, and when can the curvature or loss-sign approximation fail? | Existing S1 derivations; smooth arbitrary-optimizer step, sampling contamination, finite-step remainder, conditional sampling assumptions |
| Appendix B: methods and HVP table IV | Can the measurements be reproduced and is the cheap curvature independently checked? | Existing S2-S4; objective and normalization contracts, 1,188 HVP pairs, measurement controls |
| Spectral table V | Is low dimensionality merely a few large changes? | Frozen `05-adam/b-tracing/data/spectral_controls.json`; observed, same-energy orthogonal, and unit-row controls over the same 51/61/49 changes |
| Appendix C and late-loss table VI | Are interventions matched and endpoint claims robust to oscillation phase? | All 36 settings; mean of the last 32 states, not selected best checkpoints; shared-state and intervention integrity checks |
| Cost table VII | Does the intervention add material work? | Measured q50 gradient, reference, and controller costs on each actual host; not a time-to-loss benchmark |
| Figure 10: applied updates | What does every intervention actually do across the complete window? | Four intervention families, all settings, unchanged Adam in each row; every stored applied-update norm |
| Figure 11: activity | When do clipping and caps act? | Both clipping thresholds and all three cap quantiles; raw-gradient multiplier versus pooled fraction of targeted units capped |
| Figure 12: selected functional changes | How do conventional interventions change loss, reversal, and movement? | beta2=0.5, LR multiplier=0.1, clip=1.0 across all three models; no Adam heatmaps; gray Adam loss reference |

## Figure measurement contract

- Generator: `01_artifacts.py`; source hashes, array hashes, output hashes, and table reductions in `manifest.json`.
- Frozen input: `../../evidence/05-adam/c-intervention/data/matched-{curves.npz,results.json,costs.csv}`. Sources originate in the matched intervention study in Analysis 065; existing evidence figure 22 supplies the activity contract and figure 20 the heatmap contract.
- Windows: MLP 512-800, ResNet 256-576, Transformer 544-928; actual increments belong to the completed-state endpoint t. Every post-start update is retained without smoothing.
- Figure 10: actual parameter increment norms, logarithmic axis, common limits across families within a model. Gradient clipping factors are not Adam update-retention factors.
- Figure 11: cap-active fraction pools targeted units across all sites, rather than averaging site fractions; values come directly from the frozen reducer outputs. Legends occupy reserved space above the valid data range.
- Figure 12: consecutive representation-change cosine and per-site RMS divided by that site's unchanged-Adam maximum over the complete post-start window. A common color scale across all selected settings/models includes values above one. Loss limits match across the three settings within each model. Site selection matches the main tracing figure; no averaging across layers.
- Captions are embedded in `../../draft/sections/appendix.tex`; they state indexing, reductions, and comparison baselines.

## Interpretation and exclusions

- q25 improves the late-32 mean in all three models; ResNet q50 remains worse than control. Neither endpoint nor short late average establishes asymptotic convergence.
- Spectral controls support contributions from amplitude and repeated directions; they are not randomized significance tests. They do not alone establish reversal or causality.
- Selected conventional settings illustrate distinct responses; they are not presented as each family's best setting. Main Table III preserves the complete grid.
- All intervention comparisons use one initialization per model; effective intervention strengths differ. Cost measurements are instrumented and use different GPUs; no time-to-loss superiority claim follows.
- Excluded from the embedded appendix: exhaustive 11-setting heatmap plates, repeated Adam maps, historical MLP quantile ablations, complete held-out trajectories, and long numerical narratives. These remain available in the preserved extended record.

## Verification

- Source arrays agree exactly with all 78 plotted curves and 18 heatmaps. All 36 late-loss cells independently recomputed; costs recomputed from recorded durations/counts. Spectral rows match frozen controls.
- Full verifier retains the 144 intervention metrics, 108 main table cells, mathematical checks, and 416 protected-original checks.
- Final manuscript compiles without warnings, unresolved references, or out-of-page text; fonts embedded. All main pages reviewed in rendered contact sheets, with appendix pages and new figures additionally inspected at full render size.
- No push performed. The separate supplement PDF is byte-identical to the preceding revision.
