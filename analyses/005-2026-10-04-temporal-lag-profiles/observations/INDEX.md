# Analysis005 observations

- [001: Temporal lag profiles](001-lag-profiles.md): four cosine/Gram figures, coverage, calibration and sensitivity.
- [002: All-lag interaction](002-all-lag-interaction.md): two cumulative figures, exact aggregate and lag-one approximation checks.
- [003: Training overview](003-training-overview.md): retained 14M/31M/70M histories; loss, eta-kappa proxy, rho and raw norms at all 712 updates.
- [004: Smoothed size overlay](004-smoothed-size-overlay.md): 2x2 model-size overlays, 20-update means, update 712 excluded, and early loss/norm comparison.
- [005: Unsmoothed first400](005-unsmoothed-first400.md): separate 2x2 overlays, raw values for all four variables, updates 1-400; previous figure preserved.
- [006: Short loss difference](006-short-loss-difference.md): 1-400 variant with N=5 trailing mean of L_t-L_(t-1); other three panels remain unsmoothed.
- [007: Log-loss difference](007-log-loss-difference.md): N=5 mean of ln(L_t/L_(t-1)), approximately relative loss change; other panels remain raw.
- [008: Split update windows](008-split-update-windows.md): 4x2 layout; columns 1-100/100-400 and one metric per row, preserving the N=5 log-loss change and raw diagnostics.
- [009: Column smoothing](009-column-smoothing.md): complete means of all four metrics, N=5 at updates 1-100 and N=21 at 100-400, using raw per-update scalars.
- [010: Adam direction](010-adam-direction.md): same N=5/N=21 layout using saved adaptive direction q_d, rho_d and norms; loss means exactly preserved.
- [011: Unsmoothed Adam direction](011-unsmoothed-adam-direction.md): no additional averaging of q_d, rho_d or norms; only log-loss change keeps N=5/N=21 smoothing.
