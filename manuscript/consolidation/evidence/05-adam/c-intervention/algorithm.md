# Algorithm 2 — past-calibrated local affine-displacement cap

- **Status:** consolidated state-s notation, ready for integration; original report algorithm is preserved.
- **Inputs:** current parameters and Adam state, fixed probe, targeted modules, one fixed nonnegative budget per module; zero disables that module's cap.
- **Calibration:** for a fork at state f, pool native per-unit direct affine RMS scores over outgoing states **f−8 through f−1**; choose the module's quantile. Treatment begins at **f→f+1**. q50 is the preselected main method; q25/q75 test its sensitivity. No treatment outcome enters calibration.

1. At state s, compute the full native objective gradient and capture current module inputs with the fixed probe and normalization context, preserving the random state.
2. Advance native Adam moments/clock and form its stored parameter proposal.
3. Retain actual proposed parameter increments, including storage effects.
4. For each target module with positive budget, compute D=H Uᵀ+1vᵀ. Its n×m columns correspond to m output units.
5. Compute RMS down each column; scale the entire corresponding weight row and its bias by min(1,tau/score).
6. Use factor one for a zero score; a zero calibrated budget disables the module's cap. Keep untargeted increments native.
7. Store the capped parameters and verify the direct-score bound with the FP32 rounding tolerance. Keep the advanced Adam state.
8. Measure actual propagated motion and loss separately; they are not bounded by the local acceptance rule.

- **Scope:** both MLP hidden affines; 31 ResNet convolutions; two QKV/four feed-forward Transformer affines. These are complete interventions, each compared to unchanged Adam.
- **Cost:** probe forward plus direct local products and verification; no temporal SVD or second complete candidate-network forward is required for the local score.
- **Equations:** [dimensions, metric, bound and unbounded terms](equations.md). [Editable TeX](algorithm.tex).
- **Source:** [implementation geometry](observations/65-010-local-displacement-cap-geometry.md), [original report](report/local-displacement-cap-report.pdf).
