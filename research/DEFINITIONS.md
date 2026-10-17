# Definitions to settle

Status: manuscript orientation plus the approved Run001 measurement contract.
Run001 defines whole-model raw, unscaled accumulated gradients at each optimizer
boundary before clipping/AdamW; the requested cosines and cumulative norms are
specified in `BASELINE_PLAN.md`. Broader claims/definitions below remain open.

The working formulation (`manuscript/draft/sections/03-methods.tex`) now starts
with displacement from initialization under constant-step, plain mini-batch SGD.
It decomposes squared displacement exactly into squared gradient norms and
pairwise inner products, defines pairwise cosine `rho_{u,t}`, consecutive
cosine `rho_t`, and normalized displacement `R_T = B_T/A_T`,
and connects projected gradient change to directional curvature and sampling
noise. The older absolute parameter-norm and sample-level `C/S1/S2/S3`
decomposition remains in Appendix A for the existing image-classification results.

Its practical diagnostic `rho_t` is the cosine similarity between consecutive
mini-batch gradients evaluated along the evolving trajectory. Positive/negative
values describe reinforcement/cancellation of those vectors. The diagnostic
alone does not establish loss reduction, forgetting, or any formal stability
criterion. The current displacement formulation retains interactions beyond
lag one; it does not assume they vanish. The local loss relations retain noise
and a quadratic/Taylor approximation, so gradient reversal can coexist with
loss decrease. These manuscript relations are distinct from an experiment's
measurement contract or a convergence guarantee.

Before implementation, decide:

| Choice | Why it matters |
| --- | --- |
| Meaning of constructive/destructive | Specify which vectors, time horizon, and sign or effect define the regime. |
| Meaning of stability | Distinguish norm growth, loss behavior, trajectory sensitivity, and other possible meanings; choose an observable criterion. |
| Gradient versus actual update | Momentum, Adam/AdamW, clipping, decay, and schedules change the relationship. |
| Temporal coordinate | State whether one step means a microbatch, accumulated optimizer update, or another unit. |
| Sample unit | Documents, packed sequences, and tokens give different per-sample decompositions. |
| Parameter scope | Specify all trainable parameters or named groups/layers; count tied parameters once. |
| Exact versus approximate terms | Keep exact identities, truncated sums, sampled estimates, and residuals distinct. |
| Loss relation | State what evidence would connect the geometric diagnostic to optimization behavior. |

These choices remain open. Preserve the distinction between manuscript notation
and whichever operational definitions the new experiments eventually use.
