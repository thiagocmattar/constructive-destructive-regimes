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
decomposition is preserved in the uncompiled legacy appendix; the active draft
now follows the SGD experiments with practical directional diagnostics under Adam.

Its practical diagnostic `rho_t` is the cosine similarity between consecutive
mini-batch gradients evaluated along the evolving trajectory. Positive/negative
values describe reinforcement/cancellation of those vectors. The diagnostic
alone does not establish loss reduction, forgetting, or any formal stability
criterion. The current displacement formulation retains interactions beyond
lag one; it does not assume they vanish. The local loss relations retain noise
and a quadratic/Taylor approximation, so gradient reversal can coexist with
loss decrease. These manuscript relations are distinct from an experiment's
measurement contract or a convergence guarantee.

## Fixed-anchor lag profiles and time direction

For the user's backward anchored profiles in
[Analysis035](../analyses/035-2026-11-02-eos-backward-anchors/README.md), fix one
endpoint t and define rho_t(k) = cos(g_t, g_(t-k)), k=1..256. The reference
gradient g_t stays fixed for the entire curve. Increasing lag walks backward
through earlier states; each point uses one pair, without averaging endpoints.
The gradient is the full-objective raw gradient before optimizer transformation.

Analysis033 and Analysis034's historical anchored profiles instead fix a window
start s and measure cos(g_s, g_(s+k)). Label these forward anchored profiles.
Cosine symmetry exchanges the two vectors in a particular pair; it does not
make forward and backward windows at the same numerical state equivalent.
Neither definition is a moving adjacent profile cos(g_(u-1), g_u): there the
pair separation remains one and u moves along the trajectory. Its horizontal
coordinate counts updates after the fork, rather than pair lag.

An optimizer fork at a GD-trained state has a GD past and a new optimizer future.
For an entirely Adam backward window, place t after the Adam fork, far enough
that every compared state is in that continuation. Analysis035 uses t=s+256
for both optimizers, with starts s=500/2000/3000 and endpoints t=756/2256/3256.
Endpoint-gradient norms or adjacent cosines cannot recover the other
retrospective pairs; the relevant vectors or products must be measured.

## Remaining definition choices

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

The [Adam extension](../analyses/028-2026-10-30-adam-diagnostics/README.md)
reuses norms, adjacent cosine and accumulation ratios for explicitly identified
raw-gradient, first-moment and adaptive-direction streams. Under clipping,
moments use the clipped optimizer input. Stream accumulation is not actual
parameter displacement: the latter requires applied increments, including LR
weights and decay. Raw-gradient changes observe the sampled loss response along
the optimizer's chosen path; the SGD directional-curvature interpretation does
not carry over automatically. These distinctions clarify measurement semantics
without defining a new stability threshold or validated scheduling rule.
