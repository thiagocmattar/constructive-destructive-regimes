# Surgical changes proposal for the constructive destructive paper

Date: 2026-10-07. Status: proposal for discussion with the advisor.

The proposed direction combines **characterization of temporal gradient
interference with controlled tests of learning-rate interventions**. Replace
the broad stability claim with a precise question: how do gradient and update
vectors accumulate across time, how does Adam transform that geometry, and
when does the diagnosis help select a better training intervention?

This keeps the paper's central intuition while changing what it promises to
explain. The strongest present result is the separation between local
cancellation, cumulative interaction, and loss. An adaptive scheduler is a
possible subsequent contribution whose usefulness must be demonstrated.

The mathematical presentation starts directly with mini-batch SGD,
\(\theta_t=\theta_{t-1}-\eta g_t\), and derives displacement and gradient
diagnostics from it. Adam is treated afterward through its moments and
preconditioning. This follows the user's preference for concrete optimizer
equations rather than a general update-vector notation.

## Reading order

| Document | Purpose |
| --- | --- |
| [Evidence map](01-evidence-map.md) | What the current results support, including corrections to the discussion notes |
| [Surgical changes](02-surgical-changes.md) | Concrete changes to the existing sections, equations, claims, and figures |
| [Mathematical core](03-mathematical-core.md) | Displacement, lag decomposition, cumulative coherence, and the Adam momentum comparison |
| [Experiment plan](04-experiment-plan.md) | Ordered next steps, matched controls, measurements, and criteria for interpreting results |
| [Positioning and references](05-positioning-and-references.md) | Contribution boundaries and primary literature to address |

## Direction preserved from the discussion

The advisor discussion identified three routes: define and diagnose instability;
characterize a learning mechanism; or use the diagnosis to modulate step size.
The user's preference is a combination of the second and third. The first
would require an explicit stability definition and experiments measuring it,
and is deferred in this proposal.

The intended ideas are:

- Replace absolute parameter norm with displacement from initialization.
- Keep adjacent cosine as a practical local measurement and use lag profiles
  to establish what it misses.
- Combine local cosine with cumulative sum of norms \(A\), norm of the vector
  sum \(B\), and the coherence ratio \(R=B/A\).
- Test whether Adam's temporal averaging reduces \(A\), increases \(B\), and
  narrows their relative gap before preconditioning. These are separate
  predictions; the proposed overlay must allow each to fail.
- Use short, matched LR interventions to investigate whether the geometry
  helps identify useful changes without repeating entire training prefixes.
- Introduce the intuition with a controlled two-dimensional optimization
  example, likely the Rosenbrock function recalled in the discussion.
- Make gradient-history compression and its calibration explicit in the
  methodology.

“Leg” is interpreted as **lag**, and “fixed orientations” as the existing
**fixed origins** coverage control. The ratio is consistently \(B/A\), matching
the stored metrics, rather than its reciprocal. “Before preconditioning” means
the first moment \(m_t\), with \(\hat m_t\) also reported to separate temporal
averaging from initialization bias. These interpretations preserve the intended
scientific comparison.

## Proposed central claim

> Temporal interference depends on the vector being measured and the temporal
> horizon. In the measured Pythia/MiniPile trajectories, negative adjacent
> raw-gradient alignment coexists with positive cumulative interaction.
> Adam's adaptive directions have different coherence from raw gradients, and
> learning-rate interventions change local cancellation. These diagnostics
> characterize optimization geometry; their value for selecting interventions
> must be assessed against loss at a matched training budget.

The title can retain continuity with the existing paper:
**Constructive and Destructive Training Regimes Through Temporal Gradient
Interference**.

An alternative that emphasizes the revised contribution is:
**Temporal Gradient Interference Across Lags and Optimizer Directions**.
Add a scheduling claim to the title only after a scheduling method has been
evaluated.

## What makes the proposal stronger

The stability criticism is addressed by changing the object of study to
measured geometry. The lag-one criticism is addressed by measuring the omitted
interactions and retaining the exact all-lag identity. The practical extension
asks whether that characterization can improve a decision about step size.

Two findings must remain prominent. The middle-window profiles show negative
lags 1 and 2 followed by a positive lag-3 rebound. Also, held LR produces more
negative adjacent alignment and lower validation loss in the completed
intervention. Together, these results motivate a paper about the usefulness
and limits of diagnostics; they do not support maximizing positive cosine
as the training objective.

For a selective journal such as TNNLS, the contribution needs to go beyond
renaming a norm expansion or introducing cosine. The proposed package is a
clear mathematical accounting, calibrated temporal measurements, optimizer
component comparisons, and replicated interventions with independent loss
outcomes. This is an editorial recommendation, not a prediction of acceptance
or a claim that the journal requires a new optimizer.

## Recommended next steps

1. Audit reconstruction of the momentum comparison from the saved sketches;
   specify the exact comparison figure before considering a replay.
2. Build the controlled quadratic and Rosenbrock examples to distinguish
   useful oscillation from harmful overshooting.
3. Replicate the existing LR intervention and temporal profiles with the
   declared coverage, then test short LR probes at selected checkpoints.
4. Implement a feedback scheduler only if the probes show predictive value
   beyond training phase, the existing schedule, and direct loss measurements.

The detailed designs are proposals, not executed configurations. No new run,
analysis, or hypothesis ID is consumed by this folder. The working TeX,
supplied reflections, original archive, reviewer files, and experiment evidence
are preserved. Navigation records link this proposal; no training job or cloud
spending is part of this task.
