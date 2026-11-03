# Writing principles for the TNNLS revision

These principles synthesize the author's spoken feedback, its clarification
("methodology" means the methodology of writing), and the annotated review PDF.
They govern presentation of the existing experiments; experimental designs,
event selection, source figures and measured values stay fixed.

## Central argument

Adjacent-gradient cosine identifies constructive (`rho_t > 0`) and destructive
(`rho_t < 0`) regimes. Destructive training can continue to reduce the objective.
The combination of reversal and strong gradient-norm growth explains, in the
local GD model, how that regime develops loss-increasing steps. The Adam results
show the same diagnostic pattern around substantial loss spikes despite adaptive
updates. The projected-change diagnostic summarizes this sign-and-magnitude
interaction; inspecting its components reveals the anatomy behind the signal.

The practical contribution is inexpensive, interpretable monitoring of training
dynamics, including early warning signs of developing spikes. The paper should
make this value visible from the abstract through the conclusion.

## Principles

1. **Lead with the contribution.** State the problem, the diagnostic insight,
   and the evidence supporting it. Avoid generic introductions to deep learning.
2. **Keep one argument across the paper.** Progress from regime definitions to
   local loss behavior, controlled characterization, and practical Adam evidence.
   Each section must advance a question left by the preceding one.
3. **Use stable terminology.** A regime is defined by adjacent raw-gradient
   cosine. Describe positive and negative momentum cosines as momentum alignment;
   name the vector stream whenever a diagnostic could be ambiguous.
4. **Explain before formalizing.** Give the practical meaning of each variable
   and event anchor before its equation. Follow equations with their consequence.
5. **Make each paragraph earn its space.** Use a claim, its explanation/evidence,
   and an implication. Discuss why a result matters instead of narrating every
   curve, code decision, or measurement detail.
6. **State supported findings affirmatively.** Replace repeated "not X" sentences
   with what a measurement establishes. Place essential scope once, near the
   relevant inference or in the measurement appendix.
7. **Make affordability concrete.** State vector operation/storage complexity
   and reuse of training gradients. Do not invent measured runtime savings.
8. **Make temporal evidence concrete.** Show the early diagnostic and the later
   loss outcome with exact timing. Use "early warning of a developing spike"
   for the observed lead; reserve predictive accuracy for evaluated forecasting.
9. **Separate the scientific objects through definitions.** The GD directional
   curvature, finite-step projected change, Adam update and raw-gradient regime
   each have a clear role. Essential assumptions belong with the equation.
10. **Use literature to locate the contribution.** Credit prior gradient-product
    diagnostics, hypergradient adaptation and EoS. Emphasize the joint account
    of regime, amplification, cumulative motion and Adam spike anatomy.
11. **Design for selective reading.** Captions define axes, samples and table
    columns. Use readable rounding, centered grouped columns and direct headings.
12. **Finish with a research consequence.** Explain how the diagnostic motivates
    timing of clipping, step-size or data-mixing interventions. Their evaluation
    remains future work in this paper.

## Scientific language decisions

- "Gradients grow indefinitely" in the spoken intuition is expressed as
  "rapid/strong gradient amplification": the recorded excursions recover, so
  the evidence supports large finite growth, not mathematical unboundedness.
- "Predictive" is supported here by temporal lead and peak-magnitude association.
  State those observations positively and specifically. A held-out forecast
  accuracy or intervention benefit is a separate future result.
- Under Adam, `eta kappa-hat` measures projected raw-gradient change along the
  adaptive path. The reference at two has its formal loss interpretation in GD;
  Adam evidence establishes the observed warning pattern.
- Negative pre-rise cosine occurs in 117/127 defined events, and gradient norms
  grow to the loss peak in all 129 qualifying events. Use those actual strengths
  of evidence rather than replacing the first count with "all".
- The table uses the author's explicit rounding examples: no decimal for
  magnitudes >=10, one for [1,10), two below 1; counts are integers and learning
  rates use powers of ten. This resolves the shorthand "two significant digits"
  in favor of the specific requested display rule.
