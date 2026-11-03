# Author feedback: second pass

Source: `manuscript/draft/revisions/2026-11-03-original-argument-feedback/revised-manuscript+feedback.pdf`
SHA-256: `83e2f4c3e036470276ab69d809863fa255d34716a84e5f6da2ff8294ecc3693c`

30 annotations, transcribed verbatim. C04 highlights the colon after the proportionality statement; its selected-text field is empty because it is punctuation.

## C01 - page 3

no need to cite what isn't there. delete.

Selected passage: Llama 3 reports few spikes and no divergence interventions, while DeepSeek-V3 reports no irrecoverable spikes or checkpoint rollbacks [27], [28].

## C02 - page 3

delete

Selected passage: For plain SGD, these are also the cosines between successive parameter updates. They describe how learning changes the parameters; the associated loss deter- mines whether those changes improve the objective.

## C03 - page 3

noise

Selected passage: sampling

## C04 - page 4

, so

Selected passage: (punctuation only)

## C05 - page 4

The point here is that overwriting is not restricted to cathastropic forgetting. Even in iid setting, successive updates may overwrite previous learned structure.

Selected passage: Gra- dient conflicts also arise in catastrophic forgetting: continual- learning methods constrain updates against gradients of earlier tasks to preserve their losses [29]. Establishing forgetting in our setting would additionally require measuring retention of learned behavior.

## C06 - page 4

remove the break line and the definition as a function of gradient dot products.

Selected passage: = −g⊤ t−1(gt −gt−1) ∥gt−1∥2

## C07 - page 4

remove.

Selected passage: O(p) storage and arithmetic for p parameters, with no additional backward pass, Hessian–vector product, or eigensolver.

## C08 - page 4

remove.

Selected passage: Cosine attenuation also depends on normalization: sampling increases gradient variability relative to the direction shared between batches.

## C09 - page 5

remove. keep only the rho condition.

Selected passage: ⇐⇒ g⊤ t−1gt < −∥gt−1∥2.

## C10 - page 5

delete. simplify.

Selected passage: Prior training calibrations selected contrasting regimes; test loss did not guide selection. Plain GD uses no momentum, weight decay, clipping, or schedule. These three cases test recurrence across objectives and architectures, conditional on their matched initializations.

## C11 - page 5

Remove the subsectrion. we can present figure 1 following the previous paragraph.

Selected passage: A. Curvature and Oscillatory Convergence

## C12 - page 5

remove "retained measurements"

Selected passage: (retained measurements)

## C13 - page 6

regime.

Selected passage: alignment

## C14 - page 8

Is no clear in the document if rho here is the cosine similarity between raw gradients (g), m or d. Is it the same metric as before (SGD), the paragraph is just modifying its understanding? I don't get it. If so, we don't need to re-state the equation. PS: "this is normalized projected raw gradient change along the realized path" -> I don't think so. if the realized step is d, then the definition is wrong. preconditioning and m changes the step direction.

Selected passage: The projected-change diagnostic remains exactly com- putable as ηbκt = 1 ∥gt∥ ∥gt−1∥. (25) Under Adam, this is normalized projected raw-gradient change along the realized path; it is not the SGD directional-curvature estimate. A value above two requires negative cosine and a sufficiently large consecutive norm ratio. Thus, reversal and norm growth explain the diagnostic’s increase algebraically, rather than supplying independent confirmations of instability. The reference at two recalls the GD analysis, but does not identify an Adam stability boundary.

## C15 - page 8

The central point here is:Although d_t determines the step, g_t is still informative about the local loss slope, and its variation along the adaptive trajectory.I.e. make more intuitive: we're measuring how the loss landscape is changing with the adam steps.

Selected passage: Although dt determines the step, the raw gradients still reveal variation in local slope along the adaptive trajectory. Their consecutive alignment and magnitudes distinguish directional reversal from a sharp increase in local slope. The first moment shows how this changing input propagates through the optimizer’s memory.

## C16 - page 8

no need to state training / exp setup in a different section.

Selected passage: A. Controlled Full-Batch Training

## C17 - page 8

bad word. too complex. use a simpelr one

Selected passage: interspersed

## C18 - page 8

remove

Selected passage: Their location in the middle or later part of training is an outcome of this maximum- amplitude rule, not an additional temporal filter.

## C19 - page 8

remove.

Selected passage: with model- dependent timing relative to the raw gradient.

## C20 - page 8

central message:- all models enter with negative raw cosine- gradients spike sharply- eta kappa hat exceeds 2 before the loss maximum (peak)- descend separates de models. resnet and trasnsformers goes back to constructive regimes. mlp oscilates between them.just explain this logic. no need to overly explain "associate the developing etc etc".

Selected passage: All three examples in Fig. 6 enter the pronounced rise with negative raw-gradient cosine. During the rise, the gra- dient norm increases sharply and ηbκ exceeds two before the loss maximum. Momentum norms also grow, with model- dependent timing relative to the raw gradient. In the Trans- former, the first crossing of two within the excursion is at state 788, when loss is only 0.40% above its pre-rise level. The +10% crossing follows at 790; raw-gradient and momentum norms reach their event maxima at 792 and 793, before loss peaks at 794. The diagnostic thus precedes the pronounced spike and its maximum, while the first small loss increase has already occurred. The descent separates the models. ResNet and Transformer move toward positive gradient and momentum cosines, lower projected change and a comparatively smooth loss decline. The MLP continues to oscillate: raw cosine remains mostly nega- tive while momentum cosine changes sign, and loss and norms decrease through repeated rebounds. Recovery therefore need not require every raw-gradient pair to become constructive. These examples associate the developing loss spike with gradient reversal and rapid norm amplification, even though the parameter steps are adaptive. An above-two value marks a strong local gradient response consistent with the observed excursion; it is not proof of a linear instability of Adam. The early response suggests an opportunity to anticipate further growth and intervene. Whether it supplies a reliable trigger, or whether acting on it improves training, requires evidence beyond selected examples.

## C21 - page 8

return is not defined. is when the loss reaches back to a_I+1?

Selected passage: return

## C22 - page 10

if p_i is the peak, why do we need min after arg max?

Selected passage: min

## C23 - page 10

remove this excerpt.

Selected passage: This is an operational amplitude filter chosen for interpretation, not a separation inferred from a density minimum or an optimizer stability threshold. The density retains all amplitudes so that the selection remains visible in context.

## C24 - page 10

Should we add a short ilustration? marking the points of a, c, p, etc. I think it would help the reader intuition.

Selected passage: ci := min{s : ai < s Ls ≤Lai}, pi := min arg max ai<s<ci Ls. (26)

## C25 - page 10

remove this excerpt.

Selected passage: These observations support recurrent sign-and-magnitude structure, rather than a universal pre-onset crossing. Recovery is also heterogeneous: event-averaged raw cosines remain negative for the MLP and the eligible ResNet groups, and are near zero for the Transformer. The positive recovery segments in the illustrations therefore do not describe every event.

## C26 - page 10

This count of events is confusing. Use percentages.

Selected passage: Pre-rise raw cosine is negative in 117 of 127 defined events, and the gradient norm at the loss peak exceeds its pre-rise value in all 129. Projected change is already above two before the first loss increase in 21 of 127 defined cases. Its full-event maximum precedes the loss peak in 82 of 129 events, coincides in 29 and follows in 18.

## C27 - page 10

remove.

Selected passage: It is an event descriptor, not Eq. (25), which pairs a cosine with a consecutive norm ratio. Its magnitude incorporates the future peak and cannot be used unchanged as an online predictor.

## C28 - page 10

remove.

Selected passage: The association is consistent with the qualitative anatomy, with substantial remaining model-dependent dispersion. It does not establish a linear relation on the original loss scale, an Adam stability criterion or a causal feedback loop. Both axes use loss-defined endpoints, the sample is conditional on future amplitude and recovery, and events within a trajectory can be dependent.

## C29 - page 10

the main point here is that destructive regimes with peak gradients explain a good part of the spikes.

Selected passage: To relate the initial regime to amplification during an event, we plot ρai+1 ∥gpi+1∥ ∥gai+1∥ against Lpi Lai . (27) This combines the pre-rise consecutive cosine with the subse- quent rise norm ratio. It is an event descriptor, not Eq. (25), which pairs a cosine with a consecutive norm ratio. Its magnitude incorporates the future peak and cannot be used unchanged as an online predictor. Figure 8 relates larger spikes to the combination of a destructive initial gradient regime and norm amplification. The per-rate fits explain 67–69% of log-loss-ratio variation for the MLP, 90–94% for ResNet, and 38–54% for the Transformer. The association is consistent with the qualitative anatomy, with substantial remaining model-dependent dispersion. It does not establish a linear relation on the original loss scale, an Adam stability criterion or a causal feedback loop. Both axes use loss-defined endpoints, the sample is conditional on future amplitude and recovery, and events within a trajectory can be dependent.

## C30 - page 10

remove.

Selected passage: Our observations are consistent with in- vestigating norm control, without identifying those theoretical conditions or establishing that clipping would help these Adam runs. Clipping Adam’s input changes both moment streams; its effect on the adaptive step cannot be inferred from the raw norm alone.
