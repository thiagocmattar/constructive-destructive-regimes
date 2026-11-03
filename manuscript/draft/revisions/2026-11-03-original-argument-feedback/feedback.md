# Extracted author feedback

All 41 annotations are retained verbatim, including the six text notes.

Source: `manuscript/draft/revisions/2026-11-03-original-argument-restored/revised-manuscript+feedback.pdf`.
SHA-256: `b915602794e99461750ae326228dec10c6dbd5f1cd0a620935fda6dc53a5e8bf`.

## C01 - original page 2

Intuitively, stable optimization occurs when updates reinforce each other and gradually shift parameters toward a point of minimum loss. Instability therefore arises when updates overshoot, oscillate, or partially undo the progress made by previous steps~\cite{polyak1964methods, bertsekas1997nonlinear}. Understanding this behavior requires examining the mechanisms that govern how updates evolve from one iteration to the next. Two factors have traditionally played a central role in explaining these dynamics: the \textit{local curvature} of the loss landscape and the \textit{stochasticity} introduced by mini-batch sampling.

**Selected passage or anchor:** Training stability concerns how perturbations and loss evolve under repeated updates. Overshoot and oscillation are useful symptoms, but can also accompany sustained objective progress. Understanding these dynamics requires examining both the local curvature of the loss landscape and the stochas- ticity introduced by mini-batch sampling [10], [11], [13].

## C02 - original page 2

Remove "background"

**Selected passage or anchor:** II. BACKGROUND: STABILITY ANALYSIS OF SGD

## C03 - original page 2

Remove repetitive cientations and avoid the breaking line with a single element e.g. [11]. Maybe we can improve the citation structure here as well.

**Selected passage or anchor:** Training can be interpreted as a discrete-time dynamical system in which the local curvature of the loss function determines the stability of the updates [10], [11]. The curvature indicates how sensitive the loss is to parameter changes in different directions: when curvature is small, updates tend to produce smooth descent toward a minimum, whereas in directions of high curvature the same step can overshoot the minimum and induce oscillatory behavior [10], [11], [17]. For this reason, classical analyses of gradient descent relate stabil- ity to the spectrum of the Hessian of the loss function, whose eigenvalues quantify directional curvature and indicate how sensitive optimization is to the choice of learning rate [10], [11].

## C04 - original page 2

avoid the notation e_. its been used only here. clean.

**Selected passage or anchor:** Consider a small perturbation et−1 around θt−1. Under the lo- cal linearization of the full-batch GD update, this perturbation evolves as et ≈Jtet−1 = (I −ηHt)et−1,

## C05 - original page 2

This is very good. Please note here we're making our point:
- monitoring / measuring hessian values is very expensive and not practical.
We may make this stronger and keep carrying this argument over the paper, as we gradually builds our argument.

**Selected passage or anchor:** Although this suggests monitoring λmax(Ht) as a stability diagnostic, estimating leading Hessian eigenvalues online in large neural networks is computationally expensive, and mini- batch estimates provide only noisy, pointwise proxies of the local geometry.

## C06 - original page 3

we may sharpen it. there exists a problem here of estimating noise. But is this relevant for our argument?

**Selected passage or anchor:** This makes clear why Σ(θt−1) is generally not a viable online diagnostic: computing it requires access to the full gradient µ and all per-sample gradients {hi}N i=1 at each training step, which is impractical and incurs at least a full-pass cost per step.

## C07 - original page 3

Why is this citation relevant? Maybe because it introduces the idea of "regimes"? But why would we spend space on the paper with the proposed equation? Does it relates with our noise atenuation argument? Or not? Evaluate it critically before editing.

**Selected passage or anchor:** The break-even point framework of Jastrzebski et al. [12] is an attempt to extend the classical curvature-based stability condition to SGD. The central idea is to analyze stability along the sharpest direction of the loss landscape while explic- itly accounting for mini-batch stochasticity. More precisely, the framework proposes a one-dimensional necessary-and- sufficient condition for SGD stability at iteration t, (1 −ηλmax(Ht))2 + s2 t η2(N b(N (6) where λmax(Ht) is the top Hessian eigenvalue, N is the dataset size, b is the batch size, and s2 t denotes the variance across training examples of the curvature measured along the leading Hessian direction. Large s2 t means different mini- batches see substantially different local curvature, while small s2 t means they see similar curvature. In this sense, Eq. (6) can be read as a stochastic extension of the deterministic stability condition in Eq. (3), so that the first term captures curvature-induced amplification, while the second captures the destabilizing effect of mini-batch stochasticity. Under this definition, the break-even point is defined as the first iteration at which the left-hand side of Eq. (6) reaches 1. Intuitively, it marks the end of an early phase during which the choice of learning rate and batch size can strongly shape the later optimization trajectory. In this sense, a central conjecture of this framework is that, prior to this point, hyperparameters largely determine the maximum curvature and stochasticity levels encountered later in training, thereby setting the regime in which the subsequent dynamics unfold.

## C08 - original page 3

This is very central to our work. What is the EoS argument for coexistance of local instability and convergence? What works comes after that?

**Selected passage or anchor:** For instance, the Edge-of-Stability (EoS) phenomenon challenges the local- quadratic picture even in deterministic gradient descent (GD). Across architectures and tasks, full-batch GD often exhibits progressive sharpening, with the largest eigenvalue of the Hessian, λmax(Ht), rising until it hovers near (and some- times above) the nominal threshold 2/η (Eq. 3), while op- timization continues to make progress despite short-term non- monotonicity [13]–[15]. Thus, successful training can coexist with local linear instability for extended periods, indicating that sharpness alone does not fully explain and determine trajectory-level behavior.

## C09 - original page 3

again, strenghten our argument.

**Selected passage or anchor:** Together, these results clarify both the value and the limita- tion of curvature–noise interaction criteria. Quantities such as the leading Hessian eigenvalue λmax(Ht) and the gradient- noise covariance Σ(θt−1) characterize local geometry and stochasticity, but can be expensive to estimate online. Their pointwise values also leave open how successive updates interact and accumulate over a training trajectory. This moti- vates complementary diagnostics that describe directional re- inforcement and cancellation using quantities available during optimization.

## C10 - original page 3

but maybe its a little repetitive, and we can move it to the end of the section

**Selected passage or anchor:** This diagnostic:

## C11 - original page 3

This is an important section, but I think we can improve the writing. this is too verbose, repetitive and giving excessive detail to unimportant things (like floating point precision.. which is not relevant to our paper). Additionaly, we have a rich literature of open-weights technical reports, which may bring more citations and arguments for our central point: "how does the practicioners deal with spikes and local instability?". Then we may connect to our central argument:
- connect curvature stability with cheap diagnostics which doesn't require full hessian computation (our contribution).
- some folks already record norms, but how it connects with stability? we define on the next section.

**Selected passage or anchor:** D. Practical Stability Analysis in Large-Model Pretraining Large-model training reports approach stability through observable symptoms and interventions on the training recipe. An instructive example is PaLM: Section 5.1 reports approx- imately 20 loss spikes during training of the 540B model despite global gradient clipping. The authors recovered by returning to a checkpoint about 100 steps before a spike and skipping 200–500 batches. Replaying those batches from a different parameter state did not reproduce the spikes, suggesting an interaction between data and model state rather than a property of the batches alone [24]. Reports accompanying released model weights provide fur- ther operational evidence. During OPT-175B training, loss divergence was associated with failures of the dynamic loss scale and growing final-layer activation norms. Recovery involved reducing the learning rate and restarting from an earlier checkpoint; lowering the gradient-clipping threshold from 1.0 to 0.3 also improved stability early in training [25]. These observations motivate monitoring the numerical state of training alongside the loss. OLMo 2 makes this diagnostic process explicit by measur- ing a rolling spike score on training loss and gradient norm, and using controlled ablations to study initialization, normal- ization, and optimizer settings. Its revised recipe combines global gradient clipping with changes including query–key normalization, reordered normalization, and a logit regular- izer (z-loss) [26]. In Llama 3, the authors report a stable pretraining recipe with few loss spikes and no interventions for divergence; they use learning-rate warmup and a staged batch- size schedule, and emphasize FP32 gradient accumulation and reduction for numerical stability [27]. These reports document both failure recovery and recipes that avoid recurrent failures, rather than a universal instability mechanism. Loss, gradient norm, and activation statistics are practical monitoring signals, but gradient magnitude alone does not describe the relationship between successive directions. A large gradient can reinforce or oppose its predecessor, and a sequence of individually bounded steps can still exhibit cumulative cancellation. We therefore seek a complementary description that can be computed from training gradients and interpreted through their temporal geometry. The following section develops parameter displacement and consecutive- gradient cosine similarity for this purpose, connecting the latter to directional curvature and gradient

## C12 - original page 3

maybe we can remove it and keep only the regimes

**Selected passage or anchor:** CHARACTERIZING

## C13 - original page 3

Elaborate a little more. This is good.

**Selected passage or anchor:** Loss spikes and oscillatory updates motivate practical diag- nostics that can be computed from gradients already available during training.

## C14 - original page 3

We can improve the text here. Pensa que isso é uma introdução de seção. A gente começa ali falando, por exemplo, da loss spikes e oscillatory updates que motivam protocol diagnostics. Isso é muito bom.

Eu acho que, no final, a gente pode deixar esse pedaço de texto mais fluido. Por exemplo: "parameter displays gradient magnitudes and alignment curvature simplifies." A gente não precisa ficar citando as coisas. Eu acho que a gente tem que seguir uma linha mais coerente de raciocínio e leitura.

**Selected passage or anchor:** We develop these diagnostics by relating parameter displacement, gradient magnitudes and alignment,

## C15 - original page 4

delete

**Selected passage or anchor:** through η2∥gt∥2.

## C16 - original page 4

delete. wtf is first-order decrease? inclination? steeper?

**Selected passage or anchor:** first-order

## C17 - original page 4

I think we can simplify here. gradient squared norm is non-negative, so it always move parameter displacement. The step is taken over the loss steepest direction. So, larger gradient norm -> higher inclination (what's the best word here? in pt-br would be inclinação)

**Selected passage or anchor:** The terms ∥gt∥2 are nonnegative and determine the squared lengths of individual SGD steps through η2∥gt∥2. In full- batch GD, a larger gradient norm indicates a steeper local loss and, therefore, a larger first-order decrease along the negative gradient.

## C18 - original page 4

Very good. Give more emphasis to the destructive interference. Its the first time in the paper that it appears in the equations.

**Selected passage or anchor:** The cross-step terms g⊤ u gt describe how these steps interact – positive inner products reinforce the displacement, whereas negative inner products cancel part of it. Large gradient norms can therefore coexist with cancellation along the training trajectory, and destructive interference (i.e. persistent g⊤ u gt < 0) can limit displacement growth.

## C19 - original page 4

Not isolate, but make the alignment explicit. Isolate the norms and alignment. Oh, I see, isolate the dot-product terms. ok. If I didn't get at first, the reader won't either.

**Selected passage or anchor:** To isolate directional alignment from gradient magnitudes,

## C20 - original page 4

Reforce the paper main argument. The constructive-destructive behaviours are encoded in SGD equations. Parameter displacement is higher when alignment is positive. Persistent negative rho caps / limits parameter displacement, and learning.

**Selected passage or anchor:** For fixed gradient norms, positive ρu,t increases the squared net displacement, negative ρu,t contributes cancellation, and orthogonal gradients have no pairwise contribution. In prac- tice, ρt provides an online diagnostic of the interaction be- tween successive updates: persistent positive or negative values characterize constructive or destructive regimes.

## C21 - original page 4

Here we connect the constructive-destructive alignment with curvature. State the subsection goal.

**Selected passage or anchor:** A. Directional Curvature and Consecutive Alignment

## C22 - original page 4

No need. We're defining terms here.

**Selected passage or anchor:** Positive curvature reduces the component of the next gradient along the preceding direction.

## C23 - original page 4

Box it. Main result. Rho is directly connected with curvature.

**Selected passage or anchor:** ρt ≈∥gt−1∥ ∥gt∥(1 −ηκt + εt).

## C24 - original page 4

Let's make like this: in full batch GD, noise is zero. Gradient norms ratio is always positive. So, \rho is proportional \propto 1-\etakappa (write in-line). So, \etakappa > 1 -> \rho <0. And \eta kappa < 1 -> \rho >0. So, \etakappa = 1 is the critical regime that separates constructive and destructive regime (note again how I'm getting back to the main argument / point).

**Selected passage or anchor:** In full-batch GD, εt = 0. Within this first-order approximation, ηκt < 1 retains a positive projection, while ηκt > 1 reverses it: the next update partially opposes the preceding step. At ηκt = 1, the new gradient has lost its component along the old direction. It is orthogonal if nonzero; if it vanishes, cosine is undefined. The norm ratio determines how this projected change translates into the magnitude of ρt.

## C25 - original page 4

This should be an interpretation paragraph. The constructive / destructive regimes allow two types of interpretation:
1) representation learning: gradients are opposing previous learned structure and overwriting previous learned knowledge. So the network keeps overwriting and deleting things. This has a connection with cathastrophic forgetting, because if the step is too big we can critically overwrite this previous learned structure (and threfore forget it); but
2) it also has an optimization intepretation: which is overshooting the minimum. As the optimization gets to the basin, a too big learning rate overshoots the minimum. No need to use the \alpha notation. We can be qualitative here. Maybe we even have some citations to connect.

This is a strong discussion, so we should move the diagnostics (next) to another section.

**Selected passage or anchor:** The threshold at one also has an optimization interpre- tation. Along the full-gradient direction, the local quadratic loss model is L(θt−2 −αgt−1) ≈L(θt−2) −α∥gt−1∥2 + α2κt∥gt−1∥2/2. For κt > 0, its directional minimum is at α = 1/κt. Negative alignment therefore indicates that the step has passed this directional minimum under the approximation. Other gradient components may remain: this threshold identi- fies directional overshoot, rather than a stationary point or the loss-increase boundary derived below.

## C26 - original page 4

Another section. How to estimate curvature without computing hessians. Reforce the central argument here.

Additionaly, discuss that large model training may use big batchs such as noise -> 0. But next we'll discuss the noise effect (connect to the next subsection).

**Selected passage or anchor:** This interpretation leads to a measurable diagnostic. Rear- ranging Eq. (15) gives ηbκt := 1 ∥gt∥ −εt. (16)

## C27 - original page 5

connected with the previous comment

**Selected passage or anchor:** The left-hand side uses only consecutive gradients: it measures their finite-step projected change without a Hessian evaluation. In full GD it estimates directional curvature when the Taylor remainder is small; in SGD it also contains sampling noise. It characterizes the realized step, whereas Eq. (3) concerns perturbations along the most curved local direction.

## C28 - original page 5

I think we can be more compact here. The central point is that noise acts as an atenuation on the sequential alignment of gradients. We can remove Eq E[cos(u+e, v+c]... make the argument, and use the equations to make the point:

batching -> systematic nonpositive term -> atenuation

No need to use the "noise energy" idea.

**Selected passage or anchor:** B. Noise Contribution under Mini-Batch Sampling Unbiased sampling does not make the noise interaction van- ish. Write gt−1 = ∇L(θt−2)+ξt−1. Fresh, conditionally unbi- ased batches give E[g⊤ t−1ξt] = 0 and E[∇L(θt−2)⊤ξt−1] = 0. With finite second moments, the preceding gradient contributes its own noise energy: E[g⊤ t−1(ξt −ξt−1)] = −E[∥ξt−1∥2] (17) Thus, even unbiased sampling noise contributes a systematic nonpositive term to the expected unnormalized interaction between consecutive gradients. This contribution is directly related to mini-batch size. Consider a mini-batch of b examples sampled uniformly without replacement from N > 1 training examples. Using the per-sample gradient covariance Σ(θ) defined in Eq. (5), the sampling variance satisfies E[∥ξt−1∥2] = N −b b(N −1)E[tr Σ(θt−2)]. (18) At fixed parameters, smaller batches produce larger sampling variance and, consequently, a larger negative expected noise contribution. For full-batch gradients (b = N), this contribu- tion vanishes. Noise can weaken directional coherence by enlarging gra- dient norms relative to their shared signal. For independent estimates at two fixed states, let µ, ν be the nonzero full gradients and vµ, vν the expected squared noise norms. When sampled norms concentrate, E[cos(µ + ξ, ν + ζ)] ≈ cos(µ, ν) (1 + vµ/∥µ∥2)(1 + vν/∥ν∥2) . This attenuation moves either sign toward zero. At the same fixed state, the common full-gradient direction gives the spe- cial case E[ρt] ≈ ∥∇L(θ)∥2 ∥∇L(θ)∥2 + N−b b(N−1) tr Σ(θ). (19) Here the expected cosine is nonnegative; the approximation quantifies the dilution of a common direction by sampling. During SGD, a sampled update also changes the next state, coupling noise and curvature. The fixed-state attenuation ar- gument therefore isolates a mechanism, while the batch-size intervention in Section IV-D tests its expression along evolving trajectories.

## C29 - original page 5

the main point here is not measuring long-lag. its is this:

if learning is in destructive regimes, and it keeps overwriting, the natural question is this: how much of the path along the learning trajectory is effectively being translated in learned parameter displacement? Value -> 0, implies almost all the trajectory computation is being thrown away.

**Selected passage or anchor:** C. Normalized Displacement and Cumulative Interference While the consecutive cosine similarity ρt measures local lag-1 directional alignment, it does not capture interactions

## C30 - original page 5

No need to write 2 equations. We can mention the interpretation of R^2 in-line on the text qualitatively.

**Selected passage or anchor:** For constant-step plain SGD, the denominator represents the total distance traveled in parameter space, while the numerator measures the net displacement from initialization. Thus, RT quantifies the fraction of accumulated step length retained as net displacement. Using Eq. (10), this ratio satisfies R2 T = t=1 ∥gt∥2 + 2 1≤u<t≤T ∥gu∥∥gt∥ρu,t t=1 ∥gt∥ 2 . (21) Unlike ρt, which considers only adjacent gradients, RT incor- porates interactions across all temporal lags. When RT is near one, almost all traveled distance survives as net displacement; when it is near zero, little does. Repeated back-and-forth steps and weakly coherent sampled directions both suggest low retention. For example, equal-length orthogonal updates give RT = 1/ √ T, whereas equal opposing pairs cancel exactly. We therefore expect small RT when such behavior persists over a substantial part of an SGD path. Its value is relative to traveled distance: absolute displacement can still grow, and the sign of the complete interaction in Eq. (21) requires the full decomposition.

## C31 - original page 5

the sqrt(T) for orthogonal is a good proxy. we may give numerical example to build the intuition for readers. I think it helps explain why SGD is so bad. even orthogonal updates cancel a lot.

**Selected passage or anchor:** For total

## C32 - original page 5

I don't like the "directional overshoot". We may use conditions for spikes and loss decrease.

**Selected passage or anchor:** D. Directional Overshoot and Loss Increase

## C33 - original page 5

Shouldn't we use \Delta instead of minor delta?

**Selected passage or anchor:** δLt−1 := L(θt−1)−L(θt−2).

## C34 - original page 5

Keep readability. The natural next question is when the destructive regimes evolves into a overshoot?

**Selected passage or anchor:** Directional overshoot need not undo the loss decrease already made along a step. Define its full-objective change as δLt−1 := L(θt−1)−L(θt−2). The second-order approximation derived in Appendix A separates three contributions:

## C35 - original page 5

I think we can make this richer. Does not necessarily reformulates with \eta kappa. We want to interpret rho and its relation with gradient norms.

**Selected passage or anchor:** The magnitude term favors descent. The interference term reinforces it for positive alignment and opposes it for negative alignment: destructive interference contributes positively to the predicted loss change. With negligible noise and nonzero gradients, the approximation predicts a loss increase precisely when ρt < −∥gt−1∥ ∥gt∥. (23) This requires both negative alignment and gradient-norm growth. Neither alone suffices: the opposing component of the new gradient must exceed the preceding gradient magnitude. Large norms can therefore accompany descent, just as negative cosine can accompany a smaller loss decrease.

## C36 - original page 5

Some raw ideas (for you to polish):

1. Condition for a loss spike
A loss spike occurs when
\[
\delta\mathcal L_{t-1}>0.
\]

Substituting,
\[
-\frac{\eta}{2}\|g_{t-1}\|^2
-\frac{\eta}{2}\rho_t\|g_{t-1}\|\|g_t\|>0.
\]

For nonzero gradients, rearranging gives
\[
\boxed{
\rho_t<-\frac{\|g_{t-1}\|}{\|g_t\|}
}
\]

This is the condition for a loss increase under your second-order approximation.

Equivalently, using the inner product,
\[
\boxed{
g_{t-1}^\top g_t<-\|g_{t-1}\|^2.
}
\]

2. What does this tell us?
There are two necessary conditions:

Condition A: Gradient reversal

Since the right-hand side is negative,
\[
\rho_t<0.
\]

Thus, consecutive gradients must point in opposing directions (an angle greater than \(90^\circ\)).

Condition B: Gradient magnitude increase

Since cosine similarity satisfies \(\rho_t\geq -1\), a loss spike is possible only if
\[
\boxed{
\|g_t\|>\|g_{t-1}\|.
}
\]

In other words, the gradient must grow in magnitude following the step, and the reversal must be sufficiently strong.
3. Geometric interpretation
Let \(\phi_t\) denote the angle between the two gradients. Then
\[
\cos\phi_t<-\frac{\|g_{t-1}\|}{\|g_t\|}.
\]

For example:
Gradient magnitude ratio \(\|g_t\|/\|g_{t-1}\|\)	Required cosine similarity	Required angle
1	\(\rho_t<-1\) (impossible)	—
1.5	\(\rho_t<-0.667\)	\(>131.8^\circ\)
2	\(\rho_t<-0.5\)	\(>120^\circ\)
5	\(\rho_t<-0.2\)	\(>101.5^\circ\)
10	\(\rho_t<-0.1\)	\(>95.7^\circ\)

The larger the gradient magnitude increase, the less severe the directional reversal needs to be to produce a loss increase.

**Selected passage or anchor:** The reinforces

## C37 - original page 6

The past comment apply to here as well.

**Selected passage or anchor:** Substituting Eq. (15) with zero noise gives δLt−1 2∥gt−1∥2(ηκt −2). (24) The two thresholds distinguish overshooting the directional minimum (ηκt > 1) from overshooting far enough to raise the loss (ηκt > 2). Between them, the gradients oppose each other while the step still improves the objective. These are local quadratic predictions; finite-step nonlinear effects and the explicit SGD noise term can change the observed sign. A transient increase becomes a loss peak only if subsequent updates recover.

## C38 - original page 6

Please improve writing here. State the experiments goal. Use the standard terminology aligned with previous sections and definitions.

**Selected passage or anchor:** IV. EMPIRICAL CHARACTERIZATION OF GRADIENT INTERFERENCE We use full-batch GD to characterize directional overshoot, its effect on cumulative motion, and its coexistence with loss reduction. We then vary batch size to examine how sampling changes these signatures. The comparison covers a tanh MLP on 5,000 CIFAR-10 images, ResNet-32 on all 50,000 CIFAR-10 training images, and a word-level Transformer on WikiText-2. Each minimizes its complete fixed training objective for 3,000 updates at three constant learning rates (Table I). Within each model, rates share one frozen initialization and data preparation; they were selected by earlier training calibrations to expose contrasting dynamics, without test-loss selection. Plain GD uses no momentum, weight decay, clipping or schedule. These comparisons test whether the same geometric signatures recur across the three objectives; they do not isolate architecture ef- fects. Appendix B specifies objectives, architectures, precision and measurement details.

## C39 - original page 6

Its not "gradient interference" but the constructive / destructive regimes. We gradually:

**Selected passage or anchor:** IV.

## C40 - original page 6

- present convergence curves, show oscilation and how \eta kappa hat correctly tracks the oscilation regimes and hessian eigenvalues without having to fully compute it (PS: here you'll need to edit Fig. 1 to add hessian eigenvalue e.g. \eta lambda max to show how we track them with kappa hat without having to compute the full hessian);
- present how the oscilation happens when the network enters the destructive regimes. How gradient norms grows and displacement ratio decays strongly
- caracterize long-term lags, showing that we effectively observe oscilation, which implies the cos(g_t, g_{t-k}) oscilating between -1 and 1 with a dampening effect for even and odd k. this is a killing evidence of the convergence destructive behaviour.
- show the mini-batch attenuation effect. as we reduce batch size, we see mean \rho over N steps going to 0.
PS: I think we may remove the TABLE1 (endpoints). its not adding too much and it is ocupying too much space.

**Selected passage or anchor:** IV.

## C41 - original page 6

Apply this to all the section. I won't review subsection by subsection. Also, make sure there is no "empty spacing" on the pages. Go on.

**Selected passage or anchor:** IV.
