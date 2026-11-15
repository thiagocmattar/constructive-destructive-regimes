# Paragraph-by-paragraph review of Sections I and II

15 November 2026. Each entry was recorded before its manuscript edit. Equations and labels are protected. The sequence follows reading order.

## I-01

**Read**

```tex
Neural scaling laws motivate longer training horizons under tight compute budgets~\cite{hestness2017deep,kaplan2020scaling,henighan2020scaling,hoffmann2022chinchilla}. Learning-rate schedules, gradient clipping, and adjustments to momentum or adaptive second moments help control training dynamics~\cite{smith2017cyclical,loshchilov2016sgdr,pascanu2013difficulty,kingma2014adam}. Yet smoother loss curves do not necessarily imply faster progress. Deciding when and how to intervene requires understanding which movements an intervention suppresses and what learning those movements support.
```

**Diagnosis:** The opening changes from compute allocation to optimizer controls without a causal bridge. The technical list precedes the problem; smoother loss is introduced as an unstated objective. Progress, movement and learning are undefined.

**Proposed solution:** Begin with the observed coexistence of opposing updates and falling loss. Introduce controls in ordinary language, distinguish smoothing from loss reduction, and end with the decision the paper investigates. Remove the scaling-law detour.

**Rewrite:**

```tex
Successive parameter updates can oppose one another even while a neural network's training loss decreases over longer intervals~\cite{cohen2021edge,damian2023self}. Oscillation alone therefore does not establish that training has stopped making progress. Reducing the learning rate or limiting gradient size can help control fluctuations~\cite{smith2017cyclical,pascanu2013difficulty}, but a smoother loss curve does not necessarily mean faster loss reduction. This raises a practical question: which changes should an intervention limit to improve training?
```

## I-02

**Read**

```tex
Classical stability analysis relates update size to curvature: a step can overshoot when the learning rate is too large relative to a sharp direction~\cite{polyak1964methods,bertsekas1997nonlinear}. Stochastic extensions account for curvature variability and batch size~\cite{jastrzebski2020break}. Edge-of-Stability studies show that loss can continue to decrease while sharpness remains near or beyond the nominal $2/\eta$ threshold~\cite{cohen2021edge,arora2022understanding,damian2023self}. Under Adam, adaptive preconditioning can generate loss spikes without a progressively sharper underlying objective~\cite{bai2026spikes}. These findings establish that oscillation, threshold crossing, and loss reduction must be considered together; none alone identifies which changes within a network should be controlled.
```

**Diagnosis:** Curvature is introduced through the equally unexplained phrase sharp direction. The paragraph then adds stochastic extensions, sharpness, an undefined eta, and adaptive preconditioning. The sequence reads as a survey rather than an explanation of the decision in paragraph 1.

**Proposed solution:** Explain curvature as change in loss slope, show why it matters for overshooting, then explain why changing curvature and Adam's coordinate scaling complicate the fixed-quadratic picture. Leave the numerical threshold and sampling details for Section II.

**Rewrite:**

```tex
Curvature describes how the slope of the loss changes as the parameters move. If that slope changes rapidly, a large step can overshoot the minimum along its direction~\cite{polyak1964methods,bertsekas1997nonlinear}. Neural-network training complicates this picture because curvature changes along the trajectory. Edge-of-Stability studies explain how oscillations can coexist with loss reduction under these changing conditions~\cite{cohen2021edge,arora2022understanding,damian2023self}. Adam adds another influence: it scales each parameter update using past squared gradients. Changes in this scaling can produce loss spikes even without an increasingly curved objective~\cite{bai2026spikes}.
```

## I-03

**Read**

```tex
The remaining connection is functional. Central Flows explicitly identifies the translation from loss-landscape dynamics to architectural and learning mechanisms as an open problem~\cite{cohen2025central}. A parameter trajectory and its loss do not reveal which internal responses reverse, how large their changes become, or how their temporal structure relates to the output. Explaining spike onset also leaves open whether suppressing the associated movement preserves subsequent progress~\cite{bai2026spikes}. We therefore ask: \emph{how do oscillatory optimization trajectories manifest in internal representations, and can their measured movement guide interventions that preserve training progress?}
```

**Diagnosis:** Functional is used as a label before its meaning is explained. Loss-landscape dynamics, architectural mechanisms and temporal structure stack abstractions. The research question repeats those abstractions instead of naming what will be measured.

**Proposed solution:** Introduce representations as layer responses to inputs. Explain the missing link through concrete questions about those responses, retain the Central Flows motivation, and state the intervention question in terms of loss reduction.

**Rewrite:**

```tex
These explanations concern parameter updates and the loss, but we also need to understand what changes in the network's responses to its inputs. Central Flows identifies the connection between optimization dynamics and the roles of individual layers as an open question~\cite{cohen2025central}. At each layer, do the responses---the network's representations---change back and forth, and by how much? Explaining when a loss spike begins does not establish whether suppressing those changes improves subsequent training~\cite{bai2026spikes}. We therefore investigate how oscillatory updates change internal representations and whether limiting those changes improves loss reduction.
```

## I-04

**Read**

```tex
We approach this question through constructive and destructive training regimes. For plain constant-rate SGD, an exact displacement decomposition connects gradient magnitudes and temporal alignment to accumulated parameter change. Negative consecutive-gradient cosine identifies opposition between adjacent updates, but does not determine the sign of interactions across the complete trajectory. Local reversal, accumulated interference, and loss change are distinct properties. Normalized displacement measures how much traveled distance remains as net motion, while explicit sampling terms separate stochastic gradient measurements from full-objective geometry. Under Adam, raw-gradient opposition must additionally be distinguished from opposition between the applied updates.
```

**Diagnosis:** The paragraph introduces a decomposition, cosine, interference, retention and noise terms before explaining what each contributes. Several distinctions are merely announced. The Adam qualification is correct but leaves the reason implicit.

**Proposed solution:** Explain reinforcement and opposition first, then distinguish consecutive updates from the entire sequence. Define retained displacement in ordinary language. Give the constant-rate SGD scope and explain why Adam gradients are not its applied updates. Leave sampling detail to its contribution bullet and Section II.

**Rewrite:**

```tex
We begin by separating how updates interact from how the loss changes. An update can reinforce part of an earlier parameter change or oppose it; persistent reinforcement and opposition define the constructive and destructive regimes studied here. For stochastic gradient descent (SGD) with a constant learning rate, an exact decomposition accounts for interactions across the entire sequence of updates. This matters because opposition between consecutive updates need not imply cancellation over the full trajectory. We also measure how much of the distance traveled remains as displacement from the starting parameters. Under Adam, the same gradient-based interpretation requires care: memory and scaling alter the updates actually applied.
```

## I-05

**Read**

```tex
For a smooth, fixed objective, projecting the full-gradient difference onto a nonzero applied update and dividing by the update's squared norm recovers directional curvature averaged along that step. This identity holds for any optimizer; its full-batch GD specialization expresses curvature through gradient cosine and norm ratio. For quadratic full-batch GD, reversal and sufficiently large gradient amplification exactly characterize a loss increase on the completed step; the corresponding nonlinear relation is a local approximation. Using these signals to warn about the next Adam update is a separate empirical question.
```

**Diagnosis:** The opening is a formula described in prose: projection, normalization and nonzero updates arrive before their purpose. The rest combines an exact identity, a quadratic criterion and prediction without making their timing intuitive.

**Proposed solution:** Start with the purpose of comparing gradients before and after a step. State the exact average-curvature result and conditions without narrating its arithmetic. Introduce full-batch GD and distinguish understanding the step just taken from warning about the next step.

**Rewrite:**

```tex
Comparing gradients before and after an update also reveals how the loss's slope changed along that step. For a smooth, fixed objective and a nonzero update, gradients of the complete objective recover the average curvature along the step exactly, regardless of the optimizer. Under full-batch gradient descent (GD), this quantity depends on the gradients' relative directions and sizes. On a quadratic objective, sufficiently strong opposition between these gradients exactly characterizes a loss increase; for nonlinear objectives, the corresponding relation is a local approximation. Whether this information about a completed step warns of a loss increase on the next Adam step is tested separately.
```

## I-06

**Read**

```tex
We study these connections using multilayer perceptrons, residual networks, and Transformers on image-classification and language-modeling tasks. Full-batch and mini-batch experiments examine update interactions and sampling effects. Under Adam, fixed reference inputs expose representation changes across successive states. We then use local displacement caps to test representation movement as an intervention target, comparing matched continuations with unchanged Adam and conventional controls.
```

**Diagnosis:** Fixed reference inputs, successive states, local displacement caps and matched continuations are procedural shorthand. The reader cannot yet tell what is held fixed, what is limited, or what makes the comparison matched.

**Proposed solution:** Explain the measurement as passing the same examples through the network before and after updates. Explain the cap by its action and matching by the common saved training state. Preserve model/task coverage.

**Rewrite:**

```tex
We study multilayer perceptrons (MLPs), residual networks, and Transformers on image-classification and language-modeling tasks. Training with the full dataset and with sampled mini-batches lets us examine how sampling affects update interactions. Under Adam, we pass the same examples through successive versions of the network to measure changes in its representations. We then limit the direct changes that proposed updates would produce at selected layers. These displacement caps are compared with unchanged Adam and conventional controls, each continued from the same saved training state.
```

## I-07

**Read**

```tex
We make four contributions connecting update interactions, loss changes, representation dynamics, and intervention.
```

**Diagnosis:** The sentence repeats the preceding overview through a list of abstract nouns. Its only necessary role is to introduce the four bullets.

**Proposed solution:** Use a short signpost; let the bullets carry their substantive claims.

**Rewrite:**

```tex
We make four contributions.
```

## I-08

**Read**

```tex
\item \textbf{A framework for constructive and destructive dynamics under sampling noise.}
Oscillatory stability and consecutive-gradient diagnostics~\cite{cohen2021edge,damian2023self,chee2018diagnostics,pesme2020diagnostics} motivate distinguishing local opposition from accumulated motion. We connect update magnitudes, temporal alignment, and displacement through an exact SGD decomposition, and characterize how sampling enters the measurements. Across the studied GD trajectories, negative adjacent interactions coexist with positive accumulated interactions and continued loss reduction. Matched mini-batch continuations show how sampling modifies coherent reversal, separately from the fixed-state alignment approximation. Together, these results distinguish local reversal, cumulative interference, and displacement retention (Sections~\ref{sec:displacement}--\ref{sec:gd-experiments}).
```

**Diagnosis:** The contribution begins with a citation list and repeats magnitude, alignment and displacement without stating the new insight. Negative adjacent versus positive accumulated interactions is mathematically meaningful but unexplained. Fixed-state alignment approximation is opaque here.

**Proposed solution:** Lead with the question of whether one update's opposition predicts the whole trajectory. Explain the longer-lag result in words and distinguish resampling fixed parameters from continuing training. Retain the framework title, prior-work credit and section references.

**Rewrite:**

```tex
\item \textbf{A framework for constructive and destructive training dynamics.}
Building on studies of oscillatory training and gradient-based diagnostics~\cite{cohen2021edge,damian2023self,chee2018diagnostics,pesme2020diagnostics}, we distinguish opposition between consecutive updates from cancellation over an entire trajectory. An exact SGD decomposition relates update sizes and directions to the displacement they produce together. Our GD experiments show that reinforcement between more distant steps can outweigh opposition between consecutive steps while the loss continues to fall. We also characterize how mini-batch sampling affects gradient alignment, separating resampling at fixed parameters from changes during continued training (Sections~\ref{sec:displacement}--\ref{sec:gd-experiments}).
```

## I-09

**Read**

```tex
\item \textbf{Step-averaged curvature and practical loss-rise diagnostics.}
The additional derivatives required by curvature-based spike analyses motivate cheaper monitoring~\cite{bai2026spikes}. Using the secant quantity underlying Barzilai--Borwein step selection~\cite{tan2016bbsgd}, we connect exact full-gradient step-averaged curvature to angle and magnitude, explicit sampling contributions, and the quadratic GD loss criterion. The computation reuses training gradients with linear storage and vector operations, without additional backward passes or Hessian--vector products. In the three studied full-objective Adam trajectories, the combined angle--magnitude rule achieves the highest accuracy and warning lift among the evaluated rules for a next-step loss increase of at least $0.1\%$, tied with its algebraically equivalent projected-response form. Exact completed-step geometry and empirical next-step prediction remain separate (Sections~\ref{subsec:curvature-diagnostic}, \ref{subsec:loss}, and~\ref{subsec:characterization}).
```

**Diagnosis:** Secant quantity, full-gradient step average, linear storage, warning lift and projected response appear in one bullet. The reader cannot identify the principal result without decoding method and metric names. The timing distinction is important but repeats an already established caveat.

**Proposed solution:** State the practical measurement first, credit the BB antecedent in terms of gradient changes, and explain the prediction rule through direction and size. Define warning lift where the result is reported and describe the equivalent rule as a rewriting, not independent evidence.

**Rewrite:**

```tex
\item \textbf{Curvature measurement and warnings of loss increases.}
Curvature-based spike analyses require extra derivative computations~\cite{bai2026spikes}. We recover curvature averaged along a completed step from gradients already available during training, using parameter and gradient differences also employed in Barzilai--Borwein step selection~\cite{tan2016bbsgd}. Under GD, the result connects the gradients' relative directions and sizes to the quadratic loss-increase criterion. We separately test these signals before the next update in three Adam runs using the full training objective. For loss increases of at least $0.1\%$, the combined rule has the highest accuracy and warning lift---event frequency among warnings divided by overall event frequency---among the evaluated rules. An algebraically equivalent rewriting gives identical warnings (Sections~\ref{subsec:curvature-diagnostic}, \ref{subsec:loss}, and~\ref{subsec:characterization}).
```

## I-10

**Read**

```tex
\item \textbf{Connecting oscillatory optimization to representation dynamics.}
We address the functional characterization of the architectural question above by tracking the same reference inputs across successive training states. Separating reversal, displacement magnitude, and shared temporal structure reveals different patterns across the studied cases: reversal spans all measured MLP sites, while earlier ResNet and Transformer sites show less persistent reversal than their outputs. Decomposing the changes identifies dominant movement patterns and their reversing coefficients. These measurements connect parameter-space oscillation to internal responses without attributing their origin to a particular layer (Section~\ref{subsec:tracing}).
```

**Diagnosis:** Functional characterization of the architectural question is unnecessarily indirect. Shared temporal structure and reversing coefficients are unexplained. The result is buried between procedural language and a causal disclaimer.

**Proposed solution:** Name the measurement directly, explain temporal coordination as changes varying together over steps, and express the mode result without requiring prior SVD knowledge. Keep the architecture-specific observation bounded to the measured windows.

**Rewrite:**

```tex
\item \textbf{Measuring oscillation inside the network.}
Using the same inputs at successive steps, we measure whether layer responses change in opposing directions, how large those changes are, and how they vary together during training. In the studied windows, reversal spans all measured MLP sites, whereas earlier ResNet and Transformer sites reverse less persistently than their outputs. A small set of patterns accounts for most output movement, with repeated changes in opposite directions along those patterns. These observations describe where oscillation appears without establishing which layer causes it (Section~\ref{subsec:tracing}).
```

## I-11

**Read**

```tex
\item \textbf{Testing representation displacement as an intervention target.}
Building on controls for representation changes and unit-wise updates~\cite{kosson2024warmup,brock2021normalization}, we use local displacement caps to test whether limiting movement magnitude can improve training while reversals persist. In matched continuations across three model--dataset cases, all tested cap quantiles reduce mean training loss and accumulated upward loss variation relative to unchanged Adam, without eliminating functional reversals. Comparisons with learning-rate reduction, second-moment changes, and global gradient clipping reveal distinct tradeoffs. These findings identify local representation displacement as a promising intervention target, with endpoint and computational limitations defining the scope of the evidence (Section~\ref{sec:vc-interventions}).
```

**Diagnosis:** Cap quantiles and accumulated upward loss variation assume the reader already knows the intervention and evaluation. Second-moment changes and endpoint limitations add another layer of unexplained shorthand. The final sentence mostly restates the title.

**Proposed solution:** Refer to tested cap settings, spell out the sum of upward loss changes, describe the Adam modification by what it changes, and name final loss and computation cost directly. Preserve the unchanged-Adam comparison and intervention-target conclusion.

**Rewrite:**

```tex
\item \textbf{Testing representation displacement as an intervention target.}
Building on controls for representation changes and individual units' updates~\cite{kosson2024warmup,brock2021normalization}, we test whether limiting the size of local changes improves training while reversals persist. Across three model--dataset cases continued from matched training states, every tested cap setting reduces mean training loss and the sum of upward loss changes relative to unchanged Adam. Representation changes still reverse. Comparisons with a lower learning rate, changes to Adam's averaging of squared gradients, and gradient clipping show different tradeoffs in final loss and computation cost. The results support local displacement as a useful target for intervention in these cases (Section~\ref{sec:vc-interventions}).
```

## I-12

**Read**

```tex
Section~\ref{sec:background} reviews the related work and conceptual foundations. Section~\ref{sec:displacement} develops the regimes and their diagnostics; Section~\ref{sec:gd-experiments} examines GD and mini-batch sampling. Section~\ref{sec:adam} follows Adam loss spikes into functional oscillation, and Section~\ref{sec:vc-interventions} tests interventions. Section~\ref{sec:conclusion} concludes.
```

**Diagnosis:** The roadmap is useful and mostly clear. Follows Adam loss spikes into functional oscillation is a metaphor that obscures the two analyses covered by Section V.

**Proposed solution:** Retain the roadmap and replace only the metaphor with the actual subjects: loss spikes and representation changes.

**Rewrite:**

```tex
Section~\ref{sec:background} reviews the related work and conceptual foundations. Section~\ref{sec:displacement} develops the regimes and their diagnostics; Section~\ref{sec:gd-experiments} examines GD and mini-batch sampling. Section~\ref{sec:adam} examines Adam loss spikes and changes in network representations, and Section~\ref{sec:vc-interventions} tests interventions. Section~\ref{sec:conclusion} concludes.
```

## II-01

**Read**

```tex
Understanding oscillatory training requires distinguishing the stability of an update rule, the accumulation of parameter changes, and their effects on network representations. We review the foundations and closest approaches underlying these distinctions, then identify the questions investigated here.
```

**Diagnosis:** The opening repeats the introduction using abstract categories rather than telling the reader what this section will explain. Closest approaches and these distinctions have no specific referents.

**Proposed solution:** State the three explanatory jobs in ordinary language, in the same order as the subsections.

**Rewrite:**

```tex
This section explains why oscillating updates can accompany loss reduction, what the gradients reveal about those updates, and how their effects can be measured inside the network. The literature provides the starting point for each question and helps delimit what our analysis and interventions test.
```

## II-02

**Read**

```tex
Let $\theta_t\in\mathbb{R}^p$ denote the parameters after $t$ completed updates and $g_t$ a mini-batch gradient evaluated at $\theta_t$, before the next update. For a constant learning rate $\eta>0$, plain SGD follows
```

**Diagnosis:** The timing convention is valuable, but p is unnamed and the equation is introduced without explaining the role of eta. A short physical interpretation can make the notation easier to follow.

**Proposed solution:** Name the number of parameters and explain that plain SGD moves opposite the sampled gradient, scaled by the learning rate. Preserve the before-update timing convention and equation unchanged.

**Rewrite:**

```tex
Let $\theta_t\in\mathbb{R}^p$ collect the network's $p$ parameters after $t$ completed updates. The gradient $g_t$ is computed from a mini-batch at those parameters, before the next update. Plain SGD moves in the negative-gradient direction, with a constant learning rate $\eta>0$ setting the scale:
```

## II-03

**Read**

```tex
For full-batch GD, $g_t=\nabla\mathcal L(\theta_t)$. With $H_t:=\nabla^2\mathcal L(\theta_t)$, the Jacobian of its update map is
```

**Diagnosis:** The paragraph introduces the Hessian and update-map Jacobian solely through symbols. Their different roles are not stated, although the next stability argument depends on that distinction.

**Proposed solution:** Define the Hessian as the matrix describing changes in loss slope and the Jacobian as the first-order propagation of a small parameter perturbation through one update.

**Rewrite:**

```tex
For full-batch GD, the gradient uses the complete training objective: $g_t=\nabla\mathcal L(\theta_t)$. The Hessian $H_t:=\nabla^2\mathcal L(\theta_t)$ collects its second derivatives and describes how the loss's slope changes with the parameters. The update-map Jacobian $J_t$ describes, to first order, how a small parameter perturbation changes after one update:
```

## II-04

**Read**

```tex
On a fixed quadratic objective, a perturbation along Hessian eigendirection $i$ is multiplied by $1-\eta\lambda_i$. For a positive-definite Hessian, all components contract precisely when~\cite{polyak1964methods,bertsekas1997nonlinear}
```

**Diagnosis:** Hessian eigendirection, lambda_i, positive definite and contraction are introduced without connecting them to the preceding perturbation. This makes the threshold look like a formula to memorize.

**Proposed solution:** Explain eigenvectors as independently evolving directions on the fixed quadratic, eigenvalues as their curvatures, and contraction as shrinking perturbations. Express positive definiteness through positive curvatures.

**Rewrite:**

```tex
On a fixed quadratic objective, the Hessian's eigenvectors identify directions whose perturbations evolve independently. The eigenvalue $\lambda_i$ is the curvature in direction $i$, and one update multiplies a perturbation in that direction by $1-\eta\lambda_i$. If all these curvatures are positive, every perturbation shrinks precisely when~\cite{polyak1964methods,bertsekas1997nonlinear}
```

## II-05

**Read**

```tex
Contraction preserves sign for $0<\eta\lambda_i<1$ and alternates sign for $1<\eta\lambda_i<2$. Above two, the component alternates and grows. Thus oscillation and instability are already distinct in fixed-quadratic dynamics.
```

**Diagnosis:** Above two leaves the quantity implicit, and component lacks a clear referent. The meaning of lambda_max is not given in the equation's surrounding prose.

**Proposed solution:** Name the largest curvature, retain the three regimes, and explicitly distinguish a direction changing sign from its perturbation growing.

**Rewrite:**

```tex
Here $\lambda_{\max}$ is the largest curvature. If $0<\eta\lambda_i<1$, a perturbation in direction $i$ shrinks without changing sign. If $1<\eta\lambda_i<2$, it changes sign at every step but still shrinks. When $\eta\lambda_i>2$, it changes sign and grows. Thus back-and-forth motion can be either convergent or unstable, even on a fixed quadratic objective.
```

## II-06

**Read**

```tex
Mini-batch SGD adds sampling variability~\cite{mandt2017sgd,hu2019diffusion,jastrzebski2018three}. For independent uniform batches of $b$ examples, each sampled without replacement from $N>1$ examples,
```

**Diagnosis:** Sampling variability is named but the noise symbol in the following equation has not been explained. The reader needs to know that the comparison is made at the same parameters, not between training times.

**Proposed solution:** Explain that different batches produce different gradients at fixed parameters and define xi as the difference from the full gradient before giving the sampling assumptions.

**Rewrite:**

```tex
Different mini-batches can produce different gradients at the same parameters~\cite{mandt2017sgd,hu2019diffusion,jastrzebski2018three}. Write $\xi_t$ for the difference between a batch gradient and the gradient of the full objective. For independent uniform batches of $b$ examples, each sampled without replacement from a dataset of $N>1$ examples,
```

## II-07

**Read**

```tex
Here $\Sigma$ is the covariance of per-example gradients. Writing $h_i:=\nabla\ell_i(\theta_t)$ and $\mu:=N^{-1}\sum_i h_i=\nabla\mathcal L(\theta_t)$,
```

**Diagnosis:** The equation gives a zero-mean noise statement, but its interpretation is skipped. Covariance and per-example gradients are then introduced only by name and notation; ell_i is not identified.

**Proposed solution:** Explain unbiasedness, name ell_i as the individual example loss, and describe Sigma as variation around the mean before calling it a covariance matrix.

**Rewrite:**

```tex
The expected batch gradient equals the full gradient. Its variability depends on how much the individual example gradients differ. For example $i$ with loss $\ell_i$, let $h_i:=\nabla\ell_i(\theta_t)$ and let $\mu:=N^{-1}\sum_i h_i=\nabla\mathcal L(\theta_t)$ be their mean. The covariance matrix $\Sigma$ records their variation around this mean:
```

## II-08

**Read**

```tex
Its trace measures gradient variability; its directional structure describes the sampling perturbations. Neither is recoverable from an aggregated gradient alone. Changes in sampled-gradient alignment must therefore be distinguished from changes in full-objective curvature.
```

**Diagnosis:** Trace and directional structure need interpretation. More importantly, the final therefore does not follow from the inability to recover covariance from one batch mean; the missing reason is that resampling can change gradients without changing the full objective.

**Proposed solution:** Explain trace and directions of variability, then make the fixed-parameter resampling argument explicit. Keep the limitation on recovering per-example information from an averaged gradient.

**Rewrite:**

```tex
The trace of $\Sigma$, the sum of its diagonal entries, measures total gradient variability. The matrix also records which parameter directions vary most. These quantities require information about individual gradients that their average alone does not retain. Resampling at fixed parameters can change the observed gradient while leaving the full objective and its Hessian unchanged. A change in sampled-gradient alignment therefore does not, by itself, establish a change in that Hessian.
```

## II-09

**Read**

```tex
The geometry also evolves. The break-even framework connects early learning rates and batch sizes to later curvature and gradient variability~\cite{jastrzebski2020break}. At the Edge of Stability, full-batch GD can sustain loss reduction while sharpness fluctuates near or above $2/\eta$~\cite{cohen2021edge,arora2022understanding}. Self-stabilization explains how motion along sharp directions can reduce curvature~\cite{damian2023self}; Central Flows describes the time-averaged trajectory together with rapid oscillations~\cite{cohen2025central}. For mini-batch training, Edge of Stochastic Stability analyzes batch-dependent directional curvature and distinguishes noise-driven oscillation from instability~\cite{andreyev2025stochastic}.
```

**Diagnosis:** This is a compressed list of frameworks. Geometry, break-even, sharpness, self-stabilization, central flow and batch-dependent directional curvature are not connected to the fixed-quadratic and sampling explanations just developed.

**Proposed solution:** Separate the evolving-curvature observation from the mechanisms that explain it. Define sharpness through the already introduced largest eigenvalue, explain what averaging means in Central Flows, and specify the batch loss and gradient used by stochastic stability work. Split into two paragraphs because observation and explanation have different jobs.

**Rewrite:**

```tex
During training, the parameters move and the Hessian changes. Early learning rates and batch sizes can influence the curvature and gradient variability encountered later~\cite{jastrzebski2020break}. In full-batch neural-network training, the largest curvature, or \emph{sharpness}, often remains near or above $2/\eta$ while the loss decreases over longer intervals~\cite{cohen2021edge,arora2022understanding}. This is the Edge-of-Stability phenomenon: the fixed-quadratic threshold does not, on its own, determine the behavior of a trajectory whose curvature is changing.

Self-stabilization analyses explain how motion along a high-curvature direction can subsequently reduce that curvature and limit the oscillation~\cite{damian2023self}. Central Flows follows the trajectory averaged over rapid oscillations, while accounting for their effect on the slower motion~\cite{cohen2025central}. For mini-batch training, Edge of Stochastic Stability studies curvature of a sampled batch's loss along that batch's gradient. It distinguishes oscillations caused by sampling from instability that produces sustained growth~\cite{andreyev2025stochastic}.
```

## II-10

**Read**

```tex
Path-length analysis likewise studies optimization trajectories beyond convergence rates~\cite{gupta2021path}. These perspectives motivate distinguishing adjacent opposition from interactions accumulated across a trajectory, while accounting for sampling. Section~\ref{sec:displacement} connects these quantities through magnitude, alignment, and displacement; it does not treat oscillation alone as evidence of instability.
```

**Diagnosis:** The path-length citation appears without explaining what path length measures or why it follows the discussion of oscillation. Magnitude, alignment and displacement repeat labels instead of stating the remaining question.

**Proposed solution:** Connect back-and-forth motion to total distance traveled, distinguish that distance from the final displacement, and state the Section III question through interactions among steps.

**Rewrite:**

```tex
Back-and-forth motion also raises a question about distance: how far do the parameters travel, and how far do they end up from where they started? Path-length studies bound the total distance traveled by optimization trajectories~\cite{gupta2021path}. We examine how interactions among updates determine the displacement that remains. Section~\ref{sec:displacement} separates opposition between consecutive steps from reinforcement or cancellation across the whole sequence, including the effects of sampling.
```

## II-11

**Read**

```tex
A few Hessian outliers can be much sharper than the spectral bulk~\cite{sagun2017hessian,ghorbani2019density}. Tracking them requires additional derivative computations, even with matrix-free eigensolvers. Under Adam, Bai et al.~\cite{bai2026spikes} identify a spike mechanism involving decoupling between current squared gradients and the adaptive second-moment estimate. Their directional-curvature analysis sharpens onset detection, but additional Hessian--vector products remain a scaling constraint. Preconditioned geometry must also be distinguished from the response of raw gradients.
```

**Diagnosis:** This paragraph has both a broken chain of reasoning and unexplained jargon: Hessian outliers, spectral bulk, matrix-free eigensolvers, decoupling, directional curvature and preconditioned geometry. It changes from maximum curvature to one direction without explaining the relationship. The last sentence announces a distinction without its consequence.

**Proposed solution:** First explain maximum versus chosen-direction curvature and the extra computations involved, defining Hessian-vector products by their operation. Then give Adam's mechanism in plain language and explain why curvature after Adam's scaling is not automatically the same quantity as a raw-gradient diagnostic. Split measurement cost from the Adam application.

**Rewrite:**

```tex
The loss can curve much more strongly in a few parameter directions than in most others~\cite{sagun2017hessian,ghorbani2019density}. Estimating the largest curvature usually requires repeatedly multiplying the Hessian by selected vectors. These Hessian--vector products avoid constructing the full matrix, but still require derivative computations beyond the training gradient. Measuring curvature along one chosen direction needs fewer such operations, while retaining an additional computational cost.

Bai et al.~\cite{bai2026spikes} use directional curvature to diagnose Adam loss spikes. Adam scales updates using a running average of squared gradients; their analysis identifies spikes that arise when this average fails to track current squared gradients. The curvature they measure accounts for Adam's scaling of parameter directions. It therefore differs from simply observing how the unscaled gradients change. Their additional derivative computations remain a limitation for monitoring larger models.
```

## II-12

**Read**

```tex
Consecutive-gradient products already serve other purposes. Chee and Toulis~\cite{chee2018diagnostics} study convergence diagnostics, while Pesme et al.~\cite{pesme2020diagnostics} identify limitations of the classical gradient-product test even on noisy quadratics. Hypergradient descent uses gradient information to adapt the learning rate~\cite{baydin2018hypergradient}. Barzilai--Borwein methods use parameter and gradient differences for step selection~\cite{barzilai1988two,tan2016bbsgd}; our step-direction scalar is the reciprocal of the classical BB1 step size when both are defined.
```

**Diagnosis:** The paragraph lists methods without explaining their tasks. Convergence diagnostics can mean several things, and BB1 is an unexplained variant name. The reciprocal relationship is important but arrives as a technical aside.

**Proposed solution:** Separate detecting a transition to stationary fluctuations from choosing a learning rate. Explain each purpose before naming its method, and identify BB1 as a standard step-size rule whose reciprocal is the quantity used here.

**Rewrite:**

```tex
Gradient comparisons can also help decide when to change the training procedure. Chee and Toulis~\cite{chee2018diagnostics} use products between consecutive gradients to detect when iterates have settled into fluctuations around an optimum. Pesme et al.~\cite{pesme2020diagnostics} show that the classical test can fail even for quadratic objectives with noisy gradients. These tests concern the stage of convergence, rather than whether the next update will increase the loss.

Other methods use gradient information to choose the learning rate. Hypergradient descent adjusts it through derivatives with respect to the learning rate itself~\cite{baydin2018hypergradient}. Barzilai--Borwein methods compare parameter changes with gradient changes to set a step size~\cite{barzilai1988two,tan2016bbsgd}. The curvature quantity we use along the applied step is the reciprocal of one standard rule, BB1, when both expressions are defined.
```

## II-13

**Read**

```tex
Recent work also obtains exact full-batch diagnostics without Hessian--vector products. Lee and Lee~\cite{lee2026road} compute a realized loss-balance quantity from the gradient, applied direction, and observed loss change. That quantity characterizes the completed update; it differs from directional curvature averaged along the step through a gradient difference.
```

**Diagnosis:** Realized loss-balance quantity is named without saying what is balanced. The final comparison names two abstractions instead of explaining the difference in their inputs and meaning.

**Proposed solution:** Explain Lee and Lee's quantity as comparing the observed loss change with the change predicted from the initial gradient and update. Then contrast the measured outputs: loss difference versus gradient difference, both after a completed step.

**Rewrite:**

```tex
Lee and Lee~\cite{lee2026road} obtain an exact full-batch diagnostic without Hessian--vector products by comparing the observed loss change with the change predicted from the initial gradient and applied update direction. Their quantity uses the loss measured after the update. Our curvature quantity instead uses the change in the gradient to recover curvature averaged along that update. Both describe a completed step, but they measure different aspects of its effect.
```

## II-14

**Read**

```tex
Our question is how much information about loss increases can be recovered from gradients already computed during training. Section~\ref{sec:displacement} connects the step-averaged quantity to angle and magnitude under GD, and explicitly analyzes sampling contributions. The full-gradient identity and quadratic GD loss criterion concern the completed step; Section~\ref{subsec:characterization} separately evaluates the gradient signal as a warning before the next Adam update.
```

**Diagnosis:** The opening question is clear and worth keeping. Step-averaged quantity, angle and magnitude, sampling contributions and gradient signal are abbreviated references that make the closing distinction harder to follow than necessary.

**Proposed solution:** Keep the question, identify the gradients whose size and direction change, explain sampling through replacing full gradients with batch gradients, and preserve the completed-versus-next-step distinction.

**Rewrite:**

```tex
Our question is how much information about loss increases can be recovered from gradients already computed during training. Section~\ref{sec:displacement} relates changes in gradient size and direction to curvature under GD, and examines how mini-batch sampling alters that relationship. The exact full-gradient relation describes the completed step. Section~\ref{subsec:characterization} then tests whether the observed changes help warn of a loss increase before the next Adam update.
```

## II-15

**Read**

```tex
Loss-landscape descriptions require an additional connection to what the network represents. Central Flows explicitly leaves architectural and learning-level mechanisms open~\cite{cohen2025central}. SVCCA and centered kernel alignment compare representations across layers, models, or training states~\cite{raghu2017svcca,kornblith2019similarity}. Here the measured objects are consecutive representation changes on fixed inputs: their direction, amplitude, dominant patterns, and shared temporal structure. This characterizes functional oscillation without identifying its causal source. Work on SGD catapults further shows that loss spikes can accompany feature learning~\cite{zhu2024catapults}, motivating evaluation of subsequent progress alongside spike suppression.
```

**Diagnosis:** Architectural mechanisms and a list of similarity-method names appear before the reader sees what is compared. Direction, amplitude, dominant patterns and temporal structure again form an unexplained list. The catapult sentence introduces the different issue of intervention consequences.

**Proposed solution:** Explain prior representation comparison by its inputs and purpose, acknowledge that it can follow training, and then specify our consecutive-change measurements. Move the catapult observation to the following intervention paragraph, where it motivates evaluating subsequent loss reduction.

**Rewrite:**

```tex
The next question is how these parameter updates change the responses inside the network. Central Flows identifies the need to understand how individual layers participate in the training dynamics~\cite{cohen2025central}. Existing methods compare representations across layers, models, and training states, allowing researchers to follow how those representations evolve~\cite{raghu2017svcca,kornblith2019similarity}. We examine the changes between consecutive states for the same inputs: whether they reverse direction, how large they are, and whether different layers change together over the training steps. This locates observed oscillation within the network without establishing which layer causes it.
```

## II-16

**Read**

```tex
Practical responses include checkpoint recovery, learning-rate reduction, and tighter clipping~\cite{chowdhery2022palm,zhang2022opt}. More directly related, Kosson et al.~\cite{kosson2024warmup} analyze update-induced representation changes and propose a correction in their study of warmup; adaptive gradient clipping controls unit-wise gradient norms relative to parameter norms~\cite{brock2021normalization}. These precedents motivate selective control beyond a global learning rate. Our caps use absolute local displacement budgets calibrated before intervention to test movement magnitude as a control target during ongoing oscillatory training.
```

**Diagnosis:** The list mixes recovery, clipping, warmup correction and the new cap without explaining their different actions. Warmup, unit-wise norms and calibrated budgets are unexplained. It also needs the previous paragraph's observation that suppressing spikes is not automatically beneficial.

**Proposed solution:** First explain why later training outcomes matter, using the relocated catapult citation, and describe conventional controls. Then explain warmup and two selective controls by what they limit. Describe our cap as fixed limits on direct local response changes, without implying a bound on all downstream changes.

**Rewrite:**

```tex
The effects of a loss spike matter when choosing a control. Studies of SGD catapults---large loss excursions followed by recovery---show that spikes can accompany improved feature learning~\cite{zhu2024catapults}. Practical responses nevertheless often aim to suppress spikes by restoring a checkpoint, reducing the learning rate, or limiting gradient size~\cite{chowdhery2022palm,zhang2022opt}. Evaluating such controls therefore requires measuring subsequent loss reduction as well as the fluctuations they remove.

Some controls act on individual parts of an update. Kosson et al.~\cite{kosson2024warmup} study changes in representations during learning-rate warmup, the initial period in which the learning rate is gradually increased, and propose a correction to those changes. Adaptive gradient clipping limits each unit's gradient relative to the size of its weights~\cite{brock2021normalization}. Our caps set limits before intervention and apply them to the direct response changes induced by proposed updates at fixed layer inputs. We test whether these limits improve training during ongoing oscillations.
```

## II-17

**Read**

```tex
The remaining questions are how oscillatory trajectories appear within the network and whether limiting local displacement improves training while reversals persist. Section~\ref{subsec:tracing} characterizes the functional motion, and Section~\ref{sec:vc-interventions} tests the intervention target through matched continuations. We begin with the parameter-space foundation: how step magnitudes and temporal interactions determine accumulated displacement, and how sampling affects their measurement.
```

**Diagnosis:** The closing paragraph repeats the questions and contribution framing, using functional motion, intervention target, matched continuations and parameter-space foundation as shorthand. Its needed job is to lead into Section III.

**Proposed solution:** Keep a brief pointer to the later measurements and intervention test, then state the next section's concrete task: relating individual steps to their combined parameter displacement under sampling.

**Rewrite:**

```tex
Section~\ref{subsec:tracing} measures these representation changes, and Section~\ref{sec:vc-interventions} compares interventions continued from the same training state. We first establish how individual parameter updates combine: when their directions reinforce or oppose one another, how much displacement remains after many steps, and how mini-batch sampling affects the measurements.
```

## I-04 (coherence review)

**Read**

```tex
We begin by separating how updates interact from how the loss changes. An update can reinforce part of an earlier parameter change or oppose it; persistent reinforcement and opposition define the constructive and destructive regimes studied here. For stochastic gradient descent (SGD) with a constant learning rate, an exact decomposition accounts for interactions across the entire sequence of updates. This matters because opposition between consecutive updates need not imply cancellation over the full trajectory. We also measure how much of the distance traveled remains as displacement from the starting parameters. Under Adam, the same gradient-based interpretation requires care: memory and scaling alter the updates actually applied.
```

**Diagnosis:** Reading the paragraphs together reveals that persistent reinforcement could be read as a property of all pairs of updates, whereas the manuscript's regime definition is based on consecutive gradients. The plain-SGD scope should precede that interpretation.

**Proposed solution:** Put constant-rate SGD before the regime explanation and explicitly attach constructive/destructive behavior to consecutive updates. Keep the separate all-pair decomposition and Adam qualification.

**Rewrite:**

```tex
We begin by separating how updates interact from how the loss changes. For stochastic gradient descent (SGD) with a constant learning rate, an update can reinforce or oppose part of the preceding parameter change. Persistent reinforcement or opposition between consecutive updates defines the constructive or destructive regimes studied here. An exact decomposition then accounts for interactions across the entire sequence: opposition between consecutive updates need not imply cancellation over the full trajectory. We also measure how much of the distance traveled remains as displacement from the starting parameters. Under Adam, the gradient-based interpretation requires care because memory and scaling alter the updates actually applied.
```

## I-05 (coherence review)

**Read**

```tex
Comparing gradients before and after an update also reveals how the loss's slope changed along that step. For a smooth, fixed objective and a nonzero update, gradients of the complete objective recover the average curvature along the step exactly, regardless of the optimizer. Under full-batch gradient descent (GD), this quantity depends on the gradients' relative directions and sizes. On a quadratic objective, sufficiently strong opposition between these gradients exactly characterizes a loss increase; for nonlinear objectives, the corresponding relation is a local approximation. Whether this information about a completed step warns of a loss increase on the next Adam step is tested separately.
```

**Diagnosis:** The simpler wording needs two precision repairs: recovering the exact step average requires the actual update as well as two gradients, and strong opposition could be mistaken for a sufficiently negative angle alone. The GD loss criterion also requires sufficient growth in gradient size.

**Proposed solution:** Name the actual parameter update as an input. State opposition together with sufficient gradient growth explicitly, retaining the smooth fixed-objective and completed-step conditions.

**Rewrite:**

```tex
Comparing gradients before and after an update reveals how the loss's slope changed along that step. For a smooth, fixed objective and a nonzero update, the two full gradients together with the actual parameter update recover the average curvature along the step exactly, regardless of the optimizer. Under full-batch gradient descent (GD), this quantity depends on the gradients' relative directions and sizes. For quadratic GD, opposition together with sufficient growth in gradient size exactly characterizes a loss increase; for nonlinear objectives, the corresponding relation is a local approximation. Whether this information about a completed step warns of a loss increase on the next Adam step is tested separately.
```

## I-04 (coherence review)

**Read**

```tex
We begin by separating how updates interact from how the loss changes. For stochastic gradient descent (SGD) with a constant learning rate, an update can reinforce or oppose part of the preceding parameter change. Persistent reinforcement or opposition between consecutive updates defines the constructive or destructive regimes studied here. An exact decomposition then accounts for interactions across the entire sequence: opposition between consecutive updates need not imply cancellation over the full trajectory. We also measure how much of the distance traveled remains as displacement from the starting parameters. Under Adam, the gradient-based interpretation requires care because memory and scaling alter the updates actually applied.
```

**Diagnosis:** The coherence pass confirms the scope, but the paragraph still explains the entire framework twice: first through the decomposition and again through its purpose. The contribution bullet supplies the detailed result.

**Proposed solution:** Keep the consecutive-update definition, whole-trajectory distinction and Adam qualification in fewer sentences. Retain net displacement as the concrete measurement.

**Rewrite:**

```tex
For stochastic gradient descent (SGD) with a constant learning rate, consecutive updates can reinforce or oppose one another. Persistent reinforcement or opposition defines the constructive or destructive regimes studied here. We account for interactions across the full sequence because opposition between consecutive updates need not imply cancellation over the trajectory. We also measure how much traveled distance remains as displacement from the starting parameters. Under Adam, gradient directions require a separate interpretation because memory and scaling alter the updates actually applied.
```

## I-05 (coherence review)

**Read**

```tex
Comparing gradients before and after an update reveals how the loss's slope changed along that step. For a smooth, fixed objective and a nonzero update, the two full gradients together with the actual parameter update recover the average curvature along the step exactly, regardless of the optimizer. Under full-batch gradient descent (GD), this quantity depends on the gradients' relative directions and sizes. For quadratic GD, opposition together with sufficient growth in gradient size exactly characterizes a loss increase; for nonlinear objectives, the corresponding relation is a local approximation. Whether this information about a completed step warns of a loss increase on the next Adam step is tested separately.
```

**Diagnosis:** The first sentence repeats the definition of curvature already established in paragraph 2. The final sentence can express the timing distinction more directly without losing its scope.

**Proposed solution:** Remove the repeated motivation and keep the inputs, assumptions, optimizer scope, joint direction-size condition and separate next-step test.

**Rewrite:**

```tex
For a smooth, fixed objective, two full gradients and the nonzero parameter update between them recover curvature averaged along that step exactly, regardless of the optimizer. Under full-batch gradient descent (GD), this quantity depends on the gradients' relative directions and sizes. For quadratic GD, opposition together with sufficient gradient growth exactly characterizes a loss increase; for nonlinear objectives, the corresponding relation is a local approximation. We separately test whether these observations about a completed step warn of a loss increase on the next Adam step.
```

## II-01 (coherence review)

**Read**

```tex
This section explains why oscillating updates can accompany loss reduction, what the gradients reveal about those updates, and how their effects can be measured inside the network. The literature provides the starting point for each question and helps delimit what our analysis and interventions test.
```

**Diagnosis:** The second sentence announces the function of a literature review without adding content. The first sentence can carry the full orientation.

**Proposed solution:** Retain only the three concrete explanatory tasks.

**Rewrite:**

```tex
We review why oscillating updates can accompany loss reduction, what gradients reveal about those updates, and how their effects can be measured inside the network.
```

## II-16 (coherence review)

**Read**

```tex
The effects of a loss spike matter when choosing a control. Studies of SGD catapults---large loss excursions followed by recovery---show that spikes can accompany improved feature learning~\cite{zhu2024catapults}. Practical responses nevertheless often aim to suppress spikes by restoring a checkpoint, reducing the learning rate, or limiting gradient size~\cite{chowdhery2022palm,zhang2022opt}. Evaluating such controls therefore requires measuring subsequent loss reduction as well as the fluctuations they remove.

Some controls act on individual parts of an update. Kosson et al.~\cite{kosson2024warmup} study changes in representations during learning-rate warmup, the initial period in which the learning rate is gradually increased, and propose a correction to those changes. Adaptive gradient clipping limits each unit's gradient relative to the size of its weights~\cite{brock2021normalization}. Our caps set limits before intervention and apply them to the direct response changes induced by proposed updates at fixed layer inputs. We test whether these limits improve training during ongoing oscillations.
```

**Diagnosis:** The two-paragraph rewrite now explains the controls, but several signposting sentences repeat their topic. Warmup can be defined within its name rather than through a long inserted clause. The cap's test is stated both here and in the next transition.

**Proposed solution:** Keep the definition of catapults, the reason to evaluate later loss, the warmup explanation, and the distinct control actions. Remove repeated signposting and the final restatement of the experimental question.

**Rewrite:**

```tex
SGD catapults---large loss excursions followed by recovery---can accompany improved feature learning~\cite{zhu2024catapults}. Controls such as checkpoint recovery, learning-rate reduction, and gradient clipping~\cite{chowdhery2022palm,zhang2022opt} should therefore be evaluated through subsequent loss reduction as well as the fluctuations they suppress.

Kosson et al.~\cite{kosson2024warmup} study representation changes during the initial learning-rate increase known as \emph{warmup}, and propose a correction to those changes. Adaptive gradient clipping limits each unit's gradient relative to the size of its weights~\cite{brock2021normalization}. Our caps instead limit the direct response changes induced by proposed updates at fixed layer inputs, using thresholds set before intervention. This tests movement control during ongoing oscillatory training.
```

## II-17 (coherence review)

**Read**

```tex
Section~\ref{subsec:tracing} measures these representation changes, and Section~\ref{sec:vc-interventions} compares interventions continued from the same training state. We first establish how individual parameter updates combine: when their directions reinforce or oppose one another, how much displacement remains after many steps, and how mini-batch sampling affects the measurements.
```

**Diagnosis:** The pointers to representation and intervention sections repeat the introduction roadmap and preceding paragraphs. Only the transition to the next section is needed here.

**Proposed solution:** End with the immediate analytical task, retaining the three quantities to be related.

**Rewrite:**

```tex
We first establish how individual parameter updates combine: when their directions reinforce or oppose one another, how much displacement remains after many steps, and how mini-batch sampling affects the measurements.
```
