My recommendation is:

**Build the section around how local gradient reversal, accumulated parameter movement, and loss reduction relate across timescales. Use the curvature estimate as the mathematical bridge between these observations.**

You already have an interesting result. The current organization buries it, and two interpretive gaps keep the section from making the strongest possible contribution.

I’ve prepared revised LaTeX files below. First, here is what I would change and why.

## 1. Lead with your strongest result

The most compelling statement in your current section is this:

**Within each of your three controlled learning-rate sweeps, the highest-rate run reaches the lowest training loss after 3,000 updates, despite nearly opposing consecutive gradients and net displacement of only 2.6%, 3.7%, or 5.4% of path length.**

That belongs in the opening paragraph.

It gives the reader a concrete question:

> **How does training continue making substantial progress when successive updates reverse so strongly?**

Your current opening—“We test how constructive and destructive regimes emerge…”—describes what you measured. An opening built around the result establishes why those measurements matter.

Keep the scope precise: these are the best final losses **among the tested rates, from the shared initialization for each architecture**. That is already a useful controlled comparison. You do not need to broaden it into a claim that larger learning rates generally optimize better.

### The contribution should go beyond showing that oscillations coexist with progress

That coexistence is established in the edge-of-stability literature. Cohen et al. documented short-timescale loss oscillations with continued longer-timescale descent, and Damian et al. developed a mechanism of self-stabilization. [arxiv.org](https://arxiv.org/abs/2103.00065?utm_source=chatgpt.com) Singh et al. also studied directionality at both the overall-trajectory and individual-update scales. [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37771cc0be272368102a37f202bb88d8-Abstract-Conference.html?utm_source=chatgpt.com)

Your opportunity is to make the relationship **quantitative and directly observable from training gradients**:

- How much movement cancels between neighboring updates?
- How coherently does the remaining movement accumulate?
- What does a small displacement-to-path-length ratio actually tell us?
- How does sampling change the alignment we observe?

Those questions give the section a stronger purpose than another demonstration of oscillatory descent.

## 2. Organize the section around four claims

I would use the title:

### **Gradient Reversal and Progress in GD and SGD**

Then organize it as follows.

| Subsection | Question it answers | Evidence to use |
|---|---|---|
| **The onset of sustained gradient reversal** | When does the trajectory enter an oscillatory regime, and how does consecutive alignment express that transition? | Loss, cosine, adjacent norm ratio, directional secant curvature |
| **Local reversal coexists with substantial loss reduction** | What happens to cumulative movement and the objective once reversal becomes strong? | \(R_t\), gradient norms, final training losses |
| **Reversal has structure across temporal lags** | Is negative consecutive alignment an isolated event or part of organized motion? | Odd/even lag correlations; ideally, an analysis of paired updates |
| **Observed alignment under mini-batch sampling** | How does the geometry appear when gradients are sampled? | Initialization/checkpoint continuations across batch sizes |

This preserves most of your existing figure sequence. It changes the role of the text: each subsection establishes a claim, rather than simply describing the next figure.

There is also a concrete structural omission: **the supplied experimental section promises a batch-size study but ends before discussing it.** The fourth figure has a caption, but no corresponding results subsection. Completing that argument is essential.

The main narrative should proceed as:

**Reversal emerges → reversal coexists with progress → the reversal has temporal structure → sampling changes the observed signature.**

The two places that need more care are the connection between reversal and cumulative displacement, and the interpretation of the sampling experiment.

## 3. The most valuable addition: measure what survives adjacent reversals

Your current lag subsection ends with:

> “These recurrences explain … why net progress survives repeated reversals.”

That goes beyond the evidence. The sequence

\[
v,-v,v,-v,\ldots
\]

has perfect negative odd-lag and positive even-lag alignment, yet makes zero net movement after every pair.

The lag plot establishes temporal organization. To connect that organization to cumulative movement, examine **the displacement remaining after each pair of updates**.

This can be done exactly.

### Collapse each pair of updates

For an even number of updates \(T=2J\), define

\[
h_j=g_{2j-1}+g_{2j}.
\]

Under your constant-step plain GD update,

\[
\theta_{2j}-\theta_{2j-2}=-\eta h_j.
\]

Thus \(h_j\) is the gradient sum that determines the actual movement over the pair. Components that reverse cancel within this sum.

Now define

\[
C_T=
\frac{\sum_{j=1}^{J}\|h_j\|}
{\sum_{t=1}^{T}\|g_t\|},
\]

and, when its denominator is nonzero,

\[
D_T=
\frac{\left\|\sum_{j=1}^{J}h_j\right\|}
{\sum_{j=1}^{J}\|h_j\|}.
\]

Then your existing normalized displacement factors exactly:

\[
\boxed{R_T=C_TD_T.}
\]

These two factors answer different questions:

| Quantity | Interpretation |
|---|---|
| \(C_T\) | How much path length remains after replacing every adjacent pair by its net displacement? |
| \(D_T\) | How coherently do those remaining pair displacements accumulate? |

This is the missing connection between your local cosine and your cumulative \(R_T\).

### Why this could materially strengthen the paper

Suppose the data show that \(C_T\) becomes very small, while \(D_T\) remains comparatively large. You can then make a much more informative empirical statement:

**The trajectory contains large alternating movements, but the smaller displacements remaining after each pair accumulate relatively coherently.**

That explains the geometry of how extensive immediate reversal can coexist with net movement.

If both \(C_T\) and \(D_T\) are small, you learn something different: even after combining neighboring reversals, substantial turning or cancellation remains over longer timescales.

Either result is more informative than simply reporting a small \(R_T\).

To connect this to objective reduction, show the same windows’ two-step loss changes,

\[
\mathcal L(\theta_{2j})-\mathcal L(\theta_{2j-2}).
\]

This would establish how the observed residual movement accompanies loss reduction. It would still be an empirical explanation of trajectory geometry; the algebra alone does not prove that the remaining movement must decrease the loss.

### This may require no new training

If you retained the unsmoothed cosine and norm logs, each pair norm is available from

\[
\|h_j\|^2
=
\|g_{2j-1}\|^2+\|g_{2j}\|^2
+
2\rho_{2j}\|g_{2j-1}\|\|g_{2j}\|.
\]

You already have the quantities needed for \(R_T\). Consequently, you can compute \(C_T\), then obtain \(D_T=R_T/C_T\), without recovering every gradient vector.

Use the raw observations, and check both pairing offsets so that the result does not depend on where the pairs begin.

**If I could add only one substantive analysis to this section, this would be my choice.** The decomposition itself is elementary; the contribution comes from what it reveals about the measured trajectories.

## 4. Give normalized displacement the right reference

There is another important refinement to your interpretation of \(R_T\).

A small displacement-to-path-length ratio does not by itself establish destructive cumulative interference. For equal-length orthogonal steps,

\[
R_T=\frac{1}{\sqrt T}.
\]

At \(T=3000\), that is approximately

\[
R_T\approx1.83\%.
\]

Your reported 2.6–5.4% values are therefore small relative to a straight trajectory, but their magnitude alone does not demonstrate that the **sum of interactions across all steps** is negative.

This does not undermine your observation of local reversal. The negative consecutive cosines establish that directly. It means that local reversal and cumulative interaction need to be distinguished.

### Add a reference using the observed step lengths

For the actual, unequal gradient norms, the zero-cross-term reference is

\[
R_T^{\perp}
=
\frac{\sqrt{\sum_t\|g_t\|^2}}
{\sum_t\|g_t\|}.
\]

You can add this as a reference curve to the existing \(R_T\) panel. That would immediately make the percentages more interpretable.

An equivalent scalar is

\[
Q_T
=
\frac{\left\|\sum_tg_t\right\|^2}
{\sum_t\|g_t\|^2}
=
1+
\frac{2\sum_{u<t}g_u^\top g_t}
{\sum_t\|g_t\|^2}.
\]

Its interpretation is exact:

- \(Q_T<1\): the total cross-step interaction is negative.
- \(Q_T=1\): the total cross-step interaction is zero.
- \(Q_T>1\): the total cross-step interaction is positive.

You can compute it from the existing displacement and norm records:

\[
Q_T=
\frac{\|\Delta_T\|^2}
{\eta^2\sum_t\|g_t\|^2}.
\]

This could reveal a particularly interesting result: **negative alignment between consecutive gradients may coexist with positive cumulative interaction across all lags.**

Do not claim that result before calculating it. But it is exactly the kind of distinction your current framework is well placed to investigate.

### What to change in the prose now

Keep “destructive” explicitly tied to **opposing parameter displacements**.

A small \(R_T\) describes the path’s geometry. It does not measure a fraction of useful learning retained, and it does not establish that the remaining path length was wasted computation. Likewise, evidence of opposing parameter updates alone does not establish overwriting of learned representations.

Your existing observation that the lowest-\(R_T\) runs reach the best final training losses is a strong reason to maintain that distinction throughout the paper.

## 5. Keep curvature central, but give each comparison the correct role

The curvature connection is useful. I would retain it prominently, while being precise about what constitutes independent evidence.

### A. Present cosine and secant curvature as linked descriptions

Your identity is

\[
\eta\widehat\kappa_t
=
1-\rho_t\frac{\|g_t\|}{\|g_{t-1}\|}.
\]

When consecutive gradients are nearly opposite and have similar norms,

\[
\rho_t\approx-1,
\qquad
\frac{\|g_t\|}{\|g_{t-1}\|}\approx1,
\]

the scaled secant curvature is necessarily close to two.

That is an informative interpretation of the oscillation. It should not be presented as a separate experimental confirmation of the same mechanism.

A stronger way to structure the discussion is:

1. Establish the observed gradient reversal.
2. Show how the norm ratio determines its secant-curvature interpretation.
3. Relate that interpretation to the local quadratic loss-change condition.
4. Use the Hessian checkpoints to situate the trajectory in its spectral context.

This makes the mathematics carry the argument.

### B. Distinguish the two thresholds

For nonzero consecutive gradients, the secant statistic’s threshold at one corresponds exactly to the sign change of consecutive alignment.

The threshold at two has a different status: it is the loss-change threshold under the quadratic or trapezoidal approximation. For nonlinear losses, its fidelity depends on variation along the step.

Accordingly, I would change:

> “close to the local loss-change boundary”

to:

> “close to the quadratic-model loss-change threshold of two.”

Also distinguish the individual-step statement from the displayed 50-observation trailing means. A smoothed curve near two characterizes a regime; it does not establish that each individual update lies at that boundary.

### C. Treat the spectral curves as context

There are three differences between your solid and dashed curves:

| Comparison | Secant diagnostic | Spectral checkpoint |
|---|---|---|
| Direction | Realized gradient direction | Maximizing direction |
| Location | Averaged along the step, under the smooth full-batch interpretation | At a parameter state |
| Objective | Full training objective in these GD runs | Subset objective for ResNet and Transformer |

Therefore, a numerical gap cannot be attributed solely to gradient orientation.

The current spectral measurements are still useful. They show how the observed transition relates to sharpness evolution. They do not directly validate the accuracy of a pointwise directional-curvature estimate.

If estimator fidelity remains a central paper claim, a small number of same-objective directional Hessian-vector evaluations would be a useful independent check. **That is a conditional addition, not a prerequisite for the trajectory-geometry section I am recommending.**

For the cost claim, “no additional gradient evaluations or Hessian-vector products” is precise. A claim of negligible overall overhead should be supported separately by memory and runtime measurements.

## 6. Complete the SGD subsection around a clear distinction

The batch-size experiment has a good design question: how does the observed alignment change when training continues from either initialization or a later checkpoint?

But the current files provide only the caption and protocol. They do not provide the results prose or several necessary setup details.

### State the observed result before explaining it

The subsection needs a paragraph that explicitly answers:

- Which models and starting states show positive versus negative alignment at the largest batches?
- Does reducing batch size move those observed cosines toward zero?
- Are there exceptions or meaningful differences across initialization blocks?

Write those answers directly from the figure or underlying data. Avoid a generic sentence that obscures differences between the architectures.

Then explain what the experiment isolates.

### The moving-trajectory experiment combines two effects

Changing batch size changes both:

1. The sampling variability in the gradient measurements.
2. The updates, and therefore the subsequent parameter states.

The existing experiment measures their combined effect during SGD. That is a legitimate result.

It cannot, by itself, establish that weaker sampled-gradient cosine means that the underlying full-gradient reversal has weakened.

This distinction could become a valuable contribution: **the visibility of a geometric pattern in sampled gradients can change separately from the underlying pattern.**

### One targeted control would make that distinction much stronger

Use two frozen parameter states from a GD trajectory:

1. Select an adjacent pair whose full gradients are positively aligned.
2. Select another pair whose full gradients are negatively aligned.
3. At each fixed pair, independently sample gradients at the two endpoints over the batch-size sweep.
4. Compare sampled cosine with the fixed full-gradient cosine.

Because the states are held fixed, this isolates the effect of sampling on the measurement.

Using **two different endpoints** matters: it lets you test attenuation of genuinely negative full-gradient alignment. Two gradients evaluated at one identical state cannot provide that negative-alignment reference.

Under independent sampling and concentration of gradient norms, the expected sampled cosine has the approximate form

\[
\mathbb E[\rho_{\mathrm{sampled}}]
\approx
\frac{\rho_{\mathrm{full}}}
{\sqrt{
\left(1+\frac{v_a}{\|\mu_a\|^2}\right)
\left(1+\frac{v_b}{\|\mu_b\|^2}\right)
}},
\]

where \(\mu_a,\mu_b\) are the two full gradients and \(v_a,v_b\) are their sampling-noise variances.

This gives a clean, testable expectation: either sign is attenuated toward zero as noise becomes large relative to the full gradients. The approximation should be tested rather than assumed.

The fixed-endpoint control and your existing moving-trajectory experiment would then answer complementary questions:

- **Fixed endpoints:** How does sampling change the observation of known geometry?
- **Moving trajectories:** How does the observed geometry evolve under SGD?

That is a focused explanatory experiment, with no need to introduce a new optimization algorithm.

### Fill in the missing methods details

Neither supplied file specifies:

- The learning rates used in the batch-size continuations.
- The learning rate used to produce each 2,000-update checkpoint.
- The number of repetitions within each initialization.
- Whether successive mini-batches are sampled independently or follow another sampling scheme.

Those details matter to the noise argument.

Your fixed normalization groups for ResNet are also important. When applying the finite-population variance formula, \(N\) and \(b\) need to count the actual sampling units consistently. If sampling is over fixed groups, an image count on the horizontal axis is not automatically the \(b\) in the theoretical expression.

## 7. Make several precise changes to the current wording

These are the most consequential sentence-level edits.

| Current wording or implication | Recommended change | Reason |
|---|---|---|
| “Smooth and oscillatory convergence” | “Smooth and oscillatory loss reduction” | The finite runs establish observed behavior, not asymptotic convergence. |
| “Destructive regimes therefore sustain convergence” | “Persistent local reversal coexists with substantial training-loss reduction” | This states the result without assigning reversal a causal benefit. |
| “The joint behavior directly exposes” the loss-rise condition | Say that alignment and adjacent norm changes supply the condition’s measured ingredients | Growth in a smoothed norm over many steps is different from the consecutive norm ratio in the condition. |
| “These recurrences explain … why net progress survives” | State that the lag plot establishes recurring structure; use paired displacements to quantify what survives | Alternation alone can also produce zero net motion. |
| Differences between \(\widehat\kappa\) and \(\lambda_{\max}\) explained mainly by orientation | Identify direction, step averaging, and objective differences together | All three vary in the current comparison. |
| Small \(R_T\) interpreted as retained learning | Keep its interpretation as net displacement divided by traveled distance | Learning and parameter-path geometry are different measurements. |

These changes make the section more persuasive because the reader can see exactly which conclusion follows from each observation.

I would also remove the final loss-excursion sentence from the displacement subsection. It pulls the argument away from the section’s strongest question before that question has been fully resolved.

## 8. Give each figure one main job

You do not need many more figures. You need clearer assignments for the existing ones.

### Figure 1: establish the transition and its curvature interpretation

Its main job is to show how the loss trajectory and directional curvature change with learning rate.

Keep the spectral measurements, but describe them as context. If you redraw the panel, include the threshold at one as well as two so the distinction between reversal and the quadratic loss-change condition is visually explicit.

### Figure 2: establish the separation between path geometry and objective reduction

This figure contains your strongest result.

Put the final \(R_T\) values and the within-sweep final-loss comparison in the caption. Add the zero-cross-term reference to the \(R_T\) panel if the raw norm logs are available.

If space allows one additional panel, I would prioritize the decomposition into \(C_T\) and \(D_T\). It directly answers the question raised by the combination of negative cosine and small \(R_T\).

### Figure 3: establish temporal organization

Keep the striking odd/even structure. However, a single endpoint at update 2256 should be described as a measurement at that endpoint.

To support a broader claim about persistent temporal structure, evaluate several prespecified endpoints with comparable lag histories, wherever retained gradients permit. A representative exact trace plus a summary across endpoints would be stronger than a single trace alone.

There is no need to turn this into a large new experimental grid.

### Figure 4: establish batch-size dependence of observed alignment

Its caption already contains useful information about aggregation. Preserve the distinction that the shading is the range across three initialization blocks.

The main text should report the actual model-specific trends and explain that these are moving trajectories. A compact frozen-endpoint control could then establish how much of the trend is attributable to the measurement itself.

## 9. What this would let the paper claim

With the current evidence, your defensible contribution is a controlled characterization that distinguishes:

- Consecutive gradient reversal.
- Recurring temporal structure.
- Cumulative parameter movement.
- Continued training-loss reduction.
- Batch-size dependence of observed alignment.

With the paired-displacement analysis, you can make a more explanatory contribution: **quantify where cancellation occurs and how the movement remaining after cancellation accumulates.**

With the frozen-endpoint sampling control, you can additionally distinguish **changes in the observation of gradient geometry from changes in the evolving trajectory**.

Those are meaningful scientific contributions. Their value does not depend on a predictive application.

The strongest version of this section would leave the reader understanding something they could not infer from “the loss oscillates near the edge of stability”: **how local reversals translate into cumulative movement, and when sampled-gradient measurements reveal or obscure that relationship.**

## Revised files

I prepared the following revisions:

- 04-gd-experiments.tex[04-gd-experiments.tex](sandbox:/workspace/scratch/233425c27361/revision/04-gd-experiments.tex) — reorganized section, finding-led opening, corrected interpretations, and the missing SGD subsection.
- 04-controlled-figures.tex[04-controlled-figures.tex](sandbox:/workspace/scratch/233425c27361/revision/04-controlled-figures.tex) — revised captions with clearer scientific takeaways and measurement qualifications.
- gd-sgd-references.bib[gd-sgd-references.bib](sandbox:/workspace/scratch/233425c27361/revision/gd-sgd-references.bib) — the three verified references used in the rewrite.

The draft preserves your existing figure paths and labels. It marks the missing batch-size results and setup details in author comments, and it does not present the proposed analyses as completed experiments. I checked the source structure and citation keys; full manuscript compilation and layout were not verified.

I couldn’t save the revisions back to the attached originals, so the links above provide the downloadable versions.

**My priority order would be: adopt the new argument structure, compute the paired-displacement decomposition and the reference for \(R_T\), then complete the SGD results paragraph. Add the frozen-endpoint control if you want to make a stronger attribution about sampling.** That is a focused route to a substantially stronger section.