# Paragraph-by-paragraph review

Each row records a reviewed prose unit in the initial full-paper pass at commit
`f05098a`, in reading order. Equation introductions and explanatory continuations
are tracked separately. Every unit was checked for a clear purpose, consistent
terminology, correct mathematics/evidence, and a useful transition. Retained
equations and captions were reviewed as well; see REVIEW.md. The subsequent
appendix shortening and its affected main-text passages are reviewed in
[APPENDIX_REVIEW.md](APPENDIX_REVIEW.md); the original source hashes in the JSON
record identify the initial pass, not the shortened version. The later
curvature-focused changes and review coverage are documented in
[CURVATURE_REVIEW.md](CURVATURE_REVIEW.md).

## 01-introduction.tex

| Unit | Purpose checked | Opening anchor |
| --- | --- | --- |
| 01 | Problem and practical stakes | `Neural networks often reduce training loss while successive gradients point in nea...` |
| 02 | Regime definition and central argument | `We approach this question through \emph{constructive and destructive regimes}. The...` |
| 03 | Prior work and contribution positioning | `Classical stability analysis relates step size to the Hessian spectrum, and stocha...` |
| 04 | Diagnostic framework and computation | `The formulation starts from an exact decomposition of SGD displacement into indivi...` |
| 05 | Empirical progression and principal counts | `We first characterize the regimes in controlled full-batch GD and mini-batch SGD e...` |
| 06 | Three contributions | `The paper makes three contributions:...` |
| 07 | Reader roadmap | `Section~\ref{sec:background} positions the diagnostics within stability and optimi...` |

## 02-related-works.tex

| Unit | Purpose checked | Opening anchor |
| --- | --- | --- |
| 01 | Complementary analytical perspectives | `Training dynamics reflect both the geometry encountered by an update and its inter...` |
| 02 | Parameter and gradient clock | `Let $\theta_t\in\mathbb R^p$ denote the parameters after update $t$ and $g_t$ the ...` |
| 03 | Full-gradient update Jacobian | `In full-batch GD, $g_t=\nabla\mathcal L(\theta_{t-1})$ for the training objective ...` |
| 04 | Quadratic contraction assumptions | `This matrix describes the response to a small parameter perturbation. For a fixed ...` |
| 05 | Oscillation versus amplification | `The component contracts without changing sign when $0<\eta\lambda_i<1$, alternates...` |
| 06 | EoS and prior trajectory geometry | `Deep-network trajectories repeatedly enter oscillatory regions. Cohen et al.~\cite...` |
| 07 | Fresh sampling decomposition | `Mini-batch sampling changes both the estimated direction and the trajectory follow...` |
| 08 | Finite-population covariance assumptions | `where the last equality assumes a uniformly sampled size-$b$ subset of $N>1$ examp...` |
| 09 | Variance interpretation | `Its trace measures the total squared dispersion around the full gradient. The fini...` |
| 10 | Break-even context and transition to regimes | `Curvature and stochasticity also interact. The break-even analysis of Jastrzebski ...` |
| 11 | Prior gradient-product and hypergradient diagnostics | `Successive gradient products have a substantial history in optimization diagnostic...` |
| 12 | Specific contribution beyond prior diagnostics | `Our focus is the anatomy of the regime that these temporal interactions reveal. We...` |
| 13 | Operational cost of loss spikes | `Large-model training reports show why early diagnostics matter. PaLM describes app...` |
| 14 | Directional information added to norm monitoring | `Modern recipes therefore monitor gradient norms alongside loss. OLMo~2 uses rollin...` |

## 03-methods.tex

| Unit | Purpose checked | Opening anchor |
| --- | --- | --- |
| 01 | Three diagnostic questions | `Our diagnostics answer three connected questions: which directional regime is the ...` |
| 02 | SGD displacement definition | `Consider the plain SGD update in Eq.~\eqref{eq:sgd_map_stability}, with constant l...` |
| 03 | Exact norm expansion | `The displacement is the net parameter change produced by $T$ updates. Expanding it...` |
| 04 | Magnitude and pairwise interaction meaning | `The terms $\\|g_t\\|^2$ are nonnegative and determine the squared lengths of individ...` |
| 05 | Pairwise and adjacent cosine definitions | `To isolate directional alignment from gradient magnitudes, define the cosine simil...` |
| 06 | Explicit regime and zero-norm definitions | `We call the adjacent-gradient regime \emph{constructive} when $\rho_t>0$ and \emph...` |
| 07 | Substitution into full displacement | `Substituting Eq.~\eqref{eq:rho} into Eq.~\eqref{eq:magnitude-interaction} gives...` |
| 08 | Local versus cumulative time horizon | `The exact sum retains interactions across every temporal lag. For given gradient m...` |
| 09 | Gradient Taylor expansion and sampling | `For mini-batch SGD, we explicitly incorporate stochastic gradient noise in Eq.~\eq...` |
| 10 | Remainder and projected equation | `where $r_t$ is the Taylor remainder. Neglecting $r_t$ recovers the first-order app...` |
| 11 | Directional curvature definition | `Positive curvature reduces the component of the next gradient along the preceding ...` |
| 12 | Curvature versus noise interpretation | `Thus, $\kappa_t$ measures how strongly a step changes the gradient along the direc...` |
| 13 | Cosine normalization | `Dividing Eq.~\eqref{eq:rho-expanded} by the two gradient norms gives...` |
| 14 | Directional sign threshold | `In full-batch GD, $\varepsilon_t=0$. Within this first-order approximation, $\eta\...` |
| 15 | Directional minimum and destructive regime | `The threshold at one also has an optimization interpretation. Along the full-gradi...` |
| 16 | Observable projected-change formula | `This interpretation leads to a measurable diagnostic. Rearranging Eq.~\eqref{eq:rh...` |
| 17 | Meaning and curvature approximation | `This diagnostic measures the change of gradient projected onto its preceding direc...` |
| 18 | Conditional noise expectation | `Sampling contributes to the regime through its interaction with the preceding nois...` |
| 19 | Negative expected noise contribution | `Thus, even unbiased sampling noise contributes a systematic nonpositive term to th...` |
| 20 | Exact finite-population variance | `This contribution is directly related to mini-batch size. Consider a mini-batch of...` |
| 21 | Batch-size interpretation | `At fixed parameters, smaller batches produce larger sampling variance and, consequ...` |
| 22 | Fixed-state concentrated-norm approximation | `Noise can weaken directional coherence by enlarging gradient norms relative to the...` |
| 23 | Same-state attenuation | `This attenuation moves either sign toward zero. At the same fixed state, the commo...` |
| 24 | Evolving trajectory qualification and experiment link | `Here the expected cosine is nonnegative; the approximation quantifies the dilution...` |
| 25 | Cumulative retention intuition | `The regime describes adjacent directions; normalized displacement describes their ...` |
| 26 | Path-length normalization | `For a nonzero accumulated path length under constant-step plain SGD, the denominat...` |
| 27 | All-lag retention identity | `Using Eq.~\eqref{eq:magnitude-alignment}, this ratio satisfies...` |
| 28 | Geometric examples and joint interpretation | `Thus $R_T$ incorporates interactions across all temporal lags. When $R_T$ is near ...` |
| 29 | Linear computation and storage cost | `These diagnostics are inexpensive in computational structure. For $p$ parameters, ...` |
| 30 | Loss-change definition and question | `A destructive step can still decrease loss after crossing its directional minimum....` |
| 31 | Signed loss contributions | `The magnitude term favors descent. The interference term reinforces it for positiv...` |
| 32 | Joint reversal/amplification condition | `The condition requires a destructive regime and sufficient gradient-norm growth: t...` |
| 33 | Curvature substitution | `Substituting Eq.~\eqref{eq:rho-normalized} with zero noise gives...` |
| 34 | Productive destructive interval and empirical transition | `Within the local quadratic model, $1<\eta\kappa_t<2$ is a destructive regime with ...` |

## 04-gd-experiments.tex

| Unit | Purpose checked | Opening anchor |
| --- | --- | --- |
| 01 | Purpose of the controlled sequence | `We first examine whether destructive regimes sustain descent and how their reversa...` |
| 02 | Matched objectives and rate comparison | `The comparison covers a tanh MLP on 5,000 CIFAR-10 images, ResNet-32 on all 50,000...` |
| 03 | Destructive descent in the loss curves | `Figure~\ref{fig:gd-loss} pairs loss with the projected gradient change $\eta\wideh...` |
| 04 | Diagnostic insight and appendix audit | `Consecutive gradients therefore expose an oscillatory regime that is productive ov...` |
| 05 | Regime transition and norm growth | `Figure~\ref{fig:gd-alignment-displacement} explains how the destructive regime cha...` |
| 06 | Retention and productive large steps | `The decreasing $R_t$ quantifies the cancellation associated with these reversals. ...` |
| 07 | Local destruction with longer-range reinforcement | `Local destruction coexists with reinforcement over longer horizons. The complete c...` |
| 08 | Fixed-endpoint lag interpretation | `To examine the temporal extent of reversal, we fix one endpoint gradient and compa...` |

## 05-sgd-noise-experiments.tex

| Unit | Purpose checked | Opening anchor |
| --- | --- | --- |
| 01 | Matched continuation and averaging design | `We vary batch size at initialization and after 2,000 GD updates in the same three ...` |
| 02 | Sampling changes the observed regime | `The smallest batches weaken the trained-state destructive regime in all three case...` |
| 03 | Equal-cosine contrasting outcomes and Adam transition | `For the Transformer at $b=8$, both starting states have mean cosine near $-0.26$, ...` |

## 06-adam-diagnostics.tex

| Unit | Purpose checked | Opening anchor |
| --- | --- | --- |
| 01 | Adam motivation and diagnostic equation references | `Adam provides a practical test of the central argument: can raw-gradient regimes r...` |
| 02 | First/second-moment mechanism | `Adam~\cite{kingma2014adam} averages this input through first and second moments,...` |
| 03 | Bias correction and adaptive step | `starting from zero. Bias correction gives $\widehat m_t=m_t/(1-\beta_1^t)$ and $\w...` |
| 04 | Why changing raw input matters | `with coordinatewise operations. The first moment carries information from precedin...` |
| 05 | Adaptive-path local expansion | `The connection to the loss landscape follows from the same local expansion used fo...` |
| 06 | Projected response and sign/magnitude interpretation | `Thus the diagnostic measures the gradient's projected response to the adaptive dis...` |
| 07 | Matched Adam setup | `We use the CIFAR-10/MLP, CIFAR-10/ResNet-32 and WikiText-2/Transformer objectives ...` |
| 08 | Productive trajectories interrupted by spikes | `Figure~\ref{fig:adam-overview} shows sustained loss reduction interrupted by trans...` |
| 09 | Largest-spike selection | `For each model, we select the largest recovered relative loss spike at its highest...` |
| 10 | Shared destructive rise and amplification | `All three spikes develop from a destructive raw-gradient regime. As the loss rises...` |
| 11 | Concrete early-warning chronology | `The Transformer gives a particularly clear early warning. At state 788, $\eta\wide...` |
| 12 | Constructive segments and shrinking destructive recovery | `Recovery reveals how the regime changes after amplification. In the displayed ResN...` |
| 13 | Case synthesis and population question | `These cases connect a visible loss spike to an earlier change in the gradient fiel...` |
| 14 | Fixed-baseline event intuition | `A spike is an excursion above a fixed pre-rise loss: the loss leaves that level, r...` |
| 15 | Anchor definitions and recursive event equations | `Write $\mathcal L_s=\mathcal L(\theta_s)$ for the loss after $s$ updates. For even...` |
| 16 | Search termination and loss/time statistics | `The first elevated loss occurs at $a_i+1$, and the earliest maximum defines $p_i$ ...` |
| 17 | Amplitude distribution and threshold purpose | `The distributions in Fig.~\ref{fig:adam-spike-distribution} show that the selected...` |
| 18 | Spike severity and recovery cost | `Table~\ref{tab:adam-spike-anatomy} exposes the cost of these excursions. Both mean...` |
| 19 | Population regime/amplification/timing evidence | `The regime and norm statistics identify the recurring anatomy. Pre-rise $\rho$ is ...` |
| 20 | Joint event descriptor motivation | `The preceding results suggest that spike severity depends jointly on the regime be...` |
| 21 | Signed coordinate and temporal meaning | `The horizontal coordinate preserves the sign of the pre-rise regime and weights it...` |
| 22 | Per-rate regression evidence and interpretation | `Figure~\ref{fig:adam-rho-norm-spikes} shows a clear association between this joint...` |
| 23 | Intervention rationale and matched-state comparison | `This anatomy gives a concrete rationale for testing interventions during the devel...` |
| 24 | Scope and prospective next step | `The present evidence establishes temporal warning signs and within-run association...` |

## 05-conclusion.tex

| Unit | Purpose checked | Opening anchor |
| --- | --- | --- |
| 01 | Unified diagnostic contribution | `Constructive and destructive regimes provide an interpretable view of neural-netwo...` |
| 02 | Adam warning and population evidence | `The Adam results establish the relevance of the same raw-gradient view along adapt...` |
| 03 | Practical consequence and future interventions | `The resulting framework combines inexpensive measurement with a concrete interpret...` |

## 06-loss-appendix.tex

| Unit | Purpose checked | Opening anchor |
| --- | --- | --- |
| 01 | Exact loss line integral | `The loss relation in Section~\ref{subsec:loss-connection} combines gradient magnit...` |
| 02 | Endpoint-gradient approximation | `Approximating the integral by the average of the full gradients at the two endpoin...` |
| 03 | Quadratic exactness and remainder order | `This approximation is exact for quadratic objectives. For nonlinear objectives, th...` |
| 04 | Endpoint noise substitution | `Using the stochastic gradient decomposition $g_t=\nabla\mathcal{L}(\theta_{t-1})+\...` |
| 05 | Cosine and norm factorization | `Substituting $g_{t-1}^\top g_t =\\|g_{t-1}\\|\\|g_t\\|\rho_t$ and factoring the preced...` |
| 06 | Main-equation closure | `which recovers Eq.~\eqref{eq:loss-curvature}....` |
| 07 | Projected-change substitution | `To connect this expression with directional curvature, recall from Eq.~\eqref{eq:r...` |
| 08 | Noise simplification | `Substituting this relation and using the definition of $\varepsilon_t$ in Eq.~\eqr...` |
| 09 | Curvature/regime interpretation | `Thus, the curvature-based and alignment-based expressions describe the same local ...` |

## 07-gd-diagnostics.tex

| Unit | Purpose checked | Opening anchor |
| --- | --- | --- |
| 01 | Architectures/objectives and GD controls | `The tanh MLP has two 200-unit hidden layers and minimizes half-squared error summe...` |
| 02 | Preprocessing and fixed normalization objective | `The MLP uses coordinate-wise standardization and ResNet uses channel-wise standard...` |
| 03 | Transformer targets/vocabulary/coverage | `For the Transformer, the corpus is arranged into 20 fixed streams and causal chunk...` |
| 04 | Precision and within-case interpretation | `Physical accumulation uses 1,000-image chunks for the MLP, the fixed 1,000-image g...` |
| 05 | Undefined metrics and smoothing | `Cosines with zero norms and the first unpaired gradient are undefined. Complete 50...` |
| 06 | Unsmoothed transitions | `Figure~\ref{fig:gd-raw-alignment} supplies the unsmoothed consecutive cosines unde...` |
| 07 | Temporal accounting definitions | `For the complete prefix of $T$ gradients, define the traveled distance and the tem...` |
| 08 | Positive total and adjacent destruction | `Table~\ref{tab:gd-temporal-accounting} reports these quantities at $T=3000$. The i...` |
| 09 | Retrospective loss surrogate | `For each gradient pair indexed by $t=2,\ldots,3000$, let $\delta\mathcal L_{t-1}:=...` |
| 10 | Normalized discrepancy definition | `The normalized absolute discrepancy is...` |
| 11 | Eligible gradient pairs and denominator | `The denominator is the magnitude of the first-order full-GD descent term. It is no...` |
| 12 | Sensitivity-threshold definition | `For the threshold sensitivity columns, define...` |
| 13 | Audit counts and finite-step interpretation | `A transition is eligible only when both $\|\delta\mathcal L_{t-1}\|>h_t(\tau)$ and $...` |
| 14 | Backward lag sources and replay scope | `The backward profiles use the largest GD rate in each model, with exact full-objec...` |
| 15 | Mini-batch estimator and repetitions | `The mini-batch comparison uses three initializations for each objective. The MLP r...` |
| 16 | Runtime bridge and hardware variation | `Cloud runs use PyTorch 2.8/CUDA 12.8. The Transformer uses RTX5090 and 512-sequenc...` |
| 17 | Spectral measurement coverage | `We measure the largest algebraic Hessian eigenvalue at states $0,100,\ldots,3000$ ...` |
| 18 | Directional/spectral comparison clock | `Maximum pointwise eigenvalue concerns the most curved local direction, whereas $\w...` |
| 19 | Adam precision/hardware/selection | `Adam uses the complete objectives and frozen initial tensors specified above. All ...` |
| 20 | Adam vector clock and undefined values | `At state $\theta_s$, the raw gradient is $g_{s+1}$ and the first moment $m_{s+1}$ ...` |
| 21 | Complete-event sample and aggregation | `Event extraction searches every loss state from initialization, using strict incre...` |
| 22 | Regression/KDE specification and source provenance | `The seven scatter fits use ordinary least squares with an intercept, separate coef...` |

## Main-file Appendix B

| Unit | Purpose checked | Opening anchor |
| --- | --- | --- |
| 01 | Indexing and smoothness assumptions | `We use the same indexing as Section~\ref{sec:displacement}: $g_{t-1}$ is evaluated...` |
| 02 | One-step Taylor expansion | `For $\theta_{t-1}-\theta_{t-2}=-\eta g_{t-1}$, Taylor expansion gives...` |
| 03 | Corrected curvature/alignment cross-reference | `Substituting $\nabla\mathcal{L}(\theta_{t-2})=g_{t-1}-\xi_{t-1}$ and $g_{t-1}^\top...` |
| 04 | Conditional noise products | `The first-order noise contribution combines with the noise contained in the square...` |
| 05 | First-order noise cancellation | `Thus, the contributions from noise variance cancel between the first-order terms w...` |
| 06 | Remaining second-order contribution | `Sampling contributes through the curvature-weighted second-order term. For a nonli...` |
| 07 | Exact two-step displacement | `The displacement over two SGD updates is exactly...` |
| 08 | Two-step Taylor expansion | `Expanding around the starting parameters $\theta_{t-2}$ gives...` |
| 09 | First-order projection identity | `The first-order projection can be written as...` |
| 10 | Cosine substitution and remainder scope | `Finally, $g_{t-1}^\top g_t=\\|g_{t-1}\\|\\|g_t\\|\rho_t$ yields a form of Eq.~\eqref{e...` |

## Title, abstract, contributions and captions

- Title: retains the central regime terminology and covers the Adam extension.
- Abstract: positive contribution statement, computation cost and checked quantitative evidence.
- Three contribution bullets: formulation, controlled characterization, Adam anatomy/early signs.
- All nine figure and five table captions: checked vector stream, state clock, sample, smoothing/aggregation, axes and interpretation. The Adam anatomy caption defines every column and event anchor.
- Keywords: align with optimization, stability, gradient alignment, SGD and Adam.

Completed review: **135 prose units**, plus title, keywords, contribution items and 14 captions.
