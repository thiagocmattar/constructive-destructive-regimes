"""Apply the approved local prose/notation edits; record exact replacements."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parents[1] / 'draft'
changes = []
texts = {}


def replace(file, before, after):
    text = texts.setdefault(file, (HERE / 'scratch/baseline' / file).read_text(encoding='utf-8'))
    assert before in text, (file, before[:80])
    count = text.count(before)
    texts[file] = text.replace(before, after)
    changes.append(dict(file=file, before=before, after=after, occurrences=count))


replace('sections/05-adam.tex',
    'combination as Eq.~\\eqref{eq:kappa-estimate}. For the constant-rate Adam\ntrajectories considered here, we retain its notation,',
    'combination as Eq.~\\eqref{eq:kappa-estimate}. To distinguish this\nraw-gradient projection from step-direction curvature, define the\ndimensionless response $Q_t$. For the constant-rate trajectories here,')
replace('sections/05-adam.tex', r'\eta\widehat\kappa_t', 'Q_t')
replace('sections/05-adam.tex',
    'Under Adam, $Q_t$ describes gradient reversal and growth\nby projecting the gradient response onto the preceding raw gradient.\nThat direction generally differs from the applied update. Consequently,\nthis quantity is not generally the step-direction curvature\n$\\widehat\\kappa_t^{\\mathrm{step}}$ in Eq.~\\eqref{eq:kappa-step},\ndespite the retained notation.',
    'For plain GD, $Q_t=\\eta\\widehat\\kappa_t$. Under Adam, $Q_t$\nprojects onto the preceding raw gradient, which generally differs from\nthe applied update direction. It is therefore not the step-direction\ncurvature $\\widehat\\kappa_t^{\\mathrm{step}}$ in Eq.~\\eqref{eq:kappa-step}.\nThe latter requires the actual parameter increment.')
replace('sections/05a-characterization.tex', r'\eta\widehat\kappa_t', 'Q_t')
replace('tables/diagnostic-0p1pct.tex', r'\eta\widehat\kappa_t', 'Q_t')
replace('sections/s03-adam-audit.tex', r'\eta\widehat\kappa_t', 'Q_t')
replace('sections/05a-characterization.tex',
    'After computing the raw gradient, but before applying the next update,\nwe classify whether that update will increase the loss by at least\n0.1\\%.',
    'We evaluate fixed decision rules on recorded trajectories; no classifier\nis trained. After computing the raw gradient, but before applying the\nnext update, each rule warns whether that update will increase the loss\nby at least 0.1\\%.')
replace('sections/05a-characterization.tex',
    'although the gains in overall accuracy over always predicting no\nincrease are modest.',
    'although the gains in accuracy over always predicting no increase are\nmodest. The combined rule recalls 38.6\\%/23.9\\%/55.6\\% of the target\nincreases, so high lift does not imply complete coverage. Supplementary\nSection S3 reports warning counts and recall for every rule.')

replace('sections/s01-proofs.tex',
    'The Adam overview retains the former measured quantity and labels it $\\eta\\widehat\\kappa_t$; it is not relabeled as an empirical measurement of the latter.',
    'The main Adam section names the former quantity $Q_t=1-\\rho_t\\|g_t\\|/\\|g_{t-1}\\|$ to distinguish it from step-direction curvature. Retained intervention plates use the historical label $\\eta\\widehat\\kappa_t$ for this same raw-gradient response; their numerical values are unchanged.')
replace('sections/s01-proofs.tex',
    'The raw-gradient quantity $\\eta\\widehat\\kappa_t$ continues to describe gradient reversal and growth, with the distinct interpretation explained above.',
    'The raw-gradient quantity $Q_t$ continues to describe gradient reversal and growth, with the distinct interpretation explained above.')

# A per-unit RMS score has a different symbol from change cosine c_t^ell.
replace('sections/05c-interventions.tex', r'c_{t,j}', r's_{t,j}^{\ell}')
replace('sections/05c-interventions.tex', r'c^{\mathrm{Adam},\ell}_{t,j}', r's^{\mathrm{Adam},\ell}_{t,j}')
replace('sections/s05-intervention-methods.tex', 'c_j', 's_j')
replace('sections/s05-intervention-methods.tex', 'a_j', r'\gamma_j')
replace('sections/05c-interventions.tex',
    "Budgets remain fixed during treatment, while scores\nuse each site's current inputs.",
    "Budgets remain fixed during treatment: this is prefix calibration, not a\nrolling quantile. Current inputs, scores, and per-unit scaling factors\nevolve, but the budget does not.")
replace('sections/05c-interventions.tex',
    'The temporal SVD is an analysis tool and is not computed by the cap.',
    'The cap does not identify or suppress SVD modes: it limits direct\nper-unit changes. Concentrated functional movement motivates this test,\nbut need not be concentrated in a few neurons. No SVD is computed by\nthe cap.')
replace('sections/05c-interventions.tex',
    'shared starting state; $\\mathcal P$ sums all upward loss movement,\nrather than counting peaks.',
    'shared starting state; $\\mathcal P$ sums all positive consecutive loss\nchanges. It is neither the maximum loss nor a count of peaks.')
replace('sections/05c-interventions.tex',
    'Every tested setting is shown. One initialization is used per',
    'Every tested setting is shown; cap settings and unchanged Adam are highlighted, while conventional settings are gray with family-specific markers. Full values are in Table~\\ref{tab:vc-outcomes}. One initialization is used per')

replace('sections/05b-tracing.tex',
    'The columns $v_{\\ell,j}$ of $V_\\ell$ are orthonormal patterns of\nrepresentation change. A pattern specifies which coordinates change\ntogether and in what proportions. Its coefficient varies in magnitude\nand sign from one update to the next. The diagonal entries\n$\\sigma_{\\ell,j}$ of $\\Sigma_\\ell$ are nonnegative singular values,\nordered from largest to smallest, and $U_\\ell$ describes their temporal\nvariation. Specifically,',
    'The columns $v_{\\ell,j}$ of $V_\\ell$ are fixed orthonormal patterns\nacross input-position/feature coordinates. The nonnegative singular\nvalues $\\sigma_{\\ell,j}$ are ordered from largest to smallest; their\nproducts with entries of $U_\\ell$ give signed coefficients over time:')
replace('sections/05b-tracing.tex',
    'A \\emph{mode} is the right singular vector\n$v_{\\ell,j}\\in\\mathbb R^{P_\\ell}$: the $j$th fixed pattern at site $\\ell$\nover the whole window. Its coordinates span all reference input positions\nand features; its indices identify the site and pattern, not a sample\nor training step. Equivalently, it is an eigenvector of\n$D_\\ell^\\top D_\\ell$ with eigenvalue $\\sigma_{\\ell,j}^2$.\nThe rectangular matrix $D_\\ell$ itself need not have eigenvectors.\nAt most $\\min(T_w,P_\\ell)$ independent patterns can be observed.',
    'Each \\emph{mode} $v_{\\ell,j}$ is an eigenvector of\n$D_\\ell^\\top D_\\ell$ with eigenvalue $\\sigma_{\\ell,j}^2$. The coefficient\n$\\alpha_{t,j}^\\ell$, not the eigenvalue, changes sign as movement reverses\nalong that pattern. At most $\\min(T_w,P_\\ell)$ independent patterns can\nbe observed in the window.')
replace('sections/05b-tracing.tex',
    'Thus $E_k^\\ell=0.95$ means that reconstruction using these $k$ patterns\nretains 95\\% of squared movement, leaving 5\\% squared reconstruction\nerror. This quantity is often called cumulative energy, with\n``cumulative\'\' referring to patterns, not steps. We use squared movement\ninstead of variance because the change matrix is not centered: a\npersistent drift is part of the movement being measured. Row norms are\nalso retained, so larger changes contribute more.',
    'Thus $E_k^\\ell=0.95$ leaves 5\\% squared reconstruction error. This\ncumulative energy sums patterns, not steps. Because $D_\\ell$ is uncentered\nand its rows retain their norms, the measure includes persistent drift\nand weights larger changes more heavily; it is not centered explained\nvariance.')
replace('sections/05b-tracing.tex',
    'equating feature directions at different sites.',
    "equating feature directions at different sites. It is not the fraction\nof output oscillation caused by a site.")
replace('sections/05b-tracing.tex',
    'gradient decomposition. Concentration can reflect both repeated',
    'gradient decomposition. The output needs 5/20/15 patterns to capture\n95\\% of squared movement in the 51/61/49-change MLP/ResNet/Transformer\nwindows (Table~\\ref{tab:app-spectral}). Concentration can reflect both repeated')
replace('sections/05b-tracing.tex',
    'pattern, and its amplitude shows how strongly that pattern moves.',
    'pattern, and its amplitude shows how strongly that pattern moves. The\nsecondary axis overlays the complete-objective loss and marks its peak\non the same completed-step clock. This is a retrospective comparison\nwithin each fitted window.')
replace('sections/05b-tracing.tex',
    'A coefficient crossing zero reverses direction along that fixed pattern. Energy, temporal overlap, and signed motion answer different questions;',
    'Orange coefficients use the left axis; dashed gray complete-objective loss uses the right axis, with a dot marking its window maximum. A coefficient crossing zero reverses direction along that fixed pattern. Axes are independently scaled; the overlay compares timing within a retrospectively fitted window. Energy, temporal overlap, and signed motion answer different questions;')

replace('sections/03-regimes.tex',
    'For plain SGD, gradient sizes and directions determine how parameter\nupdates combine. We first distinguish opposition between consecutive\nupdates from cancellation over the full trajectory. We then relate\nconsecutive gradients to curvature and examine how mini-batch sampling\naffects these measurements. Finally, we derive a joint condition on\ngradient reversal and growth for a loss increase over the completed step,\nexact for full-batch quadratic GD and approximate for nonlinear losses.',
    'For plain SGD, we separate adjacent opposition from accumulated\ninteractions, connect consecutive gradients to curvature and sampling,\nand derive a completed-step loss criterion. The displacement identities\nare exact; the loss criterion is exact for full-batch quadratic GD and\napproximate for nonlinear objectives.')
replace('sections/03-regimes.tex',
    'The squared norms are nonnegative contributions from individual steps.\nIn full-batch GD, the negative gradient is the direction of steepest\nlocal descent, and a larger norm means a steeper local slope. The\ncross-step terms determine how these changes accumulate. Positive inner\nproducts reinforce displacement; negative inner products produce\n\\emph{destructive interference}, with one update undoing a component\nof another. Large individual steps can therefore coexist with limited\nnet parameter change.',
    'Squared norms contribute nonnegatively. Cross-step inner products\nreinforce displacement when positive and produce \\emph{destructive\ninterference} when negative. Large steps can therefore coexist with\nlimited net parameter change.')
replace('sections/03-regimes.tex',
    'length is retained in the network as net parameter displacement, while\n$\\rho_t$ describes the interaction between neighboring steps.',
    'length remains as net parameter displacement; it does not measure\nchange in parameter norm. The cosine $\\rho_t$ concerns neighboring steps.')

def paragraph(file, prefix, after):
    text = texts.setdefault(file, (HERE / 'scratch/baseline' / file).read_text(encoding='utf-8'))
    assert text.count(prefix) == 1, prefix
    start = text.index(prefix)
    end = text.find('\n\n', start)
    replace(file, text[start:end if end >= 0 else len(text)], after)


paragraph('sections/02-related-work.tex', 'The trace of',
    'The trace of $\\Sigma$ measures total gradient variability. Individual-gradient information is needed to recover it; the batch mean alone is insufficient. Resampling at fixed parameters changes the observed gradient without changing the full objective or Hessian, so sampled-gradient alignment alone does not establish a curvature change.')
paragraph('sections/02-related-work.tex', 'Back-and-forth motion also',
    'Path-length studies bound the total distance traveled by optimization trajectories~\\cite{gupta2021path}. We instead account for how update interactions determine the displacement that remains, distinguishing adjacent opposition from reinforcement or cancellation across the sequence.')
paragraph('sections/02-related-work.tex', 'The loss can curve',
    'Neural-network losses can curve much more strongly in a few parameter directions than in most others~\\cite{sagun2017hessian,ghorbani2019density}. Estimating the largest curvature typically requires repeated Hessian--vector products (HVPs); measuring a single direction requires fewer, but still adds derivative evaluations beyond the training gradient.')
paragraph('sections/02-related-work.tex', 'Bai et al.',
    "Bai et al.~\\cite{bai2026spikes} attribute Adam spikes to decoupling between the squared gradients and their running average. Their gradient-directional diagnostic uses an HVP with a preconditioned Hessian, and they identify its monitoring cost as a scaling limitation. Our raw-gradient response and step-average secant quantity do not measure that same preconditioned curvature; they provide different diagnostics without additional derivative evaluations.")
paragraph('sections/02-related-work.tex', 'Gradient comparisons can',
    'Chee and Toulis~\\cite{chee2018diagnostics} use consecutive-gradient products to detect fluctuations around an optimum. Pesme et al.~\\cite{pesme2020diagnostics} show that the classical test can fail even for noisy quadratic objectives. These convergence-stage tests address a different target from a loss increase on the next update.')
paragraph('sections/02-related-work.tex', 'Other methods use',
    'Hypergradient descent adjusts the learning rate through its derivative~\\cite{baydin2018hypergradient}. Barzilai--Borwein methods instead use parameter and gradient differences~\\cite{barzilai1988two,tan2016bbsgd}. Our step-direction quantity is the reciprocal of BB1 when both are defined. The secant formula itself is classical; here it connects update geometry, sampling, and loss diagnostics.')
paragraph('sections/02-related-work.tex', 'Lee and Lee',
    'Lee and Lee~\\cite{lee2026road} recover a full-batch loss-balance diagnostic from the initial gradient, applied direction, and observed loss change, without HVPs. Our secant quantity uses the gradient change to recover curvature averaged over the step. Both describe completed steps, but their measured quantities differ.')
paragraph('sections/02-related-work.tex', 'Our question is',
    'Section~\\ref{sec:displacement} relates consecutive-gradient direction and size to curvature and sampling. Section~\\ref{subsec:characterization} separately evaluates whether the measured response warns of a loss increase before the next Adam update.')
paragraph('sections/02-related-work.tex', 'We first establish how', '')

for file, text in texts.items():
    (DRAFT / file).write_text(text, encoding='utf-8', newline='\n')
(HERE / 'text-replacements.json').write_text(json.dumps(changes, indent=2)+'\n', encoding='utf-8')
print(f'Applied {len(changes)} explicit replacements.')
