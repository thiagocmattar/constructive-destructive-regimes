# GD formulation report

The [English technical report](gd-formulation.pdf) develops the supplied
`formulation.md` through the empirical full-batch GD result, followed by a compact
section extending the identities to Adam momentum and controlled Rosenbrock
results. The prose follows [writing-style.md](writing-style.md):
formal explanation, consistent terminology, defined assumptions, and a progression
from identities to mechanisms and observations. The supplied Markdown inputs and
the existing main manuscript are preserved.

Editable source: [gd-formulation.tex](gd-formulation.tex).
The [build script](00_build.py) generates the
[results table](gd-results.tex) from frozen Analysis019
results, compiles two LaTeX passes, checks equations and references, and renders
every page. The [manifest](gd-formulation-manifest.json) records input, source,
evidence, figure, PDF and render hashes and the completed visual review.
Compilation intermediates and page renders remain in ignored
`manuscript/artifacts/2026-10-11-restatement/`.

## Argument and evidence

The report develops constant-step SGD displacement, the exact lag-one and
longer-lag split, a local curvature/noise model of consecutive alignment,
the unbiased-noise expectation, and full-prefix normalized displacement B/A.
The empirical section uses only GD Runs059-062 from
[Analysis019](../../analyses/019-2026-10-13-cifar10-eos-lr/README.md).
It includes a compact publication figure and one results paragraph connecting
the transition in consecutive-step interference to the curvature diagnostics,
cumulative cancellation and unexplained gradient-norm growth. The generated
results table remains a separate source asset.
The figure is a presentation revision of all 24 original unsmoothed curves;
source values, sampling and original figure bytes are unchanged. Its local
[observation](../../analyses/019-2026-10-13-cifar10-eos-lr/observations/004-gd-paper-figure.md)
records sources, reductions, coverage and limitations. No new training occurred.

The Adam section defines the adaptive direction and exact displacement
expansion, then derives consecutive-momentum alignment using stored first moments
and a first-order curvature/noise model. Normalized momentum accumulation uses
B/A consistently with SGD and is distinguished from actual Adam displacement.
Bias correction preserves adjacent cosine but can change cumulative accumulation.
The following Rosenbrock section reuses immutable Runs055-057 from Analysis017,
with two new publication figures: complete trajectories on objective contours
and an eight-panel diagnostic view. It includes the momentum-curvature estimate
obtained by rearranging Eq.25 with its tracking term retained. Stored m is used
throughout. Cosines and both projection estimates have complete 100-pair trailing
means; other series are unsmoothed. Original figures and trajectories are preserved.
The [local observation](../../analyses/017-2026-10-12-rosenbrock-kappa-identity/observations/003-manuscript-figures.md)
records reductions and the AdamW decay/finite-step interpretation limits.
The brief noise subsection adds Gaussian error to the analytic gradient at
sigma0/0.01/0.1/1/10 with one paired seed1234. A compact six-panel sigma comparison
reuses all 90 original scalar values from Analysis014; see its
[observation](../../analyses/014-2026-10-10-rosenbrock-gradient-noise/observations/008-paper-sigma-comparison.md).

The mathematical and empirical wording makes these distinctions explicit:

- Gradient energy is nonnegative; temporal interactions determine net squared
  displacement. Negative adjacent pairs need not make the all-pair sum negative.
- The sign conditions neglect a Taylor remainder. Directional curvature is a
  Rayleigh quotient, while the full-batch gradient identity measures a finite-step
  secant. Neither is generally the maximum Hessian eigenvalue.
- Fresh conditionally unbiased noise has a nonpositive expected contribution to
  the unnormalized adjacent inner product. This does not determine the expected
  normalized cosine.
- Small B/A means small net motion relative to path length. In this cohort,
  stronger cancellation coexists with lower final loss at larger learning rates.
- Oscillation across a basin is an interpretation; a spatial spiral and
  representation overwriting are not established by these scalar diagnostics.

The requested learning-rate-halving TODO remains after the GD discussion.
It proposes an intervention and does not report an executed result. The report
cites the Edge of Stability paper and its official released implementation.

## Rebuild and review

From the repository root, with retained verified Analysis019 arrays and installed
NumPy, Matplotlib, PyMuPDF, MiKTeX and Poppler:

```powershell
python analyses/019-2026-10-13-cifar10-eos-lr/11_plot_gd_paper.py
python analyses/017-2026-10-12-rosenbrock-kappa-identity/05_plot_paper.py
python analyses/014-2026-10-10-rosenbrock-gradient-noise/04_plot_sigma_paper.py
```

Inspect the current Rosenbrock Poppler renders under Analysis017 and Analysis014 `scratch/`,
then record their review and build the manuscript:

```powershell
python analyses/017-2026-10-12-rosenbrock-kappa-identity/05_plot_paper.py --accept-reviewed
python analyses/014-2026-10-10-rosenbrock-gradient-noise/04_plot_sigma_paper.py --accept-reviewed
python manuscript/2026-10-11-restatement/00_build.py
```

The builder checks that record plus PDF, script, source and derived-history hashes.
Inspect every current report-page and standalone-figure render before recording
the manuscript review:

```powershell
python manuscript/2026-10-11-restatement/00_build.py --accept-reviewed
```

MiKTeX may need access to its user configuration/cache directories outside the
repository. Automatic package installation and shell escape are disabled.
The builder checks a deterministic quadratic/noise example and displacement
identities, including the Adam momentum model and bias-correction distinction,
independently checks table values against retained NPZ arrays,
rejects unresolved references or overflowing LaTeX boxes, and confirms that
the two supplied Markdown inputs remain byte-identical. These mechanics checks
are distinct from the frozen experiment's empirical evidence.
