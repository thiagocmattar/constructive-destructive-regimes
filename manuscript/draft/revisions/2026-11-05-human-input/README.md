# Author-input revision, 5 November 2026

The supplied `input.md`, `04-gd-experiments.tex`, and
`04-controlled-figures.tex` are preserved verbatim. They are proposals; the
active edited sources live in `manuscript/draft/sections/`.

Completed: evidence audit, Analysis056's retained-data reductions, manuscript
integration, numerical/source checks, compilation and rendered-page review.
The current `bare_jrnl.pdf` and [review copy](revised-manuscript.pdf) are identical
16-page builds. The baseline source and figure hashes are in
`baseline_manifest.json`. Prior revision PDFs remain unchanged.

The proposed SGD omission is absent from the current working draft: its separate
input file already supplies the methods and results. This revision improves
that subsection rather than inserting an empty author placeholder.

The optional frozen-two-endpoint sampling experiment and same-objective HVP
check would need new measurements. This revision uses the existing evidence
and explicitly limits sampling attribution and curvature-validation claims.
It does not launch training or incur cloud costs.

## Disposition of the supplied input

| Input | Applied change and evidence |
| --- | --- |
| 1. Lead with the strongest result | Section IV opens with the highest-rate within-sweep final-loss comparison and 2.6--5.4% displacement ratios. One-initialization scope is explicit. Existing Cohen/Damian keys are retained and the verified Singh 2025 record is added. |
| 2. Four-claim organization | The GD/SGD section now proceeds through reversal onset, loss reduction and paired motion, temporal lags, then moving-SGD observations. The existing separate SGD input remains compiled as subsection IV-D. |
| 3. Paired-displacement analysis | Analysis056 computes C and D from all raw observations, with both pairing offsets. The shifted partition keeps boundary singletons so it still represents the same full path. Figure 2 adds the factors, Table II gives all nine endpoints, and Figure 10 shows both offsets' late-window loss changes. |
| 4. Proper displacement reference | Section III defines R_perp and Q; Figure 2 adds the reference. All nine endpoint Q values exceed one. Seven have negative cumulative adjacent interaction, so local reversal and total interaction have different signs. |
| 5. Curvature interpretation | Figure 1 shows thresholds one and two separately. The text separates the exact secant/alignment identity, quadratic loss approximation, smoothing and spectral direction/location/objective differences. No independent fidelity or negligible-overhead claim is inferred. |
| 6. SGD results and protocol | Existing 546-trajectory evidence is reaggregated independently. The text gives model-specific means, all-block sign checks, reference/continuation rates, repetitions, independent subset sampling, fixed ResNet groups and token-weighted Transformer scope. It treats changes as combined sampling/trajectory effects. |
| 7. Sentence-level corrections | Removed causal/convergence wording, representation-overwriting claims and the assertion that odd/even recurrence explains nonzero progress. The abstract, selected introduction/contribution paragraphs, Section III and conclusion now agree with the revised distinctions. The original opening and Adam empirical section are preserved. |
| 8. Figure assignments | Three main GD figures are regenerated from retained evidence; the SGD PDF is unchanged. All nine existing backward endpoints support the added lag summary, while the exact 2256 trace remains. One supplementary loss figure is added. |
| 9. Scope of the contribution | The paper quantifies cancellation and residual accumulation, distinguishes them from objective decrease, and keeps the sampling interpretation conditional. It claims no new controller, generalization result or causal benefit from reversal. |

The optional fixed-two-endpoint sampling experiment would isolate measurement
attenuation, but is not needed for the combined-effect claim retained here.
The same-objective HVP proposal would independently test estimator fidelity;
the revised paper uses existing spectra as context and does not claim that test.
Neither optional measurement is represented as completed. The proposed extra
bibliography file was not present in the supplied folder; its three keys were
resolved against existing records and the primary proceedings.

## Reproduction and verification

From the repository root:

```powershell
python analyses/056-2026-11-05-gd-paired-displacement/01_analyze.py
python analyses/056-2026-11-05-gd-paired-displacement/02_verify.py
python analyses/056-2026-11-05-gd-paired-displacement/03_plot.py
python manuscript/draft/revisions/2026-11-05-human-input/01_build_and_verify.py
```

Close the current PDF tab before replacing it if Reader holds a write lock.
During this task only that tab was closed; other documents remained open.
The build uses pdflatex, BibTeX and two further pdflatex passes, with shell
escape disabled. Poppler renders every page into ignored scratch.

The [analysis observation](../../../../analyses/056-2026-11-05-gd-paired-displacement/observations/001-paired-displacement.md)
records sources, reductions, coverage, captions and limits. Checks cover 43
source hashes, 65 synthetic vector cases, 26,991 high-precision pair-norm
comparisons, 2,304 backward pairs and all 546 moving-SGD continuations.
The new plots have 99 curves / 191,826 values; 133,668 retained-curve values
agree with their prior inputs to maximum absolute difference 8.89e-16.
The previous 240 quadratic loss checks pass. Thirty-eight displayed scalar
checks, all-rate endpoint tables and the pairing/lag claims pass.

All 16 pages and all four new figures were visually reviewed. The six pages
changed by the final prose/reference-balance pass were inspected again.
There are ten figures, two tables, no unresolved references/citations and no
overfull boxes. The final page has balanced reference columns. Input proposals,
previous figure files, the Adam source and the loss appendix retain their
baseline hashes. `verification.json`, `evidence_checks.json` and `review.json`
bind the source, checks, current PDF and visual review.

Primary reference checks: [Cohen et al.](https://arxiv.org/abs/2103.00065),
[Damian et al.](https://arxiv.org/abs/2209.15594), and
[Singh et al.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37771cc0be272368102a37f202bb88d8-Abstract-Conference.html).
The new record supports only the stated prior trajectory/update-scale
directionality work, not the paired-factorization empirical findings.
