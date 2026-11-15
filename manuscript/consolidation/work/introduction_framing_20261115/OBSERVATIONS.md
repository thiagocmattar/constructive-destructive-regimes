# Introduction revision: contribution framing

15 November 2026. Author-authorized implementation of the surgical introduction
plan based on [CONTRIBUTION_FRAMING.md](../../CONTRIBUTION_FRAMING.md).

## Scope and result

Revised only the [introduction source](../../draft/sections/01-introduction.tex)
and the draft README, and rebuilt the [main manuscript](../../draft/manuscript.pdf).
The PDF remains 22 pages, with the introduction ending on page 2. The abstract,
Section II onward, bibliography source, tables, figures, original evidence and
supplement remain byte-for-byte unchanged. Citation numbering and early-page
text flow change automatically with the revised introduction.

The revision preserves the existing progression from practical motivation to
stability literature, functional gap, analytical framework, experiments and
contributions. It sharpens the gap and consolidates five contributions into four:

1. **Framework:** local reversal, cumulative interference and displacement
   retention; explicit sampling effects rather than an inferred curvature
   reduction. The key empirical result is coexistence of negative adjacent
   interactions, positive accumulated interactions and loss reduction.
2. **Diagnostic:** the two former curvature/loss bullets are merged. Full-gradient
   step-averaged curvature, the quadratic GD retrospective criterion and the
   empirical Adam outgoing-step warning remain distinct. Barzilai--Borwein is
   credited as a secant antecedent. The accuracy/lift result is stated for the
   evaluated rules, three full-objective trajectories and 0.1% event threshold;
   the equivalent projected-response rule is explicitly a tie.
3. **Representations:** Central Flows motivates the functional characterization.
   Fixed-input changes distinguish reversal, amplitude and temporal structure.
   Introductory mode counts are removed, while the original Section V-B results
   remain intact. Measurement does not establish a causal source layer.
4. **Intervention target:** caps test local representation displacement as a
   promising control target. Matched improvements relative to unchanged Adam
   and persistence of reversals remain the empirical claims; endpoint and
   computational tradeoffs retain their qualification.

The opening is shortened, repeated procedural detail is removed, and the
section roadmap is retained. The source whitespace-token count falls from 1,102
to 911 (17.3%); this is a consistent source-length measure, not a typeset word
count. Neither the Q25 spectral follow-up nor the implicit-regularization
hypothesis is added. No new experiments or Section II restructuring were done.

## Claim support and boundaries

- The gap and literature interpretation follow the discussion record above,
  particularly its Central Flows, Adam-spike and Barzilai--Borwein passages.
  Existing bibliography entries are reused; no new literature-survey claim is
  introduced.
- The decomposition, smooth fixed-objective identity and noise qualifications
  follow [Section III](../../draft/sections/03-regimes.tex). Raw-gradient cosine
  has an applied-update interpretation for plain SGD; Adam is explicitly
  distinguished. The identity requires a nonzero applied update.
- The local-versus-accumulated interaction result and sampling evidence are in
  [Section IV](../../draft/sections/04-gd-sgd.tex).
- The diagnostic result is checked against every evaluated row of
  [Table II](../../draft/tables/diagnostic-0p1pct.tex), with timing and equivalent
  rules defined in [Section V-A](../../draft/sections/05a-characterization.tex).
  No universal prediction guarantee, independent validation of equivalent
  formulas, or general optimizer superiority is claimed.
- The representation and intervention bullets retain the accepted bounded
  findings in the framing record. They do not claim increased capacity,
  newly discovered causal mechanisms, or a new general-purpose optimizer.

## Verification and reproduction

[Build record](build.json) and [verification](verification.json) record the final
PDF hash, source hash, scope checks, references, contribution count and diagnostic
table comparisons. The build runs pdflatex, BibTeX and two further pdflatex passes;
it rejects warnings, overfull/underfull boxes and unresolved references. The
final introduction is reviewed at page scale, followed by all 22 pages in
contact sheets to check reflow, floats and references.

```powershell
python manuscript/consolidation/work/introduction_framing_20261115/01_build.py
python manuscript/consolidation/work/introduction_framing_20261115/02_verify.py
```

Before/after PDFs, the original introduction, baseline hashes, build logs and
renders are retained in ignored `scratch/`. Scope verification requires those
local baseline manifests. Pre-existing unrelated working-tree edits are
preserved and excluded from the revision commit. The main compiled PDF is tracked in this consolidation draft and is committed
alongside the revised source and build record.
