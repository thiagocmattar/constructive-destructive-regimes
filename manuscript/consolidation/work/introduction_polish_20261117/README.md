# Minor introduction polish

17 November 2026. The author requested rollback of the broader replacement and a minor polish of their own wording, including resolution of the bracketed comments. Restored the author's three paragraphs from `introduction-before.tex`, made local grammar/spelling corrections, and resolved all five bracketed notes above `%%---- stop here`:

- Loss-spike reference: Bai et al.
- Controls: replaced mismatched normalization/regularization citations with learning-rate scheduling, clipping, and Adam second-moment references. Removed the unsupported suggestion of dynamic tuning as an established practice.
- Adam: named the optimizer and cited Kingma and Ba.
- Landscapes: cited Li et al. and added [Ly and Gong (2025)](https://www.nature.com/articles/s41467-025-58532-9), qualifying multifractal structure as reported in studied settings.
- Author argument: made the connection to representation learning explicitly the authors' argument, through changes in internal responses.

The original order, motivation, and paragraph structure remain. Every revised paragraph occupies one source line. Everything from `stop here` onward is byte-identical to the author's draft. Other scientific sources, tables, figures, abstract, and supplementary PDF are unchanged.

The existing main PDF is locked, so the current compiled output is [manuscript-polished.pdf](../../draft/manuscript-polished.pdf): 20 pages, 44 numbered equations, no compilation warnings. Rendered pages were inspected for reflow. [Verification](verification.json) records scope and hashes. The superseded broad rewrite and its alternate PDF remain untracked because automatic approval review blocked the cleanup deletion; they are not the current revision.

```powershell
python manuscript/consolidation/work/introduction_polish_20261117/01_build.py manuscript
python manuscript/consolidation/work/introduction_polish_20261117/02_verify.py
```

The build writes `manuscript.pdf` when available and uses `manuscript-polished.pdf` if it is locked. No experiments were run.
