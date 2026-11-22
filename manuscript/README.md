# Journal manuscript

[Read the paper](bare_jrnl/bare_jrnl.pdf) or [edit the LaTeX source](bare_jrnl/bare_jrnl.tex). The journal version comes from the November consolidation. Its [sections](bare_jrnl/sections/), [figures](bare_jrnl/figures/), [tables](bare_jrnl/tables/), and [bibliography](bare_jrnl/references.bib) are kept together.

From `manuscript/bare_jrnl/`, run `latexmk -pdf -interaction=nonstopmode -halt-on-error bare_jrnl.tex`.

- [45-minute presentation](bare_jrnl/presentation/presentation.pdf) and [speaker notes](bare_jrnl/presentation/speaker-notes.pdf). The [presentation guide](bare_jrnl/presentation/README.md) covers editing, rebuilding and figure sources.
- [Supporting evidence](bare_jrnl/evidence/README.md) contains the figure and table data and the records used to assemble the paper. Dates and paths in frozen manifests refer to the original layout.

## Earlier versions

The local [archive](archive/) is Git-ignored. The previous journal source is in [archive/draft/](archive/draft/). The eight dated review, proposal and research folders keep their original names directly under `archive/`, alongside the original submission ZIP, source inventory, and [former manuscript guide](archive/README.md).

The [consolidation snapshot](archive/consolidation/) preserves the full workspace as it stood at promotion, including the annotated PDFs, author notes, supplement, intermediate edits, and figure-generation scripts. To resolve an old path beginning with `manuscript/consolidation/` or `manuscript/draft/`, insert `archive/` after `manuscript/`. Previously tracked originals are also available at commit `de5e602`. Local-only feedback and build files are preserved in this checkout's archive.
