# Archived review-session draft

Archived on 5 November 2026 at the author's request. The active manuscript was restored to its pre-review state at commit `89ec0718690ef4a6ebef6f4eeea469ffe90b835a`.

- [Read the archived reviewed manuscript](reviewed-manuscript.pdf).
- [Recover its TeX, bibliography, figures, tables and supporting files](source-snapshot.zip).
- [File inventory and SHA-256 verification](manifest.json).

The ZIP preserves 52 files under their repository-relative paths, including the reviewed PDF and the three source files introduced during the review. It also includes existing external figure dependencies referenced by the saved sources. Extract into a separate directory; the manuscript entry point is `manuscript/draft/bare_jrnl.tex`. The usual installed LaTeX/IEEEtran dependencies are still required to rebuild.

The [synthesis and review records](../../draft/revisions/2026-11-05-submission-synthesis/README.md) and [experiment-priority reviews](../../draft/revisions/2026-11-05-experiment-priorities/README.md) remain preserved as historical records. They describe the archived revision, not the restored active manuscript. The full Git history is retained; nothing was reset or rewritten.

The restored active PDF is byte-identical to the saved pre-review PDF in `manuscript/draft/revisions/2026-11-05-human-input-adam/revised-manuscript.pdf`. Every pre-review non-revision manuscript source/asset matches commit `89ec071`; the review-only source additions were removed from the active tree after being verified in this archive. Original supplied previous-submission files and unrelated repository work remain untouched.
