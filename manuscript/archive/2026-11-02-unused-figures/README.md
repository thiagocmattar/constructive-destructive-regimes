# Unused manuscript figures, archived 2026-11-02

The user requested a cleaner active figure directory. These 26 files were not
included by the active `manuscript/draft/bare_jrnl.tex` input tree. They are
preserved byte-for-byte under their original relative `figures/` paths.
[manifest.json](manifest.json) records original and archived locations, sizes,
SHA-256 hashes, the five active figures at archive time and the traversed TeX sources.

Four references in inactive legacy sections now point here, so those sections
still resolve their figures when compiled from `manuscript/draft`. Their prose
is unchanged. Supplied original assets and historical experiment figures are
preserved, not deleted. Original manuscript ZIP/review sources remain intact.

Only compiled figures belong in `manuscript/draft/figures/`. The GD/SGD generator
remains under `draft/revisions/2026-11-02-feedback/`; the Adam revision uses
`draft/revisions/2026-11-02-adam-feedback/` and adds two compiled figures from
Analysis036. Its verifier checks the seven current input PDFs using the LaTeX
recorder, alongside the historical archive hashes. Older analysis-specific
generators target historical layouts and may repopulate obsolete files there.
To restore an asset deliberately, copy it to
its manifest-listed original path and update the active inventory and TeX use.

The 26 th asset is the superseded MLP-only compact SGD draft copy. Analysis037
replaces it with the completed three-case figure; its original Analysis027
source PDF and experiment evidence remain unchanged.
