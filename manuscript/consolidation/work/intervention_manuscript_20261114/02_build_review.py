"""Compile and render the consolidation-only V-C excerpt after packaging layouts."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import fitz

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
DRAFT = BASE / "draft"
BUILD = HERE / "scratch/latex"


def run(command, name, cwd=DRAFT):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (BUILD / name).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        print((result.stdout + result.stderr)[-5000:])
        raise RuntimeError(f"{command[0]} failed: {result.returncode}")


BUILD.mkdir(parents=True, exist_ok=True)
command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={BUILD}", "intervention-review.tex"]
run(command, "pass1.txt")
run(["bibtex", str(BUILD / "intervention-review")], "bibtex.txt")
run(command, "pass2.txt")
run(command, "pass3.txt")
log = (BUILD / "intervention-review.log").read_text(encoding="utf-8", errors="replace")
warnings = [line.strip() for line in log.splitlines() if re.search(r"Overfull|Underfull|undefined|multiply defined|LaTeX Warning", line)]
target = DRAFT / "intervention-review.pdf"
shutil.copyfile(BUILD / "intervention-review.pdf", target)
render = HERE / "scratch/review"
render.mkdir(parents=True, exist_ok=True)
with fitz.open(target) as doc:
    for i, page in enumerate(doc):
        page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False).save(render / f"page-{i+1}.png")
    result = dict(pages=doc.page_count, warnings=warnings, pdf=target.relative_to(BASE).as_posix(),
                  render_directory=render.relative_to(BASE).as_posix())
(HERE / "data/build_review.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(result, indent=2))
