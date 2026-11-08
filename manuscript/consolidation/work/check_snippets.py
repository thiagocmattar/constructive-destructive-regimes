"""Compile consolidation snippets in LaTeX draft mode (no PDF authored)."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/"manuscript/consolidation"
OUT=BASE/"evidence"
TMP=ROOT/"tmp/pdfs/consolidation-snippets"
TMP.mkdir(parents=True,exist_ok=True)
groups=["03-regimes","05-adam/a-characterization","05-adam/b-tracing","05-adam/c-intervention"]
paths=[]
parts=[r"\documentclass[10pt]{article}",
       r"\usepackage[T1]{fontenc}",
       r"\usepackage[margin=0.75in]{geometry}",
       r"\usepackage{amsmath,amssymb,booktabs,graphicx,algorithm,algpseudocode}",
       r"\begin{document}"]
for group in groups:
    parts.append(r"\clearpage")
    for name in ("equations.tex","algorithm.tex"):
        path=OUT/group/name
        if path.exists():
            paths.append(path)
            parts.append(r"\input{"+path.as_posix()+"}")
for group in ["04-gd-sgd", *groups]:
    for path in sorted((OUT/group/"tables").glob("*.tex")):
        paths.append(path)
        parts.append(r"\clearpage")
        if r"\begin{table" in path.read_text(encoding="utf-8"):
            parts.append(r"\input{"+path.as_posix()+"}")
        else:
            parts.append(r"\noindent\resizebox{\textwidth}{!}{\input{"+path.as_posix()+"}}")
parts.append(r"\end{document}")
(TMP/"snippets.tex").write_text("\n".join(parts)+"\n",encoding="utf-8")
latex=shutil.which("pdflatex")
assert latex, "pdflatex unavailable"
for _ in range(2):
    result=subprocess.run([latex,"-draftmode","-interaction=nonstopmode","-halt-on-error","snippets.tex"],
                          cwd=TMP,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if result.returncode:
        print(result.stdout[-5000:])
        raise SystemExit(result.returncode)
log=(TMP/"snippets.log").read_text(encoding="utf-8",errors="replace")
bad=[line for line in log.splitlines() if ("undefined" in line.lower() or "multiply defined" in line.lower()
                                         or line.startswith("!"))]
overfull=[line for line in log.splitlines() if "Overfull" in line]
assert not bad,bad
assert not (TMP/"snippets.pdf").exists(), "Draft check should not author a PDF."
result={"date":"2026-11-08","mode":"pdflatex draft mode; two passes; no PDF output",
        "tex_files_checked":len(paths),"errors":bad,"overfull_warnings":overfull,
        "files":{p.relative_to(BASE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(OUT/"snippet_verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
print(json.dumps({k:v for k,v in result.items() if k!="files"},indent=2))
