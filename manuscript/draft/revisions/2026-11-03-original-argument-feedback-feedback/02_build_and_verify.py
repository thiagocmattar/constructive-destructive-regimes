"""Check this annotation-driven revision and build its separate review PDF."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import fitz
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1]
SCRATCH = ROOT / "tmp/pdfs/manuscript-original-argument-feedback-feedback-2026-11-03"
PREVIOUS = HERE.parent / "2026-11-03-original-argument-restored"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_preservation():
    manifest = read(HERE / "preservation_manifest.json")
    for path, digest in manifest["protected_sources"].items():
        assert sha(ROOT / path) == digest, path
    old = subprocess.check_output(["git", "show", manifest["starting_commit"] +
                                   ":manuscript/draft/bare_jrnl.tex"], cwd=ROOT).decode("utf-8")
    current = (DRAFT / "bare_jrnl.tex").read_text(encoding="utf-8")
    for pattern in [r"\\title\{[^\n]*\}", r"\\begin\{abstract\}.*?\\end\{abstract\}",
                    r"\\begin\{IEEEkeywords\}.*?\\end\{IEEEkeywords\}"]:
        assert re.search(pattern, old, re.S).group() == re.search(pattern, current, re.S).group()
    return len(manifest["protected_sources"])


def check_loss_conditions():
    rng = np.random.default_rng(2026110341)
    errors, cases = [], 0
    for _ in range(40):
        a = rng.normal(size=(6, 6))
        h = a.T @ a + np.eye(6)
        x = rng.normal(size=6)
        g = h @ x
        norm = np.linalg.norm(g)
        kappa = (g @ h @ g)/(g @ g)
        for scaled in [.2, .8, 1.2, 1.8, 2.2, 3.]:
            eta = scaled/kappa
            y = x-eta*g
            next_g = h @ y
            next_norm = np.linalg.norm(next_g)
            rho = (g @ next_g)/(norm*next_norm)
            actual = .5*(y @ h @ y-x @ h @ x)
            predicted = -eta/2*(g @ g+g @ next_g)
            errors.append(abs(actual-predicted)/max(1, abs(actual)))
            errors.append(abs(1-rho*next_norm/norm-eta*kappa))
            assert (actual > 0) == (rho < -norm/next_norm) == (g @ next_g < -g @ g)
            assert (rho < 0) == (scaled > 1)
            if actual > 0:
                assert rho < 0 and next_norm > norm
            cases += 1
    orthogonal = np.eye(100)
    retention = np.linalg.norm(orthogonal.sum(0))/np.linalg.norm(orthogonal, axis=1).sum()
    assert abs(retention-.1) < 1e-15
    assert np.isclose(np.degrees(np.arccos(-.5)),120)
    assert round(float(np.degrees(np.arccos(-.2))),1) == 101.5
    assert max(errors) < 1e-10
    return dict(quadratic_cases=cases, maximum_scaled_error=max(errors),
                orthogonal_100_step_retention=retention, angle_examples_degrees=[120,101.5])


def check_sources():
    methods = (DRAFT / 'sections/03-methods.tex').read_text(encoding='utf-8')
    adam = (DRAFT / 'sections/06-adam-diagnostics.tex').read_text(encoding='utf-8')
    appendix = (DRAFT / 'sections/06-loss-appendix.tex').read_text(encoding='utf-8')
    feedback = read(HERE / 'feedback.json')
    assert len(feedback['comments']) == 30
    assert sha(ROOT / feedback['source']) == feedback['sha256']
    expected_ids = {f'C{i:02}' for i in range(1,31)}
    assert {c['id'] for c in feedback['comments']} == expected_ids
    review = (HERE/'REVIEW.md').read_text(encoding='utf-8')
    dispositions = re.findall(r'^\| (C\d\d) \|', review, re.M)
    assert len(dispositions) == 30 and set(dispositions) == expected_ids
    assert not re.search(r'- \[ \] C\d', (HERE/'TASKS.md').read_text())
    assert methods.count(r'\boxed{') == 2
    assert r'\Longleftrightarrow' not in methods
    assert 'eq:adam-projected-change' not in adam+appendix
    assert 'eq:adam-gradient-response' in appendix
    forbidden = ['interspersed', 'not an additional temporal filter',
                 'not proof of a linear instability', 'not a separation inferred',
                 'rather than a universal', 'cannot be used unchanged',
                 'does not establish a linear relation', 'without identifying those',
                 'normalized projected raw-gradient change along the']
    flat = ' '.join(adam.split())
    assert all(phrase not in flat for phrase in forbidden)
    assert r'\subsection{Controlled Full-Batch Training}' not in adam
    assert r'p_i&\in\operatorname*{arg\,max}' in adam
    assert r'\mathcal L_s\leq\mathcal L_{a_i}' in adam
    return dict(annotation_count=30, boxed_relations=2,
                old_adam_equation_removed=True, literal_deletions_checked=True)


def check_percentages():
    path = ROOT / 'analyses/036-2026-11-02-controlled-adam-diagnostics/spike_anatomy/events_gt10.csv'
    events = pd.read_csv(path)
    defined = events.rho_raw_before.notna()
    lead = events.eta_max_lead_to_loss_peak
    counts = [int((events.rho_raw_before < 0).sum()),
              int((events.gradient_norm_peak > events.gradient_norm_before).sum()),
              int((events.eta_kappa_hat_raw_before > 2).sum()),
              int((lead > 0).sum()), int((lead == 0).sum()), int((lead < 0).sum())]
    denominators = [int(defined.sum()), len(events), int(defined.sum()),
                    len(events), len(events), len(events)]
    assert counts == [117,129,21,82,29,18]
    assert denominators == [127,129,127,129,129,129]
    percentages = [100*n/d for n,d in zip(counts, denominators)]
    section = (DRAFT/'sections/06-adam-diagnostics.tex').read_text(encoding='utf-8')
    for value in percentages:
        token = ('100' if value == 100 else f'{value:.1f}') + r'\%'
        assert token in section, token
    return dict(source=path.relative_to(ROOT).as_posix(), source_sha256=sha(path),
                numerators=counts, denominators=denominators, percentages=percentages)


def main():
    preservation = check_preservation()
    source_checks = check_sources()
    percentages = check_percentages()
    spec = importlib.util.spec_from_file_location("previous_evidence", PREVIOUS / "01_build_and_verify.py")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    evidence, record = previous.check_evidence()
    record["new_mathematical_checks"] = check_loss_conditions()
    figure = read(HERE / "curvature_comparison_manifest.json")
    schematic = read(HERE / "spike_definition_manifest.json")
    assert figure["status"] in ["passed", "awaiting_visual_review"]
    baseline = read(ROOT / "analyses/024-2026-10-17-eos-gd-lr-calibration/curvature_comparison_manifest.json")
    assert figure["curves"] == baseline["curves"]
    assert sha(ROOT / schematic["pdf"]) == sha(ROOT / schematic["manuscript_copy"]) == schematic["pdf_sha256"]
    assert sha(ROOT / schematic["generator"]) == schematic["generator_sha256"]
    assert sha(ROOT / figure["generator"]) == figure["generator_sha256"]
    for path, digest in figure["sources"].items():
        assert sha(ROOT / path) == digest, path
    assert sha(ROOT / figure["pdf"]) == sha(ROOT / figure["manuscript_copy"]) == figure["pdf_sha256"]
    assert figure["spectral_points"] == 279 and figure["plotted_points"] == 53838
    with fitz.open(ROOT/figure["pdf"]) as document:
        assert len(document) == 1 and not document[0].get_images()
        for font in document[0].get_fonts(full=True):
            assert font[2] != "Type3" and document.extract_font(font[0])[3]
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tex = ["pdflatex", "--disable-installer", "-no-shell-escape", "-recorder",
           "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
           "-output-directory=" + str(SCRATCH), "bare_jrnl.tex"]
    for index, command in enumerate([tex, ["bibtex", str(SCRATCH/"bare_jrnl")], tex, tex], 1):
        result = subprocess.run(command, cwd=DRAFT, capture_output=True, timeout=120)
        (SCRATCH / f"build-{index}.txt").write_bytes(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f"Build pass {index} failed; see {SCRATCH}")
    log = (SCRATCH / "bare_jrnl.log").read_text(errors="replace")
    for extra in range(3):
        if "Rerun to get cross-references right" not in log:
            break
        result = subprocess.run(tex, cwd=DRAFT, capture_output=True, timeout=120)
        (SCRATCH / f"build-extra-{extra+1}.txt").write_bytes(result.stdout+result.stderr)
        assert result.returncode == 0
        log = (SCRATCH / "bare_jrnl.log").read_text(errors="replace")
    problems = [line for line in log.splitlines() if any(token in line.lower()
                for token in ["overfull", "undefined", "rerun to get", "multiply defined"])]
    assert not problems, "\n".join(problems)
    expected = {str((ROOT/f["target"]).resolve()) for f in evidence["figures"]}
    expected |= {str((ROOT/f["target"]).resolve()) for f in
                 read(previous.previous.RETAINED / "figure_manifest.json")["figures"]}
    expected.remove(str((DRAFT/"figures/analysis024/gd_loss_projected_change.pdf").resolve()))
    expected.add(str((ROOT/figure["manuscript_copy"]).resolve()))
    expected.add(str((ROOT/schematic["manuscript_copy"]).resolve()))
    recorded = set()
    tables = set()
    for line in (SCRATCH/"bare_jrnl.fls").read_text().splitlines():
        if line.startswith("INPUT "):
            path = Path(line[6:])
            path = path if path.is_absolute() else DRAFT/path
            if path.suffix == ".pdf" and path.resolve().is_relative_to((DRAFT/"figures").resolve()):
                recorded.add(str(path.resolve()))
            if path.suffix == ".tex" and path.resolve().is_relative_to((DRAFT/"tables").resolve()):
                tables.add(path.name)
    assert recorded == expected and len(recorded) == 9
    assert tables == {"analysis036_spike_anatomy.tex"}
    pdf = SCRATCH/"bare_jrnl.pdf"
    with fitz.open(pdf) as document:
        record["pages"] = len(document)
        first = " ".join(document[0].get_text().split())
        opening = "Neural scaling laws have ushered in new paradigms for interpreting deep learning dynamics"
        assert first.split("I. INTRODUCTION",1)[1].lstrip().startswith(opening)
        record["introduction_opening_verified_in_pdf"] = opening
        assert "??" not in " ".join(page.get_text() for page in document)
        record["page_text_lengths"] = [len(page.get_text()) for page in document]
    subprocess.run(["pdftoppm","-r","100","-png",str(pdf),str(SCRATCH/"review")],
                   check=True,capture_output=True,timeout=120)
    for target in [DRAFT/"bare_jrnl.pdf", HERE/"revised-manuscript.pdf"]:
        shutil.copyfile(pdf,target)
    assert check_preservation() == preservation
    record.update(status="awaiting_visual_review", protected_files=preservation,
                  source_checks=source_checks, active_figures=9, active_tables=1,
                  percentages=percentages,
                  spectral_points=279, new_figure_points=53838,
                  pdf_sha256=sha(pdf), verifier_sha256=sha(__file__),
                  source_sha256={str(p.relative_to(ROOT)).replace("\\","/"):sha(p)
                      for p in [DRAFT/"bare_jrnl.tex",DRAFT/"bibliography.bib",
                                *sorted((DRAFT/"sections").glob("*.tex"))]},
                  build="pdflatex/BibTeX; no overfull boxes, undefined or multiply defined references")
    (HERE/"verification.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:record[k] for k in ["status","pages","protected_files","active_figures",
                                          "active_tables","spectral_points","pdf_sha256"]},indent=2))


if __name__ == "__main__":
    main()
