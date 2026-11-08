"""Package selected paper evidence without mutating scientific sources.

Run --build after reviewing the catalogue; run --verify for drift/link checks.
Copies preserve bytes except Markdown links, which are rebased explicitly.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote, unquote

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "manuscript/consolidation"
OUT = BASE / "evidence"
CATALOG = Path(__file__).with_name("evidence_catalog.json")
MD_LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\n]+)(\))")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def relative(path, parent):
    return Path(os.path.relpath(path, parent)).as_posix()


def rebase_markdown(source, target, mapped):
    text = source.read_text(encoding="utf-8-sig")
    def repl(match):
        raw = match[2]
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", raw) or raw.startswith("#"):
            return match[0]
        path, sep, anchor = raw.partition("#")
        original = (source.parent / unquote(path)).resolve()
        destination = mapped.get(original, original)
        link = quote(relative(destination, target.parent), safe="/:.-_")
        return match[1] + link + (sep + anchor if sep else "") + match[3]
    text = MD_LINK.sub(repl, text)
    header = (
        "> Evidence snapshot; original scientific content retained. Relative links were rebased.\n"
        f"> Source: [{source.name}]({relative(source, target.parent)}).\n\n"
    )
    return header + text


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def table(folder, name, columns, rows, caption):
    folder.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write(folder / (name + ".csv"), buffer.getvalue())
    lines = ["# " + name.replace("-", " ").capitalize(), "", caption, "",
             "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows]
    write(folder / (name + ".md"), "\n".join(lines))
    def tex(x):
        return str(x).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    body = [r"\begin{tabular}{@{}" + "l" * len(columns) + r"@{}}",
            r"\toprule", " & ".join(tex(c) for c in columns) + r" \\", r"\midrule"]
    body += [" & ".join(tex(row[c]) for c in columns) + r" \\" for row in rows]
    body += [r"\bottomrule", r"\end{tabular}"]
    write(folder / (name + ".tex"), "% Requires booktabs. Units/scope/caption: companion Markdown.\n" + "\n".join(body))


def derived_tables():
    va = OUT / "05-adam/a-characterization"
    vb = OUT / "05-adam/b-tracing"
    vc = OUT / "05-adam/c-intervention"
    severity = [r for r in read_csv(va / "data/severity.csv")
                if r["tier"] == "2" and r["target"] == "future_step"
                and r["subset"] == "all" and r["horizon"] == "1"
                and float(r["minimum_relative_increase"]) == .1]
    assert len(severity) == 3
    order = {"mlp": 0, "resnet": 1, "transformer": 2}
    severity.sort(key=lambda r: order[r["task"]])
    assert [int(r["positives"]) for r in severity] == [168, 12, 26]
    rows = []
    for r in severity:
        row = {"Case": r["task"], "Rises": r["positives"]}
        for label, key in [("rho<0", "negative_cosine"), ("r>1", "norm_growth"),
                           ("Both", "reversal_and_growth"), ("Q>=2", "Q_ge2")]:
            assert int(r[key+"_tp"]) + int(r[key+"_fn"]) == int(r["positives"])
            assert int(r[key+"_fp"]) + int(r[key+"_tn"]) == int(r["negatives"])
            row[label] = r[key+"_tp"]
        rows.append(row)
    table(va / "tables", "severity-10pct", ["Case", "Rises", "rho<0", "r>1", "Both", "Q>=2"], rows,
          "Signal at state s, individual relative loss increase >=10% at s→s+1. "
          "Counts are eligible updates, not independent spike events. High-LR cases only; "
          "the complete data/severity.csv retains non-target alarms, precision and all thresholds.")

    signs = [r for r in read_csv(va / "data/one-step-summary.csv")
             if r["tier"] == "2" and r["timing"] == "outgoing" and float(r["relative_tolerance"]) == 0]
    assert len(signs) == 3
    signs.sort(key=lambda r: order[r["task"]])
    rows = []
    for r in signs:
        rows.append({"Case": r["task"], "Increases": r["increases"], "Decreases": r["decreases"],
                     "rho<0 up/down": r["destructive_increases"]+"/"+r["destructive_decreases"],
                     "r>1 up/down": r["growth_increases"]+"/"+r["growth_decreases"],
                     "Both up/down": r["reversing_growth_increases"]+"/"+r["reversing_growth_decreases"],
                     "Q>=2 up/down": r["Q_ge2_increases"]+"/"+r["Q_ge2_decreases"]})
    table(va / "tables", "next-step-signs", list(rows[0]), rows,
          "Strict outgoing increases/decreases after the decision-time signal. "
          "Each signal cell counts flagged increases / flagged decreases; no loss-magnitude threshold. "
          "All nine arms, incoming controls and tolerance checks remain in data/one-step-summary.csv.")

    spectra = json.loads((vb / "data/spectral_controls.json").read_text(encoding="utf-8"))
    rows = []
    for c in spectra["cases"]:
        rows.append({"Case": c["case"], "T": c["rows"],
                     "Observed top1 %": f'{100*c["observed"]["top1"]:.2f}',
                     "Observed r95": c["observed"]["rank95"],
                     "Orthogonal top1 %": f'{100*c["same_energy_orthogonal"]["top1"]:.2f}',
                     "Orthogonal r95": c["same_energy_orthogonal"]["rank95"],
                     "Unit-row top1 %": f'{100*c["unit_row_directional"]["top1"]:.2f}',
                     "Unit-row r95": c["unit_row_directional"]["rank95"]})
    table(vb / "tables", "spectral-controls", list(rows[0]), rows,
          "Original selected output windows. Orthogonal reference preserves every step norm; "
          "unit-row normalization removes magnitude weighting. These are deterministic descriptive "
          "controls, not a statistical null or quiet-window contrast.")

    metrics = read_csv(vc / "data/transfer_summary.csv")
    heldout = read_csv(vc / "data/transfer_heldout_evaluations.csv")
    rows, absolute = [], []
    for run, begin, end in [("272","448","1152"),("273","264","576"),("274","552","928"),
                            ("282","448","1152"),("283","448","1152")]:
        pair = [r for r in metrics if r["run"] == run and r["begin"] == begin and r["end"] == end
                and r["arm"] in ("control", "functional_trunk_q50")]
        assert len(pair) == 2, (run, len(pair))
        control = next(r for r in pair if r["arm"] == "control")
        cap = next(r for r in pair if r["arm"] != "control")
        test = [r for r in heldout if r["run"] == run and r["state"] == end and r["split"] == "test"
                and r["worker"] == cap["worker"] and r["arm"] in ("control","functional_trunk_q50")]
        assert len(test) == 2, (run, len(test))
        tc = next(r for r in test if r["arm"] == "control")
        ti = next(r for r in test if r["arm"] != "control")
        # Task.evaluate counts images/tokens for loss reduction, but counts
        # complete 1,000-image BN groups / sequences as objective units.
        expected_coverage = {"mlp": (10000, 10000), "resnet": (10000, 10),
                             "transformer": (245550, 7020)}[cap["case"]]
        for item in test:
            assert (int(item["loss_units"]), int(item["objective_units"])) == expected_coverage, item
        pct = 100 * (float(cap["mean_loss"])/float(control["mean_loss"]) - 1)
        test_pct = 100 * (float(ti["loss"])/float(tc["loss"]) - 1)
        assert abs(pct-float(cap["mean_loss_pct"])) < 1e-8
        assert abs(test_pct-float(ti["loss_change_pct"])) < 1e-8
        changes = {}
        for metric in ("tail32_mean", "max_loss", "positive_variation"):
            changes[metric] = 100 * (float(cap[metric])/float(control[metric]) - 1)
            assert abs(changes[metric]-float(cap[metric+"_pct"])) < 1e-8
        rows.append({"Case":cap["case"],"Seed":cap["seed"],"States":begin+"--"+end,
                     "Mean %":f"{pct:+.2f}","Tail32 %":f'{changes["tail32_mean"]:+.2f}',
                     "Max %":f'{changes["max_loss"]:+.2f}',
                     "PV %":f'{changes["positive_variation"]:+.2f}',
                     "Test %":f"{test_pct:+.2f}"})
        for r in pair:
            t = next(v for v in test if v["arm"] == r["arm"])
            absolute.append({k:r[k] for k in ["case","seed","run","worker","arm","begin","end","mean_loss",
                                             "tail32_mean","max_loss","positive_variation"]}
                            | {"test_loss":t["loss"],"test_units":t["loss_units"],
                               "test_objective_units":t["objective_units"],
                               "test_unit_kind":"tokens" if r["case"]=="transformer" else "images"})
    table(vc / "tables", "distributed-cap-outcomes", list(rows[0]), rows,
          "Distributed q50 cap relative to its own unchanged Adam baseline; negative is lower. "
          "Mean/tail32/max/PV use the stated training-loss states; test is complete terminal held-out loss. "
          "MLP has three seeds; larger models have one. This is not a matched ranking against historical LR/clipping/beta2 studies. "
          "Absolute paired values and exact source worker identities are in distributed-cap-pairs.csv.")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(absolute[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(absolute)
    write(vc / "tables/distributed-cap-pairs.csv", buffer.getvalue())
    return {"severity_rows":len(severity), "sign_rows":len(signs), "spectral_cases":len(spectra["cases"]),
            "cap_baseline_pairs":len(rows), "complete_test_evaluations":len(absolute)}


def build():
    catalogue = json.loads(CATALOG.read_text(encoding="utf-8"))
    assets = catalogue["assets"]
    mapped = {}
    for item in assets:
        source, target = ROOT/item["source"], OUT/item["destination"]
        assert target.resolve().is_relative_to(OUT.resolve()), target
        assert source.is_file(), source
        for key in ("observation", "generator"):
            if item.get(key):
                assert (ROOT/item[key]).is_file(), item[key]
        assert source.resolve() not in mapped, source
        mapped[source.resolve()] = target.resolve()
    records = []
    for item in assets:
        source, target = ROOT/item["source"], OUT/item["destination"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".md":
            write(target, rebase_markdown(source, target, mapped))
            transform = "Markdown source header and rebased local links"
        else:
            target.write_bytes(source.read_bytes())
            transform = "byte-identical copy"
        record = dict(item, source_sha256=sha(source), packaged_sha256=sha(target),
                      bytes=target.stat().st_size, transformation=transform)
        record["generator_sha256"] = sha(ROOT/item["generator"]) if item["generator"] else None
        records.append(record)
    tables = derived_tables()
    for section, info in catalogue["sections"].items():
        folder = OUT/section
        items = [r for r in records if r["section"] == section]
        lines = ["# "+info["title"], "", "- **Paper claim:** "+info["claim"],
                 "- **Boundary:** "+info["boundary"],
                 "- **Authority:** the [single paper plan]("+relative(BASE/"PLAN.md",folder)+") governs selection and closure.",
                 "", "| Asset | Status | Use / source |", "|---|---|---|"]
        for r in items:
            notes = r["note"]
            if r["observation"]:
                obs = mapped.get((ROOT/r["observation"]).resolve(), ROOT/r["observation"])
                notes += " [Observation]("+relative(obs,folder)+")."
            if r["generator"]:
                notes += " [Generator]("+relative(ROOT/r["generator"],folder)+")."
            link = relative(OUT/r["destination"],folder)
            if r.get("selected_page"):
                link += "#page=" + str(r["selected_page"])
            lines.append(f'| [{r["title"]}]({link}) | {r["role"]} | {notes} |')
        for sub in ("tables",):
            extras = sorted(p for p in (folder/sub).glob("*.md"))
            if extras:
                lines += ["", "## Consolidated tables", ""]
                lines += [f"- [{p.stem.replace('-', ' ')}]({relative(p,folder)})" for p in extras]
        editable = ("equations.md","equations.tex","algorithm.md","algorithm.tex","positioning.md","contributions.md","closure.md")
        if any((folder/name).exists() for name in editable):
            lines += ["", "## Editable integration files", ""]
        for name in editable:
            if (folder/name).exists():
                lines.append(f"- [{name}]({name})")
        write(folder/"README.md","\n".join(lines))
    copied_paths = {r["destination"] for r in records}
    prepared = {p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob("*")
                if p.is_file() and p.suffix in (".md", ".tex", ".bib", ".csv")
                and p.relative_to(OUT).as_posix() not in copied_paths}
    write(OUT/"copy_manifest.json",json.dumps({"date":"2026-11-08",
          "repository_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
          "catalogue_sha256":sha(CATALOG),"copies":records,"prepared_files":prepared,
          "derived_table_checks":tables},indent=2,ensure_ascii=False))
    print(json.dumps({"packaged_assets":len(records),"bytes":sum(r["bytes"] for r in records),"table_checks":tables},indent=2))


def verify():
    manifest = json.loads((OUT/"copy_manifest.json").read_text(encoding="utf-8"))
    errors, pdfs, pages, links = [], 0, 0, 0
    import fitz
    if sha(CATALOG) != manifest["catalogue_sha256"]:
        errors.append("Catalogue changed since package build")
    for rel, expected in manifest["prepared_files"].items():
        if sha(OUT/rel) != expected:
            errors.append("Prepared file drift: "+rel)
    for r in manifest["copies"]:
        source, target = ROOT/r["source"], OUT/r["destination"]
        if sha(source) != r["source_sha256"]:
            errors.append("Source drift: "+r["source"])
        if sha(target) != r["packaged_sha256"]:
            errors.append("Packaged copy drift: "+r["destination"])
        if r["generator"] and sha(ROOT/r["generator"]) != r["generator_sha256"]:
            errors.append("Generator drift: "+r["generator"])
        if r["transformation"] == "byte-identical copy" and source.read_bytes() != target.read_bytes():
            errors.append("Nonidentical copy: "+r["destination"])
        if target.suffix == ".pdf":
            with fitz.open(target) as doc:
                assert doc.page_count > 0
                if r.get("selected_page"):
                    assert 1 <= r["selected_page"] <= doc.page_count
                for page in doc:
                    page.get_pixmap(matrix=fitz.Matrix(.1,.1))
                pdfs += 1
                pages += doc.page_count
    for path in BASE.rglob("*.md"):
        if "human" in path.relative_to(BASE).parts:
            continue
        for match in MD_LINK.finditer(path.read_text(encoding="utf-8-sig")):
            raw = match[2]
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:",raw) or raw.startswith("#"):
                continue
            target = unquote(raw.split("#")[0])
            links += 1
            if not (path.parent/target).resolve().exists():
                errors.append("Broken link: "+str(path.relative_to(BASE))+" -> "+target)
    preservation = json.loads((BASE/"provenance.json").read_text(encoding="utf-8"))
    for kind,key in [("human_inputs","preserved_path"),("active_manuscript_snapshot","path")]:
        for item in preservation[kind]:
            if sha(ROOT/item[key]) != item["sha256"]:
                errors.append("Preserved input changed: "+item[key])
    result = {"date":"2026-11-08","copies_checked":len(manifest["copies"]),
              "prepared_files_checked":len(manifest["prepared_files"]),
              "pdfs_decoded":pdfs,"pdf_pages_decoded":pages,"markdown_links_checked":links,
              "human_originals_unchanged":len(preservation["human_inputs"]),
              "manuscript_files_unchanged":len(preservation["active_manuscript_snapshot"]),
              "errors":errors}
    write(OUT/"verification.json",json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--build",action="store_true")
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    if args.build:
        build()
    if args.verify:
        verify()
    if not (args.build or args.verify):
        parser.error("Choose --build or --verify")
