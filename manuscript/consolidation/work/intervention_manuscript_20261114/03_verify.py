"""Independent numerical, preservation and document checks after the review build."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
DATA = BASE / "evidence/05-adam/c-intervention/data"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


report = json.loads((DATA / "matched-results.json").read_text(encoding="utf-8"))
manifest = json.loads((HERE / "data/presentation_manifest.json").read_text(encoding="utf-8"))
old = json.loads((DATA / "matched-figure_manifest.json").read_text(encoding="utf-8"))
curves = np.load(DATA / "matched-curves.npz")
held = list(csv.DictReader((DATA / "matched-heldout.csv").open(encoding="utf-8")))
lookup = {(r["case"], r["arm"]): r for r in report["summary"]}
cases = ["mlp", "resnet", "transformer"]
arms = [a["id"] for a in report["cases"][0]["config"]["arms"]]
check(len(lookup) == 36 and len(arms) == 12, "Incomplete matrix")
for relative, digest in manifest["sources"].items():
    check(sha(BASE / relative) == digest, "Input drift: " + relative)
check(manifest["generator_sha256"] == sha(HERE / "01_prepare.py"), "Generator drift")

# Independent sums use Python fsum rather than copying the reduction code.
derived = {}
for case in cases:
    for arm in arms:
        prefix = f"{case}__{arm}__"
        loss = curves[prefix + "loss"]
        states = curves[prefix + "state"]
        row = lookup[case, arm]
        check(int(states[0]) == row["fork"] and int(states[-1]) == row["end"], "Window changed")
        np.testing.assert_array_equal(states, np.arange(row["fork"], row["end"]+1))
        values = dict(positive_variation=math.fsum(max(float(b)-float(a), 0.) for a, b in zip(loss[:-1], loss[1:])),
                      mean_loss=math.fsum(float(x) for x in loss[1:])/(len(loss)-1), final_loss=float(loss[-1]))
        terminal = [v for v in held if v["case"] == case and v["arm"] == arm and v["split"] == "test" and int(v["state"]) == row["end"]]
        check(len(terminal) == 1, "Ambiguous terminal test result")
        values["terminal_test_loss"] = float(terminal[0]["loss"])
        for metric, value in values.items():
            check(math.isclose(value, row[metric], rel_tol=2e-12, abs_tol=2e-12), f"Metric mismatch: {case}/{arm}/{metric}")
        derived[case, arm] = values

metrics = ["positive_variation", "mean_loss", "final_loss", "terminal_test_loss"]
table_text = (HERE / "tables/matched-main-outcomes.tex").read_text(encoding="utf-8")
table_cells = 0
for record in manifest["table"]["rows"]:
    arm = record["arm"]
    expected, full = [], []
    for case in cases:
        for metric in metrics:
            ratio = derived[case, arm][metric] / derived[case, "control"][metric]
            effect = 100*(1-ratio) if metric == "positive_variation" else 100*(ratio-1)
            expected.append("0" if arm == "control" else f"{effect:+.2f}")
            full.append(effect)
    check(record["display_values"] == expected, "Rounded table cells differ: " + arm)
    np.testing.assert_allclose(record["values"], full, rtol=1e-10, atol=2e-10)
    check(" & ".join(expected) + r" \\" in table_text, "TeX table values differ: " + arm)
    table_cells += len(expected)
for case in cases:
    values = [f"{derived[case, 'control'][m]:.5g}" for m in metrics]
    check(", ".join(values) in table_text, "Absolute Adam reference missing")
check(table_cells == 144, "Wrong table coverage")

tradeoff, motion = manifest["figures"]
old_points = { (p["case"], p["arm"]): p for out in old["outputs"] if out["file"].startswith("18-") for p in out["points"] }
for point in tradeoff["points"]:
    source = old_points[point["case"], point["arm"]]
    check(point["x"] == source["x"] and point["y"] == source["y"], "Scientific point moved")
check(len(tradeoff["points"]) == 33 and tradeoff["baseline_count"] == 3, "Scatter coverage changed")
label_obstructions = []
for label in tradeoff["labels"]:
    left, bottom, width, height = label["bounds_pixels"]
    for point in tradeoff["points"]:
        if point["case"] != label["case"]: continue
        x, y = point["display_xy_pixels"]
        if left-2 <= x <= left+width+2 and bottom-2 <= y <= bottom+height+2:
            label_obstructions.append([label["case"], label["arm"], point["arm"]])
check(not label_obstructions, f"Callout boxes obscure plotted points: {label_obstructions}")

old_motion = next(v for v in old["outputs"] if v["file"].startswith("19-"))
map_values = 0
for item in motion["maps"]:
    case, arm, metric = item["case"], item["arm"], item["metric"]
    if metric == "relative_rms":
        baseline = curves[f"{case}__control__stage_change_rms"][:, 1:]
        denominator = np.array([max(row) for row in baseline])
        values = curves[f"{case}__{arm}__stage_change_rms"][:, 1:] / denominator[:, None]
        np.testing.assert_array_equal(denominator, item["rms_denominator"])
        check(values.max() <= motion["rms_color_maximum"], "Color scale clips RMS")
    else:
        values = curves[f"{case}__{arm}__stage_rho"][:, 1:]
        check(np.isfinite(values).all() and values.min() >= -1 and values.max() <= 1, "Undefined/out-of-range cosine")
    check(list(values.shape) == item["shape"] and array_sha(values) == item["values_sha256"], "Heatmap values changed")
    old_case = next(v for v in old_motion["columns"] if v["case"] == case)
    check(item["window"] == old_case["window"] and item["baseline_peak"] == old_case["baseline_peak_state"], "Heatmap window/marker changed")
    np.testing.assert_array_equal(item["rms_denominator"], old_case["baseline_stage_rms_maximum"])
    map_values += values.size
for item in motion["lines"]:
    values = curves[f"{item['case']}__{item['arm']}__loss"]
    states = curves[f"{item['case']}__{item['arm']}__state"]
    check(array_sha(values) == item["y_sha256"] and array_sha(states) == item["x_sha256"], "Loss line changed")
check(len(motion["maps"]) == 12 and len(motion["lines"]) == 6, "Functional coverage changed")

source_preservation = []
for number in range(17, 24):
    path = next((BASE / "evidence/05-adam/c-intervention/figures").glob(f"{number}-*.pdf"))
    original = subprocess.check_output(["git", "show", "HEAD:"+path.relative_to(ROOT).as_posix()], cwd=ROOT)
    check(hashlib.sha256(original).hexdigest() == sha(path), "Historical source PDF changed")
    source_preservation.append(path.relative_to(BASE).as_posix())

pdf_checks = []
for item in manifest["figures"]:
    path = HERE / "figures" / item["file"]
    check(sha(path) == item["sha256"], "Figure hash drift")
    with fitz.open(path) as doc:
        check(doc.page_count == 1, "Main figure is not one page")
        page = doc[0]
        check(abs(page.rect.width/72-7.15) < 1e-4, "Main figure width changed")
        ordinary_sizes = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    check(x0 >= -.3 and y0 >= -.3 and x1 <= page.rect.width+.3 and y1 <= page.rect.height+.3, "Clipped PDF label")
                    if "Arial" in span["font"]: ordinary_sizes.append(span["size"])
        check(ordinary_sizes and min(ordinary_sizes) >= 8.99, "Main ordinary labels below 9pt")
        embedded = [bool(doc.extract_font(font[0])[3]) for font in page.get_fonts(full=True)]
        check(all(embedded), "Unembedded figure font")
        pdf_checks.append(dict(file=item["file"], pages=1, minimum_ordinary_font_points=min(ordinary_sizes), all_fonts_embedded=True))

build = json.loads((HERE / "data/build_review.json").read_text())
check(build["pages"] == 3 and not build["warnings"], "Review build/page check failed")
draft = BASE / "draft/intervention-review.pdf"
with fitz.open(draft) as doc:
    check(doc.page_count == 3, "Review PDF page drift")
    text = "\n".join(page.get_text() for page in doc).replace("\N{MINUS SIGN}", "-")
    for token in ["+3551.08", "+27.52", "-32.18", "Algorithm 2", "TABLE I", "Fig. 1.", "Fig. 2."]:
        check(token in text, "Missing compiled content: " + token)
    check("??" not in text, "Unresolved rendered reference")
    table_region = text[text.index("TABLE I"):text.index("PV reports reduction")]
    rendered_cells = Counter(re.findall(r"[+-]?\d+(?:\.\d+)?", table_region))
    expected_cells = Counter(x for row in manifest["table"]["rows"] for x in row["display_values"])
    check(all(rendered_cells[value] >= count for value, count in expected_cells.items()), "Compiled table cell missing")

# Explicit output-motion readouts behind the narrative, derived from all updates.
claims = {}
for case in cases:
    baseline = curves[f"{case}__control__stage_change_rms"][-1, 1:]
    capped = curves[f"{case}__cap_q50__stage_change_rms"][-1, 1:]
    claims[case] = dict(maximum_output_rms_reduction_percent=100*(1-max(capped)/max(baseline)),
                       squared_output_motion_change_percent=100*(sum(float(v)**2 for v in capped)/sum(float(v)**2 for v in baseline)-1),
                       baseline_reversal_percent=100*np.mean(curves[f"{case}__control__stage_rho"][-1, 1:] < 0),
                       q50_reversal_percent=100*np.mean(curves[f"{case}__cap_q50__stage_rho"][-1, 1:] < 0))
check(all(lookup[c, a]["mean_loss_change_percent"] < 0 and lookup[c, a]["pv_reduction_percent"] > 0 for c in cases for a in ["cap_q25", "cap_q50", "cap_q75"]), "Nine-case claim failed")
for arm, expected in [("control", 161), ("cap_q50", 70)]:
    delta = curves[f"mlp__{arm}__mse_delta_loss"]
    linear = curves[f"mlp__{arm}__mse_linear"]
    quadratic = curves[f"mlp__{arm}__mse_quadratic"]
    rising = delta > 1e-8
    check(np.count_nonzero(rising) == expected, "MLP rise count changed")
    check(np.all(linear[rising] < 0) and np.all(quadratic[rising] > -linear[rising]), "MLP loss-accounting claim failed")
result = dict(status="passed", summary_metrics_recomputed=144, table_cells_checked=table_cells,
              baseline_absolute_references_checked=12, compiled_table_cells_checked=144, scatter_points_checked=33, baseline_origins=3,
              heatmap_values_checked=map_values, loss_values_checked=sum(v["n"] for v in motion["lines"]),
              original_intervention_pdfs_unchanged=source_preservation, callout_obstructions=label_obstructions,
              pdf_checks=pdf_checks, review_pages=3, latex_warnings=[], output_motion_claims=claims,
              source_manifest_sha256=sha(HERE / "data/presentation_manifest.json"),
              draft_source_sha256=sha(BASE / "draft/05c-interventions.tex"), draft_pdf_sha256=sha(draft),
              verifier_sha256=sha(__file__), visual_review="Recorded separately after final rendered-page inspection")
(HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({k:result[k] for k in ["status", "summary_metrics_recomputed", "table_cells_checked", "heatmap_values_checked", "loss_values_checked", "review_pages", "latex_warnings"]}, indent=2))
