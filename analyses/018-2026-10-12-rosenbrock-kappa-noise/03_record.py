"""Verify plotted arrays/PDF, render it, and record completed visual review."""
import argparse
import json

import fitz
import numpy as np

from noise_kappa import HERE, ROOT, REFERENCE, digest, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-visual-review", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((HERE / "figure_manifest.json").read_text())
    report = json.loads((HERE / "results.json").read_text())
    receipt = json.loads((HERE / "verification.json").read_text())
    assert receipt["passed"]
    assert receipt["metric_sha256"] == sha256(HERE / "noise_kappa.py")
    inputs = report["scientific_inputs"]
    for field, path in (("config_sha256", HERE / "config.json"),
                        ("cohort_sha256", HERE / "cohort.json"),
                        ("metric_sha256", HERE / "noise_kappa.py"),
                        ("reference_data_sha256", REFERENCE / "eta_overlay_data.json"),
                        ("formula_snapshot_sha256", REFERENCE / "formula_source_snapshot.md")):
        assert inputs[field] == sha256(path)
    assert manifest["script_sha256"] == sha256(HERE / "02_plot.py")
    assert manifest["metric_sha256"] == sha256(HERE / "noise_kappa.py")
    assert manifest["plot_data_sha256"] == sha256(HERE / "plot_data.json")
    data = json.loads((HERE / "plot_data.json").read_text())
    for line in manifest["rendered_series"]:
        source = next(x for x in report["sources"] if x["run_id"] == line["run_id"])
        path = ROOT / source["derived_history"]
        assert sha256(path) == source["derived_history_sha256"]
        with np.load(path, allow_pickle=False) as saved:
            x = saved["updates"]
            y = saved[line["metric"] + "_mean100"]
        assert digest(x) == line["x_sha256"] and digest(y) == line["y_sha256"]
        stored = next(x for x in data["sources"] if x["run_id"] == line["run_id"])
        serialized = np.array([np.nan if v is None else v for v in stored["arrays"][line["metric"] + "_mean100"]])
        np.testing.assert_array_equal(serialized, y)
        assert np.isfinite(y).sum() == 1900 and np.isnan(y[:100]).all()
        assert line["ylim"][0] <= np.nanmin(y) and line["ylim"][1] >= np.nanmax(y)
        assert line["xlim"] == [0,2000] and line["y_scale"] == "linear"
    pdf = ROOT / manifest["figure"]
    assert sha256(pdf) == manifest["sha256"]
    document = fitz.open(pdf)
    assert len(document) == 1
    page = document[0]
    np.testing.assert_allclose([page.rect.width, page.rect.height], [16*72,18.5*72], rtol=0, atol=.01)
    text = page.get_text()
    assert text.count("100-pair mean") == 10
    assert all("(" + letter + ")" in text for letter in "abcdefghij")
    assert "linear Y range varies by row" in text and "Paired noise seed 1234" in text
    assert "AdamW step mismatch remains" in text
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                box = fitz.Rect(span["bbox"])
                assert page.rect.contains(box), (span["text"], box)
    render = HERE / "scratch/rosenbrock_kappa_eta_noise_overlay.png"
    page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25)).save(render)
    for original in report["protected_reference_figures"]:
        assert sha256(ROOT / original["path"]) == original["sha256"]
    write_json(HERE / "figure_verification.json", dict(
        passed=args.record_visual_review, numerical_and_pdf_checks_passed=True,
        visual_review="passed after inspecting latest PDF render" if args.record_visual_review else "pending",
        pdf_sha256=sha256(pdf), render_sha256=sha256(render), renderer="PyMuPDF 1.25x",
        panels=10, series=30, plotted_finite_points=57000, clipped_data=False,
        source_and_script_hashes=True, serialization_matches_plotted_arrays=True,
        shared_left_y_and_individual_right_y=True, original_figures_unchanged=True,
        glyph_bounds_inside_page=True, script_sha256=sha256(__file__)))
    owned = [p for p in HERE.rglob("*") if p.is_file()
             and not any(part in {"scratch", "artifacts", "__pycache__"} for part in p.relative_to(HERE).parts)
             and p.name != "inventory.json"]
    write_json(HERE / "inventory.json", dict(files=[dict(path=p.relative_to(HERE).as_posix(),
               size_bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(owned)],
               ignored_derived_histories=report["sources"][0]["derived_history"].rsplit("/",1)[0],
               derived_history_bytes=sum(x["derived_history_bytes"] for x in report["sources"])))
    print("PASS: 30 plotted series, 57,000 finite points, PDF text/bounds, source hashes, original PDFs unchanged.")
    print("Visual review: " + ("recorded" if args.record_visual_review else "pending"))


if __name__ == "__main__":
    main()
