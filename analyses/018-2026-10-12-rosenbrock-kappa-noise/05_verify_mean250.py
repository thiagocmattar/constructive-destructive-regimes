"""Check all 250-pair windows against direct vectors and review the separate PDF."""
import argparse
import json

import fitz
import numpy as np

from noise_kappa import HERE, ROOT, digest, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-visual-review", action="store_true")
    args = parser.parse_args()
    report = json.loads((HERE / "results.json").read_text())
    manifest = json.loads((HERE / "figure_manifest_mean250.json").read_text())
    data = json.loads((HERE / "plot_data_mean250.json").read_text())
    config = json.loads((HERE / "config_mean250.json").read_text())
    assert config["source_config_sha256"] == sha256(HERE / config["source_config"])
    assert config["source_script_sha256"] == sha256(HERE / config["source_script"])
    assert config["smoothing_window"] == manifest["window"] == data["smoothing_window"] == 250
    assert manifest["script_sha256"] == sha256(HERE / "04_plot_mean250.py")
    assert manifest["metric_sha256"] == sha256(HERE / "noise_kappa.py")
    assert manifest["plot_data_sha256"] == sha256(HERE / "plot_data_mean250.json")
    assert json.loads((HERE / "verification.json").read_text())["passed"]
    assert json.loads((HERE / "figure_verification.json").read_text())["passed"]
    checks = []
    for source in report["sources"]:
        run = ROOT / source["directory"]
        history_path = run / "artifacts/attempts/001-local/history.npz"
        assert sha256(history_path) == source["history_sha256"]
        assert sha256(run / "config.json") == source["config_sha256"]
        assert sha256(run / "verification.json") == source["verification_sha256"]
        assert sha256(ROOT / source["derived_history"]) == source["derived_history_sha256"]
        with np.load(history_path, allow_pickle=False) as history:
            g = history["raw_vectors"]
            signal = history["signal_vectors"] if "signal_vectors" in history else g
        q = np.r_[np.nan, [-float(u @ (v-u)) / float(u @ u) for u,v in zip(g[:-1],g[1:])]]
        corrected = np.r_[np.nan, [-float(u @ (v-s)) / float(u @ u)
                         for u,s,v in zip(g[:-1],signal[:-1],signal[1:])]]
        stored = next(row for row in data["sources"] if row["run_id"] == source["run_id"])
        means = {}
        for metric, direct in (("gradient_only", q), ("noise_corrected", corrected)):
            expected = np.r_[np.full(250, np.nan),
                             [sum(float(x) for x in direct[i-249:i+1])/250 for i in range(250,2000)]]
            observed = np.array([np.nan if x is None else x for x in stored["arrays"][metric + "_mean250"]])
            np.testing.assert_allclose(observed, expected, rtol=5e-10, atol=2e-12, equal_nan=True)
            assert np.isnan(observed[:250]).all() and np.isfinite(observed[250:]).all()
            assert np.isfinite(observed).sum() == 1750
            line = next(row for row in manifest["rendered_series"]
                        if row["run_id"] == source["run_id"] and row["metric"] == metric)
            assert digest(np.arange(1,2001)) == line["x_sha256"]
            assert digest(observed) == line["y_sha256"]
            assert line["finite_points"] == 1750 and line["points"] == 2000
            assert line["ylim"][0] <= np.nanmin(observed) <= np.nanmax(observed) <= line["ylim"][1]
            assert line["xlim"] == [0,2000] and line["y_scale"] == "linear"
            means[metric] = float(observed[-1])
        checks.append(dict(run_id=source["run_id"], sigma=source["sigma"],
                           learning_rate=source["learning_rate"], means_per_metric=1750,
                           final250_pair_means=means, source_hashes_verified=True))
    for path, expected in manifest["protected_originals"].items():
        assert sha256(ROOT / path) == expected
    for original in report["protected_reference_figures"]:
        assert sha256(ROOT / original["path"]) == original["sha256"]
    pdf = ROOT / manifest["figure"]
    assert sha256(pdf) == manifest["sha256"]
    document = fitz.open(pdf)
    assert len(document) == 1
    page = document[0]
    text = page.get_text()
    assert text.count("250-pair mean") == 10
    assert "average 250 pairs" in text and "t = 251" in text and "1,750 means" in text
    assert "100-pair mean" not in text
    assert all("(" + letter + ")" in text for letter in "abcdefghij")
    np.testing.assert_allclose([page.rect.width, page.rect.height], [16*72,18.5*72], rtol=0, atol=.01)
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                assert page.rect.contains(fitz.Rect(span["bbox"])), span["text"]
    render = HERE / "scratch/rosenbrock_kappa_eta_noise_overlay_mean250.png"
    page.get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(render)
    write_json(HERE / "verification_mean250.json", dict(passed=args.record_visual_review,
               numeric_and_pdf_checks_passed=True, window=250, first_complete_mean_update=251,
               trajectories=15, means_per_metric=26250, plotted_series=30, plotted_finite_points=52500,
               all_windows_checked_against_direct_vectors=True, original_100_pair_version_unchanged=True,
               previous_reference_pdfs_unchanged=True, no_clipping=True, checks=checks,
               visual_review="passed after inspecting latest render" if args.record_visual_review else "pending",
               pdf_sha256=sha256(pdf), render_sha256=sha256(render), renderer="PyMuPDF 1.25x",
               config_sha256=sha256(HERE / "config_mean250.json"), script_sha256=sha256(__file__)))
    owned = [p for p in HERE.rglob("*") if p.is_file()
             and not any(part in {"scratch", "artifacts", "__pycache__"} for part in p.relative_to(HERE).parts)
             and p.name != "inventory.json"]
    write_json(HERE / "inventory.json", dict(files=[dict(path=p.relative_to(HERE).as_posix(),
               size_bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(owned)],
               ignored_derived_histories=report["sources"][0]["derived_history"].rsplit("/",1)[0],
               derived_history_bytes=sum(x["derived_history_bytes"] for x in report["sources"])))
    print("PASS: 52,500 plotted 250-pair means; original 100-pair PDF/data unchanged.")
    print("Visual review: " + ("recorded" if args.record_visual_review else "pending"))


if __name__ == "__main__":
    main()
