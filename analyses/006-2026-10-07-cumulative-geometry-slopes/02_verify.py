"""Verify geometry, reduced artifacts, source preservation, and render PDFs."""

import csv
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    results = json.loads((HERE / "results.json").read_text())
    assert results["script_sha256"] == sha(HERE / "01_analyze.py")
    assert results["design_sha256"] == sha(HERE / "design.json")
    for path, expected in results["source_hashes_before"].items():
        assert sha(ROOT / path) == expected == results["source_hashes_after"][path]
    for path, expected in results["outputs_sha256"].items():
        assert sha(HERE / path) == expected
    with (HERE / "curve_data.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 712
    assert [int(float(r["update"])) for r in rows] == list(range(1, 713))
    assert rows[0]["raw_delta_ratio"] == rows[0]["adaptive_delta_ratio"] == ""
    assert int(float(rows[-1]["sequences"])) == 311
    for name in ("raw", "adaptive"):
        a, b, da, db = [np.array([float(r[name + "_" + k]) for r in rows])
                        for k in ("A", "B", "delta_A", "delta_B")]
        np.testing.assert_allclose(np.cumsum(da), a, rtol=1e-12)
        np.testing.assert_allclose(np.cumsum(db), b, rtol=1e-12)
        last = results["streams"][name]["terminal_windows"]["100"]
        np.testing.assert_allclose(last["mean_delta_A"], (a[710] - a[610]) / 100, rtol=1e-12)
        np.testing.assert_allclose(last["mean_delta_B"], (b[710] - b[610]) / 100, rtol=1e-12)
    weights = np.array([float(r["supervised_tokens"]) for r in rows[-31:]])
    losses = np.array([float(r["training_loss"]) for r in rows[-31:]])
    np.testing.assert_allclose(float(rows[-1]["loss_trailing31"]), np.average(losses, weights=weights), rtol=1e-12)
    synthetic = {}
    for name, vectors in {
        "aligned": [[1, 0], [2, 0], [3, 0]],
        "alternating": [[1, 0], [-1, 0], [1, 0]],
        "flat_half_ratio_with_growing_B": [[1, np.sqrt(3)], [1, -np.sqrt(3)], [.25, np.sqrt(63) / 4]],
    }.items():
        vectors = np.array(vectors)
        a = np.cumsum(np.linalg.norm(vectors, axis=1))
        b = np.linalg.norm(np.cumsum(vectors, axis=0), axis=1)
        da, db = np.diff(a), np.diff(b)
        observed = np.diff(b / a)
        predicted = (db - (b[:-1] / a[:-1]) * da) / a[1:]
        np.testing.assert_allclose(observed, predicted, atol=1e-14)
        assert np.all(np.abs(db) <= da + 1e-14)
        if name == "aligned":
            np.testing.assert_allclose(observed, 0, atol=1e-14)
            assert np.all(db > 0)
        if name == "flat_half_ratio_with_growing_B":
            np.testing.assert_allclose(b[-2:] / a[-2:], [.5, .5], atol=1e-14)
            np.testing.assert_allclose([da[-1], db[-1]], [2, 1], atol=1e-14)
        synthetic[name] = {"A": a.tolist(), "B": b.tolist(), "ratio": (b / a).tolist()}
    z = 1.0
    quadratic_loss = []
    for _ in range(5):
        quadratic_loss.append(z * z / 2)
        z -= .5 * z
    assert all(a > b for a, b in zip(quadratic_loss, quadratic_loss[1:]))
    synthetic["aligned_SGD_constant_ratio_and_decreasing_loss"] = {
        "constant_ratio": 1.0, "losses": quadratic_loss}
    rendered = []
    for path in sorted((HERE / "figures").glob("*.pdf")):
        with fitz.open(path) as document:
            assert len(document) == 1
            page = document[0]
            assert page.get_text().strip()
            target = HERE / "scratch" / (path.stem + "_pdf.png")
            page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).save(target)
            rendered.append(str(target.relative_to(HERE)).replace("\\", "/"))
    report = {"passed": True, "source_evidence_unchanged": True,
              "csv_rows": len(rows), "synthetic_cases": synthetic,
              "pdf_renders": rendered, "verification_script_sha256": sha(Path(__file__))}
    (HERE / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "source_evidence_unchanged": True,
                      "csv_rows": len(rows), "synthetic_cases": len(synthetic),
                      "pdf_renders": rendered}, indent=2))


if __name__ == "__main__":
    main()
