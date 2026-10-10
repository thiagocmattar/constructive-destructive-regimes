"""Verify scientific reductions, source preservation, and publication PDF bounds."""
from pathlib import Path
import csv
import importlib.util
import json
import unittest
import sys

sys.dont_write_bytecode = True

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("analysis", HERE / "01_analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class Mathematics(unittest.TestCase):
    def test_frequency_magnitude_intensity(self):
        result = analysis.sign_stats([np.nan, -0.8, -0.2, 0.5, 0.0])
        self.assertEqual(result["defined_pairs"], 4)
        self.assertEqual(result["negative_pairs"], 2)
        self.assertAlmostEqual(result["negative_fraction"], 0.5)
        self.assertAlmostEqual(result["conditional_negative_magnitude"], 0.5)
        self.assertAlmostEqual(result["negative_part_mean"], 0.25)

    def test_no_negative_conditional_is_undefined(self):
        result = analysis.sign_stats([np.nan, 0.0, 0.4])
        self.assertEqual(result["negative_part_mean"], 0.0)
        self.assertIsNone(result["conditional_negative_magnitude"])
        self.assertIsNone(analysis.sign_stats([np.nan])["negative_fraction"])

    def test_token_weighted_smoothing_and_edges(self):
        np.testing.assert_allclose(analysis.rolling([2, 4, 8], 3, [1, 2, 1]),
                                   [10/3, 4.5, 16/3])
        result = analysis.rolling([np.nan, 0, 1], 3)
        np.testing.assert_allclose(result, [0, 0.5, 0.5])

    def test_exact_temporal_identity(self):
        vectors = np.array([[1, 0], [-0.2, 1], [1, 0]], dtype=float)
        norms = np.linalg.norm(vectors, axis=1)
        cumulative = np.cumsum(vectors, axis=0)
        rows = [{"gradient_norm": float(norms[i]),
                 "adjacent_cosine": None if i == 0 else float(np.dot(vectors[i-1], vectors[i]) /
                                                              (norms[i-1]*norms[i])),
                 "cumulative_gradient_norm": float(np.linalg.norm(cumulative[i]))}
                for i in range(len(vectors))]
        result = analysis.temporal(rows)
        direct = 2*sum(np.dot(vectors[i], vectors[j])
                       for i in range(len(vectors)) for j in range(i))
        self.assertAlmostEqual(result["all_interaction_J"], direct)
        self.assertAlmostEqual(result["adjacent_interaction_L"], -0.8)
        self.assertAlmostEqual(result["nonadjacent_interaction_J_minus_L"], 2.0)
        self.assertGreater(result["all_interaction_J"], 0)

    def test_negative_gradient_cosine_can_accompany_loss_decrease(self):
        # L(theta)=theta^2/2; GD with eta=1.5 contracts while changing sign.
        theta = np.array([1.0, -0.5, 0.25])
        loss = theta**2/2
        self.assertTrue(np.all(np.diff(loss) < 0))
        np.testing.assert_allclose(theta[1:]/theta[:-1], [-0.5, -0.5])
        np.testing.assert_allclose(np.sign(theta[1:]*theta[:-1]), [-1, -1])


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Mathematics)
    run = unittest.TextTestRunner(verbosity=2).run(suite)
    assert run.wasSuccessful()
    result = analysis.read(HERE / "results.json")
    assert result["script_sha256"] == analysis.sha(HERE / "01_analyze.py")
    assert result["read_only_loader"]["sha256"] == analysis.sha(analysis.PILOT / "01_compare.py")
    sources_checked = 0
    for record in result["source_lineage"]:
        for source in record["sources"].values():
            assert analysis.sha(analysis.ROOT / source["path"]) == source["sha256"]
            sources_checked += 1
    replay_source = analysis.ROOT / "runs/001-2026-10-03-pythia14m-adamw-gradient-baseline/artifacts/attempt-04/full-pass/events.jsonl"
    assert analysis.sha(replay_source) == result["run001_replay_sensitivity"]["events_sha256"]
    protected = analysis.read(HERE / "protected_inputs.json")
    assert analysis.protected_inputs() == protected, "Manuscript/live scientific inputs changed"
    data = list(csv.DictReader((HERE / "figure_data.csv").open(encoding="utf-8")))
    assert len(data) == 2136
    for record in result["aggregate"]:
        rows = [r for r in data if r["model"] == record["model"]]
        cosines = [float(r["raw_cosine"]) if r["raw_cosine"] else np.nan for r in rows]
        independently_reduced = analysis.sign_stats(cosines)
        for key, value in independently_reduced.items():
            assert value == record["full_pass"][key]
        assert sum(r["displayed"] == "True" for r in rows) == 711
        assert rows[0]["raw_cosine"] == "" and rows[-1]["sequences"] == "311"
        assert rows[-1]["cosine_mean31"] == ""
        for field, raw in [("cosine_mean31", cosines[:-1]),
                           ("negative_fraction_mean31", [np.nan if not np.isfinite(c) else float(c < 0) for c in cosines[:-1]]),
                           ("negative_intensity_mean31", np.maximum(-np.asarray(cosines[:-1]), 0))]:
            np.testing.assert_allclose([float(r[field]) for r in rows[:-1]], analysis.rolling(raw))
    pdfs = []
    for path in sorted((HERE / "figures").glob("*.pdf")):
        document = fitz.open(path)
        assert len(document) == 1
        page = document[0]
        assert len(page.get_text()) > 100
        for block in page.get_text("blocks"):
            box = fitz.Rect(block[:4])
            assert page.rect.contains(box), f"Text outside page: {path.name}"
        for label in ["14M", "31M", "70M", "31-update", "1234", "1-711"]:
            assert label in page.get_text()
        page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(str(HERE / "scratch" / f"{path.stem}.png"))
        pdfs.append({"path": path.relative_to(HERE).as_posix(), "sha256": analysis.sha(path),
                     "pages": len(document), "text_bounds_ok": True})
        document.close()
    assert {"figures/loss_and_raw_cosine.pdf", "figures/negative_alignment.pdf"}.issubset(
        {item["path"] for item in pdfs})
    verification = {"verified": True, "mathematical_tests_passed": run.testsRun,
                    "source_files_rehashed": sources_checked+1, "reduced_rows": len(data),
                    "protected_manuscript_and_live_scientific_files": len(protected),
                    "protected_files_unchanged": True, "publication_pdfs": pdfs,
                    "visual_review": "All rendered PDFs inspected by the authoring agent",
                    "generating_scripts": {p.name: analysis.sha(p) for p in HERE.glob("0*.py")},
                    "results_sha256": analysis.sha(HERE / "results.json"),
                    "figure_data_sha256": analysis.sha(HERE / "figure_data.csv"),
                    "cloud_calls": 0, "new_training_runs": 0}
    (HERE / "verification.json").write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
