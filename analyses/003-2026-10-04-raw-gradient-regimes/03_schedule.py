"""Descriptive LR/recovery analysis of completed pilot logs; no interventions."""
from pathlib import Path
import csv
import importlib.util
import json
import sys
import unittest

sys.dont_write_bytecode = True
import numpy as np
import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


analysis = module(HERE / "01_analyze.py", "raw_analysis")
FROZEN = ROOT / "runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/artifacts/attempt-02/source/baseline"
FROZEN_TRAINING = FROZEN.parent / "training.py"
recipe = module(FROZEN / "recipe.py", "frozen_recipe")


def recovery(cosines, width=31, hold=31):
    """First held positive centered mean after midpoint, with full windows only."""
    means = analysis.rolling(cosines, width)
    half = width // 2
    for i in range(356, len(cosines)-half-hold+1):
        if np.all(means[i:i+hold] > 0):
            return i+1
    return None


def phase(rows, first, last):
    selected = rows[first-1:last]
    return {"first_update": first, "last_update": last,
            **analysis.sign_stats([r["adjacent_cosine"] for r in selected]),
            "loss_token_weighted": float(np.average([r["loss"] for r in selected],
                                        weights=[r["supervised_tokens"] for r in selected])),
            "first_preceding_lr": rows[first-2]["learning_rate"] if first > 1 else None,
            "last_preceding_lr": rows[last-2]["learning_rate"]}


class Mathematics(unittest.TestCase):
    def test_positive_scaling_preserves_cosine(self):
        u, v = np.array([1., 2.]), np.array([-3., 1.])
        cosine = lambda x, y: np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
        self.assertAlmostEqual(cosine(u, v), cosine(0.1*u, 0.001*v))

    def test_quadratic_schedule_changes_sign_without_instability(self):
        # Fixed positive curvature lambda=1: both eta values contract parameters.
        theta = 1.0
        for eta, sign in [(1.5, -1), (0.5, 1)]:
            following = (1-eta)*theta
            self.assertEqual(np.sign(theta*following), sign)
            self.assertLess(following**2/2, theta**2/2)
            theta = following

    def test_recovery_rejects_brief_excursion_and_incomplete_windows(self):
        values = -np.ones(711)
        values[360:370] = 1
        values[450:] = 1
        self.assertEqual(recovery(values, width=1, hold=31), 451)
        values = -np.ones(711)
        values[-20:] = 1
        self.assertIsNone(recovery(values, width=1, hold=31))


def main():
    tests = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Mathematics))
    assert tests.wasSuccessful()
    cohort, matched = analysis.load_cohort()
    base = ROOT / "runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/artifacts/attempt-02/full-pass"
    cfg = analysis.read(base / "resolved_config.json")["training"]
    schedule = np.array([r["learning_rate"] for r in cohort[0]["rows"]])
    expected = [recipe.recipe_learning_rate(i, peak=cfg["peak_learning_rate"], max_steps=712,
                warmup_fraction=cfg["warmup_fraction"], minimum=cfg["minimum_learning_rate"])
                for i in range(1, 713)]
    # Linux capture versus Windows math.cos can differ at machine precision.
    np.testing.assert_allclose(schedule, expected, rtol=1e-14, atol=0)
    floor = next(i+1 for i, rate in enumerate(schedule)
                 if i > 8 and rate == cfg["minimum_learning_rate"])
    assert floor == 575
    reduced, aggregates = [], []
    for rec in cohort:
        rows = rec["rows"]
        np.testing.assert_array_equal([r["learning_rate"] for r in rows], schedule)
        c, _ = analysis.arrays(rows[:-1])
        means = analysis.rolling(c)
        recovered = recovery(c)
        windows = {str(w): recovery(c, width=w) for w in [15, 31, 61]}
        aggregate = {"model": rec["label"], "run_id": rec["run_id"],
                     "full_pass_median_cosine": analysis.sign_stats(analysis.arrays(rows)[0])["median_cosine"],
                     "prefix_1_534_median_cosine": analysis.sign_stats(c[:534])["median_cosine"],
                     "recovery_center_update": recovered,
                     "preceding_lr_at_recovery": rows[recovered-2]["learning_rate"] if recovered else None,
                     "recovery_width_sensitivity": windows,
                     "phases": [phase(rows, a, b) for a, b in
                                [(179, 356), (357, 500), (501, 575), (576, 711),
                                 (576, 620), (621, 665), (666, 711)]]}
        aggregates.append(aggregate)
        for i, row in enumerate(rows):
            reduced.append({"model": rec["label"], "update": row["update"],
                            "current_update_lr": row["learning_rate"],
                            "preceding_update_lr": rows[i-1]["learning_rate"] if i else None,
                            "raw_cosine": row["adjacent_cosine"],
                            "centered_mean31": float(means[i]) if i < 711 else None,
                            "training_loss": row["loss"], "displayed": i < 711})
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), layout="constrained")
    axes[0].plot(np.arange(2, 712), schedule[:710]*1e3, color="#444444", lw=2)
    axes[0].set(title="(a) Identical LR schedule at every size",
                ylabel="Preceding optimizer LR (units of 0.001)", ylim=(-0.02, 1.04))
    for rec, color, style in zip(cohort, analysis.COLORS, analysis.STYLES):
        c, _ = analysis.arrays(rec["rows"][:-1])
        axes[1].plot(np.arange(1, 712), analysis.rolling(c), color=color, ls=style,
                     lw=2, label=rec["label"])
    axes[1].set(title="(b) Different late alignment at the same LR",
                ylabel="Adjacent raw-gradient cosine (31-update mean)", ylim=(-0.6, 1.02))
    axes[1].axhline(0, color="#666666", lw=1, ls=":")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.axvspan(576, 711, color="#888888", alpha=0.10)
        ax.axvline(576, color="#666666", lw=0.8, ls=":")
        ax.set(xlabel="Current gradient / optimizer boundary", xlim=(1, 711))
        ax.grid(alpha=0.18)
    axes[0].text(582, 0.48, "Fixed LR\n0.0001", fontsize=10)
    fig.suptitle("LR decay is consistent with recovery, but does not isolate its cause\nSeed 1234; updates 1-711; shading: preceding update already at LR floor", fontsize=12)
    pdf = HERE / "figures/schedule_and_raw_cosine.pdf"
    fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    doc = fitz.open(pdf)
    assert len(doc) == 1
    page = doc[0]
    for block in page.get_text("blocks"):
        assert page.rect.contains(fitz.Rect(block[:4]))
    page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(str(HERE / "scratch/schedule_and_raw_cosine.png"))
    doc.close()
    with (HERE / "schedule_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reduced[0]))
        writer.writeheader()
        writer.writerows(reduced)
    result = {"hypothesis": "H004", "scope": "Exploratory schedule association; no causal identification",
              "script_sha256": analysis.sha(Path(__file__)),
              "source_lineage": [{k: v for k, v in rec.items() if k not in ["rows", "summary"]}
                                 for rec in cohort],
              "frozen_recipe": {"path": (FROZEN / "recipe.py").relative_to(ROOT).as_posix(),
                                "sha256": analysis.sha(FROZEN / "recipe.py")},
              "frozen_training": {"path": FROZEN_TRAINING.relative_to(ROOT).as_posix(),
                                  "sha256": analysis.sha(FROZEN_TRAINING)},
              "schedule_config": cfg, "lr_floor_first_optimizer_update": floor,
              "first_gradient_pair_preceded_by_floor_lr": floor+1,
              "recovery_contract": "First 31 consecutive strictly positive centered means after update356; full windows only; descriptive, retrospective, nonpermanent",
              "statistics": "Unsmoothed cosines; equal pair weights. Phase losses supervised-token weighted.",
              "coverage": {"rows": 2136, "figure_updates": [1, 711], "all_data_updates": [1, 712]},
              "aggregate": aggregates,
              "verification": {"mathematical_tests_passed": tests.testsRun,
                               "all_three_recorded_schedules_equal": True,
                               "schedule_matches_frozen_implementation": True,
                               "recomputed_schedule_max_absolute_error": float(np.max(np.abs(schedule-np.array(expected)))),
                               "pdf_text_bounds_ok": True,
                               "figure_sha256": analysis.sha(pdf)}}
    (HERE / "schedule_results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
