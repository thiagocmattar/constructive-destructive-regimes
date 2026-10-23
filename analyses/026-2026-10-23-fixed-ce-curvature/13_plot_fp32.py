"""Focused FP32-arm figure from frozen evidence; no training or GPU work."""
import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.fixed_ce_curvature.core import save_json, sha, stamp

ARMS = ("native32", "fixed32")
COLORS = {"native32": "#D55E00", "fixed32": "#0072B2"}
LABELS = {"native32": "Native CE (FP32)", "fixed32": "Fixed CE (FP32)"}
STYLES = {"native32": "--", "fixed32": "-"}
MANIFEST = HERE / "fp32_figure_manifest.json"
DATA = HERE / "fp32_figure_data.json"
PDF = HERE / "figures/fixed_ce_curvature_fp32.pdf"
RENDER = ROOT / "tmp/analysis026/fixed_ce_curvature_fp32.png"


def blocks(x, y):
    """Arithmetic means in consecutive 100-record bins; retain a short tail."""
    x, y = np.asarray(x), np.asarray(y, dtype=float)
    assert x.shape == y.shape and np.isfinite(y).all()
    return {"x": [float(x[i:i+100].mean()) for i in range(0, len(x), 100)],
            "y": [float(y[i:i+100].mean()) for i in range(0, len(y), 100)],
            "counts": [len(y[i:i+100]) for i in range(0, len(y), 100)]}


def reduce_evidence():
    old_manifest = json.loads((HERE / "figure_manifest.json").read_text())
    assert old_manifest["source_results_sha256"] == sha(HERE / "results.json")
    assert old_manifest["source_data_sha256"] == sha(HERE / "figure_data.json")
    old = json.loads((HERE / "figure_data.json").read_text())
    results = json.loads((HERE / "results.json").read_text())
    entries = [r for r in json.loads((HERE / "cohort.json").read_text()) if r["arm"] in ARMS]
    entries.sort(key=lambda r: (ARMS.index(r["arm"]), r["seed"]))
    assert len(entries) == 20
    inventory = {r["path"]: r for r in json.loads((HERE / "artifact_inventory.json").read_text())["files"]}
    sources, records = [], []
    for entry in entries:
        run = ROOT / "runs" / entry["run"]
        output = run / "artifacts/attempt-01"
        for name in ("config.json", "metrics.jsonl.gz", "curvature.jsonl"):
            path = output / name
            relative = path.relative_to(ROOT).as_posix()
            proof = inventory[relative]
            assert path.stat().st_size == proof["bytes"] and sha(path) == proof["sha256"]
            sources.append(proof)
        config = json.loads((output / "config.json").read_text())
        assert config["arm"] == entry["arm"] and config["seed"] == entry["seed"]
        assert config["precision"] == "fp32-model-and-moments"
        eta = config["optimizer"]["lr"]
        assert eta == 0.001
        norms, cosines, losses = [], [], []
        with gzip.open(output / "metrics.jsonl.gz", "rt") as stream:
            for state, line in enumerate(stream):
                row = json.loads(line)
                assert row["gradient_state"] == state
                norms.append(row["raw"]["norm"])
                cosines.append(row["raw"]["cosine"])
                losses.append(row["common_logits_loss"])
        assert len(losses) == 10000 and cosines[0] is None
        norms = np.asarray(norms)
        rho = np.asarray(cosines[1:])
        assert np.isfinite(norms).all() and (norms > 0).all()
        assert np.isfinite(rho).all() and (np.abs(rho) <= 1 + 1e-12).all()
        q = 1 - rho * norms[1:] / norms[:-1]
        # Independent projected-difference algebra using the implied inner product.
        dot = rho * norms[1:] * norms[:-1]
        np.testing.assert_allclose(q, (norms[:-1]**2 - dot) / norms[:-1]**2,
                                   atol=5e-13, rtol=5e-13)
        np.testing.assert_array_equal(q > 1, rho < 0)
        raw = [json.loads(s) for s in (output / "curvature.jsonl").read_text().splitlines()]
        regular = sorted((r for r in raw if r["checkpoint"].startswith("state-")),
                         key=lambda r: r["state"])
        assert [r["state"] for r in regular] == old["states"] and len(regular) == 23
        assert all(r["ordinary"]["precision"] == "whole-model-fp64" and
                   r["ordinary"]["objective_examples"] == 480 and
                   r["ordinary"]["relative_residual"] <= .005 for r in regular)
        seed, arm = entry["seed"], entry["arm"]
        lambdas = np.asarray([r["ordinary"]["value"] for r in regular])
        kappas = np.asarray([r["gradient_curvature"] for r in regular])
        np.testing.assert_array_equal(lambdas, old["curves"][arm]["ordinary_seed_curves"][seed])
        np.testing.assert_array_equal(kappas, old["curves"][arm]["gradient_seed_curves"][seed])
        np.testing.assert_array_equal(np.asarray(losses)[::20], old["curves"][arm]["loss_seed_curves"][seed])
        assert np.isfinite(kappas).all() and (kappas > 0).all()
        assert (kappas <= lambdas * 1.01).all()
        result = next(r for r in results["runs"] if r["arm"] == arm and r["seed"] == seed)
        assert regular[-1]["train_loss"] == result["final_common_loss"]
        records.append({**entry, "eta": eta,
            "loss": blocks(np.arange(10000), losses),
            "terminal_loss": result["final_common_loss"],
            "eta_lambda": (eta * lambdas).tolist(), "eta_kappa": (eta * kappas).tolist(),
            "eta_kappa_hat": blocks(np.arange(1, 10000), q),
            "late1000_eta_kappa_hat_mean": float(q[-1000:].mean()),
            "late1000_eta_kappa_hat_gt1_fraction": float((q[-1000:] > 1).mean()),
            "dense_eta_kappa_hat_range": [float(q.min()), float(q.max())]})
    data = {"utc": stamp(), "arms": list(ARMS), "seeds": list(range(10)),
        "states": old["states"], "eta": .001, "records": records,
        "loss_binning": "100 pre-update states per arithmetic mean; separate state10000 dots",
        "proxy_binning": "100 consecutive gradient pairs per arithmetic mean; last bin retains99",
        "proxy_definition": "q_at_state_s=1-rho(g_s,g_(s-1))*norm(g_s)/norm(g_(s-1)); s=1..9999; g_s is the training gradient at theta_s",
        "kappa_definition": "g^T H g / norm(g)^2; common full-FP64 objective gradient and Hessian at each retained state",
        "source_results_sha256": sha(HERE / "results.json"), "source_files": sources}
    save_json(DATA, data)
    return data


def plot(data):
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
                         "legend.fontsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), sharex=True)
    curve_checks = 0
    for arm in ARMS:
        selected = [r for r in data["records"] if r["arm"] == arm]
        assert [r["seed"] for r in selected] == list(range(10))
        series = [
            (selected[0]["loss"]["x"], [r["loss"]["y"] for r in selected]),
            (data["states"], [r["eta_lambda"] for r in selected]),
            (data["states"], [r["eta_kappa"] for r in selected]),
            (selected[0]["eta_kappa_hat"]["x"], [r["eta_kappa_hat"]["y"] for r in selected])]
        for panel, (ax, (x, y)) in enumerate(zip(axes.ravel(), series)):
            values = np.asarray(y, dtype=float)
            assert values.shape == (10, len(x)) and np.isfinite(values).all()
            for row in values:
                artist, = ax.plot(x, row, color=COLORS[arm], ls=STYLES[arm],
                                  lw=.6, alpha=.23, rasterized=False)
                np.testing.assert_array_equal(artist.get_xdata(), x)
                np.testing.assert_array_equal(artist.get_ydata(), row)
                curve_checks += 1
            median = np.median(values, axis=0)
            artist, = ax.plot(x, median, color=COLORS[arm], ls=STYLES[arm],
                              lw=2, label=LABELS[arm], zorder=4,
                              marker="o" if panel in (1, 2) else None, ms=2.6)
            np.testing.assert_array_equal(artist.get_ydata(), median)
            curve_checks += 1
        axes[0, 0].scatter(np.full(10, 10000), [r["terminal_loss"] for r in selected],
                           s=11, color=COLORS[arm], alpha=.55, zorder=5, clip_on=False)
    titles = ["(a) Training loss", "(b) Maximum Hessian curvature",
              "(c) Gradient-direction Hessian curvature", "(d) Consecutive-gradient proxy"]
    labels = ["Mean cross-entropy", r"$\eta\lambda_{\max}$",
              r"$\eta\kappa=\eta\,g^\top H g/\|g\|^2$", r"$\eta\widehat{\kappa}_t$"]
    for index, (ax, title, label) in enumerate(zip(axes.ravel(), titles, labels)):
        ax.set_title(title, loc="left", fontweight="bold", pad=9)
        ax.set_ylabel(label)
        ax.grid(alpha=.17, lw=.55)
        ax.set_xlim(0, 10100)
        ax.set_xticks([0, 2500, 5000, 7500, 10000])
        ax.ticklabel_format(axis="x", style="plain")
        if index < 3:
            ax.set_yscale("log")
    axes[1, 1].set_yscale("symlog", linthresh=.1, linscale=.7)
    axes[1, 1].axhline(1, color=".35", lw=.8, ls=":")
    axes[1, 1].axhline(0, color=".65", lw=.6)
    axes[1, 1].text(.03, .96, r"$1-\rho_t\,\|g_t\|/\|g_{t-1}\|$", transform=axes[1, 1].transAxes,
                       va="top", fontsize=11, bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    for ax in axes[1]:
        ax.set_xlabel("Completed updates (parameter state)")
    figure.suptitle("Fixed cross-entropy: FP32 training comparison", y=.985,
                    fontsize=15, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .947),
                  ncol=2, frameon=False)
    figure.text(.5, .018,
        r"Mod-31 Transformer | 10 paired seeds | AdamW, $\eta=10^{-3}$ | thin: seeds; thick: median." "\n"
        "Loss / proxy: 100-state / 100-pair means (last proxy bin: 99); loss dots: final state. Proxy axis: symmetric log.\n"
        "Hessian panels: 23 measured states, all 480 train pairs, common FP64 evaluator. "
        r"$\widehat{\kappa}$ is an AdamW trajectory proxy, not a Hessian estimate.",
        ha="center", va="bottom", fontsize=8.1, linespacing=1.55)
    figure.subplots_adjust(left=.095, right=.98, top=.86, bottom=.19, hspace=.34, wspace=.29)
    figure.canvas.draw()
    bounds = figure.get_tightbbox(figure.canvas.get_renderer()).transformed(figure.dpi_scale_trans)
    assert bounds.x0 >= 0 and bounds.y0 >= 0
    assert bounds.x1 <= figure.bbox.width + 1 and bounds.y1 <= figure.bbox.height + 1
    PDF.parent.mkdir(exist_ok=True)
    figure.savefig(PDF, metadata={"Title": "Fixed CE: FP32 training, loss and eta-scaled curvature",
                                 "CreationDate": None, "ModDate": None})
    plt.close(figure)
    RENDER.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([shutil.which("pdftoppm"), "-r", "150", "-singlefile", "-png", str(PDF),
                    str(RENDER.with_suffix(""))], check=True, capture_output=True,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    save_json(MANIFEST, {"utc": stamp(), "status": "awaiting_visual_review",
        "script_sha256": sha(__file__), "data_sha256": sha(DATA), "pdf_sha256": sha(PDF),
        "pdf_bytes": PDF.stat().st_size, "render": RENDER.relative_to(ROOT).as_posix(),
        "render_sha256": sha(RENDER), "arms": list(ARMS), "panels": 4,
        "coverage": {"runs": 20, "seeds_per_arm": 10, "scalar_states_per_run": 10000,
                     "gradient_pairs_per_run": 9999, "curvature_states_per_run": 23,
                     "train_pairs_per_hessian": 480},
        "checks": ["60 frozen input sizes/SHA256", "20 complete clocks and FP32 configurations",
                   "loss/lambda/kappa unchanged from original verified reduction",
                   "eta from all20 optimizer configs", "proxy projected-difference algebra and sign identity",
                   "no zero gradient denominators", "all 88 plotted curves equal source/median arrays",
                   "all23 measured Hessian states; no interpolated new measurements", "figure bounds"],
        "plotted_curves_checked": curve_checks, "new_training": False, "gpu_rental_usd": 0})
    assert curve_checks == 88


def accept_reviewed():
    manifest = json.loads(MANIFEST.read_text())
    for key, path in (("script", Path(__file__)), ("data", DATA), ("pdf", PDF), ("render", RENDER)):
        assert manifest[key + "_sha256"] == sha(path)
    data = json.loads(DATA.read_text())
    assert data["arms"] == list(ARMS) and len(data["records"]) == 20
    for record in data["source_files"]:
        path = ROOT / record["path"]
        assert path.stat().st_size == record["bytes"] and sha(path) == record["sha256"]
    manifest.update(status="passed", visual_review=
        "Current Poppler render reviewed: four readable panels, two FP32 arms, all10 seeds, eta-scaled labels, common evaluator disclosure, signed proxy scale, and no clipped or overlapping content.")
    save_json(MANIFEST, manifest)
    print("FP32 figure: numeric, source, rendered-output and visual checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept-reviewed", action="store_true")
    arguments = parser.parse_args()
    if arguments.accept_reviewed:
        accept_reviewed()
    else:
        plot(reduce_evidence())
        print(PDF)
