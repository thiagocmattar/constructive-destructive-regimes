"""Plot the retained Analysis005 training scalars; no training or sketch reduction."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COLORS = {"14M": "#2166ac", "31M": "#d95f02", "70M": "#1b9e77"}
FIELDS = ("loss", "eta_kappa_proxy", "rho", "raw_gradient_norm")
PDF = HERE / "figures/training_overview.pdf"
MANIFEST = HERE / "training_overview_manifest.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(norm, rho):
    """The dimensionless gradient-only expression; undefined pairs stay NaN."""
    q = np.full(len(norm), np.nan)
    valid = (norm[:-1] > 0) & (norm[1:] > 0) & np.isfinite(rho[1:])
    q[1:] = np.where(valid, 1 - rho[1:] * np.divide(
        norm[1:], norm[:-1], out=np.full(len(norm) - 1, np.nan),
        where=norm[:-1] > 0), np.nan)
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-reviewed", action="store_true",
                        help="Record visual acceptance after inspecting the current Poppler render.")
    args = parser.parse_args()
    if args.accept_reviewed:
        record = json.loads(MANIFEST.read_text())
        assert sha(PDF) == record["outputs"]["figures/training_overview.pdf"]["sha256"]
        assert sha(__file__) == record["script_sha256"]
        for name, spec in record["outputs"].items():
            assert sha(HERE / name) == spec["sha256"]
        render = HERE / "scratch/training_overview-1.png"
        assert render.is_file()
        record["visual_review"] = dict(
            passed=True, renderer="Poppler pdftoppm", render_sha256=sha(render),
            inspection="All twelve panels, shared scales, formulas, endpoint diamonds and notes are legible; no overlap or clipping.")
        MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("Recorded visual acceptance of current PDF.")
        return

    # Small cases exercise the quadratic-GD identity, zero norms, and growing norms.
    np.testing.assert_allclose(identity(np.array([2., 1.]), np.array([np.nan, 1.]))[1], .5)
    np.testing.assert_allclose(identity(np.array([1., 2.]), np.array([np.nan, 1.]))[1], -1.)
    assert np.isnan(identity(np.array([0., 1.]), np.array([np.nan, np.nan]))).all()
    assert np.isnan(identity(np.array([1., 0.]), np.array([np.nan, np.nan]))).all()

    previous = json.loads((HERE / "results.json").read_text())
    cohort = json.loads((HERE / "cohort.json").read_text())
    preserved = {p.relative_to(HERE).as_posix(): sha(p) for p in [
        HERE / "results.json", HERE / "curve_data.csv", HERE / "cohort.json",
        HERE / "artifact_validation.json", *sorted((HERE / "figures").glob("*.pdf"))
    ] if p != PDF}
    data, rows, sources, summaries = [], [], [], []
    matched_protocol = None
    for case in cohort:
        run = ROOT / "runs" / case["run"]
        attempt = json.loads((run / "prelaunch/lease.json").read_text())["attempt"]
        base = run / "artifacts" / attempt / "full-pass"
        receipt_path = run / "prelaunch" / f"{attempt}-retrieval.json"
        receipt = json.loads(receipt_path.read_text())
        old = next(r for r in previous["sources"] if r["run"] == case["run"])
        assert receipt["verified"] and receipt["scientific_verification"]["verified"]
        assert sha(receipt_path) == old["receipt_sha256"]
        inventory = {r["path"]: r for r in json.loads((base.parent / "inventory.json").read_text())["files"]}
        files = {}
        for name in ("events.jsonl", "resolved_config.json", "input_identity.json",
                     "summary.json", "temporal/exact_pairs.jsonl"):
            path = base / name
            expected = inventory["full-pass/" + name]
            digest = sha(path)
            assert path.stat().st_size == expected["bytes"]
            assert digest == expected["sha256"] == old["sources"][name]
            files[name] = dict(bytes=path.stat().st_size, sha256=digest)
        cfg = json.loads((base / "resolved_config.json").read_text())
        input_identity = json.loads((base / "input_identity.json").read_text())
        # The 14M config omits the tokenizer name; all three pinned input identities include it.
        data_contract = dict(cfg["data"])
        data_contract.setdefault("tokenizer", input_identity["train"]["tokenizer_name"])
        protocol = dict(training=cfg["training"], data=data_contract, seeds=cfg["seeds"],
                        measurements=cfg["measurements"],
                        input_identity=input_identity)
        if matched_protocol is None:
            matched_protocol = protocol
        assert protocol == matched_protocol
        events = [json.loads(line) for line in (base / "events.jsonl").read_text().splitlines()]
        updates = [e for e in events if e["event"] == "update"]
        assert [e["update"] for e in updates] == list(range(1, 713))
        assert [e["gradient_step"] for e in updates] == list(range(712))
        assert all(e["overflow_retries"] == 0 for e in updates)
        assert all(e["sequences"] == 1024 for e in updates[:-1]) and updates[-1]["sequences"] == 311
        assert sum(e["input_tokens"] for e in updates) == 1491711416
        assert cfg["seeds"] == dict(model=1235, data_order=1235)
        assert cfg["training"]["optimizer"] == "adamw"
        norm = np.array([e["gradient_norm"] for e in updates], dtype=np.float64)
        rho = np.array([np.nan if e["adjacent_cosine"] is None else e["adjacent_cosine"] for e in updates])
        loss = np.array([e["loss"] for e in updates])
        q = identity(norm, rho)
        assert np.isfinite(norm).all() and (norm > 0).all() and np.isfinite(loss).all()
        assert np.isnan(rho[0]) and np.isnan(q[0])
        assert np.isfinite(rho[1:]).all() and np.isfinite(q[1:]).all()
        # Independent direct-dot form from exact calibration pairs, never projected dots.
        pairs = [json.loads(line) for line in (base / "temporal/exact_pairs.jsonl").read_text().splitlines()]
        adjacent = sorted([p for p in pairs if p["lag"] == 1], key=lambda p: p["target_update"])
        assert [p["target_update"] for p in adjacent] == list(range(2, 713))
        dot = np.array([p["exact_dot"] for p in adjacent])
        np.testing.assert_array_equal([p["source_norm"] for p in adjacent], norm[:-1])
        np.testing.assert_array_equal([p["target_norm"] for p in adjacent], norm[1:])
        np.testing.assert_allclose(dot / (norm[:-1] * norm[1:]), rho[1:], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(1 - dot / norm[:-1] ** 2, q[1:], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(q[1:] > 1, rho[1:] < 0)
        series = dict(size=case["size"], update=np.arange(1, 713), loss=loss,
                      eta_kappa_proxy=q, rho=rho, raw_gradient_norm=norm)
        data.append(series)
        for i, e in enumerate(updates):
            rows.append(dict(size=case["size"], run=case["run"][:3], update=e["update"],
                **{name: float(series[name][i]) if np.isfinite(series[name][i]) else "" for name in FIELDS},
                learning_rate=e["learning_rate"], supervised_tokens=e["supervised_tokens"],
                sequences=e["sequences"], partial_batch=i == 711))
        checkpoints = {}
        for state in ("initial", "midpoint", "final"):
            folder = base / "checkpoints" / state
            assert (folder / "model/model.safetensors").is_file() and (folder / "recovery.pt").is_file()
            checkpoints[state] = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size)
                                  for p in sorted(folder.rglob("*")) if p.is_file()]
            for path in folder.rglob("*"):
                if path.is_file():
                    assert path.stat().st_size == inventory[path.relative_to(base.parent).as_posix()]["bytes"]
        sources.append(dict(run=case["run"], size=case["size"], base=base.relative_to(ROOT).as_posix(),
                            receipt_sha256=sha(receipt_path), files=files, checkpoints_present=checkpoints))
        negative = np.flatnonzero(rho < 0)
        summaries.append(dict(size=case["size"], run=case["run"][:3], updates=712, adjacent_pairs=711,
            first_negative_pair_update=int(negative[0] + 1) if len(negative) else None,
            negative_pairs=int(len(negative)), negative_pair_fraction=float(len(negative) / 711),
            initial_training_loss=float(loss[0]), final_batch_training_loss=float(loss[-1]),
            final_validation_loss=receipt["scientific_verification"]["final_validation_loss"],
            eta_kappa_range=[float(np.nanmin(q)), float(np.nanmax(q))],
            raw_gradient_norm_range=[float(norm.min()), float(norm.max())],
            final_raw_gradient_norm=float(norm[-1])))

    csv_path = HERE / "training_overview_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Check the exported decimals before comparing each plotted array below.
    with csv_path.open(newline="", encoding="utf-8") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 2136
    for series in data:
        selected = [r for r in exported if r["size"] == series["size"]]
        for field in FIELDS:
            np.testing.assert_array_equal([float(r[field]) if r[field] else np.nan for r in selected], series[field])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "path.simplify": False})
    fig, axes = plt.subplots(4, 3, figsize=(12.6, 10.5), sharex=True, sharey="row")
    curves = []
    for col, series in enumerate(data):
        color = COLORS[series["size"]]
        for row, field in enumerate(FIELDS):
            ax = axes[row, col]
            x, y = series["update"], series[field]
            line, = ax.plot(x, y, color=color, lw=1.05)
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
            ax.scatter(x[-1], y[-1], color=color, marker="D", s=23, zorder=5, clip_on=False)
            ax.grid(axis="y", alpha=.18)
            ax.set_xlim(0, 725)
            ax.set_xticks([1, 200, 400, 600, 712])
            if field == "eta_kappa_proxy":
                ax.axhline(1, color=".4", ls="--", lw=.8)
            if field == "rho":
                ax.axhline(0, color=".4", ls="--", lw=.8)
                ax.set_ylim(-1.05, 1.05)
            if field == "raw_gradient_norm":
                ax.set_yscale("log")
                ax.set_xlabel("Optimizer update")
            curves.append(dict(size=series["size"], metric=field, points=int(np.isfinite(y).sum()),
                               arrays_sha256=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()))
        axes[0, col].set_title(f"Pythia-{series['size']}  |  Run{cohort[col]['run'][:3]}", color=color, pad=12)
    for row, label in enumerate(["Training loss", r"$\eta\widehat\kappa_t$ (proxy)",
                                 r"Raw adjacent cosine $\rho_t$", r"Raw gradient $\|g_t\|_2$"]):
        axes[row, 0].set_ylabel(label, labelpad=10)
    fig.suptitle("MiniPile training dynamics across model sizes", fontsize=16, y=.985)
    fig.text(.5, .945, r"$q_t=\eta\widehat\kappa_t=1-\rho_t\,\|g_t\|_2/\|g_{t-1}\|_2$"
             "   |   unsmoothed; seed 1235", ha="center", fontsize=11)
    fig.text(.085, .052, "All 712 updates; first pair undefined. Diamonds: partial final batch (311 vs. 1,024 sequences).", fontsize=9)
    fig.text(.085, .032, "AdamW + changing minibatches: q is a gradient-change proxy; it does not establish curvature or stability.", fontsize=9)
    fig.text(.085, .012, "Dashed references: q = 1 and rho = 0. Shared Y scales within rows; gradient norms use log Y.", fontsize=9)
    fig.subplots_adjust(left=.085, right=.985, top=.89, bottom=.12, hspace=.24, wspace=.10)
    PDF.parent.mkdir(exist_ok=True)
    fig.savefig(PDF)
    plt.close(fig)
    assert sum(c["points"] for c in curves) == 8538
    assert all(sha(HERE / p) == digest for p, digest in preserved.items())
    results = dict(cohort=summaries, updates=2136, defined_adjacent_pairs=2133,
                   finite_plotted_points=8538, formula="1 - rho[t] * norm(g[t]) / norm(g[t-1])",
                   reduction="unsmoothed; all accepted updates; no sketch-based quantities", training_launched=False)
    result_path = HERE / "training_overview_results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    outputs = {p.relative_to(HERE).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
               for p in (PDF, csv_path, result_path)}
    record = dict(script_sha256=sha(__file__), sources=sources, preserved=preserved, outputs=outputs,
        checks=dict(source_sizes_and_sha256=True, matched_data_order_and_training_protocol=True,
                    contiguous_updates=True, direct_dot_identity_pairs=2133,
                    reversal_equivalence_pairs=2133, csv_roundtrip=True, plotted_arrays=True,
                    finite_plotted_points=8538, retained_checkpoint_sizes_only=True,
                    original_outputs_unchanged=True), curves=curves, visual_review=None)
    MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
