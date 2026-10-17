"""Overlay saved Adam-direction geometry, optionally smoothing only loss change."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STEM = "training_size_adam_direction_split_n5_n21"
REFERENCE = "training_size_overlay_log_loss_diff_split_n5_n21"
FIELDS = ("log_loss_diff", "direction_change", "adaptive_rho", "adaptive_norm")
RANGES = ((1, 100), (100, 400))
COLORS = ("#2166ac", "#d95f02", "#1b9e77")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def spec(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=sha(path))


def trailing_mean(values, window):
    result = np.full(len(values), np.nan)
    result[window - 1:] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-reviewed", action="store_true")
    parser.add_argument("--loss-only-smoothing", action="store_true",
                        help="Leave q_d, rho_d and direction norms unsmoothed; retain N=5/N=21 loss means.")
    args = parser.parse_args()
    stem = "training_size_adam_direction_split_loss_only_n5_n21" if args.loss_only_smoothing else STEM
    pdf = HERE / "figures" / (stem + ".pdf")
    manifest = HERE / (stem + "_manifest.json")
    data = HERE / (stem + "_data.csv")
    export_names = {f: f if args.loss_only_smoothing and f != "log_loss_diff" else f + "_mean" for f in FIELDS}
    if args.accept_reviewed:
        record = json.loads(manifest.read_text())
        assert sha(__file__) == record["script_sha256"]
        for item in (record["output"], record["plot_data"]):
            assert spec(ROOT / item["path"]) == item
        render = HERE / "scratch" / (stem + "-1.png")
        assert render.is_file()
        record["visual_review"] = dict(passed=True, renderer="Poppler pdftoppm",
            render_sha256=sha(render), inspection="All eight panels, solid model colours, direction definitions, independent Y limits and row-specific smoothing notes are legible without overlap or clipping.")
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("Recorded visual acceptance of the Adam-direction figure.")
        return

    reference_manifest = HERE / (REFERENCE + "_manifest.json")
    reference = json.loads(reference_manifest.read_text())
    assert reference["visual_review"]["passed"]
    for item in (reference["output"], reference["plot_data"]):
        p = HERE / item["path"]
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"]
    with (HERE / reference["plot_data"]["path"]).open(newline="", encoding="utf-8") as stream:
        reference_rows = {(r["size"], int(r["column"]), int(r["update"])): r
                          for r in csv.DictReader(stream)}
    raw_loss_csv = HERE / reference["source_csv"]
    assert sha(raw_loss_csv) == reference["source_csv_sha256"]
    with raw_loss_csv.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    preserved = {p.relative_to(HERE).as_posix(): sha(p) for p in [
        *sorted((HERE / "figures").glob("*.pdf")), raw_loss_csv,
        reference_manifest, HERE / reference["plot_data"]["path"],
        HERE / "15_plot_split_windows.py"] if p != pdf}
    if args.loss_only_smoothing:
        previous = json.loads((HERE / (STEM + "_manifest.json")).read_text())
        assert previous["visual_review"]["passed"]
        for item in (previous["output"], previous["plot_data"]):
            assert spec(ROOT / item["path"]) == item
            preserved[(ROOT / item["path"]).relative_to(HERE).as_posix()] = item["sha256"]
        preserved[STEM + "_manifest.json"] = sha(HERE / (STEM + "_manifest.json"))
    cohort = json.loads((HERE / "cohort.json").read_text())
    accepted = {r["run"]: r for r in json.loads((HERE / "results.json").read_text())["sources"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "path.simplify": False})
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex="col")
    sources, curves, plot_rows = [], [], []
    loss_points_checked, raw_direction_points = 0, 0
    for member, color in zip(cohort, COLORS):
        run, size = member["run"], member["size"]
        run_path = ROOT / "runs" / run
        attempt = json.loads((run_path / "prelaunch/lease.json").read_text())["attempt"]
        receipt = run_path / "prelaunch" / (attempt + "-retrieval.json")
        assert sha(receipt) == accepted[run]["receipt_sha256"]
        assert json.loads(receipt.read_text())["verified"]
        base = run_path / "artifacts" / attempt
        inventory = {r["path"]: r for r in json.loads((base / "inventory.json").read_text())["files"]}
        source_files = []
        for name in ("events.jsonl", "resolved_config.json", "input_identity.json", "summary.json"):
            p = base / "full-pass" / name
            assert sha(p) == accepted[run]["sources"][name] == inventory["full-pass/" + name]["sha256"]
            assert p.stat().st_size == inventory["full-pass/" + name]["bytes"]
            source_files.append(spec(p))
        config = json.loads((base / "full-pass/resolved_config.json").read_text())
        assert config["seeds"] == {"model": 1235, "data_order": 1235}
        assert config["measurements"]["adaptive_direction_dtype"] == "float64_from_FP32_moments"
        assert config["measurements"]["adaptive_capture"] == "after_optimizer_step_from_actual_updated_FP32_moments"
        assert config["measurements"]["adaptive_moment_input"] == "globally_clipped_gradient"
        events = [json.loads(line) for line in (base / "full-pass/events.jsonl").read_text().splitlines()]
        updates = [e for e in events if e["event"] == "update"]
        np.testing.assert_array_equal([e["update"] for e in updates], np.arange(1, 713))
        selected = updates[:400]
        assert all(e["sequences"] == 1024 for e in selected)
        assert selected[0]["learning_rate"] == 0
        x = np.arange(1, 401)
        loss = np.array([e["loss"] for e in selected])
        norm = np.array([e["adaptive"]["direction_norm"] for e in selected])
        rho = np.array([np.nan if e["adaptive"]["adjacent_cosine"] is None else e["adaptive"]["adjacent_cosine"] for e in selected])
        assert np.isfinite(loss).all() and (loss > 0).all()
        assert np.isfinite(norm).all() and (norm > 0).all()
        assert np.isnan(rho[0]) and np.isfinite(rho[1:]).all() and (np.abs(rho[1:]) <= 1).all()
        loss_diff = np.r_[np.nan, np.diff(np.log(loss))]
        q = np.r_[np.nan, 1 - rho[1:] * norm[1:] / norm[:-1]]
        old = [r for r in raw_rows if r["size"] == size]
        np.testing.assert_array_equal([int(r["update"]) for r in old], x)
        np.testing.assert_array_equal([float(r["loss"]) for r in old], loss)
        np.testing.assert_array_equal([float(r["log_loss_diff"]) if r["log_loss_diff"] else np.nan for r in old], loss_diff)
        values = dict(zip(FIELDS, (loss_diff, q, rho, norm)))
        sources.append(dict(run=run, size=size, attempt=attempt, receipt=spec(receipt),
                            files=source_files, capture=config["measurements"]))
        for col, (first, last) in enumerate(RANGES):
            window = (5, 21)[col]
            displayed = {field: raw if args.loss_only_smoothing and field != "log_loss_diff" else trailing_mean(raw, window)
                         for field, raw in values.items()}
            keep = (x >= first) & (x <= last)
            for row, field in enumerate(FIELDS):
                window_x, window_y = x[keep], displayed[field][keep]
                indices = np.flatnonzero(keep)[np.isfinite(window_y)]
                if args.loss_only_smoothing and row != 0:
                    np.testing.assert_array_equal(window_y, values[field][keep])
                    raw_direction_points += len(indices)
                else:
                    np.testing.assert_allclose(window_y[np.isfinite(window_y)],
                        [np.mean(values[field][i - window + 1:i + 1]) for i in indices], rtol=1e-12, atol=1e-12)
                if row == 0:
                    np.testing.assert_allclose(window_y[np.isfinite(window_y)],
                        np.log(loss[indices] / loss[indices - window]) / window, rtol=1e-12, atol=1e-12)
                    old_y = [reference_rows[(size, col + 1, int(t))]["log_loss_diff_mean"] for t in window_x]
                    np.testing.assert_array_equal(window_y, [float(v) if v else np.nan for v in old_y])
                    loss_points_checked += len(indices)
                line, = axes[row, col].plot(window_x, window_y, color=color, ls="-", lw=1.15,
                                           label="Pythia-" + size)
                np.testing.assert_array_equal(line.get_xdata(), window_x)
                np.testing.assert_array_equal(line.get_ydata(), window_y)
                assert line.get_linestyle() == "-" and line.get_color() == color
                expected = ((100 if row == 3 else (95 if row == 0 else 99)) if args.loss_only_smoothing
                            else (96 if row == 3 else 95)) if col == 0 else 301
                assert np.isfinite(window_y).sum() == expected
                curves.append(dict(size=size, metric=field, column=col + 1,
                    updates=[first, last], smoothing_window=1 if args.loss_only_smoothing and row != 0 else window, finite_points=expected,
                    minimum=float(np.nanmin(window_y)), maximum=float(np.nanmax(window_y)),
                    arrays_sha256=hashlib.sha256(window_x.tobytes() + window_y.tobytes()).hexdigest()))
            for i in np.flatnonzero(keep):
                windows = dict(loss_window=window, direction_window=1) if args.loss_only_smoothing else dict(window=window)
                plot_rows.append(dict(size=size, column=col + 1, **windows, update=int(x[i]), **{
                    export_names[field]: float(displayed[field][i]) if np.isfinite(displayed[field][i]) else "" for field in FIELDS}))
    labels = ("Change in ln(training loss)",
              "Projected direction change\n" + r"$q_t^{(d)}$",
              r"Adam-direction cosine $\rho_t^{(d)}$", r"Adam direction $\|d_t\|_2$")
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label)
        for col, (first, last) in enumerate(RANGES):
            ax = axes[row, col]
            ax.set_xlim(first, last)
            ax.set_xticks([1, 25, 50, 75, 100] if col == 0 else [100, 150, 200, 250, 300, 350, 400])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
            if row == 0:
                ax.axhline(0, color=".4", ls=":", lw=.9)
            elif row in (1, 2):
                # A distant reference must not compress small direction changes/cosines.
                limits = ax.get_ylim()
                for level in ((0, 1) if row == 1 else (0,)):
                    if limits[0] <= level <= limits[1]:
                        ax.axhline(level, color=".4", ls=":", lw=.9)
                ax.set_ylim(limits)
            else:
                ax.set_yscale("log")
                ticks = ([1000, 1500, 2000, 3000, 5000, 7000] if col == 0 else
                         [1200, 1400, 1600, 1800, 2000, 2200, 2400])
                ax.yaxis.set_major_locator(FixedLocator(ticks))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
        if row == 0:
            qualifier = "loss " if args.loss_only_smoothing else ""
            axes[row, 0].set_title("Updates 1-100  |  " + qualifier + "N=5", pad=12, fontsize=13)
            axes[row, 1].set_title("Updates 100-400  |  " + qualifier + "N=21", pad=12, fontsize=13)
    for ax in axes[-1]:
        ax.set_xlabel("Optimizer update")
    fig.suptitle("Pythia training dynamics: Adam adaptive directions", y=.985, fontsize=16)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .955),
               ncol=3, frameon=False, handlelength=3)
    fig.text(.5, .062, r"$d_t=\widehat m_t/(\sqrt{\widehat v_t}+\epsilon)$; $q_t^{(d)}=1-\rho_t^{(d)}\|d_t\|/\|d_{t-1}\|$ (direction change, not measured curvature).",
             ha="center", fontsize=9)
    note = (r"Loss change only: N=5 left, N=21 right; $\ln L_t-\ln L_{t-1}$. Other rows unsmoothed. Seed 1235. LR and decay excluded from d."
            if args.loss_only_smoothing else r"Complete trailing means: N=5 left, N=21 right. Loss uses $\ln L_t-\ln L_{t-1}$. Seed 1235. LR and decay excluded from d.")
    fig.text(.5, .043, note, ha="center", fontsize=9)
    fig.text(.5, .025, "Y limits vary by column; norm row uses log Y. Update 100 appears twice; loss smoothing spans the boundary.", ha="center", fontsize=9)
    detail = ("Norms start at update 1; direction pairs at 2; left loss means at 6. No extra averaging of Adam directions."
              if args.loss_only_smoothing else "Left means start at update 5 (norm) / 6 (other rows). Direction cosines use a narrow Y range; all three are positive here.")
    fig.text(.5, .008, detail, ha="center", fontsize=9)
    fig.subplots_adjust(left=.12, right=.98, top=.89, bottom=.125, hspace=.20, wspace=.23)
    fig.savefig(pdf)
    plt.close(fig)
    point_count = 4791 if args.loss_only_smoothing else 4755
    assert sum(c["finite_points"] for c in curves) == point_count
    assert loss_points_checked == 1188
    assert raw_direction_points == (3603 if args.loss_only_smoothing else 0)
    assert all(c["minimum"] > 0 for c in curves if c["metric"] == "adaptive_rho")
    with data.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)
    with data.open(newline="", encoding="utf-8") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == len(plot_rows) == 1203
    for field in FIELDS:
        key = export_names[field]
        np.testing.assert_array_equal([float(r[key]) if r[key] != "" else np.nan for r in plot_rows],
                                      [float(r[key]) if r[key] else np.nan for r in exported])
    assert all(sha(HERE / name) == digest for name, digest in preserved.items())
    record = dict(script_sha256=sha(__file__), sources=sources,
        reference_manifest=spec(reference_manifest), preserved=preserved, curves=curves,
        output=spec(pdf), plot_data=spec(data), columns=list(RANGES), row_metrics=list(FIELDS),
        smoothing_windows=[5, 21], smoothing_scope="loss change only" if args.loss_only_smoothing else "all rows",
        direction_smoothing_window=1 if args.loss_only_smoothing else [5, 21],
        y_limits="independent by column; narrow cosine Y ranges; log norm",
        direction_change_formula="q_d[t] = 1 - rho_d[t] * norm_d[t] / norm_d[t-1]",
        direction_change_interpretation="normalized projected direction change; no measured Hessian or SGD curvature claim",
        checks=dict(accepted_source_hashes_and_bytes=True, accepted_updates_per_model=712,
            visible_updates_per_model=400, full_batches_only=True, plotted_arrays=True,
            finite_plotted_points=point_count, independent_means_checked=1188 if args.loss_only_smoothing else 4755,
            raw_direction_points_checked=raw_direction_points,
            loss_points_identical_to_raw_gradient_reference=loss_points_checked,
            log_loss_telescoping=True, csv_roundtrip=True, previous_figures_unchanged=True,
            smoothing_not_restarted_at_boundary=True, q_computed_before_smoothing=True), visual_review=None)
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["checks"], indent=2))


if __name__ == "__main__":
    main()
