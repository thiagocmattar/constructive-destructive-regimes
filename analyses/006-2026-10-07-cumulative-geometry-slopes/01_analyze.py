"""Reanalyze Run002 cumulative-norm slopes without changing source evidence."""

import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry"
SOURCE = RUN / "artifacts/attempt-02/full-pass"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def slope(values):
    values = np.asarray(values, dtype=float)
    x = np.arange(len(values), dtype=float)
    x -= x.mean()
    return float(np.dot(x, values - values.mean()) / np.dot(x, x))


def trailing(values, window, regression=False):
    output = np.full(len(values), np.nan)
    for stop in range(window - 1, len(values)):
        selected = values[stop - window + 1:stop + 1]
        if np.all(np.isfinite(selected)):
            output[stop] = slope(selected) if regression else np.mean(selected)
    return output


def finite_difference(values):
    return np.diff(np.r_[0.0, values])


def summarize(stream, loss, first, last, loss_weights):
    selected = slice(first - 1, last)
    a, b, ratio = (stream[k][selected] for k in ("A", "B", "ratio"))
    da, db = (stream[k][selected] for k in ("delta_A", "delta_B"))
    log_a, log_b = (stream[k][selected] for k in ("delta_log_A", "delta_log_B"))
    margin = stream["marginal_ratio"][selected]
    prefix_cos = stream["cosine_to_previous_sum"][selected]
    return {
        "first_update": first, "last_update": last, "updates": last - first + 1,
        "mean_delta_A": float(da.mean()), "mean_delta_B": float(db.mean()),
        "ols_A_slope": slope(a), "ols_B_slope": slope(b),
        "slope_ratio_sum_delta_B_over_sum_delta_A": float(db.sum() / da.sum()),
        "ratio_first": float(ratio[0]), "ratio_last": float(ratio[-1]),
        "ratio_change": float(ratio[-1] - ratio[0]),
        "ratio_ols_slope": slope(ratio),
        "ratio_mean_backward_difference": float(np.nanmean(stream["delta_ratio"][selected])),
        "mean_delta_log_A": float(np.nanmean(log_a)),
        "mean_delta_log_B": float(np.nanmean(log_b)),
        "mean_delta_log_ratio": float(np.nanmean(log_b - log_a)),
        "mean_plateau_required_delta_B": float(np.nanmean(stream["plateau_required_delta_B"][selected])),
        "median_marginal_ratio": float(np.median(margin)),
        "negative_delta_B_updates": int(np.sum(db < 0)),
        "negative_prefix_dot_updates": int(np.sum(prefix_cos < 0)),
        "mean_cosine_to_previous_sum": float(np.nanmean(prefix_cos)),
        "mean_loss": float(np.average(loss[selected], weights=loss_weights[selected])),
        "loss_ols_slope": slope(loss[selected]),
    }


def main():
    design = json.loads((HERE / "design.json").read_text())
    paths = [
        SOURCE / "events.jsonl", SOURCE / "summary.json", SOURCE / "verification.json",
        SOURCE / "resolved_config.json", SOURCE / "input_identity.json",
        RUN / "config.json", RUN / "04_analyze.py",
        RUN / "figures/figure_data.json", RUN / "figures/raw_adaptive_comparison.pdf",
    ]
    protected = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths}
    if not json.loads((SOURCE / "verification.json").read_text())["verified"]:
        raise ValueError("Source run must be scientifically verified.")
    figure_data = json.loads((RUN / "figures/figure_data.json").read_text())
    if protected[str((SOURCE / "events.jsonl").relative_to(ROOT)).replace("\\", "/")] != figure_data["source_events_sha256"]:
        raise ValueError("Events differ from the source figure's recorded identity.")
    events = [json.loads(line) for line in (SOURCE / "events.jsonl").read_text().splitlines()]
    rows = [e for e in events if e["event"] == "update"]
    if rows != figure_data["figure_rows"]:
        raise ValueError("Source figure rows differ from original update events.")
    assert [e["update"] for e in rows] == list(range(1, 713))
    assert all(e["sequences"] == 1024 for e in rows[:-1]) and rows[-1]["sequences"] == 311
    loss = np.array([e["loss"] for e in rows])
    loss_weights = np.array([e["supervised_tokens"] for e in rows])
    loss_trailing31 = trailing(loss * loss_weights, 31) / trailing(loss_weights, 31)
    steps = np.arange(1, 713)
    streams, checks = {}, {}
    for name in ("raw", "adaptive"):
        vectors = rows if name == "raw" else [e["adaptive"] for e in rows]
        norm_key = "gradient_norm" if name == "raw" else "direction_norm"
        b_key = "cumulative_gradient_norm" if name == "raw" else "cumulative_direction_norm"
        a = np.array([e["cumulative_norm_sum"] for e in vectors])
        b = np.array([e[b_key] for e in vectors])
        norm = np.array([e[norm_key] for e in vectors])
        ratio = b / a
        da, db = finite_difference(a), finite_difference(b)
        previous_a, previous_b = np.r_[0, a[:-1]], np.r_[0, b[:-1]]
        previous_ratio = np.r_[np.nan, ratio[:-1]]
        dr = np.r_[np.nan, np.diff(ratio)]
        required = previous_ratio * da
        identity_dr = (db - required) / a
        log_a, log_b = np.r_[np.nan, np.diff(np.log(a))], np.r_[np.nan, np.diff(np.log(b))]
        prefix_cos = np.full(712, np.nan)
        prefix_cos[1:] = (b[1:]**2 - previous_b[1:]**2 - norm[1:]**2) / (2 * previous_b[1:] * norm[1:])
        streams[name] = {
            "A": a, "B": b, "ratio": ratio, "vector_norm": norm,
            "delta_A": da, "delta_B": db, "delta_ratio": dr,
            "marginal_ratio": db / da, "plateau_required_delta_B": required,
            "delta_log_A": log_a, "delta_log_B": log_b,
            "cosine_to_previous_sum": prefix_cos,
        }
        checks[name] = {
            "max_relative_delta_A_vs_norm_error": float(np.max(np.abs(da - norm) / norm)),
            "max_ratio_difference_identity_error": float(np.nanmax(np.abs(dr - identity_dr))),
            "max_saved_ratio_error": float(np.max(np.abs(ratio - [e["cumulative_ratio"] for e in vectors]))),
            "triangle_increment_bound_max_excess": float(np.max(np.abs(db) - da)),
            "prefix_cosine_min": float(np.nanmin(prefix_cos)),
            "prefix_cosine_max": float(np.nanmax(prefix_cos)),
        }
        assert checks[name]["max_relative_delta_A_vs_norm_error"] < 1e-10
        assert checks[name]["max_ratio_difference_identity_error"] < 1e-12
        assert np.all(np.abs(db) <= da + 1e-8)
        assert np.nanmax(np.abs(prefix_cos)) <= 1 + 1e-8
    fixed = design["summary_intervals"]
    results = {
        "analysis_id": "006", "hypothesis_id": "H006",
        "source_hashes_before": protected,
        "source_sizes_bytes": {str(p.relative_to(ROOT)).replace("\\", "/"): p.stat().st_size for p in paths},
        "design_sha256": sha(HERE / "design.json"), "script_sha256": sha(Path(__file__)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "coverage": {"primary_updates": 711, "retained_updates": 712, "seed": 1234,
                     "initial_ratio_derivative": None, "fixed_validation_evaluations": 2,
                     "training_supervised_tokens": int(sum(e["supervised_tokens"] for e in rows)),
                     "validation_supervised_tokens": 693329, "validation_documents": 500},
        "mathematical_checks": checks,
        "validation": {e["event"]: e for e in events if e["event"] != "update"},
        "streams": {},
        "loss_smoothing_sensitivity": {},
    }
    for name, stream in streams.items():
        minimum = int(np.argmin(stream["ratio"][:711]) + 1)
        results["streams"][name] = {
            "fixed_intervals": [summarize(stream, loss, lo, hi, loss_weights) for lo, hi in fixed],
            "terminal_windows": {str(w): summarize(stream, loss, 712 - w, 711, loss_weights) for w in design["terminal_windows"]},
            "terminal_with_partial_sensitivity": {str(w): summarize(stream, loss, 713 - w, 712, loss_weights) for w in design["terminal_windows"]},
            "ratio_minimum": {"update": minimum, "ratio": float(stream["ratio"][minimum - 1]),
                              "centered61_summary": summarize(stream, loss, minimum - 30, minimum + 30, loss_weights)},
            "full_pass": summarize(stream, loss, 1, 712, loss_weights),
            "smoothing_sensitivity_last_displayed": {
                str(w): {"mean_delta_A": float(trailing(stream["delta_A"][:711], w)[-1]),
                         "mean_delta_B": float(trailing(stream["delta_B"][:711], w)[-1]),
                         "ratio_ols_slope": float(trailing(stream["ratio"][:711], w, True)[-1])}
                for w in design["sensitivity_windows"]},
        }
    for w in design["sensitivity_windows"]:
        results["loss_smoothing_sensitivity"][str(w)] = {
            "last_displayed_trailing_mean": float(trailing(loss[:711], w)[-1]),
            "last_displayed_ols_slope": float(trailing(loss[:711], w, True)[-1]),
        }
    results["late_loss_adjacent100_blocks"] = {
        "updates512_611": {"mean": float(loss[511:611].mean()), "ols_slope": slope(loss[511:611])},
        "updates612_711": {"mean": float(loss[611:711].mean()), "ols_slope": slope(loss[611:711])},
    }
    fields = {"update": steps, "sequences": [e["sequences"] for e in rows],
              "supervised_tokens": [e["supervised_tokens"] for e in rows],
              "learning_rate": [e["learning_rate"] for e in rows], "training_loss": loss,
              "loss_trailing31": loss_trailing31, "loss_ols31_slope": trailing(loss, 31, True)}
    for name, stream in streams.items():
        fields.update({name + "_" + key: value for key, value in stream.items()})
        for key in ("delta_A", "delta_B", "plateau_required_delta_B"):
            fields[name + "_" + key + "_trailing31"] = trailing(stream[key], 31)
        fields[name + "_ratio_ols31_slope"] = trailing(stream["ratio"], 31, True)
    with (HERE / "curve_data.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(fields))
        writer.writeheader()
        for i in range(712):
            writer.writerow({k: None if not np.isfinite(v[i]) else float(v[i]) for k, v in fields.items()})
    figures, scratch = HERE / "figures", HERE / "scratch"
    figures.mkdir(exist_ok=True)
    scratch.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    x = steps[:711]
    colors = {"A": "#d2771c", "B": "#2462a3", "raw": "#2462a3", "adaptive": "#d2771c"}

    def save(fig, name):
        fig.savefig(figures / (name + ".pdf"))
        fig.savefig(scratch / (name + ".png"), dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    for col, (name, stream) in enumerate(streams.items()):
        label = "Raw gradients" if name == "raw" else "Adam adaptive directions"
        axes[0, col].set_title(label)
        for key in ("A", "B"):
            axes[0, col].plot(x, stream[key][:711], color=colors[key], ls="-" if key == "A" else "--", label=key)
            values = stream["delta_" + key][:711]
            axes[1, col].plot(x, values, color=colors[key], alpha=.15, lw=.65)
            axes[1, col].plot(x, trailing(values, 31), color=colors[key], ls="-" if key == "A" else "--",
                              label="Slope of " + key + " (31-update mean)", lw=1.5)
        axes[1, col].plot(x, trailing(stream["plateau_required_delta_B"][:711], 31), color="#333333", ls=":",
                          label="B slope required for flat B/A", lw=1.3)
        axes[2, col].plot(x, stream["ratio"][:711], color="#28794d", label="Cumulative B/A")
        recent_ratio = trailing(stream["delta_B"][:711], 31) / trailing(stream["delta_A"][:711], 31)
        axes[2, col].plot(x, recent_ratio, color="#8052a0", ls="--", label="Recent slope ratio (31 updates)")
        axes[2, col].axhline(0, color="black", lw=.6)
        axes[2, col].set_ylim(-.1, 1.05)
        for row in range(3):
            axes[row, col].grid(alpha=.18)
            axes[row, col].legend(fontsize=8, loc="best")
        axes[0, col].set_ylabel("Cumulative norm")
        axes[1, col].set_ylabel("Norm increment per update")
        axes[2, col].set(ylabel="Dimensionless ratio", xlabel="Optimizer update")
    fig.suptitle("Run002: cumulative A/B and their per-update slopes\n"
                 "Pythia-14M, seed 1234; updates 1-711; Adam direction excludes LR and decay", fontsize=12)
    save(fig, "cumulative_slopes")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    axes[0].plot(x, loss[:711], color="#999999", alpha=.25, lw=.6, label="Changing-batch training loss")
    axes[0].plot(x, loss_trailing31[:711], color="#222222", lw=1.5, label="Trailing 31-update token-weighted mean")
    axes[1].plot(x, trailing(loss[:711], 31, True), color="#222222", label="Trailing 31-update least-squares loss slope")
    axes[1].set_yscale("symlog", linthresh=.001)
    axes[1].axhline(0, color="black", ls=":", lw=.8)
    for name, stream in streams.items():
        axes[2].plot(x, trailing(stream["ratio"][:711], 31, True), color=colors[name],
                     ls="-" if name == "raw" else "--", label=name + " B/A slope (trailing 31 updates)")
    axes[2].set_yscale("symlog", linthresh=.0001)
    axes[2].axhline(0, color="black", ls=":", lw=.8)
    for ax in axes:
        ax.grid(alpha=.18)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Training cross entropy")
    axes[1].set_ylabel("Loss units per update\n(symmetric log scale)")
    axes[2].set(ylabel="Ratio change per update\n(symmetric log scale)", xlabel="Optimizer update")
    fig.suptitle("Loss slowing and cumulative-ratio slopes\n"
                 "One changing-batch trajectory; no intermediate fixed-validation evaluations", fontsize=12)
    save(fig, "ratio_slope_and_loss")
    protected_after = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths}
    assert protected == protected_after, "Source evidence changed."
    results["source_hashes_after"] = protected_after
    results["source_evidence_unchanged"] = True
    results["outputs_sha256"] = {str(p.relative_to(HERE)).replace("\\", "/"): sha(p)
                                for p in [HERE / "curve_data.csv", *sorted(figures.glob("*.pdf"))]}
    (HERE / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"source_unchanged": True, "checks": checks,
                      "last100": {k: v["terminal_windows"]["100"] for k, v in results["streams"].items()},
                      "loss_sensitivity": results["loss_smoothing_sensitivity"],
                      "ratio_minima": {k: v["ratio_minimum"] for k, v in results["streams"].items()}}, indent=2))


if __name__ == "__main__":
    main()
