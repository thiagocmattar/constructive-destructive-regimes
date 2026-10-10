"""Read completed runs only; reduce loss and raw-gradient alignment locally."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import platform
import sys

sys.dont_write_bytecode = True

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT = ROOT / "analyses/001-2026-10-03-model-scale-geometry"
COLORS = ["#0072B2", "#D55E00", "#009E73"]
STYLES = ["-", "--", "-."]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protected_inputs():
    paths = [p for p in (ROOT / "manuscript").rglob("*")
             if p.is_file() and "artifacts" not in p.parts]
    # Scientific inputs of the live replications only, never their changing logs.
    for number in range(5, 11):
        run = next((ROOT / "runs").glob(f"{number:03d}-*"))
        paths += list(run.glob("*.py")) + list(run.glob("*config*.json"))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def arrays(rows):
    return (np.array([np.nan if r["adjacent_cosine"] is None else
                      r["adjacent_cosine"] for r in rows], dtype=float),
            np.array([r["gradient_norm"] for r in rows], dtype=float))


def rolling(values, width=31, weights=None):
    """Centered descriptive mean; shrink edge windows; ignore undefined values."""
    values = np.asarray(values, dtype=float)
    weights = np.ones(len(values)) if weights is None else np.asarray(weights)
    result = np.full(len(values), np.nan)
    for i in range(len(values)):
        start, stop = max(0, i-width//2), min(len(values), i+width//2+1)
        v, w = values[start:stop], weights[start:stop]
        valid = np.isfinite(v)
        if valid.any():
            result[i] = np.average(v[valid], weights=w[valid])
    return result


def sign_stats(cosines):
    c = np.asarray(cosines, dtype=float)
    c = c[np.isfinite(c)]
    negative = c[c < 0]
    if not len(c):
        return {"defined_pairs": 0, "negative_pairs": 0,
                "negative_fraction": None, "negative_part_mean": None,
                "conditional_negative_magnitude": None,
                "mean_cosine": None, "median_cosine": None}
    frequency = float(np.mean(c < 0))
    intensity = float(np.maximum(-c, 0).mean())
    magnitude = float((-negative).mean()) if len(negative) else None
    if magnitude is not None:
        assert np.isclose(intensity, frequency*magnitude, atol=1e-14)
    return {"defined_pairs": len(c), "negative_pairs": len(negative),
            "negative_fraction": frequency, "negative_part_mean": intensity,
            "conditional_negative_magnitude": magnitude,
            "mean_cosine": float(c.mean()), "median_cosine": float(np.median(c)),
            "fraction_below_threshold": {str(t): float(np.mean(c < -t))
                                         for t in [0.05, 0.10, 0.25]}}


def temporal(rows):
    c, n = arrays(rows)
    energy = float(np.dot(n, n))
    lag_one = float(2*np.sum(c[1:]*n[:-1]*n[1:]))
    exact = float(rows[-1]["cumulative_gradient_norm"]**2)
    all_pairs = exact-energy
    return {"energy_E": energy, "adjacent_interaction_L": lag_one,
            "all_interaction_J": all_pairs, "nonadjacent_interaction_J_minus_L": all_pairs-lag_one,
            "exact_resultant_squared_B2": exact, "lag_one_approximation_E_plus_L": energy+lag_one,
            "lag_one_relative_error": (energy+lag_one-exact)/exact,
            "adjacent_interaction_over_energy": lag_one/energy,
            "all_interaction_over_energy": all_pairs/energy}


def summary(rows):
    c, _ = arrays(rows)
    last = rows[-31:]
    return {**sign_stats(c), "updates": len(rows),
            "last_31_training_loss": float(np.average([r["loss"] for r in last],
                                              weights=[r["supervised_tokens"] for r in last])),
            "last_training_batch_loss": rows[-1]["loss"],
            "terminal_raw_ratio": rows[-1]["cumulative_ratio"],
            "temporal": temporal(rows)}


def load_cohort():
    spec = importlib.util.spec_from_file_location("pilot_compare", PILOT / "01_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cohort, contract = module.load_cohort()  # Read-only loader with matched-protocol checks.
    frozen = read(PILOT / "results.json")
    for current, previous in zip(cohort, frozen["source_lineage"]):
        assert current["sources"] == previous["sources"], "Completed source changed"
        assert [r["update"] for r in current["rows"]] == list(range(1, 713))
        assert all(r["adjacent_cosine"] is not None for r in current["rows"][1:])
        assert current["rows"][0]["adjacent_cosine"] is None
        assert current["summary"]["completed_supervised_tokens"] == 1490983041
        assert current["summary"]["final_validation"]["supervised_tokens"] == 693329
    for key in ["sequences", "input_tokens", "supervised_tokens", "cumulative_input_tokens",
                "cumulative_supervised_tokens", "learning_rate"]:
        reference = [r[key] for r in cohort[0]["rows"]]
        assert all([r[key] for r in record["rows"]] == reference for record in cohort)
    return cohort, contract


def save_figure(fig, name):
    fig.savefig(HERE / "figures" / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def plots(cohort):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.3), layout="constrained")
    for record, color, style in zip(cohort, COLORS, STYLES):
        rows = record["rows"][:-1]
        x = np.array([r["update"] for r in rows])
        c, _ = arrays(rows)
        loss = rolling([r["loss"] for r in rows], weights=[r["supervised_tokens"] for r in rows])
        ax[0].plot(x, loss, color=color, ls=style, lw=2, label=record["label"])
        ax[1].plot(x, rolling(c), color=color, ls=style, lw=2, label=record["label"])
    ax[0].set(title="(a) Training loss", ylabel="Mean next-token cross entropy (nats)")
    ax[1].set(title="(b) Adjacent raw-gradient alignment", ylabel="Cosine similarity", ylim=(-0.6, 1.02))
    ax[1].axhline(0, color="#666666", ls=":", lw=1)
    ax[1].text(22, -0.54, "Negative = local vector cancellation", fontsize=9, color="#555555")
    for a in ax:
        a.set(xlabel="Optimizer update (matched data order)", xlim=(1, 711))
        a.grid(alpha=0.18)
        a.legend(frameon=False)
    fig.suptitle("Pythia 14M / 31M / 70M on MiniPile\n31-update centered means; seed 1234; full-sized updates 1-711", fontsize=12)
    save_figure(fig, "loss_and_raw_cosine")

    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.3), layout="constrained")
    for record, color, style in zip(cohort, COLORS, STYLES):
        c, _ = arrays(record["rows"][:-1])
        indicator = np.where(np.isfinite(c), (c < 0).astype(float), np.nan)
        x = np.arange(1, len(c)+1)
        ax[0].plot(x, rolling(indicator), color=color, ls=style, lw=2, label=record["label"])
        ax[1].plot(x, rolling(np.maximum(-c, 0)), color=color, ls=style, lw=2, label=record["label"])
    ax[0].set(title="(a) Frequency of negative pairs", ylabel="Fraction with cosine < 0", ylim=(-0.02, 1.02))
    ax[1].set(title="(b) Negative alignment intensity", ylabel="Mean max(-cosine, 0)", ylim=(-0.01, 0.55))
    for a in ax:
        a.set(xlabel="Optimizer update (matched data order)", xlim=(1, 711))
        a.grid(alpha=0.18)
        a.legend(frameon=False)
    fig.suptitle("How often and how strongly do adjacent raw gradients oppose?\n31-update centered means; seed 1234; full-sized updates 1-711", fontsize=12)
    save_figure(fig, "negative_alignment")


def main():
    for folder in ["figures", "scratch", "observations"]:
        (HERE / folder).mkdir(exist_ok=True)
    snapshot = HERE / "protected_inputs.json"
    if not snapshot.exists():
        snapshot.write_text(json.dumps(protected_inputs(), indent=2)+"\n", encoding="utf-8")
    cohort, contract = load_cohort()
    aggregate, data = [], []
    for record in cohort:
        rows = record["rows"]
        c, n = arrays(rows)
        full = summary(rows)
        display = summary(rows[:-1])
        quarters = [{"first_update": a, "last_update": b,
                     **sign_stats(c[a-1:b]),
                     "training_loss_token_weighted": float(np.average([r["loss"] for r in rows[a-1:b]],
                                                    weights=[r["supervised_tokens"] for r in rows[a-1:b]]))}
                    for a, b in [(1, 178), (179, 356), (357, 534), (535, 712)]]
        windows = {str(w): {"minimum_smoothed_cosine": float(np.nanmin(rolling(c[:-1], w))),
                           "fraction_centers_with_negative_smoothed_cosine": float(np.mean(rolling(c[:-1], w) < 0))}
                   for w in [15, 31, 61]}
        aggregate.append({"model": record["label"], "run_id": record["run_id"],
                          "parameters": record["parameters"], "full_pass": full,
                          "full_sized_updates": display, "quarters": quarters,
                          "smoothing_sensitivity": windows,
                          "initial_validation_loss": record["summary"]["initial_validation"]["loss"],
                          "final_validation_loss": record["summary"]["final_validation"]["loss"]})
        display_c = c[:-1]
        indicator = np.where(np.isfinite(display_c), (display_c < 0).astype(float), np.nan)
        fields = {"loss_mean31": rolling([r["loss"] for r in rows[:-1]],
                                        weights=[r["supervised_tokens"] for r in rows[:-1]]),
                  "cosine_mean31": rolling(display_c), "negative_fraction_mean31": rolling(indicator),
                  "negative_intensity_mean31": rolling(np.maximum(-display_c, 0))}
        for i, row in enumerate(rows):
            data.append({"model": record["label"], "parameters": record["parameters"],
                         "update": row["update"], "input_tokens": row["input_tokens"],
                         "supervised_tokens": row["supervised_tokens"], "sequences": row["sequences"],
                         "cumulative_input_tokens": row["cumulative_input_tokens"],
                         "training_loss": row["loss"], "raw_cosine": row["adjacent_cosine"],
                         "raw_gradient_norm": float(n[i]), "raw_ratio": row["cumulative_ratio"],
                         "displayed": i < 711,
                         **{k: float(v[i]) if i < 711 else None for k, v in fields.items()}})
    replay_base = ROOT / "runs/001-2026-10-03-pythia14m-adamw-gradient-baseline/artifacts/attempt-04/full-pass"
    assert read(replay_base / "verification.json")["verified"]
    replay = [json.loads(line) for line in (replay_base / "events.jsonl").read_text().splitlines()
              if json.loads(line)["event"] == "update"]
    assert len(replay) == 712
    replay_results = {"note": "Same-seed 14M replay sensitivity; not an independent seed replicate",
                      "full_pass": summary(replay), "full_sized_updates": summary(replay[:-1]),
                      "final_validation_loss": read(replay_base / "summary.json")["final_validation"]["loss"],
                      "events_sha256": sha(replay_base / "events.jsonl")}
    result = {"schema_version": 1, "hypothesis": "H003", "script_sha256": sha(Path(__file__)),
              "read_only_loader": {"path": (PILOT / "01_compare.py").relative_to(ROOT).as_posix(),
                                   "sha256": sha(PILOT / "01_compare.py")},
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "matplotlib": matplotlib.__version__},
              "matched_contract": contract,
              "source_lineage": [{k: v for k, v in r.items() if k not in ["rows", "summary"]} for r in cohort],
              "coverage": {"statistics_updates": [1, 712], "figure_updates": [1, 711],
                           "seed": 1234, "smoothing": "31-update centered mean; shrunken edges; nulls omitted",
                           "loss_smoothing": "Supervised-token weighted; equal weights for displayed full batches",
                           "negative_statistics": "Computed from unsmoothed cosine values; equal pair weights"},
              "aggregate": aggregate, "run001_replay_sensitivity": replay_results}
    (HERE / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (HERE / "figure_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    plots(cohort)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
