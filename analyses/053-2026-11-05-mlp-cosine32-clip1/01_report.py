"""Compare each treatment with the matched MLP baseline and test combined-policy identity."""
import hashlib
import importlib.util
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "runs"
NEW = RUNS / "230-2026-11-05-mlp-cosine32-clip1"
PAYLOAD = Path("artifacts/spark/payload")
BASE = RUNS / "223-2026-11-04-multispike-interventions" / PAYLOAD / "worker"
CLIP = RUNS / "228-2026-11-04-mlp-global-clipping" / PAYLOAD / "worker"
COMBINED = NEW / PAYLOAD / "worker"
ARMS = {"Baseline": BASE / "control", "Clipping only": CLIP / "global_clip1",
        "Cosine only": BASE / "cosine32_half", "Cosine + clipping": COMBINED / "cosine32_half_clip1"}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_sources():
    count = 0
    for worker in (BASE, CLIP, COMBINED):
        payload = worker.parent
        receipt = json.loads((payload.parent / "receipt.json").read_text())
        assert receipt["verified"] and receipt["status"] == "completed"
        assert file_hash(payload.parent / "output.tar.gz") == receipt["sha256"]
        for entry in json.loads((payload / "transfer_inventory.json").read_text())["files"]:
            path = payload / entry["path"]
            assert path.stat().st_size == entry["bytes"] and file_hash(path) == entry["sha256"]
            count += 1
    return count


def main():
    files_checked = verify_sources()
    prefix = read_rows(BASE / "prefix/boundaries.jsonl")[:-1]
    data = {name: prefix + read_rows(path / "boundaries.jsonl") for name, path in ARMS.items()}
    metrics = {name: read_rows(path / "metrics.jsonl") for name, path in ARMS.items()}
    for name in ARMS:
        assert [row["state"] for row in data[name]] == list(range(801))
        assert [row["state"] for row in metrics[name]] == list(range(512, 800))
    for left, right in zip(data["Cosine only"][512:], data["Cosine + clipping"][512:]):
        assert {k: v for k, v in left.items() if k != "utc"} == {k: v for k, v in right.items() if k != "utc"}
    assert metrics["Cosine only"] == metrics["Cosine + clipping"]
    for offset, row in enumerate(metrics["Cosine + clipping"]):
        expected = .001 * (.5 + .25 * (1 + np.cos(np.pi * min(offset, 32) / 32)))
        assert np.isclose(row["applied_lr"], expected, rtol=0, atol=1e-18)
        assert row["clip_factor"] == 1 and row["raw_gradient_norm"] < 1
        assert row["optimizer_input_norm"] == row["raw_gradient_norm"]
    verifier = load_helper("verify_terminal", NEW / "03_verify.py")
    checkpoints = [torch.load(ARMS[name] / "end.pt", map_location="cpu", weights_only=False)
                   for name in ["Cosine only", "Cosine + clipping"]]
    for key in ["model", "optimizer", "rng", "previous_gradient", "current_gradient", "completed_updates"]:
        verifier.eq(checkpoints[0][key], checkpoints[1][key])
    detector = load_helper("spikes", ROOT / "analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py")
    summary = {}
    for name, boundaries in data.items():
        loss = np.array([row["loss"] for row in boundaries])
        events = [event for event in detector.excursions(loss) if event["peak_state"] > 512
                  and loss[event["peak_state"]] > 1.1 * loss[event["before_state"]]]
        summary[name] = {"mean": float(loss[513:].mean()), "maximum": float(loss[513:].max()),
                         "last32": float(loss[-32:].mean()), "endpoint": float(loss[-1]),
                         "peak_states_gt10": [event["peak_state"] for event in events],
                         "clipped_updates": sum(row["clip_factor"] < 1 for row in metrics[name]),
                         "largest_raw_input": max(row["raw_gradient_norm"] for row in metrics[name])}

    panels = [("loss", "Training loss (log scale)"), ("rho", r"Raw gradient $\rho_t$"),
              ("raw_gradient_norm", "Gradient norm"), ("eta_kappa_hat", r"$\eta\widehat{\kappa}_t$"),
              ("applied_lr", "Applied LR / base LR")]
    fig, axes = plt.subplots(5, 3, figsize=(16, 10), sharex="col", layout="constrained")
    values = []
    for col, arm in enumerate(list(ARMS)[1:]):
        for ax, (key, label) in zip(axes[:, col], panels):
            for name, color in [("Baseline", "#777777"), (arm, "#0072B2")]:
                stop = 800 if key == "applied_lr" else 801
                states = np.arange(448, stop)
                yy = [data[name][i][key] / (.001 if key == "applied_lr" else 1.) for i in states]
                values.extend((arm, name, key, i, y) for i, y in zip(states, yy))
                ax.plot(states, yy, color=color, lw=.95, label=name,
                        drawstyle="steps-post" if key == "applied_lr" else "default")
            if key == "raw_gradient_norm" and arm != "Cosine only":
                inputs = metrics[arm]
                vals = [row["optimizer_input_norm"] for row in inputs]
                ax.plot(range(512, 800), vals, color="#009E73", lw=1.05, label="After clipping")
                values.extend((arm, arm, "optimizer_input_norm", row["state"], val) for row, val in zip(inputs, vals))
                ax.legend(["Baseline raw", "Treatment raw", "Adam input"], loc="upper left", fontsize=7)
            ax.axvspan(448, 512, color="#efefef", zorder=-3)
            ax.axvline(512, color="#555555", lw=.7)
            ax.grid(alpha=.15)
            ax.set(xlim=(448, 800), ylabel=label)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, col].set_title(arm)
        axes[0, col].set_yscale("log")
        axes[4, col].set(xlabel="Completed optimizer updates", xticks=[448, 512, 544, 608, 672, 736, 800])
    fig.suptitle("MLP / CIFAR-10     |     Cosine decay to half LR and global clipping at 1.0", fontsize=15)
    fig.supxlabel("Gray = baseline, blue = treatment, green = recorded Adam input. Each panel uses its own y range.\n"
                  "Shaded prefix through state 512 on the same Spark. The cosine reaches half LR at state 544.", fontsize=10)
    pdf = HERE / "figures/mlp_cosine32_clip1.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    with fitz.open(pdf) as doc:
        assert len(doc) == 1 and not doc[0].get_images()
        doc[0].get_pixmap(matrix=fitz.Matrix(1.35, 1.35)).save(HERE / "scratch/preview.png")
    pd.DataFrame(values, columns=["panel", "series", "metric", "state", "value"]).to_csv(HERE / "plotted_values.csv", index=False)
    result = dict(summary=summary, matched_cosine_boundaries=289, matched_cosine_updates=288,
                  exact_cosine_endpoint_checkpoint_fields=True, verified_files=files_checked,
                  plotted_values=len(values), pdf_sha256=file_hash(pdf))
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
