"""Regenerate the gradient figure, omitting the final partial batch from display."""

import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def identity(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    run = Path(__file__).resolve().parent
    figures, scratch = run / "figures", run / "scratch"
    source = figures / "figure_data.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    evidence = run / "artifacts" / "attempt-04" / "full-pass"
    if not json.loads((evidence / "verification.json").read_text())["verified"]:
        raise ValueError("Require accepted scientific verification.")
    if identity(evidence / "events.jsonl") != data["source_events_sha256"]:
        raise ValueError("Recorded source identity changed.")
    all_rows = data["figure_rows"]
    if [r["update"] for r in all_rows] != list(range(1, 713)):
        raise ValueError("Require the complete original 712-update reduction.")
    rows = all_rows[:-1]
    if all_rows[-1]["sequences"] != 311 or any(r["sequences"] != 1024 for r in rows):
        raise ValueError("Unexpected batch coverage.")
    output = figures / "gradient_evolution.pdf"
    original = figures / "gradient_evolution_all_updates.pdf"
    if not original.exists():
        shutil.copy2(output, original)
    scratch.mkdir(exist_ok=True)
    x = np.array([r["update"] for r in rows])

    def values(key):
        return np.array([np.nan if r[key] is None else r[key] for r in rows], dtype=float)

    cosine = values("adjacent_cosine")
    rolling = np.array([np.nanmedian(cosine[max(1, i - 24):i + 1]) if i else np.nan
                        for i in range(len(rows))])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x, cosine, color="#2462a3", alpha=.5, lw=.7, label="Every displayed update")
    ax.plot(x, rolling, color="#133c68", lw=1.6, label="Trailing 25-update median")
    ax.axhline(0, color="black", lw=.8, ls="--")
    ax.set(ylabel="Adjacent raw-gradient cosine", ylim=(-1.05, 1.05))
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(x, values("cumulative_norm_sum"), color="#d2771c", lw=1.5,
            label=r"$A_t=\sum_{u\leq t}\|g_u\|_2$")
    ax.plot(x, values("cumulative_gradient_norm"), color="#2462a3", lw=1.5, ls="--",
            label=r"$B_t=\|\sum_{u\leq t}g_u\|_2$")
    ax.set(ylabel="Cumulative gradient norms", ylim=(0, None))
    ax.legend(fontsize=9)
    axes[1, 0].plot(x, values("cumulative_ratio"), color="#28794d", lw=1.5)
    axes[1, 0].set(ylabel=r"Cumulative ratio $R_t=B_t/A_t$", ylim=(0, 1.05))
    axes[1, 1].plot(x, values("gradient_norm"), color="#8052a0", lw=.9)
    axes[1, 1].set(ylabel=r"Current raw-gradient norm $\|g_t\|_2$", ylim=(0, None))
    for ax in axes.flat:
        ax.set_xlabel("Optimizer update")
        ax.grid(alpha=.18)
    fig.suptitle("Pythia-14M / MiniPile: raw-gradient geometry under AdamW\n"
                 "Seed 1234; updates 1–711; final partial batch (step 712) omitted", fontsize=12)
    fig.savefig(output)
    fig.savefig(scratch / "gradient_evolution.png", dpi=180)
    plt.close(fig)
    metadata = {
        "source_figure_data_sha256": identity(source),
        "source_events_sha256": data["source_events_sha256"],
        "script_sha256": identity(Path(__file__)),
        "source_updates": len(all_rows), "displayed_updates": len(rows),
        "excluded_updates": [{"update": 712, "sequences": 311,
                              "reason": "User-requested display omission of final partial batch"}],
        "filter_scope": "All four gradient-evolution panels only; no source data or statistics changed",
        "cumulative_values": "Original prefix values through each displayed update; no recomputation",
        "pdf_sha256": identity(output), "preserved_full_pass_pdf_sha256": identity(original),
        "figure_rows": rows,
    }
    (figures / "gradient_evolution_plot_data.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k != "figure_rows"}, indent=2))


if __name__ == "__main__":
    main()
