"""Verify selected horizon counts directly from raw records and render the audits."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "derived/forward_and_severity"
SPEC = importlib.util.spec_from_file_location("source055", HERE / "01_analyze.py")
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)
FLAGS = ("negative_cosine", "norm_growth", "reversal_and_growth", "Q_ge2")
LABELS = (r"Reversal: $\rho<0$", r"Growth: $r>1$", r"Both reversal + growth", r"Threshold: $Q\geq2$")
COLORS = ("#0072B2", "#B68100", "#7B3294", "#D55E00")
STYLES = ("-", "--", "-.", ":")
ARMS = ("mlp:0.001", "resnet:0.003", "transformer:0.003")
NAMES = ("MLP · LR 0.001", "ResNet · LR 0.003", "Transformer · LR 0.003")


def read_csv(name):
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_raw(rows):
    """Scalar raw-file check at h=1,3,16, independent of the vectorized reductions."""
    conditions = SOURCE.read(SOURCE.RUN / "results.json")["conditions"]
    raw = {}
    for condition in conditions:
        folder = ROOT / condition["source"]
        metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        final = json.loads((folder / "boundaries.jsonl").read_text().splitlines()[-1])
        losses = [m["loss_before"] for m in metrics] + [final["loss"]]
        arm = f'{condition["task"]}:{condition["lr"]:g}'
        raw[arm] = (metrics, losses)
    checked_rows = checked_states = 0
    for row in rows:
        h = int(row["horizon"])
        if h not in (1, 3, 16):
            continue
        metrics, losses = raw[row["arm"]]
        last = 1008 if row["support"] == "common_16_step" else 1024 - h
        minimum = float(row["minimum_relative_increase"])
        n = positives = 0
        alarms = [0] * 4
        tp = [0] * 4
        for s in range(1, last + 1):
            if row["subset"] == "not_already_rising" and losses[s] > losses[s - 1]:
                continue
            if row["target"] == "future_step":
                numerator, denominator = losses[s + h] - losses[s + h - 1], losses[s + h - 1]
            elif row["target"] == "net_change":
                numerator, denominator = losses[s + h] - losses[s], losses[s]
            else:
                numerator, denominator = max(losses[s + 1:s + h + 1]) - losses[s], losses[s]
            change = numerator / denominator
            positive = change > 0 if minimum == 0 else change >= minimum
            previous = metrics[s - 1]["streams"]["raw"]["norm"]
            current = metrics[s]["streams"]["raw"]["norm"]
            rho = metrics[s]["streams"]["raw"]["cosine"]
            flags = (rho < 0, current > previous, rho < 0 and current > previous,
                     rho < -previous / current)
            n += 1
            positives += int(positive)
            for j, flag in enumerate(flags):
                alarms[j] += int(flag)
                tp[j] += int(flag and positive)
            checked_states += 1
        assert n == int(row["states"]) and positives == int(row["positives"])
        for j, flag in enumerate(FLAGS):
            assert alarms[j] == int(row[f"{flag}_alarms"])
            assert tp[j] == int(row[f"{flag}_tp"])
        checked_rows += 1
    return dict(independent_raw_summary_rows=checked_rows,
                independent_raw_decision_classifications=checked_states,
                checked_horizons=[1, 3, 16], all_nine_arms=True,
                independent_angle_norm_predicate=True)


def select(rows, arm, **criteria):
    selected = [r for r in rows if r["arm"] == arm and
                all(r[key] == str(value) for key, value in criteria.items())]
    sortkey = "minimum_relative_increase" if "horizon" in criteria else "horizon"
    return sorted(selected, key=lambda r: float(r[sortkey]))


def figure(rows, severity=False, target="future_step", subset="all", memory=False):
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.6), sharey="row")
    fig.subplots_adjust(left=.075, right=.985, top=.73, bottom=.18, wspace=.18, hspace=.38)
    for col, arm in enumerate(ARMS):
        criteria = dict(target=target, subset=subset, support="all_available" if severity else "common_16_step")
        if memory:
            criteria["support"] = "common_16_step_history"
        if severity:
            criteria["horizon"] = 1
        else:
            criteria["minimum_relative_increase"] = .1
        selected = select(rows, arm, **criteria)
        assert len(selected) == (9 if severity else 16)
        x = np.arange(len(selected)) if severity else np.array([int(r["horizon"]) for r in selected])
        for line, (flag, color, style) in enumerate(zip(FLAGS, COLORS, STYLES)):
            for ax, metric in zip(axes[:, col], ("recall", "precision")):
                y = np.array([float(r[f"{flag}_{metric}"]) if r[f"{flag}_{metric}"] else np.nan for r in selected])
                ax.plot(x, y * 100, color=color, ls=style, marker="o", ms=3,
                        lw=1.7, label=LABELS[line], alpha=.93)
        axes[1, col].plot(x, [float(r["base_rate"]) * 100 for r in selected],
                          color=".45", ls=(0, (2, 2)), lw=1.25, label="Unconditional rise frequency")
        if severity:
            counts = [int(r["positives"]) for r in selected]
            for pos, n in zip(x, counts):
                axes[0, col].text(pos, 1.035, str(n), ha="center", va="bottom",
                                  transform=axes[0, col].get_xaxis_transform(),
                                  fontsize=7, color="#A03030" if n < 10 else ".25")
            axes[0, col].set_title(NAMES[col] + "\nTarget counts above curve (red: <10)", pad=24, fontsize=9)
        else:
            counts = [int(r["positives"]) for r in selected]
            alarms = [int(selected[0][f"{flag}_alarms"]) for flag in FLAGS]
            alarm_text = (f'Q alarms: {selected[0]["Q_ge2_alarms"]}–{selected[-1]["Q_ge2_alarms"]}' if memory else
                          "Alarms D/G/DG/Q: " + "/".join(map(str, alarms)))
            axes[0, col].set_title(NAMES[col] + f'\nDecisions: {selected[0]["states"]}; rises: {min(counts)}–{max(counts)}'
                                  + "\n" + alarm_text, fontsize=8.5, pad=10)
        for ax in axes[:, col]:
            ax.set_ylim(-3, 104)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis="y", alpha=.20)
            ax.spines[["top", "right"]].set_visible(False)
            if severity:
                ax.set_xticks(x, [">0", ".001", ".1", "1", "5", "10", "25", "50", "100"], rotation=35)
            else:
                ax.set_xticks([1, 2, 4, 8, 12, 16])
                ax.set_xlim(.6, 16.4)
            ax.tick_params(labelsize=8)
        xlabel = "Past states in warning window (w)" if memory else "Future update offset h"
        axes[1, col].set_xlabel("Minimum next-step increase (%)" if severity else xlabel, fontsize=9)
    axes[0, 0].set_ylabel("Coverage of rises (%)\nTP / all target rises", fontsize=9)
    axes[1, 0].set_ylabel("Precision of flags (%)\nTP / all flagged decisions", fontsize=9)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .905),
               ncol=3, fontsize=8.5, frameon=False)
    if severity:
        title = "How does pre-update coverage depend on rise magnitude?"
        subtitle = "State s diagnostics evaluated against step s → s+1 · all 1,023 eligible states per model"
        footer = "Strict >0 for any rise; other thresholds are minimum relative changes. Empty endpoints have no target rises.\nOne retained initialization per model; dependent updates. All nine model/LR arms are preserved in the data tables."
    else:
        titles = {"future_step": "Loss increase at a specific future update",
                  "net_change": "Net loss increase relative to the decision state",
                  "any_above_current": "Any loss rise above the decision loss within the window"}
        formulas = {"future_step": "[L(s+h) − L(s+h−1)] / L(s+h−1) ≥ 10%",
                    "net_change": "[L(s+h) − L(s)] / L(s) ≥ 10%",
                    "any_above_current": "max over k=1…h of [L(s+k) − L(s)] / L(s) ≥ 10%"}
        title = titles[target]
        subtitle = formulas[target] + (" · only states with L(s) ≤ L(s−1)" if subset != "all" else "")
        footer = "Same decision states s=1…1008 at every horizon; diagnostics fixed at s. Gray line: target frequency without a diagnostic.\nCurves are exploratory; no horizon was chosen as a validated optimum. Window-target positives increase with horizon by construction."
    if memory:
        title = "Does retaining a warning cover later loss increases?"
        subtitle = "Next-step increase ≥10% · flag if the condition held at any of the last w states, including the current state"
        footer = "Each target update counted once; same states s=16…1023 at every width. Only past/current gradients enter the warning.\nLonger memory raises coverage and flags more non-target steps. No winning width or universal operating threshold is claimed."
    fig.suptitle(title, y=.985, fontsize=14)
    fig.text(.53, .938, subtitle, ha="center", fontsize=10)
    fig.text(.075, .045, footer, fontsize=8.5, va="bottom", linespacing=1.6)
    return fig


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.unicode_minus": True})
    severity, forward = read_csv("severity.csv"), read_csv("horizons.csv")
    manifest = SOURCE.read(DATA / "manifest.json")
    for filename, digest in manifest["outputs"].items():
        assert SOURCE.sha(DATA / filename) == digest
    memory_manifest = SOURCE.read(DATA / "memory_manifest.json")
    for filename, digest in memory_manifest["outputs"].items():
        assert SOURCE.sha(DATA / filename) == digest
    verification = verify_raw(forward)
    path = HERE / "figures/adam_forward_and_severity.pdf"
    pages = [(severity, dict(severity=True)),
             (forward, dict(target="future_step")),
             (forward, dict(target="net_change")),
             (forward, dict(target="any_above_current")),
             (forward, dict(target="any_above_current", subset="not_already_rising")),
             (read_csv("warning_memory.csv"), dict(memory=True))]
    with PdfPages(path, metadata={"Title": "Adam diagnostics: timing and severity audit",
                                 "Author": "Constructive-destructive regimes project",
                                 "CreationDate": None, "ModDate": None}) as pdf:
        for rows, options in pages:
            fig = figure(rows, **options)
            pdf.savefig(fig)
            plt.close(fig)
    # Render every page for visual review; scratch images stay ignored.
    import fitz
    scratch = ROOT / "tmp/consolidation-forward-review"
    scratch.mkdir(exist_ok=True)
    with fitz.open(path) as document:
        assert len(document) == 6
        for i, page in enumerate(document):
            assert "MLP" in page.get_text() and "Transformer" in page.get_text()
            page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4)).save(scratch / f"page-{i+1}.png")
    verification.update(pdf=path.relative_to(ROOT).as_posix(), pages=6,
                        pdf_sha256=SOURCE.sha(path),
                        generator_sha256=SOURCE.sha(__file__),
                        inputs={**manifest["outputs"], **memory_manifest["outputs"]},
                        visual_review="Rendered all pages; human-facing layout checked separately before commit")
    SOURCE.dump(DATA / "verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
