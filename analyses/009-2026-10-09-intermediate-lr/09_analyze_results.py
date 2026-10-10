"""Verified seven-LR metrics and figures, preserving Analysis008 outputs."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import math
import string
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "analyses/008-2026-10-08-no-decay-momentum"
OUT = HERE / "complete-seven-lr"
spec = importlib.util.spec_from_file_location("old_plots", OLD / "08_plot_simplified.py")
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "pdf.fonttype": 42})


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reduce_events(events, summary):
    assert [e["update"] for e in events] == list(range(1, 713))
    assert sum(e["input_tokens"] for e in events) == 1491711416
    assert sum(e["supervised_tokens"] for e in events) == 1490983041
    assert events[-1]["sequences"] == 311
    assert summary["all_training_tokens_covered"] and summary["completed_updates"] == 712
    for phase in ("initial_validation", "final_validation"):
        assert summary[phase]["supervised_tokens"] == 693329 and summary[phase]["short_tail_included"]
    displayed = events[:711]
    assert all(e["sequences"] == 1024 for e in displayed)
    stats = {}
    for stream, (_, norm, bkey, *_rest) in plots.STREAMS.items():
        cos = plots.values(events, stream, "adjacent_cosine")
        valid = cos[np.isfinite(cos)]
        assert len(valid) == 711 and np.isnan(cos[0])
        end = events[-1][stream] if stream != "raw" else events[-1]
        assert math.isclose(end["cumulative_ratio"], end[bkey]/end["cumulative_norm_sum"], rel_tol=1e-12)
        late = plots.values(displayed[-100:], stream, "adjacent_cosine")
        stats[stream] = {"median_cosine": float(np.median(valid)), "negative_fraction": float(np.mean(valid < 0)),
                         "final_A": end["cumulative_norm_sum"], "final_B": end[bkey], "final_B_over_A": end["cumulative_ratio"],
                         "last100_median_cosine": float(np.nanmedian(late)), "last100_negative_fraction": float(np.mean(late < 0)),
                         "last100_median_norm": float(np.median(plots.values(displayed[-100:], stream, norm)))}
    performance = {"final_validation_loss": summary["final_validation"]["loss"],
                   "last100_full_training_loss": float(np.average([e["loss"] for e in displayed[-100:]], weights=[e["supervised_tokens"] for e in displayed[-100:]])),
                   "overflow_retries": sum(e["overflow_retries"] for e in events),
                   "clipping_threshold_exceeded_updates": sum(e["gradient_norm"] > 1 for e in events)}
    return displayed, stats, performance


def load_records():
    cached_path = OLD / "complete-three-lr/comparison_data.json"
    assert digest(cached_path) == "b631551121647e7653b59b6fe6b70855881c9d1b7efa978d48cefb3e923b847c"
    cached = json.loads(cached_path.read_text())["runs"]
    cohort = json.loads((OLD / "cohort.json").read_text())["runs"]
    records = []
    for r in cohort:
        run = ROOT / "runs" / r["run"]; folder = run / "artifacts/attempt-01/full-pass"
        c = cached[r["run_id"]]
        assert digest(folder / "events.jsonl") == c["sources"]["events.jsonl"]
        assert json.loads((run / "prelaunch/retrieval-receipt.json").read_text())["accepted"]
        summary = json.loads((folder / "summary.json").read_text())
        records.append({"id": r["run_id"], "lr": r["peak_lr"], "hardware": "Spark", "events": c["events"],
                        "sources": c["sources"], "summary": summary, "identity": json.loads((folder / "input_identity.json").read_text()),
                        "config": json.loads((run / "config.json").read_text())})
    receipt = json.loads((HERE / "metrics-retrieval-receipt.json").read_text())
    assert receipt["accepted"]
    inputs = HERE / "scratch/completed-metrics"
    new_cohort = json.loads((HERE / "cohort.json").read_text())["runs"]
    def completed_input(rid):
        r = next(r for r in new_cohort if r["run_id"] == rid)
        run = ROOT / "runs" / r["run"]
        proof = run / "prelaunch/retrieval-receipt.json"
        if proof.exists() and json.loads(proof.read_text())["accepted"]:
            return run / "artifacts/attempt-01/full-pass"
        return inputs / rid
    for item in receipt["files"]:
        rid, filename = item["path"].split("/", 1)
        path = completed_input(rid) / filename
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
    for r in new_cohort:
        folder = completed_input(r["run_id"])
        assert json.loads((folder / "manifest.json").read_text())["status"] == "completed"
        proof = json.loads((folder / "verification.json").read_text())
        assert proof["verified"] and proof["scope"] == "full_pass" and proof["stored_m_matches_actual_exp_avg"]
        events = [e for line in (folder / "events.jsonl").read_text().splitlines() if (e := json.loads(line))["event"] == "update"]
        sources = {Path(i["path"]).name: i["sha256"] for i in receipt["files"] if i["path"].startswith(r["run_id"] + "/")}
        records.append({"id": r["run_id"], "lr": r["peak_lr"], "hardware": "Runpod RTX4090", "events": events,
                        "sources": sources, "summary": json.loads((folder / "summary.json").read_text()),
                        "identity": json.loads((folder / "input_identity.json").read_text()),
                        "config": json.loads((ROOT / "runs" / r["run"] / "config.json").read_text())})
    records.sort(key=lambda r: r["lr"])
    for i, r in enumerate(records):
        assert r["identity"] == records[0]["identity"]
        for key in ("model", "data", "seeds", "measurements", "validation", "checkpoints"):
            assert r["config"][key] == records[0]["config"][key], key
        a, b = dict(r["config"]["training"]), dict(records[0]["config"]["training"])
        a.pop("peak_learning_rate"); b.pop("peak_learning_rate"); assert a == b
        assert all(e["learning_rate"] == r["lr"] for e in r["events"][8:])
        r["displayed"], r["geometry"], r["performance"] = reduce_events(r["events"], r["summary"])
        r["color"] = plt.get_cmap("turbo")(0.05 + .9*i/6)
        r["style"] = "--" if r["hardware"] == "Spark" else "-"
    return records


def metrics(records):
    fig, axes = plt.subplots(3, 2, figsize=(14, 13.5))
    handles = []
    for r in records:
        e = r["displayed"]; x = np.arange(1, 712); color = r["color"]; style = r["style"]
        loss = np.array([v["loss"] for v in e]); weights = np.array([v["supervised_tokens"] for v in e])
        axes[0, 0].plot(x, loss, color=color, alpha=.12, lw=.6)
        line, = axes[0, 0].plot(x, plots.smooth(loss, weights), color=color, ls=style, lw=1.8, label=f"{r['lr']:.6g} ({'S' if r['hardware']=='Spark' else 'R'})")
        handles.append(line)
        for ax, stream in ((axes[0, 1], "raw"), (axes[1, 0], "momentum"), (axes[1, 1], "adaptive")):
            cosine = plots.values(e, stream, "adjacent_cosine")
            ax.plot(x, cosine, color=color, alpha=.12, lw=.6)
            ax.plot(x, plots.smooth(cosine), color=color, ls=style, lw=1.8)
        for ax, stream in ((axes[2, 0], "raw"), (axes[2, 1], "momentum")):
            ax.plot(x, plots.values(e, stream, plots.STREAMS[stream][1]), color=color, ls=style, lw=1.1)
    titles = ["(a) Training loss", "(b) Raw-gradient adjacent cosine", "(c) Momentum adjacent cosine", "(d) Adam-direction adjacent cosine", "(e) Raw-gradient vector norm", "(f) Momentum vector norm"]
    for i, ax in enumerate(axes.flat):
        ax.set_title(titles[i]); ax.set_ylabel("Cross-entropy (nats)" if i==0 else "Cosine" if i in (1,2,3) else "Vector norm")
        if i in (1,2,3):
            ax.set_ylim(-1.05, 1.05); ax.axhline(0, color="black", ls=":", lw=.7)
        elif i in (4,5): ax.set_ylim(0, None)
    plots.format_axes(axes)
    fig.suptitle("Pythia-14M: seven warmup-only learning rates", fontsize=17, y=.988)
    fig.legend(handles=handles, title="Peak LR: S = Spark (dashed); R = Runpod RTX4090 (solid)", ncol=4, loc="upper center", bbox_to_anchor=(.5,.968), frameon=False, fontsize=10)
    fig.text(.5,.018,"Light traces: updates; bold loss: trailing 25-update token-weighted mean; bold cosine: trailing 25-update median.\n"
             "Norms are unsmoothed on linear axes. Display 1-711; partial update 712 included in numerical results.\n"
             "Same frozen initialization/order, seed 1234. Different hardware across S/R; one seed; LR also scales AdamW decay.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.91), h_pad=2.1, w_pad=2.1)
    fig.savefig(OUT / "figures/metrics_by_learning_rate.pdf"); plt.close(fig)


def geometry(records):
    # Preserve readable columns: the seven conditions span two PDF pages.
    with PdfPages(OUT / "figures/geometry_within_learning_rate.pdf") as pdf:
        for page, group in enumerate((records[:4], records[4:]), 1):
            n = len(group); fig, axes = plt.subplots(2, n, figsize=(4.8*n, 9), squeeze=False)
            for col, r in enumerate(group):
                e=r["displayed"]; x=np.arange(1,712)
                for stream in ("raw", "momentum"):
                    _, _, bkey, name, color, _ = plots.STREAMS[stream]
                    axes[0,col].plot(x,plots.values(e,stream,"cumulative_norm_sum"),color=color,lw=1.8,label=f"{name}: A")
                    axes[0,col].plot(x,plots.values(e,stream,bkey),color=color,ls="--",lw=1.8,label=f"{name}: B")
                for stream, (_,_,_,name,color,style) in plots.STREAMS.items():
                    axes[1,col].plot(x,plots.values(e,stream,"cumulative_ratio"),color=color,ls=style,lw=1.8,label=name)
                label=f"LR {r['lr']:.6g} ({'Spark' if r['hardware']=='Spark' else 'RTX4090'})"
                axes[0,col].set(title=f"({string.ascii_lowercase[col]}) {label}",ylabel="Cumulative norm",ylim=(0,None))
                axes[1,col].set(title=f"({string.ascii_lowercase[n+col]}) Cumulative B/A",ylabel="B/A",ylim=(0,1.05))
                axes[0,col].legend(fontsize=8,loc="upper left"); axes[1,col].legend(fontsize=9,loc="lower left")
            plots.format_axes(axes)
            fig.suptitle(f"Pythia-14M: geometry within each LR (page {page}/2)",fontsize=16)
            fig.text(.5,.025,"A = sum of norms; B = norm of sum. Top: raw gradient g and actual stored momentum m; solid A, dashed B.\n"
                     "Each top panel has its own LINEAR y-scale to emphasize shape. Lower row: g, m and adaptive d; common range.\n"
                     "All curves unsmoothed; updates 1-711 shown; update 712 retained in results. m uses clipped gradients; d excludes LR/decay.\n"
                     "Frozen initialization/order and seed 1234; hardware differs across original/new runs; no convergence or stability inference.",ha="center",fontsize=9)
            fig.tight_layout(rect=(0,.13,1,.92),h_pad=2.3,w_pad=1.8); pdf.savefig(fig); plt.close(fig)


def main():
    for name in ("figures", "scratch", "observations"): (OUT/name).mkdir(parents=True,exist_ok=True)
    records=load_records(); metrics(records); geometry(records)
    data={"created_at":datetime.now(timezone.utc).isoformat(),"script_sha256":digest(__file__),
          "helper_sha256":digest(OLD/"08_plot_simplified.py"),"legacy_comparison_sha256":digest(OLD/"complete-three-lr/comparison_data.json"),
          "coverage":{"source_updates":712,"display_updates":[1,711],"partial_update_retained":True,"validation_targets":693329,"seed":1234},
          "retrieval":"36 compact completed files verified; full recovery states retrieved separately",
          "runs":{r["id"]:{k:r[k] for k in ("lr","hardware","sources","performance","geometry","events")} for r in records}}
    (OUT/"comparison_data.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    fieldnames=["run","lr","hardware","validation_loss","training_loss_last100","raw_cos_last100","momentum_cos_last100","adam_cos_last100","clipped_updates"]
    rows=[dict(zip(fieldnames,[r["id"],r["lr"],r["hardware"],r["performance"]["final_validation_loss"],r["performance"]["last100_full_training_loss"],*[r["geometry"][s]["last100_median_cosine"] for s in plots.STREAMS],r["performance"]["clipping_threshold_exceeded_updates"]])) for r in records]
    with (OUT/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows,indent=2))


if __name__ == "__main__": main()
