"""Analyze completed ResNet clipping and its matched constant-LR control."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "runs/231-2026-11-05-resnet-clip1"
SPARK = RUN / "artifacts/spark/payload/worker"
ARCHIVE = ROOT / "runs/223-2026-11-04-multispike-interventions/artifacts/duqoihms8w1gle/payload/worker"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_payload(worker):
    directory = worker.parent
    assert json.loads((directory.parent / "receipt.json").read_text())["verified"]
    records = json.loads((directory / "transfer_inventory.json").read_text())["files"]
    for item in records:
        path = directory / item["path"]
        assert path.stat().st_size == item["bytes"]
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == item["sha256"]
    return len(records)


def helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    checked = verify_payload(SPARK) + verify_payload(ARCHIVE)
    manifest = json.loads((SPARK / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["exact_spark_replay"]
    replay = helper("checkpoints", ROOT / "runs/228-2026-11-04-mlp-global-clipping/03_verify.py")
    direct, resumed = [torch.load(SPARK / "qualification" / stage / "end.pt", map_location="cpu", weights_only=False) for stage in ("direct", "resumed")]
    for key in ("model", "optimizer", "rng", "current_gradient", "previous_gradient"):
        replay.eq(direct[key], resumed[key])
    detector = helper("excursions", ROOT / "analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py")
    prefix = rows(ARCHIVE / "prefix/boundaries.jsonl")[:-1]
    comparator = SPARK if manifest["baseline"] == "fresh_spark" else ARCHIVE
    arms = {"Unclipped": comparator / "control", "Clipping at 1": SPARK / "clipping"}
    summary, series, updates = {}, {}, {}
    for name, folder in arms.items():
        data = prefix + rows(folder / "boundaries.jsonl")
        metrics = rows(folder / "metrics.jsonl")
        assert [r["state"] for r in data] == list(range(577))
        assert [m["state"] for m in metrics] == list(range(256, 576))
        for m in metrics:
            gn = m["raw_gradient_norm"]
            factor = min(1., 1./gn) if name == "Clipping at 1" and gn else 1.
            assert m["clip_factor"] == factor and m["applied_lr"] == .003
            assert abs(m["optimizer_input_norm"] - gn * factor) < 2e-7
        loss = np.array([row["loss"] for row in data])
        events = []
        for e in detector.excursions(loss):
            ratio = loss[e["peak_state"]] / loss[e["before_state"]]
            if 256 < e["peak_state"] <= 576 and ratio > 1.1:
                events.append(dict(e, ratio=ratio))
        checkpoint = torch.load(folder / "end.pt", map_location="cpu", weights_only=False)
        assert checkpoint["completed_updates"] == 576
        assert all(int(s["step"]) == 576 for s in checkpoint["optimizer"]["state"].values())
        summary[name] = dict(mean_loss=float(loss[257:].mean()), tail_mean=float(loss[-32:].mean()),
                             maximum=float(loss[257:].max()), maximum_state=257+int(loss[257:].argmax()),
                             endpoint=float(loss[-1]), excursions=events, clipped=sum(r["clip_factor"]<1 for r in metrics),
                             max_input=max(r["optimizer_input_norm"] for r in metrics), max_raw=max(r["raw_gradient_norm"] for r in metrics),
                             max_step=max(r["actual_increment_norm"] for r in metrics))
        series[name], updates[name] = data, metrics
    for key in ("loss","rho","raw_gradient_norm","eta_kappa_hat","parameter_norm"):
        assert series["Unclipped"][256][key] == series["Clipping at 1"][256][key]
    panels = [("loss","Training loss (log scale)"), ("rho",r"Raw gradient $\rho_t$"),
              ("raw_gradient_norm","Gradient norm"), ("eta_kappa_hat",r"$\eta\widehat{\kappa}_t$"),
              ("applied_lr","Applied LR / base LR")]
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(9, 10.5), layout="constrained")
    values = []
    for ax,(key,label) in zip(axes,panels):
        for name, color in zip(arms, ["#777777","#0072B2"]):
            xx = list(range(192, 576 if key == "applied_lr" else 577))
            yy = [series[name][i][key] / (.003 if key == "applied_lr" else 1.) for i in xx]
            ax.plot(xx, yy, lw=1., color=color, label=name)
            values.extend((name,key,i,v) for i,v in zip(xx,yy))
        if key == "raw_gradient_norm":
            mm = updates["Clipping at 1"]
            ax.plot([r["state"] for r in mm],[r["optimizer_input_norm"] for r in mm],color="#009E73",lw=1.1)
            values.extend(("Clipping at 1","optimizer_input_norm",r["state"],r["optimizer_input_norm"]) for r in mm)
            ax.legend(["Baseline raw","Clipping run raw","After clipping"],loc="upper right",fontsize=8)
        ax.axvspan(192,256,color="#efefef",zorder=-3)
        ax.axvline(256,lw=.8,color="#444444")
        ax.set(ylabel=label,xlim=(192,576))
        ax.grid(alpha=.15)
        ax.spines[["right","top"]].set_visible(False)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[-1].set(yticks=[1.], xlabel="Completed optimizer updates",xticks=[192,256,320,384,448,512,576])
    fig.suptitle("ResNet-32 / CIFAR-10  |  Global gradient clipping before Adam",fontsize=14)
    fig.supxlabel("Shaded prefix on RTX 4090. Clipping begins on Spark at state 256.\\nFull 50,000-image objective, seed 0, constant LR 0.003.".replace("\\n","\n"),fontsize=9)
    (HERE/"figures").mkdir(exist_ok=True)
    (HERE/"scratch").mkdir(exist_ok=True)
    pdf=HERE/"figures/resnet_clip1.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    with fitz.open(pdf) as doc:
        assert len(doc)==1 and not doc[0].get_images()
        doc[0].get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(HERE/"scratch/preview.png")
    with (HERE/"plotted_values.csv").open("w",newline="") as output:
        writer=csv.writer(output)
        writer.writerow(["series","metric","state","value"])
        writer.writerows(values)
    effects={k:100*(summary["Clipping at 1"][k]/summary["Unclipped"][k]-1)
             for k in ("mean_loss","tail_mean","maximum","endpoint")}
    result=dict(summary=summary, percent_vs_baseline=effects, baseline=manifest["baseline"], verified_files=checked,
                cross_gpu_relative_gradient_error=manifest["cross_gpu_relative_error"], plotted_values=len(values))
    (HERE/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(effects,indent=2))


if __name__ == "__main__":
    main()
