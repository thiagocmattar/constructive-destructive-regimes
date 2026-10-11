"""Two publication PDFs from measured prefixes; preserve failed/missing conditions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STREAMS = {"raw": (None, "gradient_norm", "cumulative_gradient_norm"),
           "momentum": ("momentum", "momentum_norm", "cumulative_momentum_norm"),
           "adaptive": ("adaptive", "direction_norm", "cumulative_direction_norm")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def mean(values, window):
    result = np.full(len(values), np.nan)
    for i in range(len(values)):
        part = values[max(0, i-window+1):i+1]
        finite = part[np.isfinite(part)]
        if len(finite):
            result[i] = finite.mean()
            assert np.isclose(result[i], sum(finite)/len(finite), rtol=1e-14, atol=1e-14)
    return result


def serial(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def load(smoke=False):
    reference = json.loads((HERE / "reference_023.json").read_text())
    records = [dict(run_id="023", lr=.01, status="completed_reference", events=reference["events"],
                    summary=reference["summary"], sources=reference["source_sha256"],
                    config=reference["config"], identity=reference["identity"], environment=reference["environment"])]
    state_path = HERE / "queue_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(runs={})
    for item in json.loads((HERE / "cohort.json").read_text())["runs"]:
        run = ROOT / "runs" / item["run"]
        folder = run / "artifacts/attempt-01/full-pass"
        status = state.get("runs", {}).get(item["run_id"], {}).get("status", "not_started")
        events = [e for line in (folder / "events.jsonl").read_text().splitlines()
                  if (e := json.loads(line)).get("event") == "update"] if (folder / "events.jsonl").exists() else []
        if smoke:
            # A labeled rendering mechanics check, never evidence for the new LRs.
            events, status = reference["events"], "SMOKE_REFERENCE_REUSE"
        def optional(name):
            path = folder / name
            return json.loads(path.read_text()) if path.exists() else None
        summary, identity, environment = optional("summary.json"), optional("input_identity.json"), optional("environment.json")
        if status == "completed":
            proof = optional("verification.json")
            assert proof["verified"] and proof["scope"] == "full_pass" and proof["stored_m_matches_actual_exp_avg"]
            assert len(events) == 712 and summary["all_training_tokens_covered"]
        config = json.loads((run / "config.json").read_text())
        for key in ("model", "data", "seeds", "measurements", "validation", "checkpoints"):
            assert config[key] == reference["config"][key]
        assert {k:v for k,v in config["training"].items() if k != "peak_learning_rate"} == {
            k:v for k,v in reference["config"]["training"].items() if k != "peak_learning_rate"}
        if identity:
            assert identity == reference["identity"]
        if environment:
            for key in ("initial_parameter_sha256", "torch", "transformers", "numpy", "cuda", "gpu", "shared_source_sha256"):
                assert environment[key] == reference["environment"][key], key
        sources = {p.name:sha(p) for p in folder.glob("*") if p.is_file() and p.suffix != ".tmp"}
        sources["config.json"] = sha(run / "config.json")
        records.append(dict(run_id=item["run_id"], lr=item["peak_lr"], status=status, events=events,
                            summary=summary, config=config, identity=identity, environment=environment, sources=sources))
    for record, location in zip(records, (.58, .28, .03)):
        record["color"] = matplotlib.colors.to_hex(plt.get_cmap("plasma")(location))
        events = record["events"]
        assert [e["update"] for e in events] == list(range(1,len(events)+1))
        assert len(events) <= 712 and all(np.isfinite(e["loss"]) for e in events)
        if len(events) == 712:
            assert events[-1]["sequences"] == 311
            assert sum(e["input_tokens"] for e in events) == 1491711416
            assert sum(e["supervised_tokens"] for e in events) == 1490983041
        if not smoke:
            assert all(e["learning_rate"] == record["lr"] for e in events[8:])
        record["series"] = dict(loss=np.array([e["loss"] for e in events]))
        for stream, (subkey, norm_key, bkey) in STREAMS.items():
            values = [e[subkey] if subkey else e for e in events]
            for label, key in (("norm", norm_key), ("A", "cumulative_norm_sum"), ("B", bkey),
                               ("B_over_A", "cumulative_ratio"), ("cosine", "adjacent_cosine")):
                record["series"][stream+"_"+label] = np.array([np.nan if e[key] is None else e[key] for e in values])
            if len(events):
                assert np.isnan(record["series"][stream+"_cosine"][0])
                valid = record["series"][stream+"_B_over_A"]
                assert ((valid[np.isfinite(valid)] >= 0) & (valid[np.isfinite(valid)] <= 1+1e-10)).all()
            record["series"][stream+"_cosine_mean"] = mean(record["series"][stream+"_cosine"], 50 if stream == "raw" else 25)
        record["series"]["raw_norm_mean"] = mean(record["series"]["raw_norm"],10)
    return records


def line(ax, record, key, **kwargs):
    values = record["series"][key]
    x = np.arange(1,len(values)+1)
    artist, = ax.plot(x, values, **kwargs)
    np.testing.assert_array_equal(artist.get_xdata(),x)
    np.testing.assert_array_equal(artist.get_ydata(),values)
    if len(values) == 712 and np.isfinite(values[-1]):
        ax.plot(712,values[-1],"o",color=artist.get_color(),ms=3,zorder=6)
    return artist


def norms(ax, record, limits):
    series = record["series"]
    raw, smooth, momentum = (series[k] for k in ("raw_norm","raw_norm_mean","momentum_norm"))
    x = np.arange(1,len(raw)+1)
    for values, alpha, lw, color, style, label in (
        (raw,.20,.8,"#0072B2","-",None),
        (smooth,1,1.8,"#0072B2","-","Raw norm: 10-update mean"),
        (momentum,1,1.4,"#D55E00","--",r"Stored momentum $\|m_t\|$")):
        displayed = np.where(values>0,values,np.nan)
        artist, = ax.plot(x,displayed,color=color,alpha=alpha,lw=lw,ls=style,label=label,zorder=1 if alpha<1 else 4)
        np.testing.assert_array_equal(artist.get_ydata(),displayed)
        if len(values) == 712 and values[-1]>0 and alpha == 1:
            ax.plot(712,values[-1],"o",color=color,ms=3,zorder=6)
    positive = (smooth>0)&(momentum>0)
    fill = ax.fill_between(x,smooth,momentum,where=positive,color="#0072B2",alpha=.14,lw=0,zorder=2)
    if len(x) and positive.all():
        expected = np.column_stack((np.r_[x[0],x,x[-1],x[::-1],x[0]],np.r_[momentum[0],smooth,momentum[-1],momentum[::-1],momentum[0]]))
        np.testing.assert_array_equal(fill.get_paths()[0].vertices,expected)
    if not len(x):
        ax.text(.5,.5,record["status"].replace("_"," "),transform=ax.transAxes,ha="center",va="center")
    ax.set(yscale="log",ylim=limits,ylabel="Euclidean norm (log scale)")
    ax.legend(loc="best",fontsize=9,framealpha=.9)


def format_axes(axes):
    for ax in np.asarray(axes).flat:
        ax.set(xlim=(0,725),xlabel="Optimizer update t")
        ax.grid(axis="y",color="#E2E7EC",lw=.7)


def loss(ax, records):
    for record in records:
        line(ax,record,"loss",color=record["color"],lw=1.1,
             label=f"LR {record['lr']:g} ({len(record['events'])}/712)")
    ax.set_title("(a) Unsmoothed training loss",loc="left")
    ax.set_ylabel("Cross-entropy (nats)")
    ax.legend(fontsize=9,frameon=False)


def figure(records, full, destination, smoke):
    rows = 6 if full else 2
    fig, axes = plt.subplots(rows,2,figsize=(16,24 if full else 11))
    fig.subplots_adjust(left=.085,right=.97,top=.91 if full else .865,bottom=.095 if full else .17,hspace=.37,wspace=.24)
    title = "Pythia-14M on DGX Spark: higher learning rates"
    if smoke:
        title = "SMOKE ONLY: Run023 values reused for rendering mechanics"
    fig.suptitle(title,fontsize=19,y=.975)
    fig.text(.5,.947 if full else .935,"Matched initialization/order/recipe; new LR 0.02 and 0.04; completed LR 0.01 reference",ha="center",fontsize=11)
    loss(axes.flat[0],records)
    positive = np.concatenate([r["series"][key] for r in records for key in ("raw_norm","momentum_norm")])
    positive = positive[positive>0]
    limits = (float(positive.min()/1.25),float(positive.max()*1.25))
    for index, record in enumerate(records,1):
        ax = axes.flat[index]
        status = "reference" if record["status"] == "completed_reference" else record["status"]
        ax.set_title(f"({'abcdefghijkl'[index]}) Norms: LR {record['lr']:g} | {status}",loc="left")
        norms(ax,record,limits)
    if full:
        metrics = [("raw_cosine_mean","Raw-gradient adjacent cosine: 50-update mean","Cosine"),
                   ("momentum_cosine_mean","Momentum adjacent cosine: 25-update mean","Cosine"),
                   ("raw_A","Raw gradients: cumulative A and B","Cumulative norm"),
                   ("momentum_A","Momentum: cumulative A and B","Cumulative norm"),
                   ("raw_B_over_A","Raw gradients: cumulative B/A","B/A"),
                   ("momentum_B_over_A","Momentum: cumulative B/A","B/A"),
                   ("adaptive_cosine_mean","Adam direction: adjacent cosine, 25-update mean","Cosine"),
                   ("adaptive_B_over_A","Adam direction: cumulative B/A","B/A")]
        for index,(key,title,ylabel) in enumerate(metrics,4):
            ax = axes.flat[index]
            for record in records:
                if key.endswith("cosine_mean"):
                    line(ax,record,key[:-5],color=record["color"],alpha=.16,lw=.6)
                line(ax,record,key,color=record["color"],lw=1.7,label=f"LR {record['lr']:g}")
                if key in ("raw_A", "momentum_A"):
                    line(ax,record,key[:-1]+"B",color=record["color"],ls="--",lw=1.4)
            ax.set_title(f"({'abcdefghijkl'[index]}) {title}",loc="left")
            ax.set_ylabel(ylabel)
            ax.legend(loc="best",fontsize=9,frameon=False)
            if "cosine" in key:
                ax.set_ylim(-1.05,1.05)
                ax.axhline(0,color="#77818B",ls=":",lw=.8)
            elif "B_over_A" in key:
                ax.set_ylim(0,1.05)
            else:
                ax.set_ylim(0,None)
    format_axes(axes)
    footer_y = (.064,.049,.034,.02) if full else (.104,.080,.056,.031)
    footers = (
        "Loss and momentum norms are unsmoothed. Norm panels: faint raw values, trailing 10-update raw mean, fill to momentum (not uncertainty).",
        "Cosine panels: faint originals; raw mean 50 updates, momentum/Adam means 25. Available finite prefixes; undefined first cosine is a gap.",
        "Cumulative geometry is unsmoothed, linear Y: A = sum of vector norms (solid), B = norm of sum (dashed). Circles mark partial update 712.",
        "Raw g is pre-clipping; m is uncorrected Adam exp_avg from clipped g; d excludes LR/decay. One seed; no stability or causal alignment claim.")
    if not full:
        footers = (footers[0],
            "Every recorded accepted update is shown; circles mark partial final update 712 (311 sequences versus 1,024). Failed prefixes are labeled.",
            "Means use current and preceding 9 available raw norms, before log display. Common log Y limits; no normalization or positive floor.",
            footers[3])
    for y,text in zip(footer_y,footers):
        fig.text(.085,y,text,fontsize=9.5)
    stem = "training_dynamics" if full else "loss_and_norms_by_learning_rate"
    pdf = destination / "figures" / (stem+".pdf")
    fig.savefig(pdf)
    plt.close(fig)
    with fitz.open(pdf) as document:
        assert len(document) == 1
        page = document[0]
        for block in page.get_text("blocks"):
            assert page.rect.contains(fitz.Rect(block[:4]))
        for text in ("Unsmoothed training loss","Raw norm: 10-update mean","Stored momentum","LR 0.02","LR 0.04"):
            assert text in page.get_text(),text
        if full:
            for text in ("Raw-gradient adjacent cosine","Momentum adjacent cosine","cumulative A and B","cumulative B/A"):
                assert text in page.get_text(),text
        render = destination / "scratch" / (stem+"_review.png")
        page.get_pixmap(matrix=fitz.Matrix(1.1,1.1),alpha=False).save(str(render))
    return dict(path=pdf.name,sha256=sha(pdf),panels=12 if full else 4,pages=1,
                page_text_and_bounds=True,render_sha256=sha(render),norm_log_limits=limits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    destination = HERE / "scratch/rendering-smoke" if args.smoke else HERE
    for name in ("figures","scratch"):
        (destination / name).mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.titlesize":12,
                         "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    records = load(args.smoke)
    figures = [figure(records,False,destination,args.smoke),figure(records,True,destination,args.smoke)]
    data = dict(generated_at=datetime.now(timezone.utc).isoformat(),scope="rendering_smoke_only" if args.smoke else "measured_full_passes_or_explicit_failed_prefixes",
        generator_sha256=sha(__file__),reference_sha256=sha(HERE / "reference_023.json"),
        source_inventory_sha256=sha(HERE / "source_inventory.json"),
        smoothing=dict(loss=None,raw_norm=10,momentum_norm=None,raw_cosine=50,momentum_cosine=25,adaptive_cosine=25,cumulative=None),
        reduction="trailing arithmetic mean in original units; available finite prefixes; zeros omitted from logarithmic display without a floor",
        figures=figures,runs={r["run_id"]:dict(lr=r["lr"],status=r["status"],accepted_updates=len(r["events"]),
            sources=r["sources"],summary=r["summary"],color=r["color"],
            updates=[e["update"] for e in r["events"]],supervised_tokens=[e["supervised_tokens"] for e in r["events"]],
            omitted_log_zero_raw_norms=int(np.sum(r["series"]["raw_norm"]==0)),
            omitted_log_zero_momentum_norms=int(np.sum(r["series"]["momentum_norm"]==0)),
            series={k:serial(v) for k,v in r["series"].items()}) for r in records})
    save(destination / "comparison_data.json",data)
    save(destination / "figure_verification.json",dict(passed=True,scope=data["scope"],data_sha256=sha(destination / "comparison_data.json"),
        figures=figures,source_placement_and_artist_values=True,means_independently_verified=True,fill_vertices_verified_when_positive=True,
        matched_scientific_inputs=True,undefined_first_cosines_preserved=True,full_coverage_or_failure_status_declared=True,
        rendered=True,visual_review="pending_final_inspection"))
    print(json.dumps(dict(figures=figures,coverage={r["run_id"]:len(r["events"]) for r in records},smoke=args.smoke)))


if __name__ == "__main__":
    main()
