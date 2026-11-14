"""Publication views of retained evidence; no training or canonical-source edits.

Run from any directory. Outputs are packaged by ../package_evidence.py.
All progress coordinates count completed optimizer updates, s.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / "figures"
A24 = ROOT / "analyses/024-2026-10-17-eos-gd-lr-calibration"
A63 = ROOT / "analyses/063-2026-11-07-architecture-instability-tracing"
CASES = ["mlp", "resnet", "transformer"]
TITLES = ["CIFAR-10 / MLP", "CIFAR-10 / ResNet-32", "WikiText-2 / Transformer"]
COLORS = ["#0072B2", "#D55E00", "#009E73"]
XLABEL = r"Training step $s$"
STAGES = {
    "mlp": [("preactivation1", "Linear 1"), ("hidden1", "Tanh 1"),
            ("preactivation2", "Linear 2"), ("hidden2", "Tanh 2"), ("output", "Output")],
    "resnet": [("conv1.call0.output", "Conv 1"), ("layer1.4.relu.call1.output", "Stage 1"),
               ("layer2.4.relu.call1.output", "Stage 2"), ("layer3.4.relu.call1.output", "Stage 3"),
               ("avgpool.call0.output", "Pooling"), ("output.log_probabilities", "Output")],
    "transformer": [("encoder.call0.output", "Embedding"),
                    ("transformer_encoder.layers.0.self_attn.call0.output", "Attention 1"),
                    ("transformer_encoder.layers.0.norm2.call0.output", "Block 1"),
                    ("transformer_encoder.layers.1.self_attn.call0.output", "Attention 2"),
                    ("transformer_encoder.layers.1.norm2.call0.output", "Block 2"),
                    ("output.log_probabilities", "Output")],
}
AUDIT = {"date": "2026-11-14", "sources": {}, "curves": [], "figures": [],
         "clock": "Training step s = completed optimizer updates; incoming representation changes at s.",
         "smoothing": "Both GD curvature series: centered 101-step arithmetic mean on states 1..2999; edge windows truncated equally. Spectra linearly interpolated between measured 100-step states before smoothing; no extrapolation.",
         "tracing": [], "overview": []}
PLOTTED = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    path = Path(path)
    AUDIT["sources"][path.relative_to(ROOT).as_posix()] = sha(path)
    return path


def read(path):
    return json.loads(source(path).read_text(encoding="utf-8"))


def arrsha(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def curve(ax, x, y, tag, **kw):
    x, y = np.asarray(x), np.asarray(y, dtype=float)
    assert x.shape == y.shape and np.isfinite(y).all(), tag
    line, = ax.plot(x, y, **kw)
    np.testing.assert_array_equal(line.get_xdata(), x)
    np.testing.assert_array_equal(line.get_ydata(), y)
    AUDIT["curves"].append(dict(tag=tag, points=len(x), x_sha256=arrsha(x), y_sha256=arrsha(y)))
    PLOTTED[tag+":x"], PLOTTED[tag+":y"] = x, y
    return line


def style(ax):
    ax.grid(axis="y", alpha=.16, lw=.5)
    ax.tick_params(length=2.5, width=.6, pad=2)


def letter(ax, value):
    ax.text(0, 1.025, f"({value})", transform=ax.transAxes, va="bottom", fontsize=9)


def save(fig, name):
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, (name, box, fig.bbox)
    path = OUT / name
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    AUDIT["figures"].append({"file": name, "sha256": sha(path), "size_inches": list(fig.get_size_inches())})
    plt.close(fig)


def centered(y, radius=50):
    """Equal centered support for the two curvature series, including edges."""
    y = np.asarray(y, dtype=float)
    weights = np.ones(2 * radius + 1)
    result = np.convolve(y, weights, "same") / np.convolve(np.ones_like(y), weights, "same")
    independent = np.array([y[max(0, i-radius):min(len(y), i+radius+1)].mean() for i in range(len(y))])
    np.testing.assert_allclose(result, independent, rtol=2e-13, atol=2e-13)
    return result


def gd_figures():
    manifest = read(A24 / "single_rows_figure_manifest.json")
    arrays_path = source(ROOT / manifest["arrays"])
    assert sha(arrays_path) == manifest["arrays_sha256"] and manifest["status"] == "passed"
    arrays = np.load(arrays_path)
    spectral_manifest = read(A24 / "mlp_learning_rates_figure_manifest.json")
    spectral_path = source(ROOT / spectral_manifest["arrays"])
    assert sha(spectral_path) == spectral_manifest["arrays_sha256"]
    spectral = np.load(spectral_path)
    results = read(A24 / "results.json")
    records = {r["run_id"]: r for r in read(A24 / "cohort.json")["runs"]}
    runs = sorted(results["runs"], key=lambda r: (CASES.index(r["task"]), r["lr"]))
    raw = {}
    for run in runs:
        rid = run["run_id"]
        path = ROOT / "runs" / records[rid]["run"] / "artifacts/attempt-01-spark/production/metrics.jsonl"
        assert sha(source(path)) == run["source_sha256"]["metrics.jsonl"]
        rows = [json.loads(line) for line in path.read_text().splitlines()[:3000]]
        norms = np.array([r["streams"]["raw"]["norm"] for r in rows])
        rho = np.array([r["streams"]["raw"]["cosine"] for r in rows[1:]])
        q = 1 - rho * norms[1:] / norms[:-1]
        np.testing.assert_allclose(np.convolve(q, np.ones(50)/50, "valid"),
                                   arrays[f"eta_kappa_hat_smoothed_run{rid}_y"], rtol=2e-10, atol=2e-10)
        raw[rid] = q

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.25), sharex="col")
    for col, task in enumerate(CASES):
        axes[0, col].set_title(f"({chr(97+col)}) {TITLES[col]}", pad=15)
        for color, r in zip(COLORS, [r for r in runs if r["task"] == task]):
            rid = r["run_id"]
            curve(axes[0, col], arrays[f"loss_log_run{rid}_x"], arrays[f"loss_log_run{rid}_y"],
                  f"GD:loss:{rid}", color=color, lw=.8, label=rf"$\eta={r['lr']:g}$")
            s = np.arange(1, 3000)
            sx, sy = spectral[f"run{rid}_row1_x"], spectral[f"run{rid}_row1_y"]
            np.testing.assert_array_equal(sx, np.arange(0, 3001, 100))
            interpolated = np.interp(s, sx, sy)
            np.testing.assert_allclose(np.interp(sx[1:-1], s, interpolated), sy[1:-1], rtol=0, atol=0)
            curve(axes[1, col], s, centered(raw[rid]), f"GD:Q:centered:{rid}", color=color, lw=1)
            curve(axes[1, col], s, centered(interpolated), f"GD:spectrum:centered:{rid}", color=color, lw=.85, ls="--")
        axes[0, col].set_yscale("log")
        axes[0, col].yaxis.set_major_locator(FixedLocator([[.2,.3,.4,.6],[.4,.6,1,2],[6,7,8,9,10]][col]))
        axes[0, col].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axes[0, col].yaxis.set_minor_locator(NullLocator())
        axes[0, col].legend(loc="lower center", bbox_to_anchor=(.5, 1.015), ncol=3, frameon=False,
                            borderaxespad=0, fontsize=7, columnspacing=.7, handlelength=1.1, handletextpad=.25)
        axes[1, col].axhline(2, color=".45", lw=.6, ls=":")
        axes[1, col].set(xlabel=XLABEL, ylim=(-.04, [3.5,5.2,3.5][col]),
                         yticks=[0,1,2,3] if col != 1 else [0,2,4])
        letter(axes[1, col], chr(100+col))
        for ax in axes[:, col]:
            ax.set(xlim=(0,3000), xticks=[0,1000,2000,3000]); style(ax)
    axes[0,0].set_ylabel("Training loss"); axes[1,0].set_ylabel("Scaled curvature")
    fig.legend(handles=[Line2D([],[],color=".2",lw=1,label=r"$\eta\widehat\kappa_s$ (smoothed)"),
                        Line2D([],[],color=".2",lw=.9,ls="--",label=r"$\eta\lambda_{\max}$ (smoothed)")],
               loc="lower center", bbox_to_anchor=(.53,.005), ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.085,right=.97,top=.84,bottom=.22,hspace=.26,wspace=.29)
    save(fig, "gd_loss_projected_change_spectral_clean.pdf")

    fig, axes = plt.subplots(3, 3, figsize=(7.16, 4.5), sharex=True)
    for col, task in enumerate(CASES):
        axes[0,col].set_title(f"({chr(97+col)}) {TITLES[col]}", pad=15)
        for row, metric in enumerate(["rho_smoothed", "gradient_norms_smoothed", "normalized_displacement"]):
            ax = axes[row,col]
            for color, r in zip(COLORS, [r for r in runs if r["task"] == task]):
                tag = f"{metric}_run{r['run_id']}"
                # Legacy gradient index t evaluates at theta_(t-1); R_t ends at theta_t.
                x = arrays[tag+"_x"] - (1 if row < 2 else 0)
                curve(ax, x, arrays[tag+"_y"], "GD:"+tag, color=color, lw=.85, label=rf"$\eta={r['lr']:g}$")
            ax.set(xlim=(0,3000), xticks=[0,1000,2000,3000]); style(ax)
            if row == 0: ax.set(ylim=(-1.07,1.07), yticks=[-1,0,1]); ax.axhline(0,color=".5",lw=.6)
            elif row == 1: ax.set_yscale("log")
            else: ax.set(ylim=(0,1.03), yticks=[0,.5,1], xlabel=XLABEL)
            if row: letter(ax, chr(97+3*row+col))
        axes[0,col].legend(loc="lower center",bbox_to_anchor=(.5,1.015),ncol=3,frameon=False,
                            borderaxespad=0,fontsize=7,columnspacing=.7,handlelength=1.1,handletextpad=.25)
    for ax,label in zip(axes[:,0],[r"$\rho_s$",r"$\|g_s\|$",r"$R_s$"]): ax.set_ylabel(label)
    fig.subplots_adjust(left=.085,right=.97,top=.89,bottom=.10,hspace=.30,wspace=.29)
    save(fig,"gd_reversal_norm_retention.pdf")

    profiles = read(ROOT / "analyses/035-2026-11-02-eos-backward-anchors/results.json")["profiles"]
    fig, axes = plt.subplots(2,3,figsize=(7.16,3.15),sharey=True)
    for col,task in enumerate(CASES):
        p = next(p for p in profiles if (p["job"]["optimizer"],p["job"]["task"],p["job"]["start"]) == ("gd",task,2000))
        assert p["endpoint"] == 2256
        x,y = np.array([r["lag"] for r in p["rows"]]), np.array([r["cosine"] for r in p["rows"]])
        for row,limit in enumerate([256,32]):
            ax = axes[row,col]
            curve(ax,x[:limit],y[:limit],f"backward:{task}:{limit}",color=COLORS[0],lw=.6,marker=None if row==0 else "o",ms=1.8)
            ax.set(xlim=(1,limit),ylim=(-1.05,1.05),yticks=[-1,0,1],
                   xticks=[1,64,128,192,256] if row==0 else [1,8,16,24,32],
                   xlabel=r"Backward lag $k$ (steps)"+(" - zoom" if row else ""))
            ax.axhline(0,color=".5",lw=.6);style(ax)
            if row: letter(ax,chr(100+col))
            if col==0: ax.set_ylabel(r"$\cos(g_s,g_{s-k})$")
        axes[0,col].set_title(f"({chr(97+col)}) {TITLES[col]}",pad=6)
    fig.subplots_adjust(left=.085,right=.97,top=.90,bottom=.14,wspace=.2,hspace=.60)
    save(fig,"gd_backward_compact.pdf")

    # Retain the already independently checked grouped SGD values and uncertainty.
    previous = read(ROOT / "manuscript/draft/revisions/2026-11-02-feedback/figure_manifest.json")
    AUDIT["sgd_values"] = previous["sgd_values"]
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.15),sharey=True)
    for col,task in enumerate(CASES):
        ax=axes[col]
        for j,start in enumerate([0,2000]):
            entry=next(v for v in previous["sgd_values"] if v["task"]==task and v["checkpoint"]==start)
            # Values use the physical batch units preserved by the preceding generator.
            x=np.array(entry["plotted_batches"])
            y=np.array(entry["mean"])
            np.testing.assert_array_equal(y,np.mean(entry["initialization_means"],axis=0))
            curve(ax,x,y,f"SGD:{task}:{start}",color=COLORS[j],marker=["o","s"][j],ms=3,lw=.8,
                  label="At initialization" if j==0 else "After 2,000 GD updates")
            ax.fill_between(x,entry["minimum"],entry["maximum"],color=COLORS[j],alpha=.13,lw=0)
            ax.plot(x[-1],y[-1],marker=["o","s"][j],ms=4,mfc="white",mec=COLORS[j],lw=0)
        ax.set_xscale("log");ax.set_xticks(x,[f"{int(v):,}"+("\n(full)" if i==len(x)-1 else "") for i,v in enumerate(x)])
        ax.xaxis.set_minor_locator(NullLocator());ax.tick_params(axis="x",labelsize=7.5)
        ax.set(ylim=(-1.08,1.08),yticks=[-1,0,1],xlabel=r"Batch size $b$ ("+("sequences" if task=="transformer" else "images")+")")
        ax.set_title(f"({chr(97+col)}) {TITLES[col]}",pad=6);ax.axhline(0,color=".5",lw=.6);style(ax)
    axes[0].set_ylabel(r"Mean cosine $\rho_s$")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.53,1.005),ncol=2,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.085,right=.97,top=.76,bottom=.28,wspace=.18)
    save(fig,"sgd_batch_size_compact.pdf")


def adam_overview():
    path=source(ROOT/"analyses/055-2026-11-05-adam-directional-amplification/01_analyze.py")
    spec=importlib.util.spec_from_file_location("retained_adam",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    arms,sources=module.load_arms();AUDIT["sources"].update(sources)
    fig,axes=plt.subplots(4,3,figsize=(7.16,4.65),sharex="col")
    for col,case in enumerate(CASES):
        arm=next(a for a in arms if a["task"]==case and a["tier"]==2)
        xs=[np.arange(1025),np.arange(1,1024),np.arange(1,1024),np.arange(1024)]
        ys=[arm["loss"],arm["Q"][1:],arm["rho"][1:],arm["gradient"]/arm["gradient"][0]]
        qmax=float(np.max(ys[1]));qtop=np.ceil(qmax*1.06*2)/2
        AUDIT["overview"].append({"case":case,"Q_limits":[0,qtop],"negative_Q_outside_view":int(np.sum(ys[1]<0)),
                                  "min_Q":float(np.min(ys[1])),"max_Q":qmax})
        for row,(x,y) in enumerate(zip(xs,ys)):
            ax=axes[row,col];curve(ax,x,y,f"Adam:{case}:{row}",color=COLORS[2],lw=.65)
            for event in arm["events"]:
                if float(event["peak_change_percent"])<=10:continue
                a,p=int(event["before_state"]),int(event["peak_state"])
                if event["recovery_state"]:ax.axvspan(a,p,color=".5",alpha=.12,lw=0,zorder=-1)
                else:ax.axvspan(a,p,facecolor="none",edgecolor=".55",ls=":",lw=.6,zorder=-1)
            ax.set(xlim=(0,1024),xticks=[0,256,512,768,1024]);style(ax)
            if row==0:ax.set_title(f"({chr(97+col)}) {TITLES[col]}\n"+rf"$\eta={arm['lr']:g}$",pad=6)
            else:letter(ax,chr(97+3*row+col))
            if row in (0,3):
                ax.set_yscale("log");span=np.log10(ax.get_ylim()[1]/ax.get_ylim()[0])
                ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5) if span<2 else (1,),numticks=4))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v:g}"));ax.yaxis.set_minor_locator(NullLocator())
            elif row==1:
                ax.set(ylim=(0,qtop),yticks=np.arange(0,qtop+.1,1));ax.axhline(2,color=".45",ls="--",lw=.6)
            else:ax.set(ylim=(-1.06,1.06),yticks=[-1,0,1]);ax.axhline(0,color=".45",ls="--",lw=.6)
        axes[-1,col].set_xlabel(XLABEL)
    for ax,label in zip(axes[:,0],["Training loss",r"$\eta\widehat\kappa_s$",r"$\rho_s$",r"$\|g_s\|/\|g_0\|$"]):ax.set_ylabel(label)
    fig.subplots_adjust(left=.095,right=.97,top=.88,bottom=.10,hspace=.35,wspace=.32)
    save(fig,"00-high-lr-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf")


def trace_data():
    data={}
    for case in CASES:
        r=read(A63/(("mlp_450_500" if case=="mlp" else case)+"_results.json"))
        x=np.asarray(r["states"])+1
        bounds={b["state"]:b for b in r["boundaries"]}
        if case=="mlp":
            full=read(A63/"mlp_results.json")
            full_bounds={b["state"]:b for b in full["boundaries"]}
            assert all(b==full_bounds[s] for s,b in bounds.items())
            bounds=full_bounds
        loss=np.array([bounds[int(s)]["loss"] for s in x]);probe=np.array([bounds[int(s)]["probe_loss"] for s in x])
        if case=="mlp":np.testing.assert_array_equal(loss,probe)
        names,labels=zip(*STAGES[case]);rho=np.array([r["curves"][n]["rho"] for n in names],dtype=float)
        rms=np.array([r["curves"][n]["change_rms"] for n in names],dtype=float)
        assert rho.shape==rms.shape==(len(names),len(x)) and np.isfinite(rms).all()
        denom=rms.max(axis=1);assert (denom>0).all()
        relative=rms/denom[:,None]
        peak=int(np.argmax(loss));base=int(np.argmin(loss[:peak]))
        crossings=np.flatnonzero(loss[base+1:peak+1]>=1.1*loss[base]);assert len(crossings)
        loss_cross=base+1+int(crossings[0]);neg=np.flatnonzero(rho[-1]<0);assert len(neg)
        first_negative=int(neg[0]);left_censored=first_negative==0
        if not left_censored:assert rho[-1,first_negative-1]>=0
        record=dict(case=case,source_outgoing_window=[int(x[0]-1),int(x[-1]-1)],displayed_completed_steps=[int(x[0]),int(x[-1])],
                    stages=[{"source":n,"label":l} for n,l in STAGES[case]],rms_window_maxima=denom.tolist(),
                    baseline_step=int(x[base]),baseline_loss=float(loss[base]),first_rise_step=int(x[base+1]),
                    peak_step=int(x[peak]),loss_threshold=1.1,loss_threshold_step=int(x[loss_cross]),
                    first_observed_output_reversal_step=int(x[first_negative]),reversal_left_censored=left_censored,
                    heatmap_values=int(rho.size+rms.size),rho_sha256=arrsha(rho),rms_sha256=arrsha(rms))
        AUDIT["tracing"].append(record)
        data[case]=dict(result=r,x=x,loss=loss,probe=probe,names=names,labels=labels,rho=rho,relative=relative,record=record)
    return data


def markers(ax, record):
    for key,color in [("first_observed_output_reversal_step",COLORS[0]),("loss_threshold_step",".15")]:
        line=ax.axvline(record[key],color=color,ls=(0,(3,2)),lw=.8,zorder=5)
        line.set_path_effects([pe.Stroke(linewidth=1.7,foreground="white",alpha=.9),pe.Normal()])


def trace_figures():
    data=trace_data()
    fig,axes=plt.subplots(3,3,figsize=(7.16,4.65),sharex="col",gridspec_kw={"height_ratios":[1.1,1.45,1.45]})
    for col,case in enumerate(CASES):
        d=data[case];x=d["x"];record=d["record"];extent=[x[0]-.5,x[-1]+.5,len(d["names"])-.5,-.5]
        curve(axes[0,col],x,d["loss"],f"trace:{case}:full-loss",color=".12",lw=1,label="Full objective")
        if case!="mlp":curve(axes[0,col],x,d["probe"],f"trace:{case}:probe-loss",color=COLORS[1],lw=.9,ls="--",label="Fixed probe")
        axes[0,col].set_title(f"({chr(97+col)}) {TITLES[col]}",pad=8)
        axes[0,col].yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v:g}"));style(axes[0,col])
        axes[0,col].plot(record["loss_threshold_step"],d["loss"][np.where(x==record["loss_threshold_step"])[0][0]],"o",color=".12",ms=2.6,zorder=6)
        for row,values,cmap,vmin in [(1,d["rho"],"RdBu_r",-1),(2,d["relative"],"viridis",0)]:
            im=axes[row,col].imshow(values,aspect="auto",interpolation="nearest",extent=extent,cmap=cmap,vmin=vmin,vmax=1)
            np.testing.assert_array_equal(np.asarray(im.get_array()),values)
            axes[row,col].set_yticks(range(len(d["labels"])),d["labels"],fontsize=7.5)
            axes[row,col].tick_params(length=0,pad=2);letter(axes[row,col],chr(97+3*row+col))
            if row==1:rho_image=im
            else:rms_image=im
        ticks={"mlp":[451,465,480,501],"resnet":[281,290,300,311],"transformer":[553,565,580,601]}[case]
        for ax in axes[:,col]:ax.set(xlim=(x[0]-.5,x[-1]+.5),xticks=ticks);markers(ax,record)
        axes[-1,col].set_xlabel(XLABEL)
    axes[0,0].set_ylabel("Training loss")
    axes[1,0].set_ylabel("Change cosine",labelpad=9)
    axes[2,0].set_ylabel("Displacement RMS\n/ stage maximum",labelpad=7)
    handles=[Line2D([],[],color=".12",lw=1,label="Full objective"),Line2D([],[],color=COLORS[1],ls="--",label="Fixed probe"),
             Line2D([],[],color=COLORS[0],ls="--",label=r"First observed output $\rho<0$"),Line2D([],[],color=".15",ls="--",label="Loss +10%")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.53,.005),ncol=2,frameon=False,fontsize=7.5,columnspacing=1.4,handlelength=2)
    fig.subplots_adjust(left=.14,right=.91,top=.91,bottom=.20,hspace=.31,wspace=.55)
    box=axes[1,2].get_position();fig.colorbar(rho_image,cax=fig.add_axes([.932,box.y0,.012,box.height]),ticks=[-1,0,1])
    box=axes[2,2].get_position();fig.colorbar(rms_image,cax=fig.add_axes([.932,box.y0,.012,box.height]),ticks=[0,.5,1])
    save(fig,"00-three-model-loss-representation-motion.pdf")

    fig,axes=plt.subplots(3,3,figsize=(7.16,5.65),gridspec_kw={"height_ratios":[1,1.25,1]})
    for col,case in enumerate(CASES):
        d=data[case];r=d["result"];x=d["x"];records={v["name"]:v for v in r["stages"]};out=d["names"][-1]
        for name,label,color,ls in [(out,"Output changes",COLORS[1],"-"),("raw_gradient","Raw gradients",COLORS[0],"--")]:
            values=np.asarray(r["curves"][name]["spectrum"]);assert abs(values.sum()-1)<1e-9
            curve(axes[0,col],np.arange(1,len(values)+1),np.cumsum(values),f"spectrum:{case}:{name}",color=color,ls=ls,lw=1,label=label)
        axes[0,col].axhline(.95,color=".5",ls=":",lw=.6);axes[0,col].set(xlim=(1,len(x)),ylim=(0,1.03),yticks=[0,.5,1],xlabel="Number of modes")
        axes[0,col].set_title(f"({chr(97+col)}) {TITLES[col]}",pad=8);style(axes[0,col])
        y=np.arange(len(d["names"]))
        for key,label,color,marker in [("top1","Top-1 energy",COLORS[1],"o"),("top10","Top-10 energy",COLORS[0],"s"),
                                       ("temporal_top10_overlap","Output overlap",COLORS[2],"^")]:
            values=np.array([records[n][key] for n in d["names"]]);curve(axes[1,col],values,y,f"spectral-stages:{case}:{key}",
                color=color,marker=marker,ms=3,lw=.7,label=label)
        axes[1,col].set(xlim=(0,1.03),xticks=[0,.5,1],ylim=(len(y)-.5,-.5),xlabel="Energy fraction / overlap")
        axes[1,col].set_yticks(y,d["labels"],fontsize=7.5);style(axes[1,col]);letter(axes[1,col],chr(100+col))
        alpha=np.array(r["curves"][out]["alpha"])/np.sqrt(records[out]["coordinates"])
        curve(axes[2,col],x,alpha,f"output-coefficient:{case}",color=COLORS[1],lw=.9)
        axes[2,col].axhline(0,color=".5",lw=.6);axes[2,col].set(xlim=(x[0],x[-1]),xlabel=XLABEL)
        style(axes[2,col]);letter(axes[2,col],chr(103+col))
        axes[2,col].ticklabel_format(axis="y",style="sci",scilimits=(-2,2),useMathText=True)
    axes[0,0].set_ylabel("Cumulative energy")
    axes[2,0].set_ylabel(r"Output $\alpha_1/\sqrt{P}$")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.53,1.01),ncol=2,frameon=False,fontsize=7.5)
    fig.legend(*axes[1,0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.53,.002),ncol=3,frameon=False,fontsize=7.5)
    fig.subplots_adjust(left=.13,right=.97,top=.91,bottom=.13,hspace=.70,wspace=.55)
    save(fig,"01-three-model-temporal-structure.pdf")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":9,
        "axes.titlesize":9,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.6,
        "pdf.fonttype":42,"path.simplify":False})
    # An affine signal is unchanged inside a centered window: no smoothing phase shift.
    test=np.arange(301,dtype=float);np.testing.assert_allclose(centered(test)[50:-50],test[50:-50])
    gd_figures();adam_overview();trace_figures()
    np.savez_compressed(HERE/"plotted_arrays.npz",**PLOTTED)
    AUDIT["plotted_arrays_sha256"]=sha(HERE/"plotted_arrays.npz")
    AUDIT["generator_sha256"]=sha(__file__)
    AUDIT["status"]="numerical_checks_passed"
    AUDIT["visual_review_record"]="verification.json"
    (HERE/"figure_manifest.json").write_text(json.dumps(AUDIT,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"figures":len(AUDIT["figures"]),"curves":len(AUDIT["curves"]),"tracing":AUDIT["tracing"],"overview":AUDIT["overview"]},indent=2))


if __name__=="__main__":
    main()
