"""Five compact paper figures from terminal, retrieved GD/SGD seed cohorts."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,write_json
HERE=Path(__file__).parent
TASKS=["mlp","resnet","transformer"]
TITLES=["CIFAR-10 · tanh MLP","CIFAR-10 · ResNet-32","WikiText-2 · Transformer"]
COLOURS=["#0072B2","#D55E00","#8C4AA8"]


def jsonl(path):return [json.loads(s) for s in path.read_text().splitlines()]


def trailing(y,window=100):
    y=np.asarray(y,dtype=float)
    out=np.full_like(y,np.nan)
    for i in range(window-1,len(y)):
        values=y[i-window+1:i+1]
        if np.isfinite(values).all():out[i]=values.mean()
    return out


def load(record):
    run=ROOT/"runs"/record["run"]
    folder=run/"artifacts/attempt-01-spark/production"
    if record["kind"]=="new_training":assert json.loads((run/"retrieval_verification.json").read_text())["status"]=="passed"
    manifest=json.loads((folder/"manifest.json").read_text())
    assert manifest["status"]=="complete"
    rows=jsonl(folder/"metrics.jsonl")
    count=record["updates"]
    assert len(rows)>=count
    active=rows[:count]
    def stream(field):return np.array([r["streams"]["raw"][field] if r["streams"]["raw"][field] is not None else np.nan for r in active])
    if record["kind"]=="reused_gd_prefix":
        loss=[r["train_loss"] for r in active]
        q=[r["eta_kappa_proxy"] for r in active]
        final=rows[count]["train_loss"] if len(rows)>count else json.loads((folder/"summary.json").read_text())["final_train"]["loss"]
        objective=loss
    else:
        loss=[r["batch_objective_estimate"] for r in active]
        objective=[r["full_objective_loss"] if r["full_objective_loss"] is not None else np.nan for r in active]
        q=[r["eta_kappa_gradient_proxy"] for r in active]
        final=json.loads((folder/"summary.json").read_text())["final_train"]["loss"]
    spectral=[r for r in jsonl(folder/"curvature.jsonl") if r["state_step"]<=count]
    assert max(v for r in spectral for v in (r["hessian"]["relative_residuals"] if "hessian" in r else r["relative_residuals"]))<=.005
    eigen=[r["hessian"]["eigenvalues"][0] if "hessian" in r else r["eigenvalues"][0] for r in spectral]
    noise=jsonl(folder/"noise.jsonl") if (folder/"noise.jsonl").exists() else []
    ratio=stream("B_over_A")
    assert np.all((ratio>=0)&(ratio<=1+1e-10))
    return {"record":record,"x":np.arange(1,count+1),"loss":np.asarray(loss),"objective":np.asarray(objective),
        "final_loss":final,"cosine":stream("cosine"),"norm":stream("norm"),"ratio":ratio,
        "q":np.array([np.nan if v is None else v for v in q]),"spectral_x":np.array([r["state_step"] for r in spectral]),
        "eta_lambda":record["lr"]*np.array(eigen),"noise":noise,
        "source_sha256":{f.name:file_hash(f) for f in [folder/"metrics.jsonl",folder/"curvature.jsonl",folder/"manifest.json"]}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--accept-reviewed",action="store_true")
    args=parser.parse_args()
    mf=HERE/"training_figure_manifest.json"
    if args.accept_reviewed:
        record=json.loads(mf.read_text())
        assert file_hash(__file__)==record["script_sha256"]
        for f in record["figures"]:
            assert file_hash(ROOT/f["pdf"])==f["pdf_sha256"]
            assert file_hash(ROOT/f["render"])==f["render_sha256"]
        record.update(status="passed",visual_review="All five current Poppler renders reviewed for axes, reductions, legends and clipping.")
        write_json(mf,record);return
    assert json.loads((HERE/"retrieval_verification.json").read_text())["status"]=="passed"
    cohort=json.loads((HERE/"cohort.json").read_text())
    data=[load(r) for r in cohort["runs"]]
    plt.rcParams.update({"font.family":"serif","mathtext.fontset":"stix","font.size":9,
        "axes.labelsize":9,"axes.titlesize":10,"xtick.labelsize":8,"ytick.labelsize":8,
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,"pdf.fonttype":42})
    curves=[];figures=[]
    poppler=shutil.which("pdftoppm") or str(Path.home()/"AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdftoppm.exe")
    def line(ax,x,y,**kw):
        artist,=ax.plot(x,y,**kw)
        assert np.array_equal(artist.get_xdata(),x,equal_nan=True)
        assert np.array_equal(artist.get_ydata(),y,equal_nan=True)
        curves.append({"x_sha256":hashlib.sha256(np.asarray(x,dtype="<f8").tobytes()).hexdigest(),
            "y_sha256":hashlib.sha256(np.asarray(y,dtype="<f8").tobytes()).hexdigest(),"points":len(x)})
    def save(fig,name):
        pdf=HERE/"figures"/(name+".pdf");pdf.parent.mkdir(exist_ok=True)
        png=HERE/"scratch"/(name+".png");png.parent.mkdir(exist_ok=True)
        fig.savefig(pdf,metadata={"CreationDate":None,"ModDate":None});plt.close(fig)
        subprocess.run([poppler,"-r","150","-singlefile","-png",str(pdf),str(png.with_suffix(""))],check=True,capture_output=True)
        figures.append({"pdf":pdf.relative_to(ROOT).as_posix(),"pdf_sha256":file_hash(pdf),
            "render":png.relative_to(ROOT).as_posix(),"render_sha256":file_hash(png)})
    def base(rows,height):
        fig,axes=plt.subplots(rows,3,figsize=(8.4,height),squeeze=False)
        for row in range(rows):
            for col in range(3):
                ax=axes[row,col];ax.grid(alpha=.17,lw=.6)
                ax.text(.025,.94,f"({chr(97+row*3+col)})",transform=ax.transAxes,va="top")
                if row==0:ax.set_title(TITLES[col],pad=9)
                if row==rows-1:ax.set_xlabel("Optimizer update")
        return fig,axes
    groups=[[d for d in data if d["record"]["task"]==task and d["record"]["method"]=="gd"] for task in TASKS]
    for group in groups:assert len(group)==3
    fig,axes=base(1,2.8)
    for col,group in enumerate(groups):
        ys=[]
        for d in group:
            x=np.arange(len(d["loss"])+1);y=np.append(d["loss"],d["final_loss"])
            line(axes[0,col],x,y,color=COLOURS[col],lw=.65,alpha=.3);ys.append(y)
        line(axes[0,col],x,np.mean(ys,axis=0),color=COLOURS[col],lw=1.6)
        axes[0,col].set_ylabel("Full objective loss" if col==0 else "")
    fig.subplots_adjust(left=.075,right=.985,bottom=.19,top=.83,wspace=.3)
    save(fig,"eos_gd_loss")
    fig,axes=base(3,6.7)
    labels=[r"Adjacent cosine $\rho_t$",r"Projected change $\eta\widehat\kappa_t$",r"Sharpness $\eta\lambda_{\max}$"]
    for col,group in enumerate(groups):
        for row,field in enumerate(["cosine","q","eta_lambda"]):
            series=[]
            for d in group:
                x=d["spectral_x"] if row==2 else d["x"]
                y=d[field] if row==2 else trailing(d[field])
                if row<2:line(axes[row,col],x,d[field],color=COLOURS[col],lw=.3,alpha=.1)
                line(axes[row,col],x,y,color=COLOURS[col],lw=.7,alpha=.4);series.append(y)
            line(axes[row,col],x,np.mean(series,axis=0),color=COLOURS[col],lw=1.6)
            axes[row,col].axhline(2 if row==2 else 0,color=".4",ls="--",lw=.8)
            if col==0:axes[row,col].set_ylabel(labels[row])
    fig.subplots_adjust(left=.08,right=.985,bottom=.1,top=.93,wspace=.3,hspace=.27)
    save(fig,"eos_gd_alignment_curvature")
    fig,axes=base(2,4.5)
    for col,group in enumerate(groups):
        for row,field in enumerate(["ratio","norm"]):
            ys=[d[field] for d in group]
            for d in group:line(axes[row,col],d["x"],d[field],color=COLOURS[col],lw=.6,alpha=.3)
            line(axes[row,col],group[0]["x"],np.mean(ys,axis=0),color=COLOURS[col],lw=1.6)
            if row==1:axes[row,col].set_yscale("log")
            if col==0:axes[row,col].set_ylabel(r"Displacement $B_t/A_t$" if row==0 else r"Gradient norm $\|g_t\|$")
    fig.subplots_adjust(left=.09,right=.985,bottom=.13,top=.89,wspace=.3,hspace=.26)
    save(fig,"eos_gd_displacement_norms")
    batchcolours=["#D55E00","#8C4AA8","#0072B2","#333333"]
    fig,axes=base(3,7.0)
    summaryfig,summaryaxes=base(3,7.0)
    summaries=[]
    for col,task in enumerate(TASKS):
        cases=[d for d in data if d["record"]["task"]==task]
        bs=sorted(set(d["record"].get("batch_units") for d in cases if d["record"]["method"]=="sgd"))+[None]
        for bi,b in enumerate(bs):
            selected=[d for d in cases if d["record"].get("batch_units")==b]
            assert len(selected)==3
            label="Full batch" if b is None else f"b={b*1000 if task=='resnet' else b}"
            colour=batchcolours[bi]
            for row,field in enumerate(["objective","cosine","ratio"]):
                ys=[trailing(d[field]) if field=="cosine" else d[field] for d in selected]
                y=np.mean(ys,axis=0)
                valid=np.isfinite(y)
                line(axes[row,col],selected[0]["x"][valid],y[valid],color=colour,lw=1.2,label=label)
                if col==0:axes[row,col].set_ylabel(["Full objective loss",r"Adjacent cosine $\rho_t$",r"Displacement $B_t/A_t$"][row])
            noise=[];outcomes=[]
            for d in selected:
                late=[r for r in d["noise"] if r["state_step"]>=max(0,d["record"]["updates"]-200)]
                rms=0. if b is None else float(np.sqrt(np.mean([r["estimated_batch_gradient_noise_energy"] for r in late])))
                snr=0. if b is None else rms/float(np.mean([r["reference_gradient_norm"] for r in late]))
                noise.append(snr)
                outcomes.append([d["final_loss"],float(np.nanmean(d["cosine"][-100:])),d["ratio"][-1]])
            outcomes=np.array(outcomes)
            for row in range(3):
                summaryaxes[row,col].errorbar(np.mean(noise),outcomes[:,row].mean(),
                    yerr=[[outcomes[:,row].mean()-outcomes[:,row].min()],[outcomes[:,row].max()-outcomes[:,row].mean()]],
                    marker="s" if b is None else "o",color=colour,capsize=2,ms=4,label=label)
                summaryaxes[row,col].set_xscale("symlog",linthresh=.1)
                if row==2:summaryaxes[row,col].set_xlabel(r"Measured noise RMS / $\|\nabla F\|$")
                if col==0:summaryaxes[row,col].set_ylabel(["Final objective loss",r"Late adjacent cosine $\rho_t$",r"Final displacement $B_T/A_T$"][row])
            summaries.append({"task":task,"batch_units":b,"relative_noise_rms":noise,"final_values":outcomes.tolist()})
        axes[0,col].legend(fontsize=7,frameon=False,loc="best")
        summaryaxes[0,col].legend(fontsize=7,frameon=False,loc="best")
    for figure in (fig,summaryfig):figure.subplots_adjust(left=.09,right=.985,bottom=.1,top=.93,wspace=.3,hspace=.3)
    save(fig,"eos_sgd_batch_diagnostics");save(summaryfig,"eos_sgd_noise_summary")
    write_json(HERE/"training_results.json",{"status":"complete","horizon":cohort["horizon"],"noise_summary":summaries,
        "runs":[{"run_id":d["record"]["run_id"],"task":d["record"]["task"],"method":d["record"]["method"],
            "seed":d["record"]["seed"],"updates":len(d["x"]),"final_loss":d["final_loss"],
            "final_B_over_A":d["ratio"][-1],"source_sha256":d["source_sha256"]} for d in data]})
    write_json(mf,{"status":"awaiting_visual_review","script_sha256":file_hash(__file__),"figures":figures,"curves":curves,
        "reduction":"cosine/q complete100-pair trailing means with faint raw GD; objective, norms and B/A unsmoothed; seed means and individual GD lines",
        "uncertainty":"SGD summary error bars show the range across3seeds, not95%CI; fixed update budget, not matched data exposure",
        "source_results_sha256":file_hash(HERE/"training_results.json")})


if __name__=="__main__":main()
