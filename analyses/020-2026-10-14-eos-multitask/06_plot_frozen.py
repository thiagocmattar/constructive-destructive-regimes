"""Publication-format real-data noise control; all uncertainty stays explicit."""
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
RUN=ROOT/"runs/068-2026-10-14-eos-frozen-gradient-noise"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--accept-reviewed",action="store_true")
    args=parser.parse_args()
    manifestfile=HERE/"frozen_figure_manifest.json"
    pdf=HERE/"figures/eos_frozen_noise_comparison.pdf"
    render=HERE/"scratch/eos_frozen_noise_comparison.png"
    resultfile=RUN/"artifacts/attempt-01-spark/production/results.json"
    if args.accept_reviewed:
        record=json.loads(manifestfile.read_text())
        assert file_hash(pdf)==record["pdf_sha256"] and file_hash(render)==record["render_sha256"]
        assert file_hash(resultfile)==record["source_results_sha256"] and file_hash(__file__)==record["script_sha256"]
        record.update(status="passed",visual_review="Current Poppler render inspected: labels, axes, uncertainty, legends and clipping.")
        write_json(manifestfile,record);return
    assert json.loads((RUN/"retrieval_verification.json").read_text())["status"]=="passed"
    result=json.loads(resultfile.read_text())
    assert result["status"]=="complete" and len(result["conditions"])==36
    source=result["conditions"]
    for condition in source:
        condition["mean_gradient_only_projected_change"]=float(np.mean([
            r["mean_gradient_only_projected_change"] for r in condition["replicates"]]))
    assert all(c["maximum_corrected_identity_error"]<1e-7 for c in source)
    plt.rcParams.update({"font.family":"serif","mathtext.fontset":"stix","font.size":9,
        "axes.labelsize":9,"axes.titlesize":10,"xtick.labelsize":8,"ytick.labelsize":8,
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,
        "pdf.fonttype":42,"savefig.facecolor":"white"})
    fig,axes=plt.subplots(3,3,figsize=(8.4,6.65))
    colours={"mlp":"#0072B2","resnet":"#D55E00","transformer":"#8C4AA8"}
    titles={"mlp":"CIFAR-10 · tanh MLP","resnet":"CIFAR-10 · ResNet-32","transformer":"WikiText-2 · Transformer"}
    fields=["mean_cosine","mean_B_over_A","mean_gradient_only_projected_change"]
    labels=[r"Adjacent cosine $\rho$",r"Normalized accumulation $B/A$",r"Projected change $q$"]
    curves=[]
    summaries=[]
    for col,task in enumerate(colours):
        colour=colours[task]
        conditions=[c for c in source if c["task"]==task]
        batches=sorted(set(c["batch_units"] for c in conditions),reverse=True)
        groups=[[c for c in conditions if c["batch_units"]==b] for b in batches]
        x=np.array([np.mean([c["relative_noise_rms"] for c in group]) for group in groups])
        for row,(field,label) in enumerate(zip(fields,labels)):
            ax=axes[row,col]
            if row==0:ax.set_title(titles[task],pad=9)
            values=np.array([[c[field] for c in sorted(group,key=lambda c:c["seed"])] for group in groups])
            ax.fill_between(x,values.min(1),values.max(1),color=colour,alpha=.16,linewidth=0)
            for seed in (0,1,2):
                sx=np.array([next(c for c in group if c["seed"]==seed)["relative_noise_rms"] for group in groups])
                sy=values[:,seed]
                line,=ax.plot(sx,sy,color=colour,alpha=.35,lw=.7)
                assert np.array_equal(line.get_xdata(),sx) and np.array_equal(line.get_ydata(),sy)
                curves.append({"panel":[row,col],"seed":seed,"field":field,"x":sx.tolist(),"y":sy.tolist()})
            mean=values.mean(1)
            ax.plot(x,mean,"o-",color=colour,lw=1.7,ms=4,zorder=4)
            ax.plot(0,mean[0],marker="s",color=colour,ms=5,zorder=5)
            if row<2:
                ax.set_ylim(min(-.025,float(values.min())-.025),1.055)
                ax.axhline(0,color=".7",lw=.6,ls=":",zorder=0)
            else:
                ax.axhline(0,color=".3",ls="--",lw=1,label="Noise-corrected change")
                ax.set_ylim(min(-.05,float(values.min())-.06),max(1.04,float(values.max())+.06))
            ax.set_xscale("symlog",linthresh=.1)
            ax.set_xlim(-.012,max(x)*1.22)
            ax.grid(alpha=.18,lw=.6)
            ax.text(.015,1.015,f"({chr(97+row*3+col)})",transform=ax.transAxes,fontsize=9,va="bottom")
            if col==0:ax.set_ylabel(label)
            if row==2:ax.set_xlabel(r"Measured noise RMS / $\|\nabla F\|$")
        for group in groups:
            summaries.append({"task":task,"batch_units":group[0]["batch_units"],"seeds":len(group),
                "relative_noise_rms_mean":float(np.mean([c["relative_noise_rms"] for c in group])),
                "B_over_A_mean":float(np.mean([c["mean_B_over_A"] for c in group])),
                "cosine_mean":float(np.mean([c["mean_cosine"] for c in group])),
                "B_over_A_seed_range":[min(c["mean_B_over_A"] for c in group),max(c["mean_B_over_A"] for c in group)],
                "cosine_seed_range":[min(c["mean_cosine"] for c in group),max(c["mean_cosine"] for c in group)]})
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],color=".25",marker="o",lw=1.7,label="Mean over 3 initializations"),
        Line2D([],[],color=".5",lw=7,alpha=.2,label="Range across initializations"),
        Line2D([],[],color=".25",marker="s",lw=0,label="Full-batch control"),
        Line2D([],[],color=".3",ls="--",lw=1,label="Noise-corrected change = 0")],
        loc="lower center",bbox_to_anchor=(.5,.012),ncol=2,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.085,right=.985,top=.95,bottom=.155,hspace=.28,wspace=.25)
    pdf.parent.mkdir(exist_ok=True)
    render.parent.mkdir(exist_ok=True)
    fig.savefig(pdf,metadata={"CreationDate":None,"ModDate":None});plt.close(fig)
    poppler=shutil.which("pdftoppm")
    if not poppler:
        poppler=str(Path.home()/"AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdftoppm.exe")
    subprocess.run([poppler,"-r","150","-singlefile","-png",str(pdf),str(render.with_suffix(""))],check=True,capture_output=True)
    write_json(HERE/"frozen_results.json",{"status":"complete","conditions":summaries,
        "source_results_sha256":file_hash(resultfile),"within_seed_replications":32,"sequence_length":128,
        "seeds":3,"parameter_updates":0,"scope":"Frozen initialization controls only; not training dynamics."})
    write_json(manifestfile,{"status":"awaiting_visual_review","pdf_sha256":file_hash(pdf),"render_sha256":file_hash(render),
        "source_results_sha256":file_hash(resultfile),"script_sha256":file_hash(__file__),"curves":curves,
        "source_conditions":36,"metric_draws":36*32*128,"seed_uncertainty":"range, not95%CI",
        "noise_corrected_identity":"all frozen signal changes exactly zero within reported roundoff"})
    print(str(pdf))


if __name__=="__main__":main()
