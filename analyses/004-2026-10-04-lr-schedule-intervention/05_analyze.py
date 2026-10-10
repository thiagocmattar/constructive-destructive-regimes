"""Single-seed, checkpoint-matched schedule contrasts; accepted artifacts only."""
from pathlib import Path
import csv
import hashlib
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COLORS = {"original":"#0072B2", "hold":"#D55E00", "accelerated":"#009E73"}
STYLES = {"original":"-", "hold":"--", "accelerated":"-."}
LABELS = {"original":"Original decay", "hold":"Held midpoint LR", "accelerated":"Accelerated decay"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def rolling(values, width=31):
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    half = width//2
    for i in range(half,len(values)-half):
        window=values[i-half:i+half+1]
        if np.isfinite(window).all():
            result[i]=np.mean(window)
    return result


def recovery(rows,width=31,hold=31):
    # Every input pair is affected by the intervention, and the tail is omitted.
    selected=[r for r in rows if 358<=r["update"]<=711]
    means=rolling([r["adjacent_cosine"] for r in selected],width)
    for i in range(len(means)-hold+1):
        if np.all(means[i:i+hold]>0):
            later=means[i+hold:]
            return {"first_center_update":selected[i]["update"],
                "last_held_center_update":selected[i+hold-1]["update"],
                "later_negative_mean":bool(np.any(later[np.isfinite(later)]<0))}
    return {"first_center_update":None,"last_held_center_update":None,"later_negative_mean":None}


def stats(rows,first=358,last=712):
    selected=[r for r in rows if first<=r["update"]<=last]
    cosine=np.array([r["adjacent_cosine"] for r in selected])
    assert np.isfinite(cosine).all()
    negative=cosine[cosine<0]
    F=float(np.mean(cosine<0))
    D=float(np.mean(np.maximum(-cosine,0)))
    M=float(np.mean(-negative)) if len(negative) else None
    assert math.isclose(D,F*M if M is not None else 0,rel_tol=1e-14,abs_tol=1e-15)
    return {"first_update":first,"last_update":last,"pairs":len(selected),
        "F_negative_frequency":F,"D_unconditional_negative_intensity":D,
        "M_conditional_negative_magnitude":M,"median_raw_cosine":float(np.median(cosine)),
        "mean_raw_cosine":float(np.mean(cosine)),
        "training_loss_token_weighted":float(np.average([r["loss"] for r in selected],
            weights=[r["supervised_tokens"] for r in selected]))}


def expected_lr(update,cfg):
    variant=cfg["resume"]["schedule"]
    clock=update if variant=="original" else 356 if variant=="hold" else min(712,356+2*(update-356))
    t=cfg["training"]
    iteration=clock-1
    warmup=t["warmup_fraction"]*712
    bounded=min(float(iteration),712-warmup)
    value=t["peak_learning_rate"]*0.5*(math.cos(math.pi*(bounded-warmup)/712)+1)
    return max(value,t["minimum_learning_rate"])


def save_pdf(fig,name):
    pdf=HERE/"figures"/(name+".pdf")
    fig.savefig(pdf,bbox_inches="tight",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)
    with fitz.open(pdf) as doc:
        assert len(doc)==1
        for block in doc[0].get_text("blocks"):
            assert doc[0].rect.contains(fitz.Rect(block[:4]))
        doc[0].get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(str(HERE/"scratch"/(name+".png")))
    return {"path":pdf.relative_to(ROOT).as_posix(),"sha256":sha(pdf),"text_bounds_ok":True}


def main():
    (HERE/"figures").mkdir(exist_ok=True)
    (HERE/"scratch").mkdir(exist_ok=True)
    records,source,data=[],[],[]
    for row in json.loads((HERE/"cohort.json").read_text()):
        run=ROOT/"runs"/row["run"]
        receipt=json.loads((run/"prelaunch/attempt1-retrieval.json").read_text())
        assert receipt["verified"] and receipt["scientific_verification"]["verified"]
        base=run/"artifacts/attempt-01/full-pass"
        cfg=json.loads((base/"resolved_config.json").read_text())
        raw=events(base/"events.jsonl")
        rows=[r for r in raw if r["event"]=="update"]
        vals=[r for r in raw if r["event"]=="validation"]
        assert [r["update"] for r in rows]==list(range(357,713))
        assert [r["update"] for r in vals]==[356,445,534,623,712]
        np.testing.assert_allclose([r["learning_rate"] for r in rows],
            [expected_lr(r["update"],cfg) for r in rows],rtol=1e-14,atol=0)
        parent_path=ROOT/"runs"/row["parent_run"]/"artifacts/attempt-01/full-pass/events.jsonl"
        parent_events=events(parent_path)
        parent=[r for r in parent_events if r["event"]=="update"]
        combined=parent[:356]+rows
        assert [r["update"] for r in combined]==list(range(1,713))
        record={**row,"rows":rows,"combined":combined,"validation":vals,
            "primary":stats(rows),"full_batches_only_sensitivity":stats(rows,358,711),
            "phases":[stats(rows,a,b) for a,b in [(358,445),(446,534),(535,623),(624,712)]],
            "original_floor_phase":stats(rows,576,711),
            "recovery":{str(w):recovery(rows,w) for w in [15,31,61]},
            "terminal_validation_loss":vals[-1]["loss"],
            "terminal_training_loss_last31_token_weighted":stats(rows,682,712)["training_loss_token_weighted"],
            "terminal_raw_B_over_A_including_parent_prefix":rows[-1]["cumulative_ratio"],
            "terminal_adaptive_B_over_A_including_parent_prefix":rows[-1]["adaptive"]["cumulative_ratio"]}
        if row["variant"]=="original":
            record["archived_parent_primary"]=stats(parent[356:])
            record["control_vs_archive_primary_difference"]={key:record["primary"][key]-record["archived_parent_primary"][key]
                for key in ["F_negative_frequency","D_unconditional_negative_intensity","M_conditional_negative_magnitude","median_raw_cosine"]}
            parent_summary_path=parent_path.with_name("summary.json")
            parent_summary=json.loads(parent_summary_path.read_text())
            record["control_vs_archive_terminal_validation_difference"]=vals[-1]["loss"]-parent_summary["final_validation"]["loss"]
            record["original_vs_archived_parent"]={k:{"mean_absolute_error":float(np.mean([abs(a[k]-b[k]) for a,b in zip(rows,parent[356:])])),
                "maximum_absolute_error":max(abs(a[k]-b[k]) for a,b in zip(rows,parent[356:]))}
                for k in ["loss","adjacent_cosine","gradient_norm","cumulative_ratio"]}
        records.append(record)
        source.append({"run":row["run"],"events_sha256":sha(base/"events.jsonl"),
            "parent_run":row["parent_run"],"parent_events_sha256":sha(parent_path),
            "parent_summary_sha256":sha(parent_path.with_name("summary.json")),
            "archive_sha256":receipt["archive_sha256"],"acceptance_sha256":sha(run/"prelaunch/attempt1-retrieval.json")})
        display=combined[:-1]
        cosine=rolling([r["adjacent_cosine"] if r["adjacent_cosine"] is not None else np.nan for r in display])
        loss=rolling([r["loss"] for r in display])
        for i,r in enumerate(combined):
            data.append({"model":row["size"],"schedule":row["variant"],"update":r["update"],
                "inherited_prefix":r["update"]<=356,"common_first_gradient":r["update"]==357,
                "primary_contrast":358<=r["update"]<=712,"input_tokens":r["input_tokens"],
                "supervised_tokens":r["supervised_tokens"],"training_loss":r["loss"],"raw_cosine":r["adjacent_cosine"],
                "learning_rate":r["learning_rate"],"preceding_learning_rate":combined[i-1]["learning_rate"] if i else None,
                "centered_mean31_loss":float(loss[i]) if i<711 and np.isfinite(loss[i]) else None,
                "centered_mean31_cosine":float(cosine[i]) if i<711 and np.isfinite(cosine[i]) else None})
    # All conditions consume identical batches, including the final short batch.
    assert all([r["batch_indices_sha256"] for r in rec["rows"]]==
        [r["batch_indices_sha256"] for r in records[0]["rows"]] for rec in records)
    for size in ["14M","70M"]:
        group=[r for r in records if r["size"]==size]
        for rec in group[1:]:
            for key in ["loss","gradient_norm","adjacent_cosine","cumulative_ratio"]:
                assert math.isclose(rec["rows"][0][key],group[0]["rows"][0][key],rel_tol=1e-5,abs_tol=1e-6)
            assert math.isclose(rec["validation"][0]["loss"],group[0]["validation"][0]["loss"],rel_tol=1e-6,abs_tol=1e-7)
    contrasts=[]
    for size in ["14M","70M"]:
        group={r["variant"]:r for r in records if r["size"]==size}
        original=group["original"]
        for variant in ["hold","accelerated"]:
            rec=group[variant]
            contrasts.append({"model":size,"schedule":variant,"reference":"original",
                "delta_F":rec["primary"]["F_negative_frequency"]-original["primary"]["F_negative_frequency"],
                "delta_D":rec["primary"]["D_unconditional_negative_intensity"]-original["primary"]["D_unconditional_negative_intensity"],
                "delta_M":rec["primary"]["M_conditional_negative_magnitude"]-original["primary"]["M_conditional_negative_magnitude"] if all(r["primary"]["M_conditional_negative_magnitude"] is not None for r in [rec,original]) else None,
                "delta_median_raw_cosine":rec["primary"]["median_raw_cosine"]-original["primary"]["median_raw_cosine"],
                "delta_terminal_validation_loss":rec["terminal_validation_loss"]-original["terminal_validation_loss"]})
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    figures=[]
    all_cosines=[rolling([r["adjacent_cosine"] if r["adjacent_cosine"] is not None else np.nan for r in rec["combined"][:-1]])[299:] for rec in records]
    low=max(-1.02,min(float(np.nanmin(x)) for x in all_cosines)-.06)
    high=min(1.02,max(float(np.nanmax(x)) for x in all_cosines)+.06)
    for size in ["14M","70M"]:
        group=[r for r in records if r["size"]==size]
        fig,axes=plt.subplots(1,2,figsize=(11.8,4.2),layout="constrained")
        for rec in group:
            values=rec["combined"][:-1]
            x=np.arange(1,712)
            cos=rolling([r["adjacent_cosine"] if r["adjacent_cosine"] is not None else np.nan for r in values])
            loss=rolling([r["loss"] for r in values])
            variant=rec["variant"]
            for ax,y in zip(axes,[loss,cos]):
                selected=x>=357
                ax.plot(x[selected],y[selected],color=COLORS[variant],ls=STYLES[variant],lw=2,label=LABELS[variant])
        prefix=group[0]["combined"][:356]
        # Compute the shared pre-intervention trace without future branch values.
        for ax,y in zip(axes,[rolling([r["loss"] for r in prefix]),rolling([r["adjacent_cosine"] if r["adjacent_cosine"] is not None else np.nan for r in prefix])]):
            x=np.arange(1,357)
            selected=x>=300
            ax.plot(x[selected],y[selected],color="#777777",lw=1.5,label="Shared parent")
        axes[0].set(title="(a) Training loss",ylabel="Next-token cross entropy (31-update mean)")
        axes[1].set(title="(b) Raw-gradient alignment",ylabel="Adjacent cosine (31-update mean)",ylim=(low,high))
        axes[1].axhline(0,color="#666666",lw=.9,ls=":")
        axes[1].legend(frameon=True,facecolor="white",edgecolor="none",framealpha=.95,fontsize=9)
        for ax in axes:
            ax.axvline(356,color="#555555",ls=":",lw=1)
            ax.axvspan(300,356,color="#777777",alpha=.08)
            ax.set(xlim=(300,711),xlabel="Global optimizer update / gradient boundary")
            ax.grid(alpha=.17)
        fig.suptitle(f"Pythia-{size}: schedule intervention from the same update-356 checkpoint\nSeed 1235; centered full windows; final partial update omitted from curves",fontsize=12)
        figures.append(save_pdf(fig,"loss_and_raw_cosine_"+size.lower()))
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.2),layout="constrained")
    for variant in ["original","hold","accelerated"]:
        rec=next(r for r in records if r["variant"]==variant)
        axes[0].plot([r["update"] for r in rec["rows"]],[r["learning_rate"]*1e3 for r in rec["rows"]],
            color=COLORS[variant],ls=STYLES[variant],lw=2,label=LABELS[variant])
    axes[0].set(title="(a) Intervention applied after gradient capture",ylabel="Current optimizer LR (units of 0.001)",xlabel="Global optimizer update",xlim=(356,712))
    axes[0].legend(frameon=False,fontsize=9)
    for rec in records:
        variant=rec["variant"]
        axes[1].plot([v["update"] for v in rec["validation"]],[v["loss"] for v in rec["validation"]],
            color=COLORS[variant],ls=STYLES[variant],marker="o" if rec["size"]=="14M" else "s",lw=1.7,
            label=rec["size"]+": "+LABELS[variant])
    axes[1].set(title="(b) Complete validation coverage",ylabel="Token-weighted next-token cross entropy",xlabel="Global optimizer update",xlim=(350,718))
    axes[1].legend(frameon=False,fontsize=8,ncol=2)
    for ax in axes:ax.grid(alpha=.17)
    fig.suptitle("Single-seed schedule comparison: same checkpoint within each model size\nValidation: all 500 MiniPile documents / 693,329 supervised tokens, including tail",fontsize=12)
    figures.append(save_pdf(fig,"schedule_and_validation"))
    with (HERE/"curve_data.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    result={"hypothesis":"H004","scope":"Exploratory single-seed intervention conditional on a fixed trained prefix",
        "seed":1235,"source":source,"script_sha256":sha(__file__),
        "primary_pairs":[358,712],"smoothing":"Centered equal-update mean31; full windows; curves exclude partial update712",
        "statistics":"Unsmoothed cosines; equal pair weights. Aggregate training and validation loss are target-token weighted",
        "recovery_contract":"First31 consecutive strictly positive full centered means using only pairs358-711; retrospective and nonpermanent; widths15/31/61",
        "records":[{k:v for k,v in rec.items() if k not in ["rows","combined"]} for rec in records],
        "contrasts":contrasts,"figures":figures,
        "verification":{"all_batches_matched":True,"all_schedules_match_formulas":True,"first_gradient_and_validation_matched":True},
        "limits":["One seed, no cross-seed uncertainty or p-values","GPU numerical variation; control is a new replay",
            "LR also changes AdamW shrinkage","Alignment sign does not establish convergence, stability or loss mediation"]}
    (HERE/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(contrasts,indent=2))


if __name__=="__main__":main()
