"""Independent numerical, PDF, and preservation checks for the revised figures."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import fitz
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
EVIDENCE=ROOT/"manuscript/consolidation/evidence"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ah(x):return hashlib.sha256(np.asarray(x,dtype="<f8").tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def main(visual_reviewed=False):
    m=read(HERE/"figure_manifest.json")
    assert sha(HERE/"01_figures.py")==m["generator_sha256"]
    for p,digest in m["sources"].items():assert sha(ROOT/p)==digest,p
    assert sha(HERE/"plotted_arrays.npz")==m["plotted_arrays_sha256"]
    plots=np.load(HERE/"plotted_arrays.npz")
    curves={c["tag"]:c for c in m["curves"]}
    for tag,c in curves.items():
        assert ah(plots[tag+":x"])==c["x_sha256"] and ah(plots[tag+":y"])==c["y_sha256"]
    base=ROOT/"analyses/024-2026-10-17-eos-gd-lr-calibration"
    old=np.load(ROOT/read(base/"single_rows_figure_manifest.json")["arrays"])
    spectrum=np.load(ROOT/read(base/"mlp_learning_rates_figure_manifest.json")["arrays"])
    cohort=read(base/"cohort.json")["runs"]
    for r in cohort:
        rid=r["run_id"]
        # State s -> theta_s, unlike the legacy one-based gradient index.
        for metric in ["rho_smoothed","gradient_norms_smoothed","normalized_displacement"]:
            tag=f"{metric}_run{rid}";new="GD:"+tag
            np.testing.assert_array_equal(plots[new+":y"],old[tag+"_y"])
            np.testing.assert_array_equal(plots[new+":x"],old[tag+"_x"]-(0 if metric=="normalized_displacement" else 1))
        np.testing.assert_array_equal(plots[f"GD:loss:{rid}:y"],old[f"loss_log_run{rid}_y"])
        p=ROOT/"runs"/r["run"]/"artifacts/attempt-01-spark/production/metrics.jsonl"
        raw=[json.loads(line) for line in p.read_text().splitlines()[:3000]]
        q=np.array([v.get("eta_kappa_proxy",v.get("eta_kappa_gradient_proxy")) for v in raw[1:]],dtype=float)
        x=np.arange(1,3000)
        lam=np.interp(x,spectrum[f"run{rid}_row1_x"],spectrum[f"run{rid}_row1_y"])
        for name,y in [("Q",q),("spectrum",lam)]:
            # Direct means of the recorded proxy (not its norm/cosine reconstruction).
            expected=np.array([y[max(0,i-50):min(len(y),i+51)].mean() for i in range(len(y))])
            np.testing.assert_allclose(plots[f"GD:{name}:centered:{rid}:y"],expected,rtol=2e-10,atol=2e-10)
            np.testing.assert_array_equal(plots[f"GD:{name}:centered:{rid}:x"],x)
    old_adam=read(ROOT/"analyses/055-2026-11-05-adam-directional-amplification/plot_manifest.json")
    for case in ["mlp","resnet","transformer"]:
        for row,key in enumerate(["loss","Q","rho","relative_norm"]):
            original=next(c for c in old_adam["curves"] if c["figure"]=="adam_full_trajectories_high.pdf" and c["arm"].startswith(case+":") and c["key"]==key)
            assert curves[f"Adam:{case}:{row}"]["x_sha256"]==original["x_sha256"]
            assert curves[f"Adam:{case}:{row}"]["y_sha256"]==original["y_sha256"]
    expected_events={"mlp":(451,465,466,470,474,True),"resnet":(281,286,287,289,293,True),"transformer":(556,557,558,561,562,False)}
    for c in m["tracing"]:
        case=c["case"]
        assert tuple(c[k] for k in ["first_observed_output_reversal_step","baseline_step","first_rise_step","loss_threshold_step","peak_step","reversal_left_censored"])==expected_events[case]
        r=read(ROOT/"analyses/063-2026-11-07-architecture-instability-tracing"/(("mlp_450_500" if case=="mlp" else case)+"_results.json"))
        states=np.array(r["states"])+1
        np.testing.assert_array_equal(plots[f"trace:{case}:full-loss:x"],states)
        boundaries=read(ROOT/"analyses/063-2026-11-07-architecture-instability-tracing/mlp_results.json")["boundaries"] if case=="mlp" else r["boundaries"]
        lookup={v["state"]:v for v in boundaries}
        loss=np.array([lookup[int(s)]["loss"] for s in states])
        np.testing.assert_array_equal(plots[f"trace:{case}:full-loss:y"],loss)
        base_loss=lookup[c["baseline_step"]]["loss"]
        assert lookup[c["loss_threshold_step"]-1]["loss"]<1.1*base_loss<=lookup[c["loss_threshold_step"]]["loss"]
        rho=np.array([r["curves"][s["source"]]["rho"] for s in c["stages"]])
        rms=np.array([r["curves"][s["source"]]["change_rms"] for s in c["stages"]])
        assert ah(rho)==c["rho_sha256"] and ah(rms)==c["rms_sha256"]
        np.testing.assert_array_equal(rms.max(axis=1),c["rms_window_maxima"])
        output=rho[-1];negative=np.flatnonzero(output<0)[0]
        assert states[negative]==c["first_observed_output_reversal_step"]
        assert c["reversal_left_censored"]==(negative==0)
        if negative>0:assert output[negative-1]>=0
    pdfs=[]
    for record in m["figures"]:
        path=HERE/"figures"/record["file"];assert sha(path)==record["sha256"]
        with fitz.open(path) as doc:
            assert len(doc)==1
            page=doc[0];text=page.get_text()
            assert not any(v in text for v in ["Parameter state","Gradient index","Update index","50-pair mean","Rise of recovered","Recovery unobserved"])
            if record["file"] not in ["gd_backward_compact.pdf","sgd_batch_size_compact.pdf"]:assert "Training step" in text
            if record["file"]=="gd_loss_projected_change_spectral_clean.pdf":assert text.count("smoothed")==2
            for font in page.get_fonts(full=True):assert doc.extract_font(font[0])[3]
            for w in page.get_text("words"):
                assert -.1<=w[0]<w[2]<=page.rect.width+.1 and -.1<=w[1]<w[3]<=page.rect.height+.1,(path.name,w)
            page.get_pixmap(matrix=fitz.Matrix(.4,.4))
            pdfs.append(dict(file=record["file"],sha256=sha(path),pages=1,embedded_fonts=True,text_inside_page=True))
    preservation=read(ROOT/"manuscript/consolidation/provenance.json")
    for kind,key in [("human_inputs","preserved_path"),("active_manuscript_snapshot","path")]:
        for v in preservation[kind]:assert sha(ROOT/v[key])==v["sha256"],v[key]
    intervention_pdfs=[]
    for p in (EVIDENCE/"05-adam/c-intervention").rglob("*.pdf"):
        rel=p.relative_to(ROOT).as_posix()
        original=subprocess.check_output(["git","show","HEAD:"+rel],cwd=ROOT)
        assert p.read_bytes()==original,rel
        intervention_pdfs.append(rel)
    result=dict(date="2026-11-14",status="passed" if visual_reviewed else "numerical_and_pdf_checks_passed_visual_pending",
        source_hashes=len(m["sources"]),curve_arrays=len(curves),plotted_values=sum(c["points"] for c in m["curves"]),
        heatmap_values=sum(c["heatmap_values"] for c in m["tracing"]),pdfs=pdfs,
        unchanged_manuscript_files=len(preservation["active_manuscript_snapshot"]),unchanged_human_files=len(preservation["human_inputs"]),
        unchanged_intervention_pdfs=len(intervention_pdfs),
        visual_review="All seven final Poppler renders inspected: labels, stage reductions, synchronized markers, captions, color scales, legends, whitespace and clipping." if visual_reviewed else "pending")
    (HERE/"verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:v for k,v in result.items() if k!="pdfs"},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--visual-reviewed",action="store_true");a=p.parse_args();main(a.visual_reviewed)
