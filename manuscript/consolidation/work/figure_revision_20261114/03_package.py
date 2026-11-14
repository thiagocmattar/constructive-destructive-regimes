"""Register revised views and refresh their metadata; package_evidence does copies."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=ROOT/"manuscript/consolidation"
CATALOG=HERE.parent/"evidence_catalog.json"
GENERATOR=(HERE/"01_figures.py").relative_to(ROOT).as_posix()
OBS=(HERE/"OBSERVATIONS.md").relative_to(ROOT).as_posix()


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")


def main():
    catalog=json.loads(CATALOG.read_text(encoding="utf-8"));catalog["date"]="2026-11-14"
    notes={"gd_loss_projected_change_spectral_clean.pdf":"Revised: completed-step clock; both curvature curves use the same centered 101-step smoother. Sparse spectrum interpolation is explicit in the caption.",
        "gd_reversal_norm_retention.pdf":"Revised state-indexed gradient coordinates; original trailing means and cumulative R values preserved.",
        "gd_backward_compact.pdf":"Revised state-indexed notation; backward lag remains a separation axis, with the same endpoint and values.",
        "sgd_batch_size_compact.pdf":"Revised state-indexed notation; original batch axis, means and initialization ranges preserved.",
        "00-high-lr-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf":"Compact revised overview: Q axes start at zero; unchanged data/event intervals. Gray-band meanings and negative values outside the viewport are explicit in the caption."}
    for a in catalog["assets"]:
        name=Path(a["destination"]).name
        if name in notes and "/figures/" in a["destination"]:
            a.update(source=(HERE/"figures"/name).relative_to(ROOT).as_posix(),generator=GENERATOR,observation=OBS,
                     role="prepared main",note=notes[name])
        if a["section"]=="05-adam/b-tracing":
            if name in ["01-resnet-instability-trace.pdf","02-transformer-instability-trace.pdf","04-mlp-instability-trace-450-500.pdf"]:
                a.update(destination="05-adam/b-tracing/background/figures/"+name,title="Historical detailed trace: "+name,
                         role="background",note="Complete original all-stage report, preserved unchanged. The selected three-model loss/motion and temporal-structure views are now assembled separately.")
            elif a["destination"].startswith("05-adam/b-tracing/report/"):
                a["note"]="Original technical report and numbering preserved. The selected Figure 6 is the separate three-model loss/motion overview; spectra use its companion figure."
                if a["role"]=="assembly component":a["role"]="historical support"
        if a["destination"]=="04-gd-sgd/data/figure1-layout-manifest.json":
            a.update(title="Historical Figure 1 layout manifest",role="historical source record",note="Authenticates the pre-revision source. Current views use the 14 November figure-revision manifest and captions.")
    catalog["sections"]["04-gd-sgd"]["presentation"]="Figures 1-4 refreshed on 14 November. Training progress is Training step s; lag and batch axes retain their own units. [Current captions and clock conversions](../FIGURE_NOTES.md)."
    catalog["sections"]["05-adam/a-characterization"]["presentation"]="Compact Figure 5 refreshed on 14 November. Training step s; projected-change axes start at zero, with gray bands and omitted negative display range explained in the caption."
    catalog["sections"]["05-adam/b-tracing"]["presentation"]="Figure 6 is assembled: three model columns, loss/cosine/RMS rows, 5/6/6 stage endpoints and synchronized markers. SVD/overlap is a separate companion. Historical all-stage PDFs remain in background/figures."
    additions=[]
    for name,title,role,note in [
        ("00-three-model-loss-representation-motion.pdf","Figure 6: loss and stage-level representation motion","prepared main","Three model columns and three rows; completed-step alignment, fixed stage endpoints, common cosine scale, per-stage normalized RMS and declared reversal/loss markers."),
        ("01-three-model-temporal-structure.pdf","Tracing companion: spectra, stage overlap and output coefficients","prepared support","Original fixed-window fits preserved; SVD and temporal overlap separated from the main loss/motion display.")]:
        additions.append(dict(section="05-adam/b-tracing",source=(HERE/"figures"/name).relative_to(ROOT).as_posix(),
            destination="05-adam/b-tracing/figures/"+name,title=title,role=role,observation=OBS,generator=GENERATOR,note=note))
    additions += [
        dict(section="04-gd-sgd",source=OBS,destination="FIGURE_NOTES.md",title="Revised figure contracts, captions and timing observations",role="current figure record",observation=None,generator=None,note="Covers revised Figures 1-6 and the temporal-structure companion; includes every clock conversion, smoother, event marker and coverage limit."),
        dict(section="04-gd-sgd",source=(HERE/"captions.tex").relative_to(ROOT).as_posix(),destination="figure_captions.tex",title="Editable captions for seven revised figures",role="integration source",observation=OBS,generator=None,note="LaTeX caption macros; does not edit the active manuscript."),
        dict(section="04-gd-sgd",source=(HERE/"figure_manifest.json").relative_to(ROOT).as_posix(),destination="figure_revision_manifest.json",title="Current figure source and numerical manifest",role="source record",observation=OBS,generator=GENERATOR,note="Hashes sources, plotted coordinates, reductions, stage choices and completed-step markers."),
        dict(section="04-gd-sgd",source=(HERE/"verification.json").relative_to(ROOT).as_posix(),destination="figure_revision_verification.json",title="Revised figure verification",role="verification",observation=OBS,generator=(HERE/"02_verify.py").relative_to(ROOT).as_posix(),note="Independent numerical checks, PDF bounds/fonts, visual review and preservation of manuscript/human/intervention files.")]
    existing={a["destination"]:i for i,a in enumerate(catalog["assets"])}
    for a in additions:
        if a["destination"] in existing:catalog["assets"][existing[a["destination"]]]=a
        else:catalog["assets"].append(a)
    write(CATALOG,catalog)
    print(json.dumps({"assets":len(catalog["assets"]),"unique_sources":len({a["source"] for a in catalog["assets"]})}))


if __name__=="__main__":main()
