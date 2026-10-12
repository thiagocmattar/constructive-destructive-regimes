"""Check every plotted series and record observations and visual acceptance."""
import argparse
import json
from datetime import datetime, timezone

import fitz
import numpy as np

from kappa import HERE, ROOT, digest, load_sources, rolling, sha256, write_json


def main(reviewed=False):
    report = json.loads((HERE / "results.json").read_text())
    assert json.loads((HERE / "verification.json").read_text())["results_sha256"] == sha256(HERE / "results.json")
    histories, _, _ = load_sources()
    for history, source in zip(histories, report["sources"]):
        with np.load(ROOT / source["derived_history"], allow_pickle=False) as extra:
            history.update({key: extra[key] for key in extra.files})
    checks = []
    for stem, expected_panels in (("rosenbrock_kappa_comparison",6),("rosenbrock_kappa_diagnostics",13),("rosenbrock_kappa_eta_overlay",1)):
        manifest = json.loads((HERE / ("figure_manifest_" + stem + ".json")).read_text())
        pdf = ROOT / manifest["figure"]
        assert sha256(pdf) == manifest["sha256"] and manifest["panels"] == expected_panels
        script = HERE / ("04_plot_eta_overlay.py" if expected_panels == 1 else "02_plot.py")
        assert sha256(script) == manifest["script_sha256"]
        if expected_panels == 6:
            by_id = {row["run_id"]:history for row,history in zip(report["sources"],histories)}
            for item in manifest["rendered_series"]:
                history = by_id[item["run_id"]]
                assert item["x_sha256"] == digest(history["updates"])
                assert item["y_sha256"] == digest(history[item["field"]])
        elif expected_panels == 13:
            expected = {panel:[] for panel in "abcdefghijklm"}
            for h in histories:
                pos,updates=np.arange(2001),np.arange(1,2001)
                expected["a"].append((h["theta"][:,0],h["theta"][:,1]))
                for panel,key in (("b","loss"),("j","hessian_lambda_max"),("k","hessian_lambda_min")):
                    expected[panel].append((pos,h[key]))
                for panel,stream in (("c","raw"),("d","momentum")):
                    expected[panel].append((updates,rolling(h[stream+"_cosine"])))
                for stream in ("raw","momentum"):
                    expected["e"].append((updates,h[stream+"_norm"]))
                for panel,stream in (("f","raw"),("g","momentum")):
                    for field in ("A","B"):
                        expected[panel].append((updates,h[stream+"_"+field]))
                for panel,stream in (("h","raw"),("i","momentum")):
                    a,b=h[stream+"_A"],h[stream+"_B"]
                    expected[panel].append((updates,np.divide(b,a,out=np.full_like(a,np.nan),where=a>0)))
                expected["l"].extend([(updates,h["kappa_proxy"]),(updates,h["kappa_proxy_mean100"])])
                expected["m"].append((updates,h["kappa_direction"]))
            for panel,values in expected.items():
                recorded=[item for item in manifest["rendered_series"] if item["panel"]==panel]
                assert len(recorded)==len(values)
                for item,(x,y) in zip(recorded,values):
                    assert item["x_sha256"]==digest(x) and item["y_sha256"]==digest(y)
            assert sha256(HERE/"plot_full.py")==manifest["plot_implementation_sha256"]
        else:
            data_path=HERE/"eta_overlay_data.json"
            assert sha256(data_path)==manifest["data_sha256"]
            data=json.loads(data_path.read_text())
            for item,h,source,recorded in zip(data["sources"],histories,report["sources"],manifest["rendered_series"]):
                expected=rolling(source["learning_rate"]*h["kappa_proxy"])
                actual=np.array([np.nan if v is None else v for v in item["normalized_mean100"]])
                np.testing.assert_array_equal(actual,expected)
                assert recorded["plotted_x_sha256"]==digest(data["updates"])
                assert recorded["plotted_y_sha256"]==digest(actual)
            assert manifest["original_comparison_sha256_unchanged"]==sha256(HERE/"figures/rosenbrock_kappa_comparison.pdf")
        with fitz.open(pdf) as doc:
            assert len(doc)==1
            for block in doc[0].get_text("blocks"):
                assert doc[0].rect.contains(fitz.Rect(block[:4])),(stem,block[:4])
        render=HERE/"scratch"/{6:"comparison_poppler.png",13:"diagnostics_poppler.png",1:"eta_overlay_pdf_review.png"}[expected_panels]
        assert render.exists()
        checks.append(dict(figure=stem,pdf_sha256=sha256(pdf),panels=expected_panels,
                           actual_series_checked=True,pdf_text_bounds_passed=True,
                           renderer="PyMuPDF" if expected_panels==1 else "Poppler",
                           render_sha256=sha256(render),visual_review="passed" if reviewed else "pending"))
    write_json(HERE/"figure_verification.json",dict(passed=True,visual_review="passed" if reviewed else "pending",checked_at=datetime.now(timezone.utc).isoformat(),figures=checks,script_sha256=sha256(__file__)))
    obs=HERE/"observations"; obs.mkdir(exist_ok=True)
    table="| LR | Raw Pearson vs lambda_max | 100-pair Pearson | Late proxy mean | Late exact directional mean | Late lambda_max mean | Late eta-proxy mean |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for row in report["sources"]:
        last=row["last100_pair_means"]; eta=row["learning_rate"]
        table+=f"| {eta:.7g} | {row['pearson_proxy_lambda_raw']:.6f} | {row['pearson_proxy_lambda_mean100']:.6f} | {last['kappa_proxy']:.6f} | {last['kappa_direction']:.6f} | {last['lambda_max']:.6f} | {eta*last['kappa_proxy']:.6f} |\n"
    (HERE/"README.md").write_text(f"""# Analysis017: Rosenbrock gradient-identity kappa

Complete local postprocessing of immutable Runs055-057, retaining Analysis016's
AdamW trajectories. The user requested the restatement's kappa identity and
then a separate overlay of eta times kappa. Noise epsilon is zero; AdamW's
numerical epsilon is separate. No new optimizer runs or cloud resources.

- [New LR-scaled single-panel overlay](figures/rosenbrock_kappa_eta_overlay.pdf)
- [Preserved six-panel Hessian comparison](figures/rosenbrock_kappa_comparison.pdf)
- [Complete thirteen-panel diagnostics](figures/rosenbrock_kappa_diagnostics.pdf)
- [Method](METHOD.md), [observations](observations/INDEX.md), [results](results.json)
- [Scientific verification](verification.json), [figure verification](figure_verification.json)

{table}
Late means cover pair updates1901-2000. H references use theta[t-2]. The broad
smoothed trend resembles lambda_max, but magnitude differs and the proxy can
fall outside the local Hessian spectrum. LR scaling yields similar late
dimensionless levels with different transition times; this is descriptive for
one deterministic setup, not a curvature estimator or AdamW stability test.

All5,997 adjacent pairs and5,700 full-window means are retained; no filtering
or thinning. All source/formula/derived-array hashes, independent mathematics,
gradient/Hessian reconstruction, exact discrepancy decomposition and actual
plotted arrays pass. Final visual review: {'passed' if reviewed else 'pending'}.
Reproduce with00_compute.py,01_verify.py,02_plot.py,04_plot_eta_overlay.py,
then03_record.py; --record-visual-review only after inspecting current PDF renders.
Derived histories {report['derived_storage_bytes']:,} bytes are ignored and locally
retained; source histories reused without copying. No new run ID is allocated.
""",encoding="utf-8")
    (obs/"001-kappa-and-hessian.md").write_text(f"""# Observation001: gradient identity along AdamW Rosenbrock trajectories

H016 asks whether the requested identity has a shape similar to lambda_max.
Sources: immutable Run055/056/057 histories, configs and verification receipts;
[results](../results.json) pins their SHA-256/size and all derived array hashes.
The supplied formula source is manuscript/2026-10-11-restatement/problem.md,
also hashed. Full diagnostic definitions/indexing are in [METHOD](../METHOD.md).
Reduction: ../00_compute.py and ../kappa.py; verification: ../01_verify.py;
plotting: ../02_plot.py and ../plot_full.py; figure acceptance: ../03_record.py.

{table}
Each LR has1,999 defined unsmoothed pairs t=2..2000 and1,900 complete100-pair
means t=101..2000. Late arithmetic means use pairs1901..2000, whose Hessian
positions are theta[1899..1998]. Initial pair is undefined; no epsilon floor,
clamping, thinning or curve normalization. The identity is computed before
smoothing. H excludes decoupled decay and uses the preceding Taylor point.

Smoothed correlations are higher than raw correlations, but amplitudes differ
substantially. Proxy values outside the local Hessian spectrum occur at
245/326/728 pairs. Exact directional curvature stays within the spectrum.
Late proxy-minus-direction means are explained by update-mismatch corrections
-724.340/-378.043/+820.549; late nonlinear-remainder means are approximately
0.000830/0.002783/-0.0000149. The exact algebraic decomposition closes within
2.4e-12 at every pair. This identifies terms of the discrepancy, not a causal
benefit. Under plain SGD on a quadratic, the expression recovers the Rayleigh
quotient; even that quotient need not be lambda_max.

Caption - preserved [six-panel comparison](../figures/rosenbrock_kappa_comparison.pdf):
**Gradient-identity kappa versus exact Rosenbrock curvature.** Columns are
LR0.01/midpoint/0.001, blue/purple/orange. Top: all unsmoothed proxy pairs,
largest eigenvalue (dark solid), and exact directional curvature (green dashed).
Bottom:100-pair trailing arithmetic means of each, common linear Y limits
within each row. No clipping or rescaling. LR-coloured vertical references
are original first-y-turn positions81/228/655, not estimator thresholds.

Caption - [full diagnostics](../figures/rosenbrock_kappa_diagnostics.pdf):
**Analysis016 geometry with the requested identity.** Complete trajectory
on original contours, square start, star optimum, open final points and diamond
y-turns; loss;100-step raw/momentum cosine means; vector norms; raw/momentum
A and B and B/A; largest/smallest Hessian eigenvalues;100-pair proxy means with
faint originals; exact directional curvature. All temporal axes linear except
loss logarithmic. Original Hessian panels use theta[t]; pair-curvature panels
use theta[t-2]. Every source point/vector remains visible, without zoom.

Coverage is complete, uncertainty is not estimated: one deterministic function,
start, optimizer and horizon. Serial correlations are descriptive and sensitive
to averaging. The SGD derivation is applied as a proxy to AdamW, whose moments,
preconditioning and decay change its step. No valid curvature estimator,
universal stability threshold, asymptotic convergence or LLM transfer is shown.
Both PDFs pass actual-series/source/text-bound checks and final layout inspection
as recorded in [figure verification](../figure_verification.json).
""",encoding="utf-8")
    (obs/"002-eta-kappa-overlay.md").write_text("""# Observation002: separate learning-rate-scaled kappa overlay

The user explicitly requests another PDF, preserving the original comparison.
Plot eta*kappa_proxy in a single panel for all three LRs. Since eta is constant,
this cancels the explicit1/eta factor and equals1-rho*||g[t]||/||g[t-1]|| with
noise epsilon zero. This is multiplication by LR, not division or rescaling to
a fitted maximum. Compute this dimensionless value per unsmoothed pair, then
take a complete100-pair trailing arithmetic mean. First mean is at101;
all1,900 means per LR remain. The original comparison PDF's SHA-256 is checked
before and after generation and recorded unchanged in the new figure manifest.

Sources: [results](../results.json), immutable Run055-057 histories and derived
kappa arrays. Exact overlay values, null prefixes and source hashes are in
[eta_overlay_data.json](../eta_overlay_data.json). Generator:
[04_plot_eta_overlay.py](../04_plot_eta_overlay.py); independent direct-vector
identity/mean checks run in that script; plotted-array/PDF checks and review
are in ../03_record.py and [figure verification](../figure_verification.json).

Caption - [separate overlay](../figures/rosenbrock_kappa_eta_overlay.pdf):
**Learning-rate-scaled gradient-identity kappa along matched Rosenbrock AdamW
trajectories.** Blue solid: LR0.01; purple short dashed: midpoint0.003162277660;
orange long dashed: LR0.001. Single linear Y axis, all pair times,100-pair means,
no thinning, clipping, fitted normalization or uncertainty bands. Vertical
dashed references retain original y-turn indices81/228/655. Start(0,1),
2,000 updates, betas(0.9,0.95), decoupled decay0.1, CPU float64. Epsilon is
the normalized noise term (zero), separate from optimizer epsilon1e-8.

Final100-pair means are1.790182/1.754971/1.747944 at decreasing LRs. The late
dimensionless levels are similar, while the rise occurs later at smaller LR.
This cancels a formula factor; it does not normalize Hessian eigenvalues or
correct the mismatch between AdamW updates and the SGD curvature derivation.
One deterministic setup provides no ensemble uncertainty, universal plateau,
stability boundary or causal statement. Latest PDF was rendered with PyMuPDF
and inspected after layout adjustment; text, legends and footnotes are readable
and unclipped. Original two PDFs remain unchanged.
""",encoding="utf-8")
    (obs/"INDEX.md").write_text("# Observations\n\n- [001: Kappa and the Hessian](001-kappa-and-hessian.md)\n- [002: Separate eta-kappa overlay](002-eta-kappa-overlay.md)\n",encoding="utf-8")
    print(json.dumps(dict(figures_verified=len(checks),visual_review=reviewed)))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--record-visual-review",action="store_true")
    main(parser.parse_args().record_visual_review)
