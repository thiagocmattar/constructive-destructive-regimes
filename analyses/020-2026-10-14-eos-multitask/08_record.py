"""Close the verified first control, without claiming pending training results."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import write_json,file_hash
HERE=Path(__file__).parent


def main():
    record=json.loads((HERE/"frozen_figure_manifest.json").read_text())
    assert record["status"]=="passed"
    result=json.loads((HERE/"frozen_results.json").read_text())
    original=ROOT/"runs/068-2026-10-14-eos-frozen-gradient-noise/artifacts/attempt-01-spark/production"
    manifest=json.loads((original/"manifest.json").read_text())
    assert manifest["status"]=="complete"
    rows=result["conditions"]
    minimum=[min([r for r in rows if r["task"]==task],key=lambda r:r["batch_units"]) for task in ["mlp","resnet","transformer"]]
    ordered={task:sorted([r for r in rows if r["task"]==task],key=lambda r:r["batch_units"])
        for task in ["mlp","resnet","transformer"]}
    monotone=all(all(b["B_over_A_mean"]>=a["B_over_A_mean"] and b["cosine_mean"]>=a["cosine_mean"]
        for a,b in zip(group[:-1],group[1:])) for group in ordered.values())
    directional_statement=("Greater measured noise reduces the frozen positive alignment and accumulation in these conditions."
        if monotone else "The seed-mean ordering is mixed; the figure and per-seed reductions retain these counterexamples.")
    table="| Case | Smallest batch | Relative noise RMS | Mean adjacent cosine | Mean B/A |\n| --- | ---: | ---: | ---: | ---: |\n"
    for r in minimum:
        size=r["batch_units"]*1000 if r["task"]=="resnet" else r["batch_units"]
        units="images" if r["task"]!="transformer" else "sequences"
        table+=f"| {r['task']} | {size} {units} | {r['relative_noise_rms_mean']:.3f} | {r['cosine_mean']:.3f} | {r['B_over_A_mean']:.3f} |\n"
    calibration=json.loads((ROOT/"runs/067-2026-10-14-eos-multitask-calibration/calibration_results.json").read_text())
    staged=json.loads((HERE/"staged-500-plan.json").read_text())
    longer=json.loads((HERE/"paper-length-plan.json").read_text())
    actual=manifest["elapsed_seconds"]/60
    launched=(HERE/"training_launch.json").exists()
    training_status=("The500-update first GD/SGD stage is launched and running; its results remain pending. "
        "The longer paper-length horizon remains a subsequent decision." if launched else
        "The500-update first GD/SGD stage is prepared for launch; its results remain pending.")
    report=f"""# Analysis020: first verified results and production execution budget

Status: source review, mathematical verification, all three production-shaped
smokes and owned-Spark calibration complete. Run068 frozen-state control is
complete, retrieved/hash verified, numerically checked and visually reviewed.
{training_status} These control results are not a completed EOS/TNNLS reproduction.

## What the original paper tests

The three user-selected cases use two real datasets: CIFAR-10 with a tanh MLP,
CIFAR-10 with ResNet-32, and WikiText-2 with a small word-level Transformer.
The paper also tests VGG-11 and synthetic tasks; it does not test ViT.
[Exact practical setup and source qualifications](SOURCE_REVIEW.md).
ResNet's effective full update covers50,000 images, accumulated over50 fixed
ghost groups of1,000. Its Hessian uses the first5,000 images. The MLP uses5,000
images; the complete prepared language objective covers2,088,600 supervised tokens.
All tail/context exclusions inherited from tutorial stream preprocessing are
declared inRun067 inputs. No small training subset replaces the language corpus.

## Real-data gradient estimation noise

Run068 took{actual:.2f} minutes on owned Spark; rental cost zero. Three initializations
per case,32 independent128-draw sequences per batch condition,36 conditions and
147,456 recorded vector observations. At each fixed initial parameter vector,
noise is the actual sampled batch-gradient error against the complete gradient.
ResNet samples whole ghost groups to preserve its objective. There are no
parameter updates, artificial Gaussian perturbations or invented loss functions.
Normalized B/A in this experiment is gradient accumulation, not actual motion.

{table}
These values average seeds0,1,2 and32 repetitions within each seed. The publication
figure spans the range across initializations; that range is not a95% confidence
interval. Repeated sampling quantifies conditional Monte Carlo uncertainty, not
variation over datasets or a universal training law. {directional_statement}
the smallest available ResNet ghost batch retains substantial alignment.
Approaching zero requires sufficiently large noise relative to the signal.
The gradient-only projected-change statistic can be nonzero without parameter
motion. Its measured noise correction removes that contribution up to roundoff.

## Training plan and ETC

The [protocol](PROTOCOL.md) pairs three seeds and frozen initial tensors within
each case. Plain GD/SGD, constant LR, no decay/clipping/momentum/schedule/augmentation.
Rates0.01/0.2/0.1 for MLP/ResNet/Transformer; three stochastic batch sizes per
case, plus full-batch controls. Every raw adjacent pair and full-prefix norm,
energy, interaction and B/A is logged. Complete-gradient paired probes provide
the actual noise correction; vector covariance estimates include population
coverage. Ordinary largest-algebraic Hessian eigenvalues have independent
residual checks. Sparse reference/Hessian measurements are explicitly labeled.

The suite has35 new training runs plus one verified existing seed0 MLP control.
The current [500-update plan](staged-500-plan.json) estimates
{staged['estimated_hours']:.1f} hours, with a{staged['qualified_hours']:.1f}-hour allowance
and72-hour hard stop. It is a first stage, not full paper-length evidence.
The [longer plan](paper-length-plan.json) uses20k/2k/15k updates for the respective
cases and estimates{longer['estimated_hours']/24:.1f} days, qualified to
{longer['qualified_hours']/24:.1f} days. Its21-day hard cap does not cover the entire
uncertainty allowance; a longer occupation budget would need a concrete choice.
Both keep complete datasets, bounded detached processes, persistent progress,
initial/intermediate/recovery/final states and a40GB retained-storage allowance.
No rental or cloud service is authorized/used. Early calibration does not measure
late eigensolver clustering, validation overhead drift or other users' device load.

## Paper artifacts and limits

[Publication control figure](figures/eos_frozen_noise_comparison.pdf),
[figure/array manifest](frozen_figure_manifest.json),
[observation](observations/001-frozen-real-data-noise.md).
The formulation report moves the noise discussion after normalized displacement,
adds finite-population noise variance and the fixed-parameter accumulation limit,
and includes this control with its sampling/trajectory distinction. Existing
GD and Rosenbrock evidence remains unchanged. Training loss, directional-change,
sharpness and displacement comparisons across all three cases remain pending.

Three seeds at one prespecified LR per case are a replicated foundation; they
do not guarantee journal readiness. Claims must remain conditional on the tested
objectives, batch semantics, precision and horizons. Batch-size sweeps change
data exposure at a fixed update budget. Frozen controls isolate noise; moving
trajectories may reverse the ordering because signal and curvature change.
ReLU/BatchNorm cases also require care interpreting a finite gradient difference
as local Hessian curvature. No generalization claim follows from these diagnostics.
"""
    (HERE/"REPORT.md").write_text(report,encoding="utf-8")
    observations=HERE/"observations";observations.mkdir(exist_ok=True)
    (observations/"INDEX.md").write_text("# Observations\n\n- [001: Frozen real-data noise control](001-frozen-real-data-noise.md)\n",encoding="utf-8")
    (observations/"001-frozen-real-data-noise.md").write_text("""# Observation001: real-data frozen-gradient sampling

Sources: immutable Run068 config/input manifest, complete original metrics and
batch indices inits verified Spark attempt, byte/SHA256 retrieval receipt;
Analysis020 frozen_results.json and frozen_figure_manifest.json.
Generating script:06_plot_frozen.py; original experiment:Run06801_experiment.py.

Reduction: each128-draw sequence contributes its final complete-prefix B/A,
127-pair mean cosine and mean projected change. Average32 independent sequences
within each seed, then average three initializations. Horizontal positions are
the RMS sampled-gradient error divided by the complete-gradient norm. All plots
use observed reductions without smoothing. Shading is the range across seed
means; it is not a confidence interval. Conditional noise-energy Monte Carlo SE
is retained ineach original condition summary.

Coverage:36 conditions,3 architectures,2 real datasets,3 initializations,
32 sequences each,128 draws,0 optimizer updates. Full-objective references cover
5k CIFAR images forMLP,50k forResNet, and all prepared WikiText-2 training tokens.
ResNet batches contain whole fixed1k ghost groups. Their gradients are averaged
inFP64 from frozenFP32 unit gradients; no BN group context changes. All parameter
hashes remain unchanged. Each adjacent noise-corrected change is zero up to the
recorded roundoff; full batch has cosine/B/A one.

Caption: Real-data gradient-noise control at frozen initial parameters. Columns
compare CIFAR-10/MLP, CIFAR-10/ResNet-32 andWikiText-2/Transformer; rows show
adjacent cosine,128-draw normalized accumulation and gradient-only projected
change with its zero noise-corrected reference. Measured noise RMS is normalized
by the complete-gradient signal. Seed means and ranges are shown.

Interpretation: the seed-mean ordering is recorded quantitatively inREPORT.md.
Noise does not universally make cosine zero:
the initial ResNet signal remains strong relative to its minimum1k-group noise.
The projected gradient-change observable is noise contaminated even without
parameter motion. This tests a conditional mechanism; it is not evidence about
moving SGD trajectories, EOS entry, loss decrease or generalization.
"""+"\n"+table,encoding="utf-8")
    write_json(HERE/"first_stage_verification.json",{"status":"passed","completed":"calibration and frozen sampling control",
        "training_status":"staged500 launched" if launched else "staged500 prepared; not launched","figure_manifest_sha256":file_hash(HERE/"frozen_figure_manifest.json"),
        "source_results_sha256":record["source_results_sha256"],"original_completed_manifest_sha256":file_hash(original/"manifest.json")})
    print("Recorded verified frozen control; production training remains explicitly pending.")


if __name__=="__main__":main()
