"""Record terminal measured outcomes without inferring stability from geometry."""
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    data = json.loads((HERE / "comparison_data.json").read_text())
    assert data["scope"] != "rendering_smoke_only"
    rows = []
    for rid, record in data["runs"].items():
        series = record["series"]
        n = record["accepted_updates"]
        summary = record["summary"]
        row = dict(run_id=rid,peak_lr=record["lr"],status=record["status"],accepted_updates=n,
                   final_validation_loss=summary["final_validation"]["loss"] if summary else None)
        if n:
            full = min(n,711)
            start = max(0,full-100)
            row["last100_available_full_training_loss"] = float(np.average(series["loss"][start:full],weights=record["supervised_tokens"][start:full]))
        else:
            row["last100_available_full_training_loss"] = None
        for stream in ("raw","momentum","adaptive"):
            cosine = [v for v in series[stream+"_cosine"] if v is not None]
            row[stream+"_negative_adjacent_fraction"] = sum(v<0 for v in cosine)/len(cosine) if cosine else None
            row[stream+"_last_norm"] = series[stream+"_norm"][-1] if n else None
            row[stream+"_last_A"] = series[stream+"_A"][-1] if n else None
            row[stream+"_last_B"] = series[stream+"_B"][-1] if n else None
            row[stream+"_last_B_over_A"] = series[stream+"_B_over_A"][-1] if n else None
        rows.append(row)
    with (HERE / "summary.csv").open("w",newline="",encoding="utf-8") as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def number(value):
        return "not available" if value is None else f"{value:.6g}"
    table = "| Run | LR | Status | Updates | Final validation loss | Last100 available full-update loss | Raw negative pairs | Momentum negative pairs |\n| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |\n"
    for row in rows:
        table += f"| {row['run_id']} | {row['peak_lr']:g} | {row['status']} | {row['accepted_updates']} | {number(row['final_validation_loss'])} | {number(row['last100_available_full_training_loss'])} | {number(row['raw_negative_adjacent_fraction'])} | {number(row['momentum_negative_adjacent_fraction'])} |\n"
    text = f"""# Higher learning rates on DGX Spark: recorded outcomes

Automatic terminal reduction generated {data['generated_at']}. No replacement
training settings or lower-LR retries were introduced. Failed/missing conditions
remain explicitly labeled; prefix endpoints are not full-pass results.

{table}
Training loss in the figures is unsmoothed. The table's loss is token weighted
over the last up-to100 available full-sized updates, excluding partial update712.
Negative-pair fractions use all defined recorded adjacent cosines, including
the partial endpoint when present. They are descriptive fractions, not regimes
or convergence thresholds. Full-precision outcomes: [summary.csv](summary.csv).

[Compact loss and norms](figures/loss_and_norms_by_learning_rate.pdf).
[Full training dynamics](figures/training_dynamics.pdf).
[Sources and exact plotted values](comparison_data.json).
[Automated figure verification](figure_verification.json).
[Caption, reductions and limitations](observations/001-high-learning-rates.md).

Remote full-pass verification reconstructs all three streaming buffers from
the final recovery and checks actual stored momentum against optimizer exp_avg,
complete training/validation coverage and schedule. Local download receipts
separately verify byte sizes and SHA-256 before declaring retrieval complete.
Render/text/bounds checks are automatic; final human/model visual review remains
pending unless separately recorded. One seed1234, matched initialization/order,
one finite pass and LR-scaled weight decay provide no universal stability,
generalization or causal alignment claim.
"""
    (HERE / "RESULTS.md").write_text(text,encoding="utf-8")
    readme = HERE / "README.md"
    if readme.exists():
        contents = readme.read_text(encoding="utf-8")
        contents = contents.replace("Status: preparing the user-authorized full-pass queue.",
            "Status: terminal remote outcomes recorded; [results and figures](RESULTS.md) are ready. Local byte retrieval and final visual review are recorded separately.",1)
        readme.write_text(contents,encoding="utf-8")
    observation = """# Observation001: larger constant learning rates on matched Spark trajectories

Source runs: completed Run023 (LR0.01), newly authorized Runs053/054
(LR0.02/0.04). Frozen initialization, seed1234 token order, cache hashes,
architecture, optimizer/clipping/batch/precision/validation/retention match.
Only peak LR varies; LR-scaled AdamW decay displacement also changes.
Sources and generator hashes are pinned in source_inventory.json and
comparison_data.json; owning raw events and all recovery states are retained.

Caption: Pythia-14M higher-LR stress comparison on DGX Spark. The compact
4-panel figure puts unsmoothed training loss first and then one raw/momentum
norm panel per LR. The full12-panel figure adds raw/momentum/Adam adjacent
cosines, raw/momentum cumulative A and B, and all three B/A curves.
All completed trajectories show712 accepted updates, including zero-LR
update1 and the marked partial final batch at712 (311 versus1024 sequences).
Failed prefixes show only accepted updates and are labeled as failed.

Loss is the recorded supervised-token-mean cross-entropy, nats; no loss
smoothing. Norm panels share log Y, show raw originals at opacity0.20 and
a trailing10-update arithmetic mean above them, with fill (opacity0.14) to
unsmoothed momentum. Average norm units before log display; prefix windows
use available values. Nonpositive norms are counted and omitted from log
display without flooring. Fill denotes a magnitude gap, not uncertainty.
Adjacent cosines show faint originals and arithmetic means: raw50 updates,
momentum/Adam25. Available finite prefixes preserve the undefined first cosine.
Cumulative panels are unsmoothed and linear: solid A=sum||x|| and dashed
B=||sum x||, with B/A in[0,1]. No time thinning or loss-value filtering.

Raw g is whole-model accumulated/unscaled pre-clipping gradient; m is actual
uncorrected FP32 Adam exp_avg based on clipped g; d is actual-moment
bias-corrected/preconditioned direction excluding LR, decay and subtraction
rounding. All reductions/prefixes are FP64. g/m/d are different streams and
none is automatically the realized parameter displacement.

Every completed pass covers all1,491,711,416 input tokens and1,490,983,041
supervised targets. Both validation passes cover339 sequences/693,329 targets
and the short tail. Summary last100 loss is token weighted over full-sized
updates only; endpoint/negative-pair statistics declare their coverage.
No cross-seed uncertainty is estimated. Clipping/EMA, one seed/order/start,
finite budget and LR-scaled decay limit interpretation. Sign or norm growth
alone establishes neither stability nor generalization or a causal noise effect.

Generator: [03_plot.py](../03_plot.py); reduction: [04_record_results.py](../04_record_results.py).
[Data, source hashes and smoothing](../comparison_data.json),
[verification and pending visual review](../figure_verification.json),
[recorded outcomes](../RESULTS.md), [full-precision table](../summary.csv).
"""
    (HERE / "observations").mkdir(exist_ok=True)
    (HERE / "observations/001-high-learning-rates.md").write_text(observation,encoding="utf-8")
    (HERE / "observations/INDEX.md").write_text("- [001: Higher learning rates](001-high-learning-rates.md)\n",encoding="utf-8")
    for item in json.loads((HERE / "cohort.json").read_text())["runs"]:
        run = ROOT / "runs" / item["run"]
        record = data["runs"][item["run_id"]]
        note = f"\n## Recorded outcome\n\nStatus: {record['status']}; accepted updates: {record['accepted_updates']}/712.\n[Comparison and measured outcomes](../../analyses/{HERE.name}/RESULTS.md).\nRemote verification and local byte-retrieval receipts are separate; final visual\nreview is declared in the figure verification record. Scientific inputs preserved.\n"
        readme = run / "README.md"
        if readme.exists() and "## Recorded outcome" not in readme.read_text():
            contents = readme.read_text(encoding="utf-8").replace("Status: prepared;",f"Status: {record['status']};",1)
            readme.write_text(contents+note,encoding="utf-8")
        else:
            readme.write_text(f"# Run {item['run_id']}: LR {item['peak_lr']:g}\n"+note,encoding="utf-8")
        (run / "observations").mkdir(exist_ok=True)
        (run / "observations/001-high-lr.md").write_text(f"# Observation001: LR {item['peak_lr']:g}\n\n"+note+
            f"\n[Shared source/reduction/caption/coverage/limits](../../../analyses/{HERE.name}/observations/001-high-learning-rates.md).\n",encoding="utf-8")
        (run / "observations/INDEX.md").write_text("- [001: High-LR dynamics](001-high-lr.md)\n",encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
