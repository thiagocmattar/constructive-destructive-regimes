"""Check the closing-section table against retained, hash-pinned reductions.

This reads completed analysis outputs and selected raw scalar histories.
It does not rerun neural training or replace complete checkpoint audits.
"""

from pathlib import Path
import hashlib
import json
import math
import os
import statistics


ROOT = Path(__file__).resolve().parents[4]
REVISION = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(disk_path(path).read_text(encoding="utf-8"))


def disk_path(path):
    return Path("\\\\?\\" + str(path.resolve())) if os.name == "nt" else path


def main():
    selection = read_json(REVISION / "evidence_selection.json")
    sources = selection["source_results"] + selection["protocol_sources"] + selection["raw_scalar_sources"]
    for source in sources:
        assert hashlib.sha256(disk_path(ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"], source["path"]
    by_id = {row["id"]: row for row in selection["recommended_table_rows"]}
    section = (ROOT / "manuscript/draft/sections/08-implications.tex").read_text(encoding="utf-8")
    table = (ROOT / "manuscript/draft/tables/submission_interventions.tex").read_text(encoding="utf-8")
    checks = []
    raw_rows = 0
    numeric_checks = 0

    def close(actual, expected):
        nonlocal numeric_checks
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)
        numeric_checks += 1

    def raw(cohort, condition, kind):
        nonlocal raw_rows
        entries = [s for s in selection["raw_scalar_sources"] if s["cohort"] == cohort and s["condition"] == condition and s["kind"] == kind]
        assert len(entries) == 1
        rows = [json.loads(line) for line in disk_path(ROOT / entries[0]["path"]).read_text().splitlines() if line]
        raw_rows += len(rows)
        return rows

    def contains(text, literal, reason):
        assert " ".join(literal.split()) in " ".join(text.split()), (reason, literal)
        checks.append(reason)

    schedule_results = read_json(ROOT / "analyses/004-2026-10-04-lr-schedule-intervention/results.json")
    for size in ("14M", "70M"):
        row = by_id[f"pythia_schedule_{size}"]
        values = row["values"]
        assert [v["variant"] for v in values] == ["original", "hold", "accelerated"]
        for selected in values:
            actual = next(r for r in schedule_results["records"] if r["run"] == selected["run"])
            events = raw("004", selected["run"], "events")
            pairs = [r for r in events if r["event"] == "update" and 358 <= r["update"] <= 712]
            assert [r["update"] for r in pairs] == list(range(358, 713))
            cosines = [r["adjacent_cosine"] for r in pairs]
            median = statistics.median(cosines)
            intensity = statistics.fmean(max(-v, 0) for v in cosines)
            frequency = sum(v < 0 for v in cosines) / len(cosines)
            validation = [r for r in events if r["event"] == "validation" and r["update"] == 712]
            assert len(validation) == 1 and validation[0]["supervised_tokens"] == 693329 and validation[0]["short_tail_included"]
            for key, source_key, reduced in [("median_raw_cosine", "median_raw_cosine", median), ("negative_intensity", "D_unconditional_negative_intensity", intensity), ("negative_fraction", "F_negative_frequency", frequency)]:
                close(selected[key], actual["primary"][source_key])
                close(selected[key], reduced)
            close(selected["validation_loss"], actual["terminal_validation_loss"])
            close(selected["validation_loss"], validation[0]["loss"])
        contains(table, "/".join(f"{v['median_raw_cosine']:.3f}" for v in values), size + " schedule cosines")
        contains(table, "/".join(f"{v['validation_loss']:.4f}" for v in values), size + " validation losses")
        assert values[1]["validation_loss"] < values[0]["validation_loss"] < values[2]["validation_loss"]

    ablation = read_json(ROOT / "analyses/025-2026-10-17-pythia-descent-mechanism/results.json")
    effects = ablation["branch_effects"]
    assert len(effects) == 18 and all(e["tau"] > 0 for e in effects)
    penalties = []
    for size in ("14M", "31M", "70M"):
        rows = [e for e in effects if e["size"] == size and e["fork"] == 225]
        assert len(rows) == 3
        for e in rows:
            assert math.isclose(e["tau"], math.log(e["ablated_final_probe_loss"] / e["native_final_probe_loss"]), abs_tol=1e-12)
        penalties.append(100 * math.expm1(statistics.mean(e["tau"] for e in rows)))
    contains(table, "/".join(f"{p:.2f}" for p in penalties) + r"\%", "three-seed geometric-mean probe penalties")

    transformer_source = read_json(ROOT / "analyses/043-2026-11-03-spark-fixed-step-interventions/results.json")
    transformer = [next(r for r in transformer_source["comparison"] if r["case"] == "transformer" and r["condition"] == condition) for condition in ("unchanged", "fixed_half")]
    for actual, selected in zip(transformer, by_id["transformer_half_late"]["values"]):
        rows = raw("043", actual["condition"], "boundaries")
        assert [r["state"] for r in rows] == list(range(1024, 1089))
        derived = {"mean_post_loss": statistics.fmean(r["loss"] for r in rows[1:]), "final_loss": rows[-1]["loss"], "mean_raw_rho": statistics.fmean(r["rho"] for r in rows[1:])}
        for key, value in derived.items():
            close(value, actual[key])
            close(value, selected[key])
    control, half = transformer
    for metric, delta_key in [("mean_post_loss", "mean_loss_change_vs_unchanged_percent"), ("final_loss", "final_loss_change_vs_unchanged_percent")]:
        reduced = 100 * (half[metric] / control[metric] - 1)
        close(reduced, half[delta_key])
        close(reduced, by_id["transformer_half_late"]["values"][1][delta_key])
    contains(table, f"{control['mean_raw_rho']:.3f}" + r"\to" + f"{half['mean_raw_rho']:+.3f}", "Transformer raw cosine response")
    contains(table, f"{half['mean_loss_change_vs_unchanged_percent']:+.2f}" + r"\%", "Transformer mean loss")
    contains(table, f"{half['final_loss_change_vs_unchanged_percent']:+.2f}" + r"\%", "Transformer final loss")

    mlp = by_id["mlp_clip1"]
    mlp_source = read_json(ROOT / "analyses/051-2026-11-04-mlp-global-clipping/results.json")
    for selected in mlp["values"]:
        condition = selected["condition"]
        actual = mlp_source["summary"][condition]
        rows = raw("051", condition, "boundaries")
        metrics = raw("051", condition, "metrics")
        assert [r["state"] for r in rows] == list(range(512, 801))
        assert [r["state"] for r in metrics] == list(range(512, 800))
        derived = {"mean_513_800": statistics.fmean(r["loss"] for r in rows[1:]), "last32_mean": statistics.fmean(r["loss"] for r in rows[-32:]), "final_loss": rows[-1]["loss"], "max_loss_513_800": max(r["loss"] for r in rows[1:]), "mean_actual_step_norm": statistics.fmean(r["actual_increment_norm"] for r in metrics), "max_actual_step_norm": max(r["actual_increment_norm"] for r in metrics), "max_optimizer_input_norm": max(r["optimizer_input_norm"] for r in metrics)}
        for key, value in derived.items():
            close(value, actual[key])
            close(value, selected[key])
        if condition == "global_clip1":
            assert sum(r["clip_factor"] < 1 for r in metrics) == mlp_source["clipping"]["clipped_updates"] == 125
            assert max(r["optimizer_input_norm"] for r in metrics) < 1 + 2e-7
    d = {key: 100 * (mlp_source["summary"]["global_clip1"][key] / mlp_source["summary"]["unchanged"][key] - 1) for key in ("mean_513_800", "last32_mean", "final_loss", "max_loss_513_800")}
    for key, value in d.items():
        close(value, mlp_source["clipping_vs_unchanged_pct"][key])
        close(value, mlp["changes_percent"][key])
    clip = mlp_source["clipping"]
    assert clip == mlp["clipping"]
    contains(table, f"{clip['clipped_updates']}/{clip['updates']}", "MLP clipping activation")
    contains(table, f"{d['mean_513_800']:+.2f}" + r"\%", "MLP mean loss")
    contains(table, f"{1 + d['max_loss_513_800'] / 100:.2f}" + r"\times", "MLP maximum ratio")
    contains(table, f"{d['final_loss']:+.2f}" + r"\%", "MLP endpoint loss")
    for key in ("mean_actual_step_norm", "max_actual_step_norm"):
        before, after = [mlp_source["summary"][condition][key] for condition in ("unchanged", "global_clip1")]
        contains(section, f"{before:.5f}\nto {after:.5f}" if key == "max_actual_step_norm" else f"{before:.5f} to {after:.5f}", "MLP " + key)

    resnet = by_id["resnet_clip1"]
    resnet_source = read_json(ROOT / "analyses/054-2026-11-05-resnet-clip1/results.json")
    for selected in resnet["values"]:
        condition = selected["condition"]
        actual = resnet_source["summary"][condition]
        rows = raw("054", condition, "boundaries")
        metrics = raw("054", condition, "metrics")
        assert [r["state"] for r in rows] == list(range(256, 577))
        assert [r["state"] for r in metrics] == list(range(256, 576))
        derived = {"mean_loss": statistics.fmean(r["loss"] for r in rows[1:]), "tail_mean": statistics.fmean(r["loss"] for r in rows[-32:]), "maximum": max(r["loss"] for r in rows[1:]), "endpoint": rows[-1]["loss"], "clipped": sum(r["clip_factor"] < 1 for r in metrics), "max_input": max(r["optimizer_input_norm"] for r in metrics), "max_raw": max(r["raw_gradient_norm"] for r in metrics), "max_step": max(r["actual_increment_norm"] for r in metrics)}
        for key, value in derived.items():
            close(value, actual[key])
            close(value, selected[key])
    d = {key: 100 * (resnet_source["summary"]["Clipping at 1"][key] / resnet_source["summary"]["Unclipped"][key] - 1) for key in ("mean_loss", "tail_mean", "maximum", "endpoint")}
    for key, value in d.items():
        close(value, resnet_source["percent_vs_baseline"][key])
        close(value, resnet["changes_percent"][key])
    clipped = resnet_source["summary"]["Clipping at 1"]
    contains(table, f"{clipped['clipped']}/320", "ResNet clipping activation")
    for field in ("mean_loss", "maximum", "tail_mean"):
        contains(table, f"{d[field]:+.2f}" + r"\%", "ResNet " + field)
    before, after = [resnet_source["summary"][condition]["max_step"] for condition in ("Unclipped", "Clipping at 1")]
    contains(section, f"{before:.3f} to {after:.3f}", "ResNet actual maximum step")

    replication = read_json(ROOT / "analyses/003-2026-10-04-raw-gradient-regimes/replication_check.json")
    records = replication["statistics"]
    assert len(records) == 9 and all(r["raw_all_pair_interaction_J"] > 0 for r in records)
    errors = [-100 * r["lag_one_relative_error"] for r in records]
    contains(section, f"{min(errors):.2f}--{max(errors):.2f}" + r"\%", "Pythia lag-one squared-norm error range")
    for seed in {r["seed"] for r in records}:
        ordered = sorted((r for r in records if r["seed"] == seed), key=lambda r: int(r["size"][:-1]))
        assert len(ordered) == 3
        assert all(a["negative_fraction_F"] < b["negative_fraction_F"] for a, b in zip(ordered, ordered[1:]))
        assert all(a["final_validation_loss"] > b["final_validation_loss"] for a, b in zip(ordered, ordered[1:]))

    bibliography = (ROOT / "manuscript/draft/bibliography.bib").read_text(encoding="utf-8")
    for key in ("biderman2023pythia", "kaddour2023minipile"):
        assert "{" + key + "," in bibliography, key
    config = read_json(ROOT / "runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/config.json")["training"]
    assert config["warmup_fraction"] == .01 and config["peak_learning_rate"] == .001 and config["minimum_learning_rate"] == .0001
    probe = read_json(ROOT / "analyses/025-2026-10-17-pythia-descent-mechanism/probe.json")
    assert probe["excluded_online_prefix_updates"] == 350 and probe["excluded_data_order_seeds"] == [1234, 1235, 1236]
    contains(section, "disjoint from the union of the three 350-update\ntraining prefixes in this intervention study", "probe exclusion scope")
    contains(section, r"\label{sec:implications}", "section label")
    contains(section, r"\input{tables/submission_interventions}", "table input")
    assert table.count(r"\begin{tabular}") == table.count(r"\end{tabular}") == 1
    print(json.dumps({"status": "passed", "source_hash_checks": len(sources), "raw_scalar_files": len(selection["raw_scalar_sources"]), "raw_scalar_rows": raw_rows, "independent_numeric_comparisons": numeric_checks, "display_checks": len(checks), "checks": checks, "pythia_replication_rows": len(records), "preconditioning_branches": len(effects)}, indent=2))


if __name__ == "__main__":
    main()
