"""Close verified owning runs and prepare an index patch excluding unrelated work."""
import difflib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    results = json.loads((HERE / "results.json").read_text())
    verification = json.loads((HERE / "verification.json").read_text())
    assert verification["status"] == "passed" and results["coverage"]["runs"] == 30
    assert (HERE / "REPORT.md").is_file() and (HERE / "figures/fixed_ce_curvature.pdf").is_file()
    ratio = results["primary"]["fixed32_over_native32"]
    effect = f"curvature ratio {ratio['geometric_ratio']:.5g}, 95% CI [{ratio['ci95'][0]:.5g},{ratio['ci95'][1]:.5g}]"
    status = "Supports raw-Hessian flattening in this recipe" if ratio["ci95"][1] < .8 else "Raw-Hessian result conditional on this recipe"
    qualification = results["qualification"]["qualified_native_seeds"]
    row154 = "| [154](154-2026-10-23-fixed-ce-calibration/README.md) | H022 full-shape mechanics/timing; seed1000 excluded from inference | Complete; two local calibration attempts | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/README.md) |"
    rows = [row154]
    for run in results["runs"]:
        directory = ROOT / "runs" / run["run"]
        readme = directory / "README.md"
        text = readme.read_text()
        text = text.replace("Status: prepared, user-approved October23,2026; not yet executed.",
                            "Status: complete and locally size/SHA256 verified, October23,2026.")
        text += (f"\n## Completed evidence\n\n10,000 updates and23 regular whole-objective curvature states;\n"
                 f"{run['spectra']} ordinary spectra including loss crossings. Late-window\n"
                 f"geometric mean lambda_max={run['late_lambda_geometric_mean']:.12g};\n"
                 f"final common FP64 training loss={run['final_common_loss']:.12g};\n"
                 f"held-out accuracy={run['final_heldout_accuracy']:.8g}.\n"
                 f"{run['files']} retained files /{run['bytes']} bytes verified.\n"
                 "One owned RTX5070Ti Laptop GPU; zero rental; no cloud resource.\n"
                 "See Analysis026 results/verification and its completed observation for\n"
                 "paired inference, qualification, limitations and the publication figure.\n")
        readme.write_text(text, encoding="utf-8", newline="\n")
        rows.append(f"| [{run['id']}]({run['run']}/README.md) | H022 {run['arm']},seed{run['seed']} | Complete10k; local hash/math verification | Owned5070Ti; no rental | [Analysis026](../analyses/026-2026-10-23-fixed-ce-curvature/REPORT.md) |")
    h_path = "research/HYPOTHESES.md"
    a_path = "analyses/INDEX.md"
    r_path = "research/INDEX.md"
    u_path = "runs/INDEX.md"
    changes = []
    old_h = next(line for line in (ROOT / h_path).read_text().splitlines() if line.startswith("| H022 |"))
    new_h = f"| H022 | Does numerically fixed hard-target CE lead AdamW to lower common-objective Hessian curvature? | {status}; exposure-qualified seeds{qualification}/10 | [Analysis026](../analyses/{HERE.name}/REPORT.md), Runs154-184 | {effect};30 trajectories/300k updates; full480-example FP64 Hessians; paired precision control and matched-loss audit; zero rental; no Pythia transfer claim |"
    changes.append((h_path, old_h, new_h))
    old_a = next(line for line in (ROOT / a_path).read_text().splitlines() if line.startswith("| [026]"))
    new_a = f"| [026]({HERE.name}/README.md) | H022 fixed-CE curvature | Runs154-184;10 paired seeds,three loss-arithmetic arms | Complete30/30,300k updates; verified ordinary/scaled spectra; zero rental | [Conclusions]({HERE.name}/REPORT.md), [figure]({HERE.name}/figures/fixed_ce_curvature.pdf); {effect}; collapse-qualified seeds{qualification}/10 |"
    changes.append((a_path, old_a, new_a))
    old_r = (f"[Analysis026: fixed-CE curvature proposal](../analyses/{HERE.name}/README.md)\n"
             "operationalizes H022 with ten paired seeds on tiny modular addition, three CE\n"
             "arithmetic conditions, and common full-FP64 Hessian diagnostics. Primary target\n"
             "is a20% late-window maximum-curvature reduction, with confidence/scale and Adam\n"
             "controls. CPU calculus checks pass; no training evidence or resource creation.\n"
             "Live3090 price USD0.50/h; proposed four-hour/USD3 limit, runtime unmeasured.\n"
             "Next Run154 / Analysis027 / H023.")
    new_r = (f"[Analysis026: completed fixed-CE curvature experiment](../analyses/{HERE.name}/REPORT.md)\n"
             "completed30 paired conditions and300k updates on owned5070Ti. Common fully\n"
             f"FP64 full480-example Hessians give fixed/native{effect};\n"
             f"hard-collapse exposure qualifies{qualification}/10 seeds; precision/scale limits are explicit.\n"
             "All evidence is local and hash verified; publication PDF reviewed.\n"
             "Zero rental and no cloud resources created. Next Run185 / Analysis027 / H023.")
    changes.append((r_path, old_r, new_r))
    changes.append((u_path, "Next run: `154-YYYY-MM-DD-slug`.", "\n".join(rows) + "\n\nNext run: `185-YYYY-MM-DD-slug`."))
    patch = []
    for filename, old, new in changes:
        path = ROOT / filename
        before = path.read_text(encoding="utf-8")
        original = subprocess.run(["git", "show", f"HEAD:{filename}"], capture_output=True, text=True, check=True).stdout
        assert before.count(old) == 1 and original.count(old) == 1, filename
        path.write_text(before.replace(old, new), encoding="utf-8", newline="\n")
        patch.extend(difflib.unified_diff(original.splitlines(True), original.replace(old, new).splitlines(True),
                                        fromfile="a/" + filename, tofile="b/" + filename))
    (ROOT / "tmp/analysis026/closeout-records.patch").write_text("".join(patch), encoding="utf-8", newline="\n")
    print("Closed30 runs and prepared an isolated four-registry patch; not staged yet.")


if __name__ == "__main__":
    main()
