"""Final source, retained-evidence, figure and link integrity checks."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.fixed_ce_curvature.core import save_json, sha, stamp


def main():
    here = Path(__file__).resolve().parent
    status = json.loads((ROOT / "worker-control/status.json").read_text())
    assert status["status"] == "complete" and len(status["completed"]) == 30
    frozen = json.loads((here / "source_inventory.json").read_text())
    for item in frozen["files"]:
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], item["path"]
    verified = json.loads((here / "verification.json").read_text())
    assert verified["status"] == "passed"
    figure = json.loads((here / "figure_manifest.json").read_text())
    assert figure["pdf_sha256"] == sha(here / "figures/fixed_ce_curvature.pdf")
    assert figure["source_data_sha256"] == sha(here / "figure_data.json")
    assert figure["source_results_sha256"] == sha(here / "results.json")
    assert figure["script_sha256"] == sha(here / "09_plot.py")
    data = json.loads((here / "figure_data.json").read_text())
    for arm in ("native32", "fixed32", "ce64"):
        curves = data["curves"][arm]
        assert curves["loss_steps"] == list(range(0, 10000, 20))
        assert len(curves["loss_seed_curves"]) == 10
        assert all(len(r) == 500 for r in curves["loss_seed_curves"])
        assert all(len(r) == 23 for r in curves["ordinary_seed_curves"])
        assert all(r[-1] is None for r in curves["update_seed_curves"])
        assert all(r[0] is None for r in curves["adam_seed_curves"])
    files = []
    for base in sorted((ROOT / "runs").glob("*-2026-10-23-fixed-ce-*/artifacts")):
        for path in sorted(base.rglob("*")):
            if path.is_file():
                files.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                              "bytes": path.stat().st_size, "sha256": sha(path)})
    save_json(here / "artifact_inventory.json", {"utc": stamp(), "scope": "all retained calibration and scientific artifacts",
        "files": files, "total_bytes": sum(item["bytes"] for item in files),
        "note": "Owned local execution; no remote transfer or cloud resources."})
    links = 0
    markdown = list(here.glob("*.md")) + list((here / "observations").glob("*.md"))
    for path in markdown:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "#")):
                continue
            target = target.split("#", 1)[0]
            resolved = (path.parent / target).resolve()
            # This verification record is emitted below after its own link checks.
            assert resolved.exists() or resolved == (here / "output_verification.json").resolve(), (str(path), target)
            links += 1
    result = {"status": "passed", "utc": stamp(), "frozen_input_files": len(frozen["files"]),
              "retained_artifact_files": len(files), "retained_artifact_bytes": sum(f["bytes"] for f in files),
              "local_links_checked": links,
              "checks": ["frozen scientific source/config/initial/data identities unchanged",
                         "all calibration and production evidence inventoried with size/SHA256",
                         "publication PDF/data/results/script hash identities",
                         "figure state indexing and explicit undefined endpoints", "local documentation links"],
              "rental_usd": 0, "cloud_resources_created": 0}
    save_json(here / "output_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
