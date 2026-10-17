"""Verify the prepared scientific comparison and every source/input packet byte."""
from pathlib import Path
import hashlib
import json
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    cohort = json.loads((HERE / "cohort.json").read_text())
    assert len(cohort) == len({r["run"] for r in cohort}) == 27
    assert {r["size"] for r in cohort} == {"14M", "31M", "70M"}
    assert {r["seed"] for r in cohort} == {1234, 1235, 1236}
    probe = json.loads((HERE / "probe.json").read_text())
    assert len(probe["indices"]) == len(set(probe["indices"])) == 128
    assert probe["supervised_tokens"] == 262016
    for row in cohort:
        run = ROOT / "runs" / row["run"]
        config = json.loads((run / "config.json").read_text())
        c = config["causal"]
        assert c["baseline_updates"] == 350 and c["schedule_updates"] == 712
        assert c["probe_sha256"] == sha(HERE / "probe.json")
        assert config["training"]["effective_batch_sequences"] == 1024
        assert config["training"]["micro_batch_size"] == 4
        assert config["measurements"]["track_adam_momentum"]
        order = np.r_[np.random.default_rng(row["seed"]).permutation(728374), 728374].astype("<i8")
        assert c["order_sha256"] == hashlib.sha256(order.tobytes()).hexdigest()
        assert not np.intersect1d(probe["indices"], order[:350 * 1024]).size
        if row["condition"] != "baseline":
            parent_path = ROOT / "runs" / row["parent"] / "config.json"
            parent = json.loads(parent_path.read_text())
            assert c["parent_config_sha256"] == sha(parent_path)
            for key in ("model", "training", "data", "seeds", "measurements"):
                assert config[key] == parent[key]
            assert c["junction_update"] in {175, 225} and c["branch_updates"] == 50
    plan = json.loads((HERE / "worker_plan.json").read_text())
    assigned = [r for worker in plan for family in worker for r in [family["baseline"], *family["branches"]]]
    assert len(assigned) == len(set(assigned)) == 27
    assert set(assigned) == {r["run"] for r in cohort}
    assert all(worker[0]["size"] == "70M" for worker in plan)
    assert b"\r" not in (HERE / "setup.sh").read_bytes()
    for filename in ("source_packet.json", "input_packet.json"):
        packet = json.loads((HERE / filename).read_text())
        path = ROOT / packet["path"]
        assert path.stat().st_size == packet["bytes"] and sha(path) == packet["sha256"]
    inputs = json.loads((HERE / "input_packet.json").read_text())
    with tarfile.open(ROOT / inputs["path"]) as archive:
        expected = {r["path"]: r for r in inputs["files"]}
        members = [m for m in archive.getmembers() if m.isfile()]
        assert len(members) == len(expected)
        for member in members:
            item = expected[member.name]
            assert member.size == item["bytes"]
            with archive.extractfile(member) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
    for item in json.loads((HERE / "source_inventory.json").read_text())["files"]:
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
    for item in json.loads((HERE / "initial_sources.json").read_text()):
        source = ROOT / item["source"]
        attempt = source.parents[3]
        inventory = {r["path"]: r for r in json.loads((attempt / "inventory.json").read_text())["files"]}
        for name, digest in item["files"].items():
            assert inventory[(source / name).relative_to(attempt).as_posix()]["sha256"] == digest
    result = dict(passed=True, unexecuted_runs=27, matched_baselines=9, matched_branches=18,
        common_probe_training_overlap=0, original_712_update_schedule=True,
        parent_configs_matched=True, frozen_initials_match_accepted_inventories=True,
        source_packet_sha256=sha(ROOT / json.loads((HERE / "source_packet.json").read_text())["path"]),
        input_packet_sha256=inputs["sha256"], all_input_archive_payloads_verified=True,
        worker_assignment_unique=True, paid_gpu_calibration="pending spending approval")
    (HERE / "preflight.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
