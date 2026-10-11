"""Allocate the two requested conditions and freeze the minimal Spark packet."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT / "runs/023-2026-10-08-pythia14m-no-decay-large"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main():
    config = json.loads((REFERENCE / "config.json").read_text())
    folder = REFERENCE / "artifacts/attempt-01/full-pass"
    receipt = json.loads((REFERENCE / "prelaunch/retrieval-receipt.json").read_text())
    assert receipt["accepted"]
    records = []
    for rid, lr in (("053", .02), ("054", .04)):
        name = f"{rid}-2026-10-10-pythia14m-high-lr-{str(lr).replace('.', 'p')}"
        run = ROOT / "runs" / name
        run.mkdir(exist_ok=False)
        (run / "prelaunch").mkdir()
        (run / "observations").mkdir()
        cfg = copy.deepcopy(config)
        cfg.update(run_id=rid, name=name[15:], hypothesis="H014", matched_reference_run="023")
        cfg["training"]["peak_learning_rate"] = lr
        cfg["execution"].update(cohort_maximum_hours=16, cohort_additional_storage_allowance_gb=20)
        assert {k:v for k,v in cfg["training"].items() if k != "peak_learning_rate"} == {
            k:v for k,v in config["training"].items() if k != "peak_learning_rate"}
        for key in ("model", "data", "seeds", "measurements", "validation", "checkpoints"):
            assert cfg[key] == config[key]
        save(run / "config.json", cfg)
        for filename in ("01_train.py", "model_config.json"):
            shutil.copyfile(REFERENCE / filename, run / filename)
        (run / "README.md").write_text(f"""# Run {rid}: Pythia-14M, constant post-warmup LR {lr:g}

Status: prepared; user authorized training on the owned DGX Spark while away.
H014; matched frozen initialization, seed1234 order and complete MiniPile pass
with Spark Run023. Only peak LR changes; AdamW weight decay0.1 remains.
712 accepted updates; batch1024/microbatch4/sequence2048, FP16 compute and
FP32 parameters/moments. Capture raw, stored momentum and adaptive norm,
adjacent cosine, A, B and B/A at every accepted update with FP64 reductions.
Raw is pre-clipping; momentum is actual uncorrected exp_avg from clipped g.
Full initial/final validation and partial token tails are retained.
Initial/midpoint/final/every100-update recovery; separate reset calibration.
8-hour full-pass bound; failures preserve evidence and do not lower LR.
[Design and automatic completion](../../analyses/{HERE.name}/README.md).
No general stability threshold or independent-seed uncertainty is asserted.
""", encoding="utf-8")
        records.append(dict(run_id=rid, run=name, peak_lr=lr))
    names = ("events.jsonl", "summary.json", "input_identity.json", "environment.json", "verification.json")
    reference = dict(run_id="023", peak_lr=.01, config=config,
        events=[e for line in (folder / "events.jsonl").read_text().splitlines()
                if (e := json.loads(line)).get("event") == "update"],
        summary=json.loads((folder / "summary.json").read_text()),
        identity=json.loads((folder / "input_identity.json").read_text()),
        environment=json.loads((folder / "environment.json").read_text()),
        verification=json.loads((folder / "verification.json").read_text()),
        source_sha256={name:sha(folder / name) for name in names},
        source_config_sha256=sha(REFERENCE / "config.json"))
    assert len(reference["events"]) == 712 and reference["verification"]["verified"]
    save(HERE / "reference_023.json", reference)
    save(HERE / "cohort.json", dict(runs=records, reference_run_id="023",
        status="prepared", hypothesis="H014", scope="two_full_passes_constant_high_lr"))
    package(records)


def package(records):
    files = list((ROOT / "src/baseline").glob("*.py")) + list(HERE.glob("*.py"))
    files += [HERE / "cohort.json", HERE / "reference_023.json"]
    files += [ROOT / "analyses/008-2026-10-08-no-decay-momentum/02_verify.py"]
    for record in records:
        files += [ROOT / "runs" / record["run"] / name for name in ("01_train.py", "config.json", "model_config.json")]
    inventory = dict(files=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                                sha256=sha(p)) for p in sorted(files)])
    save(HERE / "source_inventory.json", inventory)
    packet = ROOT / "cache/spark/high-lr-2026-10-10-source.tar.gz"
    packet.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(packet, "w:gz") as archive:
        documentation = [HERE / "README.md"]+[ROOT / "runs" / r["run"] / "README.md" for r in records]
        for path in files + [HERE / "source_inventory.json"] + documentation:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix())
    save(HERE / "packet_identity.json", dict(bytes=packet.stat().st_size, sha256=sha(packet)))
    print(json.dumps(dict(runs=records, packet=str(packet), bytes=packet.stat().st_size, sha256=sha(packet))))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    if args.package_only:
        assert not (HERE / "launch-verification.json").exists(), "Executed inputs are frozen."
        package(json.loads((HERE / "cohort.json").read_text())["runs"])
    else:
        main()
