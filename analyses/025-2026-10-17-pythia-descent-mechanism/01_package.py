"""Create separate verified input/source packets without changing executed evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, record):
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", action="store_true")
    args = parser.parse_args()
    cohort = json.loads((HERE / "cohort.json").read_text())
    for row in cohort:
        if list((ROOT / "runs" / row["run"] / "prelaunch").glob("*-launch.json")):
            raise RuntimeError("Executed source/config packets are frozen")
    target = ROOT / "cache/runpod"
    target.mkdir(parents=True, exist_ok=True)
    if args.inputs:
        packet = target / "analysis025-inputs.tar"
        if packet.exists():
            raise RuntimeError("Input packet already exists; verify instead of replacing")
        cache = ROOT.parent / "sparsity-spillover/data/tokenized/minipile-pythia-14m-full"
        config = json.loads((ROOT / "runs" / cohort[0]["run"] / "config.json").read_text())
        sources = []
        for split in ("train", "validation"):
            for name in ("metadata.json", "tokens.int32.bin"):
                p = cache / split / name
                digest = sha(p)
                if name == "tokens.int32.bin":
                    assert digest == config["data"][split + "_sha256"]
                sources.append((p, f"inputs/cache/{split}/{name}", digest))
        for item in json.loads((HERE / "initial_sources.json").read_text()):
            for name, digest in item["files"].items():
                p = ROOT / item["source"] / name
                assert sha(p) == digest
                sources.append((p, item["destination"] + "/" + name, digest))
        with tarfile.open(packet, "w") as archive:
            for p, destination, _ in sources:
                archive.add(p, arcname=destination)
        save(HERE / "input_packet.json", dict(path=packet.relative_to(ROOT).as_posix(),
            bytes=packet.stat().st_size, sha256=sha(packet),
            files=[dict(path=destination, bytes=p.stat().st_size, sha256=digest) for p, destination, digest in sources]))
        print("Verified input packet:", packet.stat().st_size, "bytes")
    source_packet = target / "analysis025-source.tar.gz"
    files = [*sorted((ROOT / "src/baseline").glob("*.py")), ROOT / "tools/artifact_inventory.py",
             ROOT / "tests/test_descent_diagnostics.py", ROOT / "tests/test_causal_control.py"]
    files += sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.sh")) + [HERE / "README.md", HERE / "design.json", HERE / "cohort.json",
        HERE / "probe.json", HERE / "initial_sources.json", HERE / "input_packet.json",
        HERE / "worker_plan.json", HERE / "launch_plan.json", HERE / "live_price.json", HERE / "historical_timing.json"]
    for row in cohort:
        run = ROOT / "runs" / row["run"]
        files += sorted(run.glob("*.py")) + sorted(run.glob("*.json"))
    inventory = dict(files=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in files])
    save(HERE / "source_inventory.json", inventory)
    with tarfile.open(source_packet, "w:gz") as archive:
        for path in files + [HERE / "source_inventory.json"]:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix())
    save(HERE / "source_packet.json", dict(path=source_packet.relative_to(ROOT).as_posix(),
        bytes=source_packet.stat().st_size, sha256=sha(source_packet)))
    print("Source packet:", source_packet.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
