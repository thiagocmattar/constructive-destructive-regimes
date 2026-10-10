"""Verify existing token bytes and package a bounded production-shaped calibration."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
CACHE = ROOT.parent / "sparsity-spillover/data/tokenized/minipile-pythia-14m-full"
STAGE = ROOT / "cache/spark/020-input"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    STAGE.mkdir(parents=True, exist_ok=False)
    sources = [
        ("14m", "002-2026-10-03-pythia14m-adamw-adaptive-geometry"),
        ("31m", "003-2026-10-03-pythia31m-adamw-scale"),
        ("70m", "004-2026-10-03-pythia70m-adamw-scale"),
    ]
    expected = json.loads((ROOT / "runs" / sources[0][1] / "config.json").read_text())["data"]
    verified = {}
    for split in ("train", "validation"):
        path = CACHE / split / "tokens.int32.bin"
        meta = json.loads((CACHE / split / "metadata.json").read_text())
        sha = digest(path)
        assert sha == expected[f"{split}_sha256"]
        assert path.stat().st_size == meta["tokens"] * 4
        assert meta["dataset_revision"] == expected["revision"]
        assert meta["tokenizer_revision"] == expected["tokenizer_revision"]
        assert meta["block_size"] == 2048 and meta["append_eos"]
        verified[split] = {"bytes": path.stat().st_size, "sha256": sha, "metadata": meta}
    order = np.random.default_rng(1234).permutation(728374).astype(np.int64)
    prefix = order[:3072]
    tokens = np.memmap(CACHE / "train/tokens.int32.bin", mode="r", dtype="<i4")
    selected = np.asarray(tokens[prefix[:, None] * 2048 + np.arange(2048)], dtype="<i4")
    selected.tofile(STAGE / "train-prefix.int32.bin")
    shutil.copyfile(CACHE / "validation/tokens.int32.bin", STAGE / "validation.int32.bin")
    inputs = {
        "schema_version": 1, "scope": "calibration_only", "seed": 1234,
        "source_cache": verified, "full_training_input_tokens": 1491711416,
        "full_training_sequences": 728375, "full_updates": 712,
        "prefix_source_block_indices": prefix.tolist(),
        "prefix_sha256": digest(STAGE / "train-prefix.int32.bin"),
        "prefix_bytes": selected.nbytes, "prefix_sequences": len(prefix),
        "validation_sha256": digest(STAGE / "validation.int32.bin"),
        "validation_bytes": (STAGE / "validation.int32.bin").stat().st_size,
    }
    (RUN / "input_identity.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    for size, name in sources:
        dest = RUN / "recipes" / size
        dest.mkdir(parents=True, exist_ok=False)
        for filename in ("config.json", "model_config.json"):
            shutil.copyfile(ROOT / "runs" / name / filename, dest / filename)
    files = list((ROOT / "src/baseline").glob("*.py"))
    files += [RUN / "01_calibrate.py", RUN / "02_run.sh", RUN / "input_identity.json"]
    files += list((RUN / "recipes").rglob("*.json"))
    identity = {"files": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                            "sha256": digest(p)} for p in sorted(files)]}
    (RUN / "source_inventory.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    files += [RUN / "source_inventory.json"]
    packet = ROOT / "cache/spark/run020-calibration.tar.gz"
    with tarfile.open(packet, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix())
        for path in STAGE.iterdir():
            archive.add(path, arcname="calibration-input/" + path.name)
    receipt = {"bytes": packet.stat().st_size, "sha256": digest(packet)}
    (RUN / "packet_identity.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
