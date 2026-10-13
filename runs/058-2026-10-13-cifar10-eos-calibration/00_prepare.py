"""Pin official sources, verify CIFAR bytes, and freeze common CPU inputs."""
import argparse
import ast
import hashlib
import io
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import requests
import torch

from cifar_eos import make_model, tensor_hash, write_json, file_hash

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
EXPECTED_MD5 = "c58f30108f718f92721af3b95e74349a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/cifar10-eos")
    parser.add_argument("--mirror", action="store_true", help="Use MIT's unchanged archive mirror, with original MD5 verification")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    if (args.cache / "inputs.json").exists():
        raise FileExistsError("Prepared inputs are immutable; use a fresh cache for changed inputs.")
    config = json.loads((HERE / "config.json").read_text())
    archive = args.cache / "cifar-10-python.tar.gz"
    download_url = URL
    mirror_revision = None
    if args.mirror:
        response = requests.get("https://huggingface.co/api/datasets/MIT-OL-AI-D/cifar-10-python", timeout=60)
        response.raise_for_status()
        mirror_revision = response.json()["sha"]
        download_url = f"https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/resolve/{mirror_revision}/cifar-10-python.tar.gz"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with requests.get(download_url, stream=True, timeout=90) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for block in response.iter_content(2**20):
                    output.write(block)
        partial.replace(archive)
    if file_hash(archive, "md5") != EXPECTED_MD5:
        raise RuntimeError("Official CIFAR archive MD5 mismatch.")
    response = requests.get("https://api.github.com/repos/locuslab/edge-of-stability/commits/github", timeout=60)
    response.raise_for_status()
    revision = response.json()["sha"]
    source_dir = HERE / "reference"
    source_dir.mkdir(exist_ok=True)
    sources = {}
    for name in ("README.md", "src/archs.py", "src/cifar.py", "src/gd.py", "src/adam.py", "src/utilities.py"):
        response = requests.get(f"https://raw.githubusercontent.com/locuslab/edge-of-stability/{revision}/{name}", timeout=60)
        response.raise_for_status()
        path = source_dir / Path(name).name
        path.write_bytes(response.content)
        sources[name] = {"sha256": file_hash(path), "url": response.url}
    with tarfile.open(archive, "r:gz") as bundle:
        def batch(name):
            # Verified official archive; no filesystem extraction is performed.
            return pickle.load(io.BytesIO(bundle.extractfile("cifar-10-batches-py/" + name).read()), encoding="bytes")
        batches = [batch(f"data_batch_{i}") for i in range(1, 6)]
        train_bytes = np.concatenate([b[b"data"] for b in batches])
        train_labels = np.concatenate([b[b"labels"] for b in batches])
        test = batch("test_batch")
    # Match official NHWC flattening and float64 centering BEFORE std computation.
    train = train_bytes.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).reshape(50000, -1) / 255
    mean = train.mean(0)
    train -= mean
    std = train.std(0)
    if np.any(std <= 0):
        raise RuntimeError("Zero CIFAR normalization denominator.")
    xtrain = ((train[:5000] / std).reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)).astype("float32")
    xtest = test[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).reshape(10000, -1) / 255
    xtest = ((xtest - mean) / std).reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2).astype("float32")
    data = {"x_train": torch.from_numpy(xtrain.copy()), "y_train": torch.tensor(train_labels[:5000]),
            "x_test": torch.from_numpy(xtest.copy()), "y_test": torch.tensor(test[b"labels"]),
            "normalization_mean": torch.from_numpy(mean), "normalization_std": torch.from_numpy(std)}
    torch.save(data, args.cache / "data.pt")
    torch.manual_seed(config["seed"])
    model = make_model()
    torch.save(model.state_dict(), args.cache / "initial.pt")
    # Execute only official pure preprocessing functions as an independent parity check.
    tree = ast.parse((source_dir / "cifar.py").read_text())
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("center", "standardize")], type_ignores=[])
    namespace = {"np": np}
    exec(compile(selected, "official/cifar.py", "exec"), namespace)
    probe = train_bytes[:11].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).reshape(11, -1) / 255
    # Compare the frozen prepared data against independently reconstructed statistics.
    original = train_bytes.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).reshape(50000, -1) / 255
    centered, centered_probe = namespace["center"](original, probe)
    _, reference = namespace["standardize"](centered, centered_probe)
    reference = reference.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2).astype("float32")
    np.testing.assert_array_equal(reference, xtrain[:11])
    manifest = {"dataset_url": URL, "download_url": download_url, "mirror_revision": mirror_revision,
                "archive_bytes": archive.stat().st_size, "archive_sha256": file_hash(archive),
                "archive_md5": EXPECTED_MD5, "source_revision": revision, "sources": sources,
                "train_examples": 5000, "normalization_examples": 50000, "test_examples": 10000,
                "order": "official first 5000; unchanged deterministic order", "parameter_count": sum(p.numel() for p in model.parameters()),
                "initial_tensor_sha256": tensor_hash(model.parameters()), "data_tensor_sha256": {k: tensor_hash([v]) for k, v in data.items()},
                "prepared_files": {name: {"bytes": (args.cache / name).stat().st_size, "sha256": file_hash(args.cache / name)} for name in ("data.pt", "initial.pt")},
                "official_preprocessing_parity": "bitwise equal for 11 probe images"}
    write_json(args.cache / "inputs.json", manifest)
    write_json(HERE / "inputs.json", manifest)
    print(json.dumps({"parameter_count": manifest["parameter_count"], "source_revision": revision, "cache": str(args.cache)}, indent=2))


if __name__ == "__main__":
    main()
