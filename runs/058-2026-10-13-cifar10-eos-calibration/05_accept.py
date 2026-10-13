"""Safe archive retrieval, file integrity and independent final-state checks."""
import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import torch

from cifar_eos import file_hash, stamp, write_json

HERE = Path(__file__).resolve().parent


def state_checks(directory, inputs):
    final = torch.load(directory / "final.pt", map_location="cpu", weights_only=False)
    initial = torch.load(inputs / "initial.pt", weights_only=True)
    rows = [json.loads(s) for s in (directory / "metrics.jsonl").read_text().splitlines()]
    last = rows[-1]
    concatenate = lambda values: np.concatenate([v.numpy().reshape(-1).astype("float64") for v in values])
    theta = concatenate(final["model"].values())
    theta0 = concatenate(initial.values())
    for name, state in final["streams"].items():
        assert state["count"] == final["step"]
        metrics = last["streams"][name]
        np.testing.assert_allclose(np.linalg.norm(state["previous"].numpy()), metrics["norm"], rtol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(state["total"].numpy()), metrics["B"], rtol=1e-12)
        np.testing.assert_allclose(state["A"], metrics["A"], rtol=1e-12)
    np.testing.assert_allclose(final["streams"]["displacement"]["total"], theta - theta0, rtol=1e-12, atol=1e-12)
    if final["optimizer_name"] == "adam":
        group = final["optimizer"]["param_groups"][0]
        moments = [final["optimizer"]["state"][p] for p in group["params"]]
        m = concatenate([s["exp_avg"] for s in moments])
        v = concatenate([s["exp_avg_sq"] for s in moments])
        b1, b2 = group["betas"]
        mhat = m / (1 - b1**final["step"])
        direction = mhat / (np.sqrt(v / (1 - b2**final["step"])) + group["eps"])
        for name, reference in (("momentum", m), ("unbiased_momentum", mhat), ("adaptive", direction)):
            np.testing.assert_allclose(final["streams"][name]["previous"], reference, rtol=1e-12, atol=1e-13)
    else:
        residual = theta - theta0 + final["optimizer"]["param_groups"][0]["lr"] * final["streams"]["raw"]["total"].numpy()
        assert np.max(np.abs(residual)) < 2e-6
    return {"updates": final["step"], "streams_checked": list(final["streams"]), "exact_displacement_telescoping": "passed", "optimizer_state_vectors": "passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    assert file_hash(args.archive) == args.expected_sha256
    destination = HERE / "artifacts"
    if (destination / "attempt-01-spark").exists():
        raise FileExistsError("Retrieved attempt already exists.")
    with tarfile.open(args.archive) as bundle:
        for member in bundle.getmembers():
            assert member.name == "attempt-01-spark" or member.name.startswith("attempt-01-spark/")
            assert not member.issym() and not member.islnk()
            resolved = (destination / member.name).resolve()
            assert resolved.is_relative_to(destination.resolve())
        bundle.extractall(destination, filter="data")
    spark = destination / "attempt-01-spark"
    inventory = json.loads((spark / "transfer_inventory.json").read_text())
    for item in inventory["files"]:
        path = spark / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert file_hash(path) == item["sha256"]
    result = {"status": "passed", "utc": stamp(), "archive_sha256": args.expected_sha256,
              "retrieved_files_verified": len(inventory["files"]), "retrieved_bytes": sum(i["bytes"] for i in inventory["files"]),
              "states": {}}
    for device in ("rtx", "spark"):
        for method in ("gd", "adam"):
            result["states"][device + "/" + method] = state_checks(destination / ("attempt-01-" + device) / method, args.inputs)
    write_json(HERE / "retrieval_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
