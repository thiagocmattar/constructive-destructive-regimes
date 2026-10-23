"""Frozen H022 training, calibration and common-objective curvature worker."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
import torch
from src.fixed_ce_curvature.core import (AUDIT_NAMES, Geometry, Hessian, adam_vectors,
    configure, fixed_ce, flat, initialize, inventory, logit_audit, make_data,
    optimizer_for, save_json, sha, stamp, tensor_hash, training_loss)


ANALYSIS = ROOT / "analyses/026-2026-10-23-fixed-ce-curvature"
STATES = set(range(0, 10001, 500)) | {9250, 9750}
LEVELS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def nullable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def event(output, kind, **fields):
    record = {"utc": stamp(), "event": kind, **fields}
    with (output / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    print(json.dumps(record, allow_nan=False), flush=True)


def environment():
    return {"utc": stamp(), "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "scipy": scipy.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def state(model, optimizer, geometry, step, recovery=False):
    value = {"parameters": {n: p.detach().cpu() for n, p in model.state_dict().items()},
             "optimizer": optimizer.state_dict(), "step": step,
             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
             "numpy_rng": np.random.get_state(), "data_position": "full-batch-fixed-order"}
    if recovery:
        value["geometry"] = geometry.state()
    return value


def train(config, output, calibration=False):
    configure()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise RuntimeError("Refusing to overwrite an executed attempt")
    device = "cuda"
    model = initialize(config["seed"], device)
    if config.get("initial_path"):
        model.load_state_dict(torch.load(ROOT / config["initial_path"], map_location=device, weights_only=True))
    x, y, xt, yt = make_data(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    opt = optimizer_for(model)
    geometry = Geometry(parameter_count, device)
    save_json(output / "config.json", config)
    save_json(output / "environment.json", environment())
    manifest = {"status": "running", "started": stamp(), "seed": config["seed"],
                "arm": config["arm"], "parameter_count": parameter_count,
                "initial_tensor_sha256": tensor_hash(model.parameters()),
                "data_sha256": tensor_hash((x, y, xt, yt)),
                "config_sha256": sha(output / "config.json"), "updates": 0}
    for key in ("initial_tensor_sha256", "data_sha256"):
        if key in config and manifest[key] != config[key]:
            raise RuntimeError(f"Frozen input identity mismatch: {key}")
    save_json(output / "manifest.json", manifest)
    (output / "states").mkdir()
    torch.save(state(model, opt, geometry, 0), output / "states/state-000000.pt")
    horizon, arm = config["updates"], config["arm"]
    diagnostic_states = ({0, horizon} if calibration else STATES)
    matched = set()
    pending = 0
    times = []
    start = time.perf_counter()
    with gzip.open(output / "metrics.jsonl.gz", "wt", encoding="utf-8", newline="\n") as metrics:
        for step in range(horizon):
            if time.time() >= config.get("deadline_epoch", float("inf")):
                raise TimeoutError("Training admission deadline reached")
            begin = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = training_loss(logits, y, arm)
            loss.backward()
            gradient = flat([p.grad for p in model.parameters()]).detach().clone()
            audit = logit_audit(logits, y)
            prefix = torch.stack((loss.detach().double(), (logits.argmax(1) == y).double().mean()))
            # One host transfer for pre-update values; crossing states are exact pre-update states.
            measures = torch.cat((prefix, audit)).tolist()
            if not all(math.isfinite(float(v)) for v in measures[:3]):
                raise FloatingPointError("Nonfinite training objective")
            if not calibration:
                for level in LEVELS:
                    if level not in matched and measures[2] <= level:
                        label = f"loss-{level:.0e}-step-{step:06d}"
                        torch.save(state(model, opt, geometry, step), output / f"states/{label}.pt")
                        matched.add(level)
            previous = flat(model.parameters()).detach().clone()
            opt.step()
            momentum, adaptive, _ = adam_vectors(model, opt, step + 1)
            displacement = flat(model.parameters()).detach() - previous
            geometry_values = geometry.capture((gradient, momentum, adaptive, displacement)).tolist()
            if not all(math.isfinite(row[0]) for row in geometry_values):
                raise FloatingPointError("Nonfinite vector norm")
            row = {"update": step + 1, "gradient_state": step,
                   "optimization_loss": measures[0], "train_accuracy": measures[1]}
            row.update({name: nullable(value) for name, value in zip(AUDIT_NAMES, measures[2:])})
            for name, values in zip(Geometry.names, geometry_values):
                row[name] = {k: nullable(v) for k, v in zip(("norm", "cosine", "A", "B", "ratio"), values)}
            metrics.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            if pending is not None:
                torch.save({"parameters": {n: p.detach().cpu() for n, p in model.state_dict().items()},
                            "step": step + 1}, output / f"states/next-{pending:06d}.pt")
                pending = None
            completed = step + 1
            if completed in diagnostic_states:
                torch.save(state(model, opt, geometry, completed, completed == horizon),
                           output / f"states/state-{completed:06d}.pt")
                if completed < horizon:
                    pending = completed
            if completed % 250 == 0 or completed == horizon:
                torch.save(state(model, opt, geometry, completed, True), output / "recovery.pt")
                metrics.flush()
                elapsed = time.perf_counter() - start
                event(output, "progress", seed=config["seed"], arm=arm, updates=completed,
                      supervised_targets=completed * 480, common_logits_loss=row["common_logits_loss"],
                      accuracy=row["train_accuracy"], collapse=row["native_collapse_fraction"],
                      updates_per_second=completed / elapsed,
                      etc_seconds=(horizon - completed) * elapsed / completed,
                      peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9)
                save_json(output / "manifest.json", {**manifest, "updates": completed,
                          "elapsed_seconds": elapsed, "latest": row})
            times.append(time.perf_counter() - begin)
    manifest.update(status="training_complete", completed=stamp(), updates=horizon,
                    seconds=time.perf_counter() - start,
                    mean_update_seconds=float(np.mean(times[20:] or times)),
                    p90_update_seconds=float(np.quantile(times[20:] or times, 0.9)),
                    final_tensor_sha256=tensor_hash(model.parameters()),
                    peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9)
    save_json(output / "manifest.json", manifest)
    event(output, "training_complete", **{k: manifest[k] for k in ("seed", "arm", "updates", "seconds")})
    return manifest


def diagnose(output, limit=None):
    configure()
    model = initialize(999, "cuda", torch.float64)
    x, y, xt, yt = make_data("cuda")
    config = json.loads((output / "config.json").read_text())
    paths = sorted((output / "states").glob("state-*.pt"))
    paths += sorted((output / "states").glob("loss-*.pt"))
    result_path = output / "curvature.jsonl"
    done = set()
    if result_path.exists():
        done = {r["checkpoint"] for r in map(json.loads, result_path.read_text().splitlines())}
    results = []
    for path in paths[:limit] if limit else paths:
        if path.name in done:
            continue
        saved = torch.load(path, map_location="cpu", weights_only=False)
        step = saved["step"]
        model.load_state_dict(saved["parameters"])
        hessian = Hessian(model, x, y)
        record = {"checkpoint": path.name, "state": step, "utc": stamp(),
                  "train_loss": float(hessian.loss.detach()),
                  "gradient_norm": float(hessian.gradient.detach().norm()),
                  "parameter_norm": float(flat(model.parameters()).detach().norm()),
                  "layer_norms": {name: float(p.detach().norm()) for name, p in model.named_parameters()},
                  "ordinary": hessian.largest(),
                  "gradient_curvature": hessian.rayleigh(hessian.gradient.detach())}
        with torch.no_grad():
            prediction = model(xt)
            record["heldout_loss"] = float(fixed_ce(prediction, yt))
            record["heldout_accuracy"] = float((prediction.argmax(1) == yt).double().mean())
            prediction = model(x)
            record["train_accuracy"] = float((prediction.argmax(1) == y).double().mean())
        if step > 0 and path.name.startswith("state-"):
            moments = saved["optimizer"]["state"]
            v = torch.cat([moments[key]["exp_avg_sq"].reshape(-1) for key in moments]).to("cuda")
            denominator = (v / (1 - 0.98 ** step)).sqrt() + 1e-8
            record["adam_scaled"] = hessian.largest(denominator.double().rsqrt())
        next_path = output / f"states/next-{step:06d}.pt"
        if path.name.startswith("state-") and next_path.exists():
            after = torch.load(next_path, map_location="cpu", weights_only=False)["parameters"]
            delta = flat([after[n].double() - saved["parameters"][n].double()
                          for n, _ in model.named_parameters()]).to("cuda")
            hd = hessian.product(delta)
            norm2 = delta.dot(delta)
            record["actual_update_norm"] = float(norm2.sqrt())
            record["update_curvature"] = None if not bool(norm2 > 0) else float(delta.dot(hd) / norm2)
            record["linear_change"] = float(hessian.gradient.detach().dot(delta))
            record["quadratic_term"] = float(0.5 * delta.dot(hd))
            model.load_state_dict(after)
            with torch.no_grad():
                next_loss = float(fixed_ce(model(x), y))
            record["actual_loss_change"] = next_loss - record["train_loss"]
            record["taylor_remainder"] = record["actual_loss_change"] - record["linear_change"] - record["quadratic_term"]
        with result_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        event(output, "curvature", seed=config["seed"], arm=config["arm"], state=step,
              checkpoint=path.name, eigenvalue=record["ordinary"]["value"],
              residual=record["ordinary"]["relative_residual"],
              seconds=record["ordinary"]["seconds"])
        results.append(record)
        del hessian
    return results


def calibrate(output):
    start = time.perf_counter()
    manifests, diagnoses = [], []
    for arm in ("native32", "fixed32", "ce64"):
        folder = output / arm
        config = {"seed": 1000, "arm": arm, "updates": 200, "scope": "mechanics-calibration"}
        manifests.append(train(config, folder, True))
        diagnoses += diagnose(folder)
        inventory(folder)
    train_time = sum(m["mean_update_seconds"] for m in manifests) / 3 * 300000
    ordinary = max(r["ordinary"]["seconds"] for r in diagnoses)
    scaled = max((r["adam_scaled"]["seconds"] for r in diagnoses if "adam_scaled" in r), default=ordinary)
    # 690 regular states + <=150 loss crossings; 660 noninitial scaled states.
    diagnostic_time = ordinary * 840 + scaled * 660
    estimate = {"utc": stamp(), "status": "passed", "scope": "mechanics-not-evidence",
                "training_seconds": train_time, "diagnostic_seconds": diagnostic_time,
                "production_estimate_seconds": train_time + diagnostic_time,
                "with_30_percent_allowance_and_10min_retrieval_seconds":
                    1.3 * (train_time + diagnostic_time) + 600,
                "calibration_seconds": time.perf_counter() - start,
                "manifests": manifests, "ordinary_solve_seconds_max": ordinary,
                "scaled_solve_seconds_max": scaled}
    save_json(output / "calibration.json", estimate)
    inventory(output)
    print(json.dumps(estimate), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("calibrate", "train", "diagnose"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.mode == "calibrate":
        calibrate(args.output)
    elif args.mode == "train":
        train(json.loads(args.config.read_text()), args.output)
        inventory(args.output)
    else:
        diagnose(args.output, args.limit)
        inventory(args.output)


if __name__ == "__main__":
    main()
