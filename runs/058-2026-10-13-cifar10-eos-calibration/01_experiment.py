"""Production-ready GD/Adam runner; calibration is a separate explicit mode."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cifar_eos import (Geometry, adam_vectors, append_json, cosine, curvature_event, environment,
                       evaluate, file_hash, flat, full_gradient, make_model, memory, stamp,
                       synchronize, tensor_hash, write_json)

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimizer", choices=["gd", "adam"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    inputs = json.loads((args.inputs / "inputs.json").read_text())
    for name, identity in inputs["prepared_files"].items():
        if file_hash(args.inputs / name) != identity["sha256"]:
            raise RuntimeError(f"Input hash mismatch: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = config["tf32"]
    torch.backends.cudnn.allow_tf32 = config["tf32"]
    device = torch.device(args.device)
    data = {k: v.to(device) for k, v in torch.load(args.inputs / "data.pt", weights_only=True).items() if k.startswith(("x_", "y_"))}
    model = make_model().to(device)
    model.load_state_dict(torch.load(args.inputs / "initial.pt", map_location=device, weights_only=True))
    if tensor_hash(model.parameters()) != inputs["initial_tensor_sha256"]:
        raise RuntimeError("Frozen initialization mismatch.")
    options = config["optimizers"][args.optimizer]
    if args.optimizer == "gd":
        optimizer = torch.optim.SGD(model.parameters(), **options)
    else:
        options = dict(options, betas=tuple(options["betas"]))
        optimizer = torch.optim.Adam(model.parameters(), **options, foreach=False, fused=False)
    count = inputs["parameter_count"]
    names = ["raw", "displacement"] + (["momentum", "unbiased_momentum", "adaptive"] if args.optimizer == "adam" else [])
    streams = {name: Geometry(count, device) for name in names}
    policy = config["proposed_production"]
    updates = config["calibration_updates"] if args.calibrate else policy["max_updates"]
    start_step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved["config_sha256"] != file_hash(args.config) or saved["input_sha256"] != file_hash(args.inputs / "inputs.json") or saved["optimizer_name"] != args.optimizer:
            raise RuntimeError("Resume scientific identity differs.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for name, stream in streams.items():
            stream.load_state_dict(saved["streams"][name])
        start_step = saved["step"]
        torch.set_rng_state(saved["rng_cpu"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["rng_cuda"]])
    manifest = {"status": "running", "mode": "calibration" if args.calibrate else "production", "started_utc": stamp(),
                "optimizer": args.optimizer, "start_step": start_step, "requested_updates": updates,
                "config_sha256": file_hash(args.config), "input_sha256": file_hash(args.inputs / "inputs.json"),
                "source_sha256": {p.name: file_hash(p) for p in HERE.glob("*.py")}, "environment": environment(device),
                "initial_tensor_sha256": inputs["initial_tensor_sha256"], "train_examples": len(data["x_train"]),
                "test_examples": len(data["x_test"]), "history": "raw f32 vectors" if policy["record_raw_history"] else "exact scalar prefixes; no dense histories",
                "resume_checkpoint": str(args.resume) if args.resume else None}
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "resolved_config.json", config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times, checkpoint_times, curvature_times = [], [], []
    clock = time.perf_counter()
    raw_history = (args.output / "raw-gradient-history.f32").open("wb") if policy["record_raw_history"] else None

    def checkpoint(step, label):
        synchronize(device)
        t0 = time.perf_counter()
        temporary = args.output / (label + ".tmp")
        torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "optimizer_name": args.optimizer, "streams": {k: s.state_dict() for k, s in streams.items()},
                    "config_sha256": manifest["config_sha256"], "input_sha256": manifest["input_sha256"],
                    "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}, temporary)
        temporary.replace(args.output / (label + ".pt"))
        checkpoint_times.append(time.perf_counter() - t0)

    def curvature(step):
        event = curvature_event(model, data, config, step, optimizer if args.optimizer == "adam" else None)
        event["eta_lambda_max"] = options["lr"] * event["hessian"]["eigenvalues"][0]
        event["gd_reference_2_over_eta"] = 2 / options["lr"] if args.optimizer == "gd" else None
        if "paper_preconditioned_hessian" in event:
            event["eta_paper_preconditioned_lambda_max"] = options["lr"] * event["paper_preconditioned_hessian"]["eigenvalues"][0]
        append_json(args.output / "curvature.jsonl", event)
        curvature_times.append(event)
        print(json.dumps({"state_step": step, "hessian": event["hessian"]["eigenvalues"], "seconds": event["hessian"]["seconds"]}), flush=True)

    checkpoint(start_step, "initial" if not args.resume else "resumed")
    curvature(start_step)
    stop_reason = "update_budget"
    completed = start_step
    for index in range(start_step, updates):
        synchronize(device)
        begin = time.perf_counter()
        train_loss, train_acc, raw = full_gradient(model, data["x_train"], data["y_train"], config["physical_batch_size"], config["loss"])
        synchronize(device)
        backward_seconds = time.perf_counter() - begin
        if not args.calibrate and train_acc >= policy["accuracy_goal"]:
            stop_reason = "training_accuracy_goal"
            break
        begin = time.perf_counter()
        before = flat([p.detach() for p in model.parameters()]).double()
        optimizer.step()
        synchronize(device)
        optimizer_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        if not bool(torch.isfinite(before).all()) or not bool(torch.isfinite(raw).all()):
            raise RuntimeError("Nonfinite gradient/parameter.")
        vectors = {"raw": raw, "displacement": flat([p.detach() for p in model.parameters()]).double() - before}
        if args.optimizer == "adam":
            adaptive, _ = adam_vectors(optimizer, index + 1)
            vectors.update(adaptive)
        previous_raw = streams["raw"].previous
        previous_norm = None if previous_raw is None else float(previous_raw.norm())
        q = None if previous_norm is None or previous_norm == 0 else 1 - float(previous_raw.dot(raw.double())) / previous_norm**2
        stats = {name: streams[name].consume(value) for name, value in vectors.items()}
        raw_norm = stats["raw"]["norm"]
        row = {"update": index + 1, "gradient_state_step": index, "train_loss": train_loss, "train_accuracy": train_acc,
               "streams": stats, "parameter_norm_before": float(before.norm()),
               "displacement_from_initial_norm": None,
               "gradient_dot_update": float(raw.double().dot(vectors["displacement"])),
               "raw_update_cosine": cosine(raw, vectors["displacement"]),
               "raw_parameter_cosine": cosine(raw, before), "eta_kappa_proxy": q,
               "kappa_proxy": None if q is None else q / options["lr"],
               "momentum_raw_cosine": cosine(raw, vectors["momentum"]) if "momentum" in vectors else None,
               "unbiased_momentum_raw_cosine": cosine(raw, vectors["unbiased_momentum"]) if "unbiased_momentum" in vectors else None,
               "momentum_to_raw_norm_ratio": stats["momentum"]["norm"] / raw_norm if "momentum" in stats and raw_norm > 0 else None,
               "adaptive_to_raw_norm_ratio": stats["adaptive"]["norm"] / raw_norm if "adaptive" in stats and raw_norm > 0 else None,
               "adaptive_to_momentum_norm_ratio": stats["adaptive"]["norm"] / stats["momentum"]["norm"] if "momentum" in stats and stats["momentum"]["norm"] > 0 else None}
        initial_distance = streams["displacement"].total
        row["displacement_from_initial_norm"] = float(initial_distance.norm())
        if raw_history:
            raw_history.write(raw.detach().cpu().numpy().astype("<f4").tobytes())
            raw_history.flush()
        synchronize(device)
        diagnostic_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        if index % policy["test_every"] == 0:
            # Same state as the row's gradient/loss requires evaluating before update;
            # this evaluation is at theta[index+1] and is labeled explicitly.
            row["test_state_step"] = index + 1
            row["test"] = evaluate(model, data["x_test"], data["y_test"], config["physical_batch_size"], config["loss"])
        synchronize(device)
        test_seconds = time.perf_counter() - begin
        row["timing"] = {"backward_seconds": backward_seconds, "optimizer_seconds": optimizer_seconds,
                         "diagnostic_seconds": diagnostic_seconds, "test_seconds": test_seconds,
                         "total_seconds": backward_seconds + optimizer_seconds + diagnostic_seconds + test_seconds}
        append_json(args.output / "metrics.jsonl", row)
        times.append(row["timing"])
        completed = index + 1
        if not args.calibrate and completed % policy["eigen_every"] == 0:
            curvature(completed)
        if completed % policy["checkpoint_every"] == 0:
            checkpoint(completed, "recovery")
        if completed % policy["progress_every"] == 0 or args.calibrate:
            elapsed = time.perf_counter() - clock
            event = {"utc": stamp(), "completed_updates": completed, "train_loss": train_loss, "train_accuracy": train_acc,
                     "elapsed_seconds": elapsed, "updates_per_second": (completed - start_step) / elapsed,
                     "remaining_seconds_at_observed_rate": (updates - completed) * elapsed / max(1, completed - start_step), **memory(device)}
            append_json(args.output / "events.jsonl", event)
            print(json.dumps(event), flush=True)
    if not curvature_times or curvature_times[-1]["state_step"] != completed:
        curvature(completed)
    checkpoint(completed, "final")
    if raw_history:
        raw_history.close()
    final_train = evaluate(model, data["x_train"], data["y_train"], config["physical_batch_size"], config["loss"])
    final_test = evaluate(model, data["x_test"], data["y_test"], config["physical_batch_size"], config["loss"])
    warmup = config["warmup_updates"] if args.calibrate else 0
    summary = {"completed_updates": completed, "stop_reason": stop_reason, "elapsed_seconds": time.perf_counter() - clock,
               "final_train": final_train, "final_test": final_test, "memory": memory(device),
               "update_samples": times, "post_warmup_samples": times[warmup:], "checkpoint_seconds": checkpoint_times,
               "curvature_samples": curvature_times, "final_tensor_sha256": tensor_hash(model.parameters())}
    write_json(args.output / "summary.json", summary)
    manifest.update(status="complete", completed_utc=stamp(), completed_updates=completed,
                    files={p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in args.output.iterdir() if p.is_file() and p.name != "manifest.json"})
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "updates": completed, "memory": summary["memory"]}), flush=True)


if __name__ == "__main__":
    main()
