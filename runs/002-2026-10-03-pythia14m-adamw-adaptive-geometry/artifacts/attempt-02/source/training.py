"""Dense Pythia baseline with per-update raw-gradient diagnostics."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, GPTNeoXConfig

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from baseline.data_protocol import block_order, read_cache, sha256, target_count, update_microbatches
from baseline.gradient_geometry import GradientGeometry
from baseline.initialization import apply_pythia_14m_initialization, verify_recipe_model
from baseline.recipe import DynamicLossScaler, build_recipe_adamw, recipe_attention_context, recipe_learning_rate

from adam_diagnostics import adaptive_directions, norm_comparison


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_event(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"timestamp": now(), **value}, allow_nan=False) + "\n")
        stream.flush()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def forward_loss(model, batch, device):
    autocast = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    with recipe_attention_context(torch, device), autocast:
        return model(input_ids=batch, labels=batch, use_cache=False).loss


def optimizer_boundary(model, optimizer, scaler, geometry, adaptive_geometry, tokens, indices,
                       config, device, overflow_callback=None):
    """One accepted update; overflow retries use the same data and no diagnostic mutation."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    length = config["data"]["sequence_length"]
    targets = target_count(len(tokens), indices, length)
    if targets <= 0:
        raise ValueError("Optimizer update has no supervised targets.")
    for retry in range(config["training"]["maximum_overflow_retries"] + 1):
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        scale = float(scaler.scale)
        for batch, count in update_microbatches(tokens, indices, length,
                                                config["training"]["micro_batch_size"], device):
            loss = forward_loss(model, batch, device)
            if not torch.isfinite(loss.detach()):
                raise RuntimeError("Nonfinite training loss.")
            weight = count / targets
            weighted_loss += float(loss.detach()) * weight
            (loss * (weight * scale)).backward()
            del loss, batch
        with torch.no_grad():
            for p in parameters:
                if p.grad is None:
                    raise RuntimeError("Missing trainable gradient.")
                p.grad.div_(scale)
            finite = bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all())
        scale_event = scaler.update(finite=finite)
        if not finite:
            if overflow_callback:
                overflow_callback({"event": "overflow_retry", "retry": retry, **scale_event})
            continue
        start = time.perf_counter()
        metrics = geometry.consume()
        synchronize(device)
        diagnostic_seconds = time.perf_counter() - start
        clip_norm = torch.nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip_norm"],
                                                   error_if_nonfinite=True)
        optimizer.step()
        synchronize(device)
        start = time.perf_counter()
        adaptive = adaptive_geometry.consume(adaptive_directions(
            optimizer, model.named_parameters(), geometry.updates))
        adaptive["direction_norm"] = adaptive.pop("gradient_norm")
        adaptive["cumulative_direction_norm"] = adaptive.pop("cumulative_gradient_norm")
        synchronize(device)
        adaptive_diagnostic_seconds = time.perf_counter() - start
        return {"loss": weighted_loss, "supervised_tokens": targets,
                "sequences": len(indices), "input_tokens": targets + len(indices),
                "diagnostic_seconds": diagnostic_seconds + adaptive_diagnostic_seconds,
                "raw_diagnostic_seconds": diagnostic_seconds,
                "adaptive_diagnostic_seconds": adaptive_diagnostic_seconds,
                "adaptive": adaptive,
                **norm_comparison(metrics["gradient_norm"], adaptive["direction_norm"]),
                "gradient_clip_returned_norm": float(clip_norm),
                "overflow_retries": retry, **scale_event, **metrics}
    raise RuntimeError("Overflow retries exhausted; no data silently skipped.")


@torch.no_grad()
def evaluate(model, tokens, config, device):
    start = time.perf_counter()
    model.eval()
    length = config["data"]["sequence_length"]
    count = math.ceil(len(tokens) / length)
    weighted = 0.0
    targets = 0
    for batch, size in update_microbatches(tokens, np.arange(count), length,
                                           config["validation"]["batch_size"], device):
        loss = float(forward_loss(model, batch, device))
        if not math.isfinite(loss):
            raise RuntimeError("Nonfinite validation loss.")
        weighted += loss * size
        targets += size
    synchronize(device)
    model.train()
    return {"loss": weighted / targets, "supervised_tokens": targets,
            "input_tokens": len(tokens), "sequences": count,
            "seconds": time.perf_counter() - start, "short_tail_included": bool(len(tokens) % length)}


def checkpoint(output, label, model, optimizer, scaler, geometry, adaptive_geometry, position, order_hash, config_hash):
    start = time.perf_counter()
    destination = output / "checkpoints" / label
    destination.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(destination / "model", safe_serialization=True)
    state = {"schema_version": 1, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "geometry": geometry.state_dict(), "adaptive_geometry": adaptive_geometry.state_dict(),
             "data_position": position, "order_sha256": order_hash,
             "config_sha256": config_hash, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
    torch.save(state, destination / "recovery.pt")
    return time.perf_counter() - start


def build_model(config, directory, device):
    torch.manual_seed(config["seeds"]["model"])
    np.random.seed(config["seeds"]["model"])
    random.seed(config["seeds"]["model"])
    architecture = GPTNeoXConfig.from_json_file(directory / "model_config.json")
    architecture.use_cache = False
    architecture._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(architecture).float()
    initialization = apply_pythia_14m_initialization(model, torch=torch)
    model = model.to(device)
    metadata = verify_recipe_model(model)
    metadata["parameter_count"] = sum(p.numel() for p in model.parameters())
    metadata["initialization"] = initialization
    return model, metadata


def execute(args):
    directory = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_hash = sha256(config_path)
    write_json(output / "resolved_config.json", config)
    manifest = {"schema_version": 1, "run_id": "002", "status": "preparing",
                "scope": "smoke_calibration" if args.calibrate_updates else "full_pass",
                "started_at": now(), "config_sha256": config_hash}
    write_json(output / "manifest.json", manifest)
    events = output / "events.jsonl"
    try:
        train, train_meta = read_cache(Path(args.cache) / "train", config["data"]["train_sha256"], config["data"])
        validation, val_meta = read_cache(Path(args.cache) / "validation", config["data"]["validation_sha256"], config["data"])
        order = block_order(len(train), config["data"]["sequence_length"], config["seeds"]["data_order"])
        order_hash = hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()
        effective = config["training"]["effective_batch_sequences"]
        full_updates = math.ceil(len(order) / effective)
        planned_updates = min(args.calibrate_updates, full_updates) if args.calibrate_updates else full_updates
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable.")
        torch.backends.cuda.matmul.allow_tf32 = False
        model, model_metadata = build_model(config, directory, device)
        optimizer, optimizer_metadata = build_recipe_adamw(model, config["training"], torch=torch)
        geometry = GradientGeometry(model.named_parameters())
        adaptive_geometry = GradientGeometry(model.named_parameters(), previous_dtype=torch.float64)
        initial_digest = hashlib.sha256()
        for name, p in model.named_parameters():
            initial_digest.update(name.encode() + b"\0")
            initial_digest.update(p.detach().cpu().numpy().tobytes())
        initial_parameter_sha256 = initial_digest.hexdigest()
        if initial_parameter_sha256 != config["model"]["initial_parameter_sha256"]:
            raise ValueError("Initial parameters differ from the matched Run001 checkpoint.")
        scaler = DynamicLossScaler(scale=config["training"]["initial_loss_scale"],
                                   growth_interval=config["training"]["loss_scale_growth_interval"],
                                   hysteresis=config["training"]["loss_scale_hysteresis"],
                                   minimum_scale=config["training"]["minimum_loss_scale"])
        environment = {"python": __import__("sys").version, "torch": torch.__version__,
                       "transformers": __import__("transformers").__version__,
                       "numpy": np.__version__, "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       "model": model_metadata, "optimizer": optimizer_metadata,
                       "initial_parameter_sha256": initial_parameter_sha256,
                       "shared_source_sha256": {p.name: sha256(p) for p in (directory.parents[1] / "src/baseline").glob("*.py")},
                       "source_sha256": {p.name: sha256(p) for p in directory.glob("*.py")}}
        write_json(output / "environment.json", environment)
        write_json(output / "input_identity.json", {"train": train_meta, "validation": val_meta,
                   "order_sha256": order_hash, "sequence_count": len(order), "full_updates": full_updates,
                   "repeated_sequences": 0, "excluded_tail_tokens": 0})
        initial_validation = evaluate(model, validation, config, device)
        append_event(events, {"event": "initial_validation", **initial_validation})
        checkpoint_seconds = 0.0
        if not args.calibrate_updates:
            checkpoint_seconds += checkpoint(output, "initial", model, optimizer, scaler, geometry, adaptive_geometry,
                                               0, order_hash, config_hash)
        model.train()
        samples = []
        diagnostic_samples = []
        input_tokens = supervised_tokens = position = 0
        manifest.update(status="running", planned_updates=planned_updates, full_updates=full_updates,
                        completed_updates=0, completed_input_tokens=0)
        write_json(output / "manifest.json", manifest)
        for update in range(1, planned_updates + 1):
            synchronize(device)
            start = time.perf_counter()
            indices = order[position:position + effective]
            lr = recipe_learning_rate(update, peak=config["training"]["peak_learning_rate"],
                                      max_steps=full_updates, warmup_fraction=config["training"]["warmup_fraction"],
                                      minimum=config["training"]["minimum_learning_rate"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            metrics = optimizer_boundary(model, optimizer, scaler, geometry, adaptive_geometry, train, indices, config, device,
                        lambda event: append_event(events, {"update": update, **event}))
            elapsed = time.perf_counter() - start
            samples.append(elapsed)
            diagnostic_samples.append(metrics["diagnostic_seconds"])
            position += len(indices)
            input_tokens += metrics["input_tokens"]
            supervised_tokens += metrics["supervised_tokens"]
            event = {"event": "update", "update": update, "gradient_step": update - 1,
                     "learning_rate": lr, "update_seconds": elapsed,
                     "tokens_per_second": metrics["input_tokens"] / elapsed,
                     "cumulative_input_tokens": input_tokens, "cumulative_supervised_tokens": supervised_tokens,
                     "remaining_training_seconds": (planned_updates - update) * float(np.median(samples[-10:])),
                     **metrics}
            if device.type == "cuda":
                event["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                event["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            append_event(events, event)
            manifest.update(completed_updates=update, completed_input_tokens=input_tokens,
                            latest_loss=metrics["loss"], latest_event_at=now(),
                            remaining_training_seconds=event["remaining_training_seconds"])
            write_json(output / "manifest.json", manifest)
            print(json.dumps(event), flush=True)
            if not args.calibrate_updates and update == full_updates // 2:
                checkpoint_seconds += checkpoint(output, "midpoint", model, optimizer, scaler, geometry, adaptive_geometry,
                                                   position, order_hash, config_hash)
        final_validation = evaluate(model, validation, config, device)
        append_event(events, {"event": "final_validation", **final_validation})
        checkpoint_seconds += checkpoint(output, "final", model, optimizer, scaler, geometry, adaptive_geometry,
                                           position, order_hash, config_hash)
        write_json(output / "gradient_state_identity.json", {"updates": geometry.updates,
                   "norm_sum": geometry.norm_sum, "previous_norm": geometry.previous_norm,
                   "parameter_names": geometry.names, "adaptive_updates": adaptive_geometry.updates,
                   "adaptive_norm_sum": adaptive_geometry.norm_sum,
                   "adaptive_previous_norm": adaptive_geometry.previous_norm})
        summary = {"scope": manifest["scope"], "completed_updates": planned_updates,
                   "full_updates": full_updates, "completed_sequences": position,
                   "planned_sequences": len(order), "completed_input_tokens": input_tokens,
                   "completed_supervised_tokens": supervised_tokens, "train_token_count": len(train),
                   "all_training_tokens_covered": position == len(order) and input_tokens == len(train),
                   "repeated_sequences": 0, "initial_validation": initial_validation,
                   "final_validation": final_validation, "last_update": event,
                   "optimizer_step_seconds": samples[1:] or samples,
                   "full_validation_seconds": [initial_validation["seconds"], final_validation["seconds"]],
                   "diagnostic_seconds_per_update": diagnostic_samples,
                   "checkpoint_seconds": checkpoint_seconds, "order_sha256": order_hash}
        write_json(output / "summary.json", summary)
        manifest.update(status="completed", finished_at=now(), all_training_tokens_covered=summary["all_training_tokens_covered"])
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibrate-updates", type=int)
    args = parser.parse_args()
    if args.calibrate_updates is not None and args.calibrate_updates < 1:
        parser.error("Calibration updates must be positive.")
    execute(args)


if __name__ == "__main__":
    main()
