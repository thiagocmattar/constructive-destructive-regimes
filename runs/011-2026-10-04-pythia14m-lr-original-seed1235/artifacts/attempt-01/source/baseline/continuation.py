"""Matched checkpoint continuations for the six H004 LR interventions."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from baseline.data_protocol import block_order, read_cache, sha256
from baseline.gradient_geometry import GradientGeometry
from baseline.recipe import DynamicLossScaler, build_recipe_adamw, recipe_learning_rate
from baseline.training import append_event, checkpoint, evaluate, now, optimizer_boundary, synchronize, write_json


def learning_rate(update, training, resume):
    junction, budget = resume["junction_update"], resume["full_updates"]
    branch = resume["schedule"]
    if update <= junction or branch == "original":
        clock = update
    elif branch == "hold":
        clock = junction
    elif branch == "accelerated":
        clock = min(budget, junction + 2 * (update - junction))
    else:
        raise ValueError(f"Unknown continuation schedule: {branch}")
    return recipe_learning_rate(clock, peak=training["peak_learning_rate"], max_steps=budget,
        warmup_fraction=training["warmup_fraction"], minimum=training["minimum_learning_rate"])


def rng_state():
    return {"torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    if state["cuda_rng"]:
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count differs from the checkpoint.")
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def protected_evaluate(model, tokens, config, device):
    state, mode = rng_state(), model.training
    try:
        return evaluate(model, tokens, config, device)
    finally:
        restore_rng(state)
        model.train(mode)


def load_parent(config, directory, device):
    directory = Path(directory)
    expected = config["resume"]
    for relative, digest in expected["checkpoint_sha256"].items():
        if sha256(directory / relative) != digest:
            raise ValueError(f"Parent checkpoint hash differs: {relative}")
    # These are trusted local experiment states, containing Python/NumPy RNGs.
    state = torch.load(directory / "recovery.pt", map_location="cpu", weights_only=False)
    if state["config_sha256"] != expected["parent_config_sha256"]:
        raise ValueError("Recovery config differs from the declared parent.")
    if state["order_sha256"] != expected["order_sha256"]:
        raise ValueError("Recovery data order differs from the declared parent.")
    junction = expected["junction_update"]
    if state["data_position"] != junction * config["training"]["effective_batch_sequences"]:
        raise ValueError("Recovery data position is not the declared junction.")
    if any(state[key]["updates"] != junction for key in ("geometry", "adaptive_geometry")):
        raise ValueError("Diagnostic clock differs from the declared junction.")
    model = AutoModelForCausalLM.from_pretrained(directory / "model", local_files_only=True,
        attn_implementation="sdpa", dtype=torch.float32).to(device)
    model.config.use_cache = False
    if sum(p.numel() for p in model.parameters()) != config["model"]["parameter_count"]:
        raise ValueError("Parent parameter count differs.")
    optimizer, metadata = build_recipe_adamw(model, config["training"], torch=torch)
    optimizer.load_state_dict(state["optimizer"])
    if any(int(value["step"].item()) != junction for value in optimizer.state.values()):
        raise ValueError("Adam bias-correction clock differs from the junction.")
    geometry = GradientGeometry(model.named_parameters())
    adaptive = GradientGeometry(model.named_parameters(), previous_dtype=torch.float64)
    geometry.load_state_dict(state["geometry"])
    adaptive.load_state_dict(state["adaptive_geometry"])
    scaler = DynamicLossScaler()
    scaler.load_state_dict(state["scaler"])
    position = state["data_position"]
    restore_rng(state)
    model.train()
    identity = {"junction_update": junction, "data_position": position,
        "order_sha256": state["order_sha256"], "parent_config_sha256": state["config_sha256"],
        "parent_checkpoint_sha256": expected["checkpoint_sha256"],
        "raw_norm_sum_at_junction": geometry.norm_sum,
        "adaptive_norm_sum_at_junction": adaptive.norm_sum,
        "parameter_names": geometry.names, "optimizer": metadata}
    del state
    return model, optimizer, scaler, geometry, adaptive, position, identity


def execute(args):
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_hash = sha256(config_path)
    write_json(output / "resolved_config.json", config)
    scope = "resume_mechanics" if args.calibrate_updates else "midpoint_continuation"
    manifest = {"schema_version": 1, "run_id": config["run_id"], "scope": scope,
        "status": "preparing", "started_at": now(), "config_sha256": config_hash}
    write_json(output / "manifest.json", manifest)
    events = output / "events.jsonl"
    try:
        train, train_meta = read_cache(Path(args.cache) / "train", config["data"]["train_sha256"], config["data"])
        val, val_meta = read_cache(Path(args.cache) / "validation", config["data"]["validation_sha256"], config["data"])
        order = block_order(len(train), config["data"]["sequence_length"], config["seeds"]["data_order"])
        order_hash = hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()
        if order_hash != config["resume"]["order_sha256"]:
            raise ValueError("Regenerated permutation differs from parent.")
        effective = config["training"]["effective_batch_sequences"]
        full = math.ceil(len(order) / effective)
        if full != config["resume"]["full_updates"]:
            raise ValueError("Cache implies a different original schedule budget.")
        device = torch.device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        loaded = load_parent(config, args.parent or config["resume"]["remote_checkpoint"], device)
        model, optimizer, scaler, geometry, adaptive, position, identity = loaded
        junction = config["resume"]["junction_update"]
        finish = min(full, junction + args.calibrate_updates) if args.calibrate_updates else full
        write_json(output / "resume_identity.json", identity)
        write_json(output / "input_identity.json", {"train": train_meta, "validation": val_meta,
            "order_sha256": order_hash, "sequence_count": len(order), "full_updates": full,
            "prefix_sequences": position, "continuation_sequences": len(order) - position,
            "repeated_sequences": 0, "excluded_tail_tokens": 0})
        write_json(output / "environment.json", {"python": __import__("sys").version,
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "tf32": False, "optimizer": identity["optimizer"],
            "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}})
        validations = []
        evaluation = {"update": junction, **protected_evaluate(model, val, config, device)}
        validations.append(evaluation)
        append_event(events, {"event": "validation", **evaluation})
        samples, checkpoint_seconds = [], 0.0
        input_tokens = supervised_tokens = 0
        prefix_inputs = junction * effective * config["data"]["sequence_length"]
        prefix_targets = prefix_inputs - position
        manifest.update(status="running", full_updates=full, planned_updates=finish,
            completed_updates=junction, completed_input_tokens=prefix_inputs,
            branch=config["resume"]["schedule"])
        write_json(output / "manifest.json", manifest)
        for update in range(junction + 1, finish + 1):
            synchronize(device)
            started = time.perf_counter()
            indices = order[position:position + effective]
            lr = learning_rate(update, config["training"], config["resume"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            metrics = optimizer_boundary(model, optimizer, scaler, geometry, adaptive, train,
                indices, config, device,
                lambda event: append_event(events, {"update": update, **event}))
            elapsed = time.perf_counter() - started
            samples.append(elapsed)
            position += len(indices)
            input_tokens += metrics["input_tokens"]
            supervised_tokens += metrics["supervised_tokens"]
            event = {"event": "update", "update": update, "gradient_step": update - 1,
                "branch": config["resume"]["schedule"], "learning_rate": lr,
                "batch_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                "update_seconds": elapsed, "tokens_per_second": metrics["input_tokens"] / elapsed,
                "cumulative_input_tokens": prefix_inputs + input_tokens,
                "cumulative_supervised_tokens": prefix_targets + supervised_tokens,
                "continuation_input_tokens": input_tokens,
                "remaining_training_seconds": (finish-update) * float(np.median(samples[-10:])), **metrics}
            if device.type == "cuda":
                event["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                event["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            append_event(events, event)
            manifest.update(completed_updates=update, completed_input_tokens=prefix_inputs+input_tokens,
                latest_loss=metrics["loss"], latest_event_at=now(),
                remaining_training_seconds=event["remaining_training_seconds"])
            write_json(output / "manifest.json", manifest)
            print(json.dumps(event), flush=True)
            if not args.calibrate_updates and update in config["resume"]["validation_updates"]:
                evaluation = {"update": update, **protected_evaluate(model, val, config, device)}
                validations.append(evaluation)
                append_event(events, {"event": "validation", **evaluation})
            if not args.calibrate_updates and update in (534, 712):
                checkpoint_seconds += checkpoint(output, f"update-{update}", model, optimizer,
                    scaler, geometry, adaptive, position, order_hash, config_hash)
        summary = {"scope": scope, "completed_updates": finish, "junction_update": junction,
            "completed_sequences": position, "planned_sequences": len(order),
            "completed_input_tokens": prefix_inputs+input_tokens,
            "completed_supervised_tokens": prefix_targets+supervised_tokens,
            "continuation_input_tokens": input_tokens, "continuation_supervised_tokens": supervised_tokens,
            "all_training_tokens_covered": position == len(order) and prefix_inputs+input_tokens == len(train),
            "repeated_sequences": 0, "validations": validations, "last_update": event,
            "optimizer_step_seconds": samples, "checkpoint_seconds": checkpoint_seconds,
            "order_sha256": order_hash}
        write_json(output / "summary.json", summary)
        manifest.update(status="completed", finished_at=now(), all_training_tokens_covered=summary["all_training_tokens_covered"])
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--parent")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibrate-updates", type=int)
    args = parser.parse_args()
    if args.calibrate_updates is not None and args.calibrate_updates < 1:
        parser.error("Mechanics update count must be positive.")
    execute(args)


if __name__ == "__main__":
    main()
