"""Matched Pythia replays and preconditioning-ablation continuations for H021."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import time

import numpy as np
import torch

from baseline.continuation import rng_state, restore_rng, protected_evaluate
from baseline.data_protocol import block_order, read_cache, sha256
from baseline.descent_diagnostics import (DisplacementMetrics, adam_components, apply_direction,
    direction_trials, fixed_probe, vector_norm)
from baseline.gradient_geometry import GradientGeometry
from baseline.recipe import DynamicLossScaler, build_recipe_adamw, recipe_learning_rate
from baseline.training import (append_event, build_model, now, optimizer_boundary,
                               synchronize, write_json)


def save_state(path, model, optimizer, scaler, geometries, displacement, update,
               order_hash, config_hash, output, data_position):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(schema_version=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(), geometry=geometries[0].state_dict(),
        adaptive_geometry=geometries[1].state_dict(), momentum_geometry=geometries[2].state_dict(),
        displacement=displacement.state_dict(), completed_updates=update, data_position=data_position,
        order_sha256=order_hash, config_sha256=config_hash, source_output=str(output), **rng_state())
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_state(path, model, optimizer, scaler, geometries, displacement, device,
                  order_hash, expected_config_hash, effective_batch=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["order_sha256"] != order_hash or state["config_sha256"] != expected_config_hash:
        raise ValueError("Checkpoint input/config identity mismatch")
    if effective_batch is not None and state["data_position"] != state["completed_updates"] * effective_batch:
        raise ValueError("Checkpoint data position mismatch")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    for geometry, key in zip(geometries, ("geometry", "adaptive_geometry", "momentum_geometry")):
        geometry.load_state_dict(state[key])
        if geometry.updates != state["completed_updates"]:
            raise ValueError("Checkpoint diagnostic clock mismatch")
    displacement.load_state_dict(state["displacement"], device)
    if any(int(s["step"].item()) != state["completed_updates"] for s in optimizer.state.values()):
        raise ValueError("Checkpoint Adam clock mismatch")
    restore_rng(state)
    return state["completed_updates"], state["source_output"]


def parameter_sha(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode() + b"\0")
        digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def prefix_events(source, destination, update):
    """A retry copies its accepted prefix; the previous attempt remains immutable."""
    for name in ("events.jsonl", "probe_events.jsonl"):
        path = source / name
        if path.exists():
            accepted = [line for line in path.read_text().splitlines()
                        if json.loads(line).get("update", 0) <= update]
            (destination / name).write_text("\n".join(accepted) + "\n", encoding="utf-8")


def execute(args):
    config_path = Path(args.config).resolve()
    directory, output = config_path.parent, Path(args.output).resolve()
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    config_hash = sha256(config_path)
    (output / "resolved_config.json").write_bytes(config_path.read_bytes())
    frozen_source = output / "source/baseline"
    frozen_source.mkdir(parents=True)
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, frozen_source / path.name)
    for path in [directory / "02_train.py", directory / "model_config.json"]:
        if path.is_file():
            shutil.copy2(path, output / "source" / path.name)
    manifest = dict(run_id=config["run_id"], status="preparing", started_at=now(),
                    config_sha256=config_hash, smoke_calibration=bool(args.calibrate_updates))
    write_json(output / "manifest.json", manifest)
    try:
        train, train_meta = read_cache(Path(args.cache) / "train", config["data"]["train_sha256"], config["data"])
        validation, val_meta = read_cache(Path(args.cache) / "validation", config["data"]["validation_sha256"], config["data"])
        order = block_order(len(train), config["data"]["sequence_length"], config["seeds"]["data_order"])
        order_hash = hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()
        if order_hash != config["causal"]["order_sha256"]:
            raise ValueError("Training order differs from frozen design")
        probe_path = (directory / config["causal"]["probe_path"]).resolve()
        if sha256(probe_path) != config["causal"]["probe_sha256"]:
            raise ValueError("Probe selection identity mismatch")
        probe = json.loads(probe_path.read_text())
        probe_indices = np.array(probe["indices"], dtype=np.int64)
        effective = config["training"]["effective_batch_sequences"]
        if np.intersect1d(probe_indices, order[:config["causal"]["baseline_updates"] * effective]).size:
            raise ValueError("Probe intersects the online training prefix")
        device = torch.device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, model_metadata = build_model(config, directory, device)
        if parameter_sha(model) != config["model"]["initial_parameter_sha256"]:
            raise ValueError("Frozen initial tensors differ")
        optimizer, optimizer_metadata = build_recipe_adamw(model, config["training"], torch=torch)
        geometries = (GradientGeometry(model.named_parameters()),
                      GradientGeometry(model.named_parameters(), previous_dtype=torch.float64),
                      GradientGeometry(model.named_parameters()))
        displacement = DisplacementMetrics()
        scaler = DynamicLossScaler(scale=config["training"]["initial_loss_scale"],
            growth_interval=config["training"]["loss_scale_growth_interval"],
            hysteresis=config["training"]["loss_scale_hysteresis"],
            minimum_scale=config["training"]["minimum_loss_scale"])
        start, end = 0, config["causal"]["baseline_updates"]
        targets = {}
        branch = config["causal"]["condition"] != "baseline"
        if branch:
            parent = (directory / config["causal"]["parent_output"]).resolve()
            parent_config = (directory / config["causal"]["parent_config"]).resolve()
            if sha256(parent_config) != config["causal"]["parent_config_sha256"]:
                raise ValueError("Parent scientific config differs")
            verification = json.loads((parent / "verification.json").read_text())
            if not verification["verified"]:
                raise ValueError("Parent has not passed verification")
            expected_start = config["causal"]["junction_update"]
            checkpoint = parent / "checkpoints" / f"update-{expected_start}.pt"
            if sha256(checkpoint) != verification["checkpoints_sha256"][checkpoint.name]:
                raise ValueError("Parent checkpoint hash differs")
            start, _ = restore_state(checkpoint, model, optimizer, scaler, geometries,
                displacement, device, order_hash, sha256(parent_config), effective)
            if start != expected_start:
                raise ValueError("Parent checkpoint junction differs")
            end = start + config["causal"]["branch_updates"]
            parent_events = [json.loads(line) for line in (parent / "events.jsonl").read_text().splitlines()]
            targets = {e["update"]: e["learning_rate"] * e["adaptive"]["direction_norm"]
                       for e in parent_events if e["event"] == "update"}
            write_json(output / "parent_receipt.json", dict(checkpoint=str(checkpoint),
                checkpoint_sha256=sha256(checkpoint), verification_sha256=sha256(parent / "verification.json"),
                parent_events_sha256=sha256(parent / "events.jsonl"), parent_config_sha256=sha256(parent_config)))
        if args.recover_from:
            recovery = Path(args.recover_from).resolve()
            start, previous = restore_state(recovery, model, optimizer, scaler, geometries,
                displacement, device, order_hash, config_hash, effective)
            prefix_events(Path(previous), output, start)
            (output / "checkpoints").mkdir(exist_ok=True)
            for junction in config["causal"]["junctions"]:
                previous_checkpoint = Path(previous) / "checkpoints" / f"update-{junction}.pt"
                if junction <= start and previous_checkpoint.is_file():
                    shutil.copy2(previous_checkpoint, output / "checkpoints" / previous_checkpoint.name)
            write_json(output / "recovery_receipt.json", dict(path=str(recovery), sha256=sha256(recovery), update=start))
        if args.calibrate_updates:
            end = min(end, start + args.calibrate_updates)
        probe_loss, _ = fixed_probe(model, train, probe_indices, config["data"]["sequence_length"],
            config["causal"]["probe_microbatch"], device)
        append_event(output / "events.jsonl", dict(event="initial_probe", update=start, loss=probe_loss))
        if not args.recover_from:
            validation_result = protected_evaluate(model, validation, config, device)
            append_event(output / "events.jsonl", dict(event="initial_validation", update=start, **validation_result))
        write_json(output / "environment.json", dict(torch=torch.__version__,
            transformers=__import__("transformers").__version__, numpy=np.__version__,
            cuda=torch.version.cuda, device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            model=model_metadata, optimizer=optimizer_metadata,
            source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}))
        write_json(output / "input_identity.json", dict(train=train_meta, validation=val_meta,
            order_sha256=order_hash, probe_sha256=sha256(probe_path), probe_indices=probe["indices"],
            probe_supervised_tokens=len(probe_indices) * (config["data"]["sequence_length"] - 1),
            original_schedule_updates=config["causal"]["schedule_updates"]))
        position, samples = start * effective, []
        model.train()
        for update in range(start + 1, end + 1):
            if args.deadline_epoch and time.time() >= args.deadline_epoch:
                raise TimeoutError("Training deadline reached; durable recovery retained")
            synchronize(device)
            tick = time.perf_counter()
            selected = order[position:position + effective]
            if len(selected) != effective:
                raise ValueError("Unexpected partial training batch")
            parameters = [p for p in model.parameters() if p.requires_grad]
            before = [p.detach().clone() for p in parameters]
            diagnostic = update in config["causal"]["measurement_updates"] or bool(args.calibrate_updates)
            if diagnostic:
                loss_before, probe_gradient = fixed_probe(model, train, probe_indices,
                    config["data"]["sequence_length"], config["causal"]["probe_microbatch"], device, gradient=True)
            learning = config["training"]
            lr = recipe_learning_rate(update, peak=learning["peak_learning_rate"],
                minimum=learning["minimum_learning_rate"], warmup_fraction=learning["warmup_fraction"],
                max_steps=config["causal"]["schedule_updates"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            training_tick = time.perf_counter()
            metrics = optimizer_boundary(model, optimizer, scaler, geometries[0], geometries[1],
                train, selected, config, device,
                lambda event: append_event(output / "events.jsonl", dict(update=update, **event)), geometries[2])
            if branch:
                vectors = adam_components(optimizer, list(model.named_parameters()), update, "momentum")
                metrics["intervention"] = apply_direction(optimizer, list(model.named_parameters()), before,
                                                           vectors, targets[update])
                del vectors
            metrics["actual_parameter_update"] = displacement.consume(before, parameters)
            synchronize(device)
            training_elapsed = time.perf_counter() - training_tick
            if diagnostic:
                for event in direction_trials(model, optimizer, before, probe_gradient, loss_before,
                        train, probe_indices, config, device, update, interventions=not branch):
                    append_event(output / "probe_events.jsonl", event)
                del probe_gradient
            synchronize(device)
            elapsed = time.perf_counter() - tick
            samples.append(elapsed)
            position += effective
            event = dict(event="update", update=update, gradient_step=update - 1, learning_rate=lr,
                cumulative_input_tokens=position * config["data"]["sequence_length"],
                cumulative_supervised_tokens=position * (config["data"]["sequence_length"] - 1),
                update_seconds=elapsed, tokens_per_second=metrics["input_tokens"] / elapsed,
                training_update_seconds=training_elapsed,
                probe_and_snapshot_seconds=elapsed-training_elapsed,
                probe_measured=diagnostic, **metrics)
            if device.type == "cuda":
                event["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            append_event(output / "events.jsonl", event)
            manifest.update(status="running", completed_updates=update, planned_end=end,
                completed_input_tokens=event["cumulative_input_tokens"], latest_loss=event["loss"],
                latest_event_at=now(), remaining_training_seconds=(end - update) * float(np.median(samples[-20:])))
            write_json(output / "manifest.json", manifest)
            print(json.dumps(event), flush=True)
            if not args.calibrate_updates:
                if not branch and update in config["causal"]["junctions"]:
                    save_state(output / "checkpoints" / f"update-{update}.pt", model, optimizer, scaler,
                               geometries, displacement, update, order_hash, config_hash, output, position)
                elif update % config["causal"]["recovery_every_updates"] == 0 and update != end:
                    save_state(output / "checkpoints/rolling.pt", model, optimizer, scaler,
                               geometries, displacement, update, order_hash, config_hash, output, position)
        final_probe, _ = fixed_probe(model, train, probe_indices, config["data"]["sequence_length"],
            config["causal"]["probe_microbatch"], device)
        final_validation = protected_evaluate(model, validation, config, device)
        if not args.calibrate_updates:
            save_state(output / "checkpoints/final.pt", model, optimizer, scaler, geometries,
                       displacement, end, order_hash, config_hash, output, position)
        summary = dict(completed_updates=end, attempted_start=start, condition=config["causal"]["condition"],
            final_probe_loss=final_probe, final_validation=final_validation,
            processed_training_sequences=position, original_schedule_updates=config["causal"]["schedule_updates"],
            mean_update_seconds=float(np.mean(samples)), probe_indices_sha256=sha256(probe_path))
        write_json(output / "summary.json", summary)
        manifest.update(status="completed", completed_at=now())
        write_json(output / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc), failed_at=now())
        write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibrate-updates", type=int, default=0)
    parser.add_argument("--recover-from")
    parser.add_argument("--deadline-epoch", type=float)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
