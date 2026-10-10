"""Three complete reference updates per model; no full-pass execution path."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import platform
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
import transformers
from baseline.training import (append_event, build_model, checkpoint, evaluate,
                               optimizer_boundary, synchronize, write_json)
from baseline.gradient_geometry import GradientGeometry
from baseline.initialization import EXPECTED_ARCHITECTURE
from baseline.recipe import DynamicLossScaler, build_recipe_adamw, recipe_learning_rate


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--source-inventory", required=True)
    args = parser.parse_args()
    assert args.attempt in ("attempt-02",)
    output = RUN / "artifacts" / args.attempt
    output.mkdir(parents=True, exist_ok=False)
    event_path = output / "events.jsonl"
    manifest = {"run_id": "020", "scope": "calibration_only", "status": "running",
                "planned_updates_per_model": 3, "completed_updates": 0}
    write_json(output / "manifest.json", manifest)
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        environment = {"platform": platform.platform(), "machine": platform.machine(),
                       "python": sys.version, "torch": torch.__version__,
                       "transformers": transformers.__version__, "numpy": np.__version__,
                       "cuda_runtime": torch.version.cuda, "gpu": props.name,
                       "compute_capability": [props.major, props.minor],
                       "reported_total_memory_bytes": props.total_memory,
                       "cpu_threads": torch.get_num_threads(),
                       "cuda_arch_list": torch.cuda.get_arch_list()}
        write_json(output / "environment.json", environment)
        inventory = json.loads((RUN / args.source_inventory).read_text())
        for item in inventory["files"]:
            path = ROOT / item["path"]
            assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
        inputs = json.loads((RUN / "input_identity.json").read_text())
        input_dir = ROOT / "calibration-input"
        assert digest(input_dir / "train-prefix.int32.bin") == inputs["prefix_sha256"]
        assert digest(input_dir / "validation.int32.bin") == inputs["validation_sha256"]
        train = np.memmap(input_dir / "train-prefix.int32.bin", mode="r", dtype="<i4")
        validation = np.memmap(input_dir / "validation.int32.bin", mode="r", dtype="<i4")
        assert len(train) == 3 * 1024 * 2048 and len(validation) == 693668
        results = {}
        for size in ("14m", "31m", "70m"):
            recipe_dir = RUN / "recipes" / size
            config = json.loads((recipe_dir / "config.json").read_text())
            if size == "14m":
                # Run002 predates the shared scale entry point; use its exact guard.
                config["model"]["expected_architecture"] = EXPECTED_ARCHITECTURE.copy()
            model, metadata = build_model(config, recipe_dir, device)
            assert metadata["parameter_count"] == config["model"].get("parameter_count", 14067712)
            optimizer, optimizer_meta = build_recipe_adamw(model, config["training"], torch=torch)
            scaler = DynamicLossScaler(scale=config["training"]["initial_loss_scale"])
            geometry = GradientGeometry(model.named_parameters())
            adaptive_geometry = GradientGeometry(model.named_parameters(), previous_dtype=torch.float64)
            initial_hash = hashlib.sha256()
            for name, p in model.named_parameters():
                initial_hash.update(name.encode() + b"\0")
                initial_hash.update(p.detach().cpu().numpy().tobytes())
            init_sha = initial_hash.hexdigest()
            append_event(event_path, {"event": "model_ready", "model": size,
                                      "parameters": metadata["parameter_count"],
                                      "initial_parameter_sha256": init_sha,
                                      "historical_initial_hash_matches": init_sha == config["model"]["initial_parameter_sha256"]})
            torch.cuda.reset_peak_memory_stats()
            initial_val = evaluate(model, validation, config, device)
            updates = []
            for update in range(1, 4):
                lr = recipe_learning_rate(update, peak=0.001, max_steps=712,
                                          warmup_fraction=0.01, minimum=0.0001)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                synchronize(device)
                start = time.perf_counter()
                metrics = optimizer_boundary(
                    model, optimizer, scaler, geometry, adaptive_geometry, train,
                    np.arange((update - 1) * 1024, update * 1024), config, device,
                    overflow_callback=lambda event: append_event(event_path, {"model": size, **event}))
                synchronize(device)
                seconds = time.perf_counter() - start
                event = {"event": "update", "model": size, "update": update,
                         "warmup": update == 1, "update_seconds": seconds,
                         "tokens_per_second": metrics["input_tokens"] / seconds,
                         "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), **metrics}
                updates.append(event)
                append_event(event_path, event)
                manifest["completed_updates"] += 1
                manifest["current_model"] = size
                write_json(output / "manifest.json", manifest)
                print(json.dumps({k: event[k] for k in ("model", "update", "update_seconds", "loss", "tokens_per_second", "diagnostic_seconds", "overflow_retries")}), flush=True)
            final_val = evaluate(model, validation, config, device)
            save_seconds = checkpoint(output, size, model, optimizer, scaler, geometry,
                                      adaptive_geometry, 3072, inputs["prefix_sha256"],
                                      digest(recipe_dir / "config.json"))
            checkpoint_files = []
            hash_start = time.perf_counter()
            for path in sorted((output / "checkpoints" / size).rglob("*")):
                if path.is_file():
                    checkpoint_files.append({"path": path.relative_to(output).as_posix(),
                                             "bytes": path.stat().st_size, "sha256": digest(path)})
            hash_seconds = time.perf_counter() - hash_start
            samples = [e["update_seconds"] for e in updates[1:]]
            # 711 full updates and a 311/1024 partial update; tail shape can differ.
            equivalent_updates = 711 + 311 / 1024
            validation_seconds = initial_val["seconds"] + final_val["seconds"]
            projected_seconds = statistics.mean(samples) * equivalent_updates + validation_seconds + 3 * save_seconds
            result = {"model": size, "parameters": metadata["parameter_count"],
                      "initial_parameter_sha256": init_sha,
                      "historical_initial_hash_matches": init_sha == config["model"]["initial_parameter_sha256"],
                      "model_metadata": metadata, "optimizer_metadata": optimizer_meta,
                      "initial_validation": initial_val, "final_validation": final_val,
                      "optimizer_step_seconds": samples, "all_updates": updates,
                      "recovery_checkpoint_seconds": save_seconds,
                      "recovery_checkpoint_hash_seconds": hash_seconds,
                      "checkpoint_inventory": checkpoint_files,
                      "estimated_three_checkpoint_bytes": 3 * sum(f["bytes"] for f in checkpoint_files),
                      "projected_full_run_seconds_excluding_setup_transfer": projected_seconds,
                      "projected_full_run_hours": projected_seconds / 3600}
            results[size] = result
            write_json(output / f"{size}-summary.json", result)
            append_event(event_path, {"event": "model_completed", "model": size,
                                      "projected_full_run_hours": projected_seconds / 3600})
            del model, optimizer, geometry, adaptive_geometry, metrics, p, group
            gc.collect()
            torch.cuda.empty_cache()
        write_json(output / "summary.json", {"scope": "calibration_only", "models": results,
                                              "environment": environment, "full_updates": 712,
                                              "full_training_input_tokens": 1491711416})
        manifest["status"] = "completed"
        write_json(output / "manifest.json", manifest)
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
