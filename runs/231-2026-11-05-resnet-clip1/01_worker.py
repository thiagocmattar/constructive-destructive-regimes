"""Run the matched ResNet clipping pilot on Spark from the archived Run223 fork."""
import importlib.util
import json
import signal
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT = HERE / "scratch/parent"
OUT = HERE / "artifacts/worker"
CFG = json.loads((HERE / "config.json").read_text())
PARENT = ROOT / CFG["parent"]
spec = importlib.util.spec_from_file_location("run223", PARENT / "01_worker.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    status = dict(status="running", started_utc=w.stamp(), results={})
    w.write(OUT / "manifest.json", status)
    started = time.monotonic()
    cancelled = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancelled.append(True))

    def stop():
        return bool(cancelled) or (HERE / "STOP").exists() or time.monotonic() - started > CFG["max_seconds"]

    try:
        for item in json.loads((HERE / "input_manifest.json").read_text())["files"]:
            path = ROOT / item["path"]
            assert path.stat().st_size == item["bytes"] and w.sha(path) == item["sha256"], str(path)
        w.configure()
        w.torch.use_deterministic_algorithms(True)
        w.torch.backends.cudnn.deterministic = True
        parent_cfg = json.loads((PARENT / "config.json").read_text())
        control = parent_cfg["control"]
        context = w.prepare("resnet", parent_cfg)
        saved = w.load(context, INPUT / "fork.pt")
        assert context["state"] == 256 and context["base"] == 0.003
        loss, gradient = context["task"].gradient()
        actual = gradient.detach().cpu()
        reference = saved["current_gradient"]
        exact_gradient = w.torch.equal(reference, actual)
        relative_error = float((reference.double()-actual.double()).norm()/reference.double().norm())
        status.update(gpu=w.torch.cuda.get_device_name(), torch=w.torch.__version__, cross_gpu_gradient_exact=exact_gradient, cross_gpu_relative_error=relative_error)
        w.write(OUT / "manifest.json", status)
        w.load(context, INPUT / "fork.pt")
        w.qualify(context, parent_cfg, OUT / "qualification", stop)
        replay = w.torch.load(OUT / "qualification/direct/end.pt", map_location="cpu", weights_only=False)
        cloud = w.torch.load(INPUT / "direct.pt", map_location="cpu", weights_only=False)
        try:
            for key in ["model","optimizer","current_gradient","previous_gradient","rng"]:
                w.equal(replay[key], cloud[key])
            exact_two = True
        except AssertionError:
            exact_two = False
        reused = exact_gradient and exact_two
        status.update(exact_spark_replay=True, cross_gpu_two_updates_exact=exact_two,
                      baseline="archived_4090" if reused else "fresh_spark")
        w.write(OUT / "manifest.json", status)
        arms = [("clipping", CFG["treatment"])]
        if not reused:
            arms.append(("control", control))
        for folder, arm in arms:
            w.load(context, INPUT / "fork.pt")
            context["threshold"] = CFG["cap"]
            context["identity"] = dict(run="231", parent_sha256=w.sha(INPUT / "fork.pt"),
                                       config_sha256=w.sha(HERE / "config.json"), manifest_sha256=w.sha(HERE / "input_manifest.json"))
            result = w.run(context, arm, OUT / folder, CFG["end"], stop)
            status["results"][folder] = result
            w.write(OUT / "manifest.json", status)
            assert result["status"] == "completed"
            records = w.rows(OUT / folder / "metrics.jsonl")
            assert [row["state"] for row in records] == list(range(256,576))
            for row in records:
                norm = row["raw_gradient_norm"]
                expected = min(1., 1./norm) if folder == "clipping" and norm else 1.
                assert row["clip_factor"] == expected and row["applied_lr"] == .003
                assert abs(row["optimizer_input_norm"]-expected*norm) < 2e-7
            context["identity"] = saved["identity"]
        status["status"] = "completed"
    except BaseException as exc:
        status.update(status="failed", exception=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        status.update(finished_utc=w.stamp(), elapsed_seconds=time.monotonic()-started)
        w.write(OUT / "manifest.json", status)


if __name__ == "__main__":
    main()
