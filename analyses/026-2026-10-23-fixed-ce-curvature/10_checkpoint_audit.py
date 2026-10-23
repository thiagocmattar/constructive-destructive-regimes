"""Exploratory same-state gradient audit from retained checkpoints; no updates.

Added after the first seed showed curvature separation without exact target
collapse. It tests pre-collapse gradient rounding and is not a primary endpoint.
"""
import json
import os
import sys
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from src.fixed_ce_curvature.core import (configure, fixed_ce, flat, initialize,
                                        make_data, save_json, stamp, tensor_hash)


def main():
    configure()
    here = Path(__file__).resolve().parent
    cohort = json.loads((here / "cohort.json").read_text())
    x, y, _, _ = make_data("cuda")
    records = []
    for run in cohort:
        model = initialize(999, "cuda")
        for step in (0, 9000, 9250, 9500, 9750, 10000):
            path = ROOT / "runs" / run["run"] / f"artifacts/attempt-01/states/state-{step:06d}.pt"
            saved = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["parameters"])
            before = tensor_hash(model.parameters())
            logits = model(x)
            z = logits.detach().requires_grad_(True)
            gradients = {}
            logit_gradients = {}
            for name, loss in (("native32", F.cross_entropy(logits, y)),
                               ("fixed32", fixed_ce(logits, y)),
                               ("ce64", F.cross_entropy(logits.double(), y))):
                gradients[name] = flat(torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)).detach().double()
            for name, loss in (("native32", F.cross_entropy(z, y, reduction="sum")),
                               ("fixed32", fixed_ce(z, y, reduction="sum")),
                               ("ce64", F.cross_entropy(z.double(), y, reduction="sum"))):
                logit_gradients[name] = torch.autograd.grad(loss, z, retain_graph=True)[0].detach().double()
            reference = gradients["ce64"]
            true_target = logit_gradients["ce64"].gather(1, y[:, None]).squeeze(1)
            native_target = logit_gradients["native32"].gather(1, y[:, None]).squeeze(1)
            target_error = (native_target - true_target).abs() / true_target.abs()
            record = {"run": run["run"], "seed": run["seed"], "trained_arm": run["arm"], "state": step,
                "native_parameter_gradient_relative_error": float((gradients["native32"] - reference).norm() / reference.norm()),
                "fixed_parameter_gradient_relative_error": float((gradients["fixed32"] - reference).norm() / reference.norm()),
                "native_logit_gradient_relative_error": float((logit_gradients["native32"] - logit_gradients["ce64"]).norm() / logit_gradients["ce64"].norm()),
                "fixed_logit_gradient_relative_error": float((logit_gradients["fixed32"] - logit_gradients["ce64"]).norm() / logit_gradients["ce64"].norm()),
                "native_target_relative_error_median": float(target_error.median()),
                "native_target_relative_error_q90": float(torch.quantile(target_error, .9)),
                "native_target_zero_fraction": float(((native_target == 0) & (true_target != 0)).double().mean()),
                "native_fixed_parameter_gradient_cosine": float(gradients["native32"].dot(gradients["fixed32"]) /
                    (gradients["native32"].norm() * gradients["fixed32"].norm()))}
            assert tensor_hash(model.parameters()) == before
            records.append(record)
    summary = {}
    for arm in ("native32", "fixed32", "ce64"):
        rows = [r for r in records if r["trained_arm"] == arm and r["state"] >= 9000]
        summary[arm] = {key: {"median": float(np.median([r[key] for r in rows])),
                              "min": min(r[key] for r in rows), "max": max(r[key] for r in rows)}
                       for key in ("native_parameter_gradient_relative_error", "fixed_parameter_gradient_relative_error",
                                   "native_logit_gradient_relative_error", "fixed_logit_gradient_relative_error",
                                   "native_target_relative_error_median", "native_target_zero_fraction")}
    save_json(here / "checkpoint_gradient_audit.json", {"utc": stamp(), "scope": "exploratory-posthoc-no-updates",
        "selection": "state0 and all five primary late states, all30 runs; added after first seed",
        "reference": "FP64 CE on the same FP32 logits; FP32 model/backprop in all three gradients",
        "does_not_establish": "causal mediation of trajectory differences or an exact full-FP64 parameter gradient",
        "records": records, "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
