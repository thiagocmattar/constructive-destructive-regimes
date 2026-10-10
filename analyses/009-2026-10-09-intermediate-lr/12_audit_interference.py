"""Audit full-prefix interaction from the seven verified scalar histories."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "complete-seven-lr"
SOURCE = OUT / "comparison_data.json"
EXPECTED_SHA = "a02a1ba7ace7a7011063767e6e7fb1c25548be0a12be06b6dad6826e5e7b2739"
STREAMS = {
    "raw": ("gradient_norm", "cumulative_gradient_norm"),
    "momentum": ("momentum_norm", "cumulative_momentum_norm"),
    "adaptive": ("direction_norm", "cumulative_direction_norm"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(SOURCE) == EXPECTED_SHA
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(SOURCE),
        "generator_sha256": sha(Path(__file__)),
        "definition": "Q=sum_t ||v_t||^2; J=(B^2-Q)/2=sum_{u<t} v_u dot v_t; I1=sum_{t>1} ||v_(t-1)|| ||v_t|| cosine_t",
        "coverage": "All 712 updates; separate endpoint at 711 excludes the partial batch. Each prefix starts at update 1, including the zero-LR first update.",
        "limits": "Exact aggregate identity evaluated from recorded FP64 scalar reductions; no individual longer-lag profile recovered; post-hoc, one seed, mixed hardware, coupled LR/decay.",
        "runs": {},
    }
    for rid, row in sorted(source["runs"].items(), key=lambda item: item[1]["lr"]):
        events = row["events"]
        assert [e["update"] for e in events] == list(range(1, 713))
        assert all(e["sequences"] == 1024 for e in events[:711])
        assert events[-1]["sequences"] == 311
        run = {
            "lr": row["lr"], "hardware": row["hardware"],
            "performance": row["performance"], "streams": {},
            "first_negative_raw_pair_update": next(e["update"] for e in events[1:] if e["adjacent_cosine"] < 0),
        }
        for stream, (norm_key, B_key) in STREAMS.items():
            values = events if stream == "raw" else [e[stream] for e in events]
            norms = np.array([e[norm_key] for e in values], dtype=np.float64)
            B = np.array([e[B_key] for e in values], dtype=np.float64)
            cosines = np.array([e["adjacent_cosine"] for e in values[1:]], dtype=np.float64)
            assert np.isfinite(norms).all() and np.isfinite(B).all() and np.isfinite(cosines).all()
            np.testing.assert_allclose(math.fsum(norms), row["geometry"][stream]["final_A"], rtol=1e-13)
            Q = np.cumsum(norms * norms)
            np.testing.assert_allclose(Q[-1], math.fsum(float(n)**2 for n in norms), rtol=1e-13)
            J = (B * B - Q) / 2
            I1 = math.fsum(float(a*b*c) for a,b,c in zip(norms[:-1],norms[1:],cosines))
            tolerance = 1e-12 * np.maximum(B*B + Q, 1)
            negative = np.flatnonzero(J < -tolerance)
            run["streams"][stream] = {
                **row["geometry"][stream],
                "Q_712": float(Q[-1]), "J_712": float(J[-1]),
                "J_711": float(J[-2]), "I1_712": I1,
                "interaction_lags_2_plus_712": float(J[-1] - I1),
                "B_squared_over_Q_712": float(B[-1]**2/Q[-1]),
                "first_negative_prefix_J_update": int(negative[0]+1) if len(negative) else None,
                "prefix_J": J.tolist(),
            }
        g, m = run["streams"]["raw"], run["streams"]["momentum"]
        run["momentum_A_reduction_vs_raw_percent"] = 100*(1-m["final_A"]/g["final_A"])
        run["momentum_B_reduction_vs_raw_percent"] = 100*(1-m["final_B"]/g["final_B"])
        # Equal-sized full updates have equal target counts; exclude partial 712.
        loss_mean = [math.fsum(e["loss"] for e in events[max(0,i-24):i+1])/len(events[max(0,i-24):i+1]) for i in range(711)]
        best = int(np.argmin(loss_mean))
        run["training_loss_trailing25_minimum"] = {"update": best+1, "loss": loss_mean[best]}
        result["runs"][rid] = run
    path = OUT / "interference_audit.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for rid,r in result["runs"].items():
        raw = r["streams"]["raw"]
        print(json.dumps({"id":rid,"lr":r["lr"],"J_712":raw["J_712"],"J_711":raw["J_711"],"first_negative_prefix":raw["first_negative_prefix_J_update"]}))


if __name__ == "__main__":
    main()
