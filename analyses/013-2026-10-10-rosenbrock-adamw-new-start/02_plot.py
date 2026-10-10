"""Render the trajectories and mark the initial downward-to-upward y-turn."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.controlled.rosenbrock_plot import plot


def trajectory_turns():
    points = []
    cohort = json.loads((HERE / "cohort.json").read_text())
    for entry in cohort:
        with np.load(ROOT / entry["directory"] / "artifacts/attempts/001-local/history.npz", allow_pickle=False) as saved:
            theta = saved["theta"]
            dy = np.diff(theta[:, 1])
            # theta[t] is the position after update t, not a gradient-history index.
            candidates = np.flatnonzero((dy[:-1] < 0) & (dy[1:] > 0)) + 1
            if not len(candidates):
                raise ValueError("No downward-to-upward y-turn: " + entry["run_id"])
            update = int(candidates[0])
            points.append(dict(run_id=entry["run_id"], learning_rate=entry["learning_rate"],
                               update=update, point=theta[update].tolist(),
                               incoming_delta_y=float(dy[update - 1]), outgoing_delta_y=float(dy[update])))
    return dict(definition="First strict local minimum of y: y[t] < y[t-1] and y[t] < y[t+1]",
                indexing="Position theta[t] after optimizer update t; no shift for vector or rolling panels",
                points=points)


if __name__ == "__main__":
    plot(HERE, title="Rosenbrock · AdamW with momentum, weight decay 0.1",
         labels=(r"$\eta=10^{-2}$", r"$\eta=3.1623\times10^{-3}$", r"$\eta=10^{-3}$"),
         colors=("#0072B2", "#8C4AA8", "#D55E00"), styles=("-", (0, (2, 2)), (0, (7, 3))),
         cosine_window=100, norm_scale="linear", trajectory_turns=trajectory_turns())
