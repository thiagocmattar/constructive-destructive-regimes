"""Render the same nine-panel figure using the new paths' actual starting point."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.controlled.rosenbrock_plot import plot


if __name__ == "__main__":
    plot(HERE, title="Rosenbrock · AdamW with momentum, weight decay 0.1",
         labels=(r"$\eta=10^{-2}$", r"$\eta=3.1623\times10^{-3}$", r"$\eta=10^{-3}$"),
         colors=("#0072B2", "#8C4AA8", "#D55E00"), styles=("-", (0, (2, 2)), (0, (7, 3))),
         cosine_window=100, norm_scale="linear")
