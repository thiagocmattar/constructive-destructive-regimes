"""Execute each new frozen condition locally; deterministic controls are read-only."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.controlled.stochastic_rosenbrock import execute

if __name__ == "__main__":
    cohort = json.loads((HERE / "cohort.json").read_text())
    for entry in cohort:
        if not entry["reused"]:
            execute(ROOT / entry["directory"])
