"""Post-training figures and observations; not an executed training input."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from plot_scale_run import run_analysis
if __name__ == "__main__":
    run_analysis(HERE)

