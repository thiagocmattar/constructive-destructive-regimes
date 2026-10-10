"""Run-local entry point for the matched scale baseline."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from baseline.training import main
if __name__ == "__main__":
    main()
