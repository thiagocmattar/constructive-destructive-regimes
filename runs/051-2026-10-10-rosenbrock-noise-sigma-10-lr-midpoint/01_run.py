"""Execute this frozen Gaussian-noise Rosenbrock condition."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.controlled.stochastic_rosenbrock import execute

if __name__ == "__main__":
    execute(Path(__file__).resolve().parent)
