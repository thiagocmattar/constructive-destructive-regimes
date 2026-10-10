"""Execute this frozen Rosenbrock condition using the shared implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.controlled.rosenbrock import execute

if __name__ == "__main__":
    execute(Path(__file__).resolve().parent)
