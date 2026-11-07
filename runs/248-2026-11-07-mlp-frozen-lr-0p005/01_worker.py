"""I5: freeze linear1 from448 and change Adam LR to0.005 at500."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks.frozen_lr_continuation import run

if __name__ == "__main__":
    run(Path(__file__).resolve().parent)
