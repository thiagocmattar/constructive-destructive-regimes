"""Run-local entry point for the authorized three-stream full pass."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
from baseline.training import main
if __name__ == "__main__":
    main()
