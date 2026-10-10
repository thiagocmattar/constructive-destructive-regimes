"""Run the independent Adam checks on CUDA with the production fused optimizer."""
from pathlib import Path
import runpy
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required for production optimizer audit.")
BaseAdamW = torch.optim.AdamW


class FusedAuditAdamW(BaseAdamW):
    def __init__(self, *args, **kwargs):
        kwargs["fused"] = True
        super().__init__(*args, **kwargs)


torch.optim.AdamW = FusedAuditAdamW
torch.set_default_device("cuda")
root = Path(__file__).resolve().parents[3]
runpy.run_path(str(root / "tests/test_run002_adaptive.py"), run_name="__main__")
